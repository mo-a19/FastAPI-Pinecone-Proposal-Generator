"""
Proposal Generation Routes

Single endpoint for intelligent proposal generation based on:
- New job description and skills
- Historical project data
- Portfolio links and feedback
- Previous successful proposals

Includes Outcome Tracking:
- Save sent proposals
- Track outcomes (viewed/hired)
- Conversion analytics

Authentication:
- PUBLIC: /generate endpoint (for index.html)
- PROTECTED: History, stats, save, update, delete endpoints
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from app.middleware.auth import verify_api_key
from app.utils.retrieval_pipeline import RetrievalPipeline
from app.utils.openai_service import OpenAIService
from app.utils.prompt_engine import PromptEngine
from app.db import get_db
from app.infra.mongodb.repositories.proposal_repo import get_sent_proposal_repo
from app.infra.mongodb.repositories.analytics_repo import get_analytics_repo
from app.config import settings

# Import services
from app.services.job_requirements_service import get_job_requirements_service, HookQuestion
from app.services.proposal_service import (
    ProposalService, 
    ProposalRequest as ServiceProposalRequest,
    get_proposal_service
)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/proposals", tags=["proposals"])


# ===================== ENUMS =====================

class ProposalOutcome(str, Enum):
    """Outcome of a sent proposal"""
    SENT = "sent"
    VIEWED = "viewed"
    HIRED = "hired"
    REJECTED = "rejected"


# ===================== REQUEST/RESPONSE MODELS =====================

class GenerateProposalRequest(BaseModel):
    """Request to generate a proposal based on new job details"""
    # Required job information
    job_title: str = Field(..., description="Job position title")
    company_name: str = Field(..., description="Client company name")
    job_description: str = Field(..., description="Full job description")
    skills_required: List[str] = Field(..., description="Required skills")
    
    # Optional job details
    industry: Optional[str] = Field(None, description="Industry category (e.g., SaaS, FinTech, HealthTech)")
    task_type: Optional[str] = Field(None, description="Type of task (e.g., full_stack, backend, frontend, mobile)")
    estimated_budget: Optional[float] = Field(None, description="Estimated budget in USD")
    project_duration_days: Optional[int] = Field(None, description="Project duration in days")
    urgent_adhoc: Optional[bool] = Field(False, description="Is this urgent/adhoc project?")
    
    # Multi-tenant context (optional for backward compatibility)
    org_id: Optional[str] = Field(None, description="Organization ID for multi-tenant data scoping")
    profile_id: Optional[str] = Field(None, description="Freelancer profile ID for personalization")
    
    # Proposal customization
    proposal_style: str = Field("professional", description="Style: professional, casual, technical, creative, data_driven")
    tone: str = Field("confident", description="Tone: confident, humble, enthusiastic, analytical, friendly")
    max_word_count: int = Field(300, ge=100, le=1500, description="Target proposal length in words (default 300, recommended 200-350)")
    
    # Timeline options
    include_timeline: bool = Field(False, description="Include timeline in proposal? If false, no timeline section added")
    timeline_duration: Optional[str] = Field(None, description="Custom timeline duration (e.g., '2-3 weeks', '1 month'). Only used if include_timeline is True")
    
    # Historical data options
    similar_projects_count: int = Field(3, ge=1, le=10, description="Number of similar past projects to reference")
    include_previous_proposals: bool = Field(True, description="Include analysis of previous proposals for similar jobs?")
    include_portfolio: bool = Field(True, description="Include portfolio links from similar projects?")
    include_feedback: bool = Field(True, description="Include client feedback/testimonials from similar projects?")
    
    # Hook question (user-selected opening question for the proposal)
    selected_hook_question: Optional[str] = Field(None, description="User-selected opening question to use as the proposal hook. If provided, this becomes the mandatory first line.")


class ProposalResponse(BaseModel):
    """Response with intelligently generated proposal"""
    success: bool
    job_title: str
    company_name: str
    generated_proposal: str
    word_count: int
    proposal_style: str
    proposal_tone: str
    
    # Historical context used
    similar_projects: List[Dict[str, Any]] = Field(description="Similar projects from history used for reference")
    previous_proposals_insights: Optional[Dict[str, Any]] = Field(None, description="Insights from previous proposals")
    portfolio_links_used: List[str] = Field(description="Portfolio links included in proposal")
    feedback_urls_used: List[str] = Field(description="Client feedback URLs included")
    
    # Retrieval insights
    insights: Optional[Dict[str, Any]] = Field(None, description="Success patterns and client values extracted from retrieval")
    
    # Quality metrics
    confidence_score: float = Field(description="Confidence score (0-1) for proposal quality")
    improvement_suggestions: Optional[List[str]] = Field(None, description="Suggestions for better proposals")
    
    metadata: Dict[str, Any] = Field(description="Additional metadata")


# ===================== MAIN ENDPOINT =====================

@router.post(
    "/generate",
    response_model=ProposalResponse,
    status_code=200,
    summary="Generate intelligent proposal from job details and history",
    responses={
        200: {"description": "Proposal generated successfully"},
        400: {"description": "Invalid job data"},
        500: {"description": "Generation error"}
    }
)
async def generate_proposal(request: GenerateProposalRequest):
    """
    Generate an intelligent AI proposal based on:
    
    1. **New Job Details** - Job title, description, skills, company
    2. **Historical Data** - Similar past projects from database
    3. **Previous Proposals** - What worked in similar situations
    4. **Portfolio & Feedback** - Links from successful similar projects
    5. **Style & Tone** - Customizable writing style and tone
    
    The AI will:
    - Find similar past projects
    - Analyze what made those proposals successful
    - Extract portfolio links and feedback URLs
    - Generate a new proposal leveraging all historical insights
    - Score proposal quality and suggest improvements
    
    **Request Example:**
    ```json
    {
        "job_title": "Full Stack Developer",
        "company_name": "TechStartup Inc",
        "job_description": "Build a modern web app with React and Node.js...",
        "skills_required": ["React", "Node.js", "MongoDB", "Docker"],
        "industry": "SaaS",
        "task_type": "full_stack",
        "proposal_style": "professional",
        "tone": "confident",
        "max_word_count": 500,
        "similar_projects_count": 3,
        "include_previous_proposals": true,
        "include_portfolio": true,
        "include_feedback": true
    }
    ```
    
    **Response Example:**
    ```json
    {
        "success": true,
        "job_title": "Full Stack Developer",
        "company_name": "TechStartup Inc",
        "generated_proposal": "Dear TechStartup Inc...",
        "word_count": 487,
        "proposal_style": "professional",
        "proposal_tone": "confident",
        "similar_projects": [...],
        "previous_proposals_insights": {...},
        "portfolio_links_used": ["https://..."],
        "feedback_urls_used": ["https://..."],
        "confidence_score": 0.92,
        "improvement_suggestions": [...],
        "metadata": {...}
    }
    ```
    """
    try:
        logger.info(f"[ProposalAPI] Generating proposal for {request.company_name} - {request.job_title}")
        
        # Initialize service and convert request
        proposal_service = _init_proposal_service()
        service_request = _convert_to_service_request(request)
        
        # Generate using service
        result = proposal_service.generate_proposal(service_request)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error_message)
        
        # Build quality score details
        quality_score_details = {"overall_score": result.confidence_score}
        
        return ProposalResponse(
            success=True,
            job_title=request.job_title,
            company_name=request.company_name,
            generated_proposal=result.generated_proposal,
            word_count=result.word_count,
            proposal_style=request.proposal_style,
            proposal_tone=request.tone,
            similar_projects=result.similar_projects,
            previous_proposals_insights=result.previous_proposals_insights,
            portfolio_links_used=result.portfolio_links_used,
            feedback_urls_used=result.feedback_urls_used,
            insights=result.insights,
            confidence_score=result.confidence_score,
            improvement_suggestions=result.improvement_suggestions,
            metadata={
                "similar_projects_used": len(result.similar_projects),
                "quality_score_details": quality_score_details
            }
        )
        
    except Exception as e:
        logger.error(f"[ProposalAPI] Error generating proposal: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate proposal: {str(e)}")


# ===================== HELPER FUNCTIONS =====================

# Service singletons to avoid re-initialization on every request
_cached_openai_service = None
_cached_pinecone_service = None
_cached_retrieval_pipeline = None
_cached_prompt_engine = None


def _init_proposal_service() -> ProposalService:
    """Initialize and configure ProposalService with cached dependencies."""
    global _cached_openai_service, _cached_pinecone_service, _cached_retrieval_pipeline, _cached_prompt_engine
    
    db = get_db()  # Already a singleton
    
    # Cache OpenAI service
    if _cached_openai_service is None:
        _cached_openai_service = OpenAIService(api_key=settings.OPENAI_API_KEY)
    
    # Cache Pinecone service
    if _cached_pinecone_service is None:
        from app.utils.pinecone_service import PineconeService
        _cached_pinecone_service = PineconeService(api_key=settings.PINECONE_API_KEY)
    
    # Cache retrieval pipeline
    if _cached_retrieval_pipeline is None:
        _cached_retrieval_pipeline = RetrievalPipeline(db, _cached_pinecone_service)
    
    # Cache prompt engine
    if _cached_prompt_engine is None:
        _cached_prompt_engine = PromptEngine()
    
    # JobRequirementsService uses its own singleton via get_job_requirements_service
    job_requirements_service = get_job_requirements_service(
        openai_service=_cached_openai_service, 
        db_manager=db
    )
    
    return get_proposal_service(
        db_manager=db,
        openai_service=_cached_openai_service,
        retrieval_pipeline=_cached_retrieval_pipeline,
        prompt_engine=_cached_prompt_engine,
        job_requirements_service=job_requirements_service
    )


def _convert_to_service_request(request: GenerateProposalRequest) -> ServiceProposalRequest:
    """Convert API request model to service request model."""
    return ServiceProposalRequest(
        job_title=request.job_title,
        company_name=request.company_name,
        job_description=request.job_description,
        skills_required=request.skills_required,
        industry=request.industry or "general",
        task_type=request.task_type or "other",
        proposal_style=request.proposal_style,
        tone=request.tone,
        max_word_count=request.max_word_count,
        include_timeline=request.include_timeline,
        timeline_duration=request.timeline_duration,
        similar_projects_count=request.similar_projects_count,
        include_previous_proposals=request.include_previous_proposals,
        include_portfolio=request.include_portfolio,
        include_feedback=request.include_feedback,
        org_id=request.org_id,
        profile_id=request.profile_id,
        selected_hook_question=request.selected_hook_question
    )


# ===================== SUGGEST HOOK QUESTIONS =====================

class SuggestQuestionsRequest(BaseModel):
    """Request to generate suggested hook questions from a job description"""
    job_description: str = Field(..., description="Full job description")
    job_title: Optional[str] = Field(None, description="Job title")
    skills_required: List[str] = Field(default_factory=list, description="Required skills")
    org_id: Optional[str] = Field(None, description="Organization ID")


class SuggestQuestionsResponse(BaseModel):
    """Response with suggested hook questions"""
    success: bool
    questions: List[Dict[str, Any]] = Field(description="List of suggested questions with category and reason")
    job_title: Optional[str] = None
    exact_task: Optional[str] = None


@router.post(
    "/suggest-questions",
    response_model=SuggestQuestionsResponse,
    summary="Generate suggested hook questions from job description",
)
async def suggest_hook_questions(request: SuggestQuestionsRequest):
    """
    Analyze a job description and return 3-4 suggested opening questions.
    
    These questions are generated from extracted job requirements and are
    designed to be used as the first line of a proposal to increase engagement.
    
    The user can select one of these or write their own, then pass it as
    `selected_hook_question` to the generate/generate-stream endpoints.
    """
    try:
        from app.utils.openai_service import OpenAIService
        from app.config import settings
        from app.db import get_db
        
        # Reuse cached services
        global _cached_openai_service
        if _cached_openai_service is None:
            _cached_openai_service = OpenAIService(api_key=settings.OPENAI_API_KEY)
        
        db = get_db()
        job_req_service = get_job_requirements_service(
            openai_service=_cached_openai_service, db_manager=db
        )
        
        # Step 1: Extract requirements (cached if already done)
        requirements = job_req_service.extract_job_requirements(
            job_description=request.job_description,
            job_title=request.job_title or "",
            skills_required=request.skills_required
        )
        
        # Step 2: Generate hook questions from requirements
        hook_questions = job_req_service.generate_hook_questions(
            requirements=requirements,
            job_description=request.job_description,
            job_title=request.job_title or ""
        )
        
        return SuggestQuestionsResponse(
            success=True,
            questions=[q.to_dict() for q in hook_questions],
            job_title=request.job_title,
            exact_task=requirements.exact_task[:200] if requirements.exact_task else None
        )
        
    except Exception as e:
        logger.error(f"[SuggestQuestions] Error: {e}", exc_info=True)
        return SuggestQuestionsResponse(
            success=False,
            questions=[],
            job_title=request.job_title
        )


# ===================== STREAMING PROPOSAL GENERATION =====================

from fastapi.responses import StreamingResponse
import asyncio
import json


# ===================== SIMILAR PAST PROPOSAL LOOKUP =====================

class SimilarJobRequest(BaseModel):
    """Request to find a similar past PROPOSAL for a given job description."""
    title: Optional[str] = Field("", description="Job title (free text)")
    description: str = Field(..., description="Job description (free text)")
    skills_required: List[str] = Field(default_factory=list, description="Skills (improves matching)")
    top_k: int = Field(3, ge=1, le=10, description="How many candidates to return")


@router.post("/similar-job")
async def find_similar_past_proposal(request: SimilarJobRequest):
    """
    Find a similar past PROPOSAL for the supplied job description.

    Source: cleaned historical (Job Post -> Proposal) corpus built from the
    Projects Data CSV (data/proposal_corpus.json). Matching uses OpenAI
    embeddings + cosine similarity (in-memory; corpus is small).
    """
    try:
        from app.services.proposal_corpus_service import ProposalCorpusService

        global _cached_openai_service
        if _cached_openai_service is None:
            _cached_openai_service = OpenAIService(api_key=settings.OPENAI_API_KEY)

        description = (request.description or "").strip()
        if not description:
            raise HTTPException(status_code=400, detail="description is required")

        service = ProposalCorpusService.get_instance(_cached_openai_service)
        result = service.find_similar(
            title=request.title or "",
            description=description,
            skills=request.skills_required or [],
            top_k=request.top_k,
        )

        if not result.get("has_match"):
            return {
                "success": True,
                "has_match": False,
                "best_match": None,
                "matches": result.get("matches", []),
                "message": "No similar past proposal found for this job description.",
            }

        return {
            "success": True,
            "has_match": True,
            "best_match": result["best_match"],
            "matches": result.get("matches", []),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SimilarProposal] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to find similar proposal: {e}")


@router.post("/generate-stream")
async def generate_proposal_stream(request: GenerateProposalRequest):
    """
    Generate proposal with real-time progress streaming via SSE.
    
    Shows step-by-step progress and streams the final proposal text.
    Uses parallel execution for extraction + retrieval (~2s saved).
    """
    async def event_generator():
        try:
            # Initialize service and convert request
            proposal_service = _init_proposal_service()
            service_request = _convert_to_service_request(request)
            
            # Use service streaming generator
            for event in proposal_service.generate_proposal_stream(service_request):
                yield event
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


# ===================== OUTCOME TRACKING MODELS =====================

class SaveProposalRequest(BaseModel):
    """Request to save a sent proposal for outcome tracking"""
    job_title: str = Field(..., description="Job title")
    company_name: str = Field(..., description="Company name")
    job_description: str = Field(..., description="Full job description")
    proposal_text: str = Field(..., description="The proposal text that was sent")
    skills_required: List[str] = Field(default_factory=list, description="Skills required for the job")
    industry: Optional[str] = Field("general", description="Industry")
    task_type: Optional[str] = Field("other", description="Task type")
    word_count: Optional[int] = Field(0, description="Word count")
    similar_projects_used: Optional[List[str]] = Field(default_factory=list, description="IDs of similar projects used")
    portfolio_links_used: Optional[List[str]] = Field(default_factory=list, description="Portfolio links included")
    confidence_score: Optional[float] = Field(0.0, description="AI confidence score")
    source: Optional[str] = Field("ai_generated", description="Source: ai_generated or manual")


class SaveProposalResponse(BaseModel):
    """Response after saving a proposal"""
    success: bool
    proposal_id: str = Field(..., description="Unique proposal ID for tracking")
    message: str
    sent_at: str


class UpdateOutcomeRequest(BaseModel):
    """Request to update proposal outcome"""
    outcome: ProposalOutcome = Field(..., description="New outcome: viewed, hired, or rejected")
    discussion_initiated: bool = Field(False, description="Did client start a discussion? (validates Message-Market Fit)")
    rejection_reason: Optional[str] = Field(None, description="Reason if rejected (budget, timing, etc.)")


class UpdateOutcomeResponse(BaseModel):
    """Response after updating outcome"""
    success: bool
    proposal_id: str
    outcome: str
    discussion_initiated: bool
    message: str


class ProposalHistoryItem(BaseModel):
    """Single item in proposal history"""
    proposal_id: str
    job_title: str
    company_name: str
    outcome: str
    sent_at: str
    viewed_at: Optional[str] = None
    hired_at: Optional[str] = None
    discussion_initiated: bool
    word_count: int
    skills_required: List[str]


class ProposalHistoryResponse(BaseModel):
    """Response with proposal history"""
    success: bool
    total: int
    items: List[ProposalHistoryItem]
    stats: Dict[str, Any]


class ConversionStatsResponse(BaseModel):
    """Response with conversion statistics"""
    success: bool
    total_sent: int
    total_viewed: int
    total_hired: int
    total_discussions: int
    view_rate: float = Field(..., description="% of proposals that were viewed")
    hire_rate: float = Field(..., description="% of proposals that led to hire")
    discussion_rate: float = Field(..., description="% that started discussions")
    message_market_fit: float = Field(..., description="% of viewed that started discussions (validates Hook+Approach)")
    view_to_hire_rate: float = Field(..., description="% of viewed that got hired")


# ===================== OUTCOME TRACKING ENDPOINTS =====================

@router.post(
    "/save",
    response_model=SaveProposalResponse,
    status_code=201,
    summary="Save a sent proposal for outcome tracking",
    responses={
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"}
    }
)
async def save_sent_proposal(
    request: SaveProposalRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Save a proposal that you've sent to a client.
    
    This enables tracking the outcome (viewed → hired) and learning from
    which proposals work best for future AI improvements.
    
    **Outcome Flow:**
    1. `sent` - Initial status when saved
    2. `viewed` - Client opened/viewed the proposal  
    3. `hired` - Client hired you! 🎉
    4. `rejected` - Client passed (track reason for learning)
    
    **Why track 'viewed'?**
    A viewed proposal validates your Hook and Approach worked.
    Even if not hired (budget/timing), the proposal text is proven effective.
    """
    try:
        repo = get_sent_proposal_repo()
        
        result = repo.save_sent_proposal({
            "job_title": request.job_title,
            "company_name": request.company_name,
            "job_description": request.job_description,
            "proposal_text": request.proposal_text,
            "skills_required": request.skills_required,
            "industry": request.industry,
            "task_type": request.task_type,
            "word_count": request.word_count,
            "similar_projects_used": request.similar_projects_used,
            "portfolio_links_used": request.portfolio_links_used,
            "confidence_score": request.confidence_score,
            "source": request.source
        })
        
        return SaveProposalResponse(
            success=True,
            proposal_id=result["proposal_id"],
            message="Proposal saved! Update the outcome when client views or hires.",
            sent_at=result["sent_at"]
        )
        
    except Exception as e:
        logger.error(f"Error saving proposal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/{proposal_id}/outcome",
    response_model=UpdateOutcomeResponse,
    summary="Update proposal outcome (viewed/hired/rejected)",
    responses={
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"}
    }
)
async def update_proposal_outcome(
    proposal_id: str,
    request: UpdateOutcomeRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Update the outcome of a sent proposal.
    
    **Outcomes:**
    - `viewed` - Client viewed your proposal (Hook worked! ✅)
    - `hired` - You got the job! 🎉
    - `rejected` - Client passed (provide reason for AI learning)
    
    **Discussion Initiated:**
    Set to `true` if client started a chat/message. This validates:
    - Hook was strong enough to click
    - Approach was professional
    - Pain point addressed correctly
    
    Even if not hired (budget/timing), these proposals have proven
    **Message-Market Fit** and should be weighted highly by AI.
    """
    try:
        repo = get_sent_proposal_repo()
        
        result = repo.update_outcome(
            proposal_id=proposal_id,
            outcome=request.outcome.value,
            discussion_initiated=request.discussion_initiated,
            rejection_reason=request.rejection_reason
        )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
        
        outcome_messages = {
            "viewed": "Great! Client viewed your proposal - Hook worked! 🎯",
            "hired": "Congratulations! You got the job! 🎉",
            "rejected": "Noted. We'll learn from this for better future proposals."
        }
        
        return UpdateOutcomeResponse(
            success=True,
            proposal_id=proposal_id,
            outcome=result["outcome"],
            discussion_initiated=result["discussion_initiated"],
            message=outcome_messages.get(result["outcome"], "Outcome updated")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating outcome: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/history",
    response_model=ProposalHistoryResponse,
    summary="Get sent proposals history with outcomes",
    responses={
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"}
    }
)
async def get_proposal_history(
    skip: int = 0,
    limit: int = 50,
    outcome: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get history of all sent proposals with their outcomes.
    
    Filter by outcome:
    - `sent` - Waiting for response
    - `viewed` - Client viewed (Message-Market Fit validated)
    - `hired` - Converted to job
    - `rejected` - Client passed
    
    Use this to:
    1. Track your proposal performance
    2. Update outcomes when you hear back
    3. Identify winning proposal patterns
    """
    try:
        repo = get_sent_proposal_repo()
        analytics = get_analytics_repo()
        
        proposals = repo.get_sent_proposals(skip=skip, limit=limit, outcome_filter=outcome)
        stats = analytics.get_conversion_stats()
        
        items = []
        for p in proposals:
            items.append(ProposalHistoryItem(
                proposal_id=p["proposal_id"],
                job_title=p.get("job_title", ""),
                company_name=p.get("company_name", ""),
                outcome=p.get("outcome", "sent"),
                sent_at=p.get("sent_at", datetime.utcnow()).isoformat() if p.get("sent_at") else "",
                viewed_at=p.get("viewed_at").isoformat() if p.get("viewed_at") else None,
                hired_at=p.get("hired_at").isoformat() if p.get("hired_at") else None,
                discussion_initiated=p.get("discussion_initiated", False),
                word_count=p.get("word_count", 0),
                skills_required=p.get("skills_required", [])
            ))
        
        return ProposalHistoryResponse(
            success=True,
            total=len(items),
            items=items,
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stats",
    response_model=ConversionStatsResponse,
    summary="Get proposal conversion statistics",
    responses={
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"}
    }
)
async def get_conversion_stats(
    api_key: str = Depends(verify_api_key)
):
    """
    Get conversion statistics for all sent proposals.
    
    **Key Metrics:**
    - `view_rate` - % of proposals clients opened
    - `hire_rate` - % that converted to jobs
    - `message_market_fit` - % of viewed that started discussions
    - `view_to_hire_rate` - Conversion from view to hire
    
    **Message-Market Fit** is crucial:
    If a client views and starts a discussion, your proposal text works.
    Not closing (budget/timing) is a logistics issue, not communication.
    """
    try:
        analytics = get_analytics_repo()
        stats = analytics.get_conversion_stats()
        
        return ConversionStatsResponse(
            success=True,
            **stats
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{proposal_id}",
    summary="Get a single sent proposal by ID",
    responses={
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"}
    }
)
async def get_sent_proposal(
    proposal_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get details of a specific sent proposal"""
    try:
        repo = get_sent_proposal_repo()
        proposal = repo.get_by_proposal_id(proposal_id)
        
        if not proposal:
            raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
        
        # Format dates
        for date_field in ["sent_at", "viewed_at", "hired_at", "outcome_updated_at", "created_at"]:
            if proposal.get(date_field):
                proposal[date_field] = proposal[date_field].isoformat()
        
        return {"success": True, "proposal": proposal}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting proposal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/{proposal_id}",
    summary="Delete a single sent proposal",
    responses={
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"}
    }
)
async def delete_proposal(
    proposal_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete a specific sent proposal by ID"""
    try:
        repo = get_sent_proposal_repo()
        deleted = repo.delete_one({"proposal_id": proposal_id})
        
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
        
        return {
            "success": True,
            "proposal_id": proposal_id,
            "message": "Proposal deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting proposal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/clear-all",
    summary="Delete all sent proposals (for testing)",
    responses={
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"}
    }
)
async def clear_all_proposals(
    api_key: str = Depends(verify_api_key)
):
    """Delete all sent proposals from database"""
    try:
        repo = get_sent_proposal_repo()
        deleted_count = repo.delete_all()
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} proposals"
        }
    except Exception as e:
        logger.error(f"Error clearing proposals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
"""
Prompt Lab Routes

Experimentation interface for testing prompt variations:
- Write custom instructions in plain English for each section
- Preview assembled prompt before generation
- Generate proposals from custom instructions
- Save/load prompt templates
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from bson import ObjectId

from app.db import get_db
from app.utils.openai_service import OpenAIService
from app.utils.pinecone_service import PineconeService
from app.utils.retrieval_pipeline import RetrievalPipeline
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/lab", tags=["lab"])

# Initialize services (lazy loading to avoid startup errors)
_db = None
_openai_service = None
_pinecone_service = None
_retrieval_pipeline = None


def get_services():
    """Lazy load services"""
    global _db, _openai_service, _pinecone_service, _retrieval_pipeline
    
    if _db is None:
        _db = get_db()
    
    if _openai_service is None:
        _openai_service = OpenAIService(
            api_key=settings.OPENAI_API_KEY,
            embedding_model=settings.OPENAI_EMBEDDING_MODEL,
            llm_model=settings.OPENAI_LLM_MODEL
        )
    
    if _pinecone_service is None:
        _pinecone_service = PineconeService(api_key=settings.PINECONE_API_KEY)
    
    if _retrieval_pipeline is None:
        _retrieval_pipeline = RetrievalPipeline(_db, _pinecone_service)
    
    return _db, _openai_service, _retrieval_pipeline


def _get_portfolio_urls(project: Dict[str, Any]) -> List[str]:
    """Extract ALL portfolio URLs from project - handles both list and string formats."""
    urls = []
    
    # Try portfolio_urls (list) first
    portfolio_urls = project.get("portfolio_urls", [])
    if portfolio_urls:
        if isinstance(portfolio_urls, list):
            urls.extend([u for u in portfolio_urls if u])
        elif isinstance(portfolio_urls, str) and portfolio_urls:
            urls.append(portfolio_urls)
    
    # Also check singular fields
    for field in ["portfolio_url", "portfolio_link"]:
        val = project.get(field)
        if val and val not in urls:
            urls.append(val)
    
    return urls if urls else ["N/A"]


def _get_portfolio_items_as_jobs(db) -> List[Dict[str, Any]]:
    """Convert portfolio items to job format for retrieval compatibility."""
    try:
        items = list(db.db.portfolio_items.find({}))
        return [
            {
                "contract_id": item.get("item_id"),
                "job_title": item.get("project_title", ""),
                "skills_required": item.get("skills", []),
                "industry": item.get("industry", "general"),
                "deliverables": item.get("deliverables", []),
                "outcome": item.get("outcome"),
                "company_name": item.get("project_title", ""),
                "portfolio_urls": [item.get("portfolio_url")] if item.get("portfolio_url") else [],
                "portfolio_url": item.get("portfolio_url"),
                "client_feedback_text": item.get("client_feedback", ""),
                "duration_days": item.get("duration_days"),
                "project_status": "completed",
            }
            for item in items
        ]
    except Exception as e:
        logger.error(f"Error fetching portfolio items: {e}")
        return []


# ===================== REQUEST/RESPONSE MODELS =====================

class SectionInstructions(BaseModel):
    """Plain English instructions for each proposal section"""
    hook: str = Field(
        default="Start with a compelling opening that shows you understand their problem. Reference a similar project you've done.",
        description="How should the hook/opening be written?"
    )
    proof: str = Field(
        default="Mention 1-2 relevant projects with specific outcomes and metrics.",
        description="How should you prove your expertise?"
    )
    approach: str = Field(
        default="Briefly explain your approach to solving their specific problem.",
        description="How should you describe your approach?"
    )
    cta: str = Field(
        default="End with a casual call to action like 'Happy to chat more about this.'",
        description="How should the proposal end?"
    )
    general: str = Field(
        default="Keep it under 300 words. Be conversational, not formal. No markdown formatting.",
        description="Any general instructions?"
    )


class PromptTemplate(BaseModel):
    """Saved prompt template"""
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    instructions: SectionInstructions
    is_default: bool = Field(False, description="Is this the default template?")


class PreviewPromptRequest(BaseModel):
    """Request to preview assembled prompt"""
    job_title: str
    job_description: str
    company_name: str = "Client"
    skills_required: List[str] = []
    instructions: SectionInstructions
    include_similar_projects: bool = Field(True, description="Include similar projects from portfolio")
    similar_projects_count: int = Field(3, ge=1, le=5)


class GenerateFromLabRequest(BaseModel):
    """Request to generate proposal from lab"""
    job_title: str
    job_description: str
    company_name: str = "Client"
    skills_required: List[str] = []
    instructions: SectionInstructions
    include_similar_projects: bool = True
    similar_projects_count: int = 3
    max_words: int = Field(300, ge=100, le=500)


class TemplateResponse(BaseModel):
    """Response for template operations"""
    id: str
    name: str
    description: Optional[str]
    instructions: SectionInstructions
    is_default: bool
    created_at: str


# ===================== HELPER FUNCTIONS =====================

def build_lab_prompt(
    job_data: Dict[str, Any],
    instructions: SectionInstructions,
    similar_projects: List[Dict[str, Any]] = None,
    max_words: int = 300
) -> str:
    """Build prompt from plain English instructions"""
    
    # Job context
    job_section = f"""
## JOB DETAILS
Title: {job_data.get('job_title', 'N/A')}
Company: {job_data.get('company_name', 'Client')}
Skills: {', '.join(job_data.get('skills_required', []))}

Description:
{job_data.get('job_description', '')}
"""
    
    # Similar projects section
    projects_section = ""
    if similar_projects:
        projects_section = "\n## YOUR RELEVANT PROJECTS (use these for proof)\n"
        for i, proj in enumerate(similar_projects[:3], 1):
            portfolio_urls = _get_portfolio_urls(proj)
            urls_str = ", ".join(portfolio_urls) if portfolio_urls else "N/A"
            projects_section += f"""
Project {i}: {proj.get('title', proj.get('project_title', proj.get('job_title', 'Project')))}
- Company: {proj.get('company', proj.get('company_name', 'N/A'))}
- Outcome: {proj.get('outcome', proj.get('project_outcome', 'Completed successfully'))}
- URLs: {urls_str}
"""
    
    # Build the prompt with user's plain English instructions
    prompt = f"""You are writing a freelance proposal. Follow these instructions EXACTLY.

{job_section}
{projects_section}

## YOUR INSTRUCTIONS (Follow these precisely)

### HOOK (Opening)
{instructions.hook}

### PROOF (Show expertise)
{instructions.proof}

### APPROACH (Your solution)
{instructions.approach}

### CTA (Call to action)
{instructions.cta}

### GENERAL RULES
{instructions.general}

## CRITICAL RULES
- Target length: {max_words} words
- Write in PLAIN TEXT only - NO markdown (**bold**, *italic*, bullets)
- NO section headers in output - write natural flowing paragraphs
- Sound like a REAL human freelancer, not AI
- ONLY reference projects listed above - do NOT invent projects
- Use actual URLs from projects above if mentioned

Now write the proposal following ALL instructions above.
"""
    return prompt


# ===================== ENDPOINTS =====================

@router.get("/templates", response_model=List[TemplateResponse])
async def get_templates():
    """Get all saved prompt templates"""
    try:
        db, _, _ = get_services()
        templates = list(db.db["prompt_templates"].find().sort("created_at", -1))
        return [
            TemplateResponse(
                id=str(t["_id"]),
                name=t["name"],
                description=t.get("description"),
                instructions=SectionInstructions(**t["instructions"]),
                is_default=t.get("is_default", False),
                created_at=t.get("created_at", datetime.utcnow()).isoformat()
            )
            for t in templates
        ]
    except Exception as e:
        logger.error(f"Error fetching templates: {e}")
        return []


@router.post("/templates", response_model=TemplateResponse)
async def save_template(template: PromptTemplate):
    """Save a new prompt template"""
    try:
        db, _, _ = get_services()
        doc = {
            "name": template.name,
            "description": template.description,
            "instructions": template.instructions.dict(),
            "is_default": template.is_default,
            "created_at": datetime.utcnow()
        }
        
        # If setting as default, unset other defaults
        if template.is_default:
            db.db["prompt_templates"].update_many({}, {"$set": {"is_default": False}})
        
        result = db.db["prompt_templates"].insert_one(doc)
        
        return TemplateResponse(
            id=str(result.inserted_id),
            name=template.name,
            description=template.description,
            instructions=template.instructions,
            is_default=template.is_default,
            created_at=doc["created_at"].isoformat()
        )
    except Exception as e:
        logger.error(f"Error saving template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """Delete a prompt template"""
    try:
        db, _, _ = get_services()
        result = db.db["prompt_templates"].delete_one({"_id": ObjectId(template_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Template not found")
        return {"success": True, "message": "Template deleted"}
    except Exception as e:
        logger.error(f"Error deleting template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preview-prompt")
async def preview_prompt(request: PreviewPromptRequest):
    """Preview the assembled prompt before generation"""
    try:
        db, openai_service, retrieval_pipeline = get_services()
        
        job_data = {
            "job_title": request.job_title,
            "job_description": request.job_description,
            "company_name": request.company_name,
            "skills_required": request.skills_required
        }
        
        # Get similar projects if requested
        similar_projects = []
        if request.include_similar_projects:
            try:
                # Fetch historical jobs from portfolio_items
                all_jobs = _get_portfolio_items_as_jobs(db)
                if all_jobs and retrieval_pipeline:
                    results = retrieval_pipeline.retrieve_for_proposal(
                        new_job_requirements=job_data,
                        all_jobs=all_jobs,
                        top_k=request.similar_projects_count
                    )
                    similar_projects = results.get("similar_projects", [])
            except Exception as e:
                logger.warning(f"Could not retrieve similar projects: {e}")
        
        # Build prompt
        prompt = build_lab_prompt(
            job_data=job_data,
            instructions=request.instructions,
            similar_projects=similar_projects
        )
        
        return {
            "prompt": prompt,
            "similar_projects_found": len(similar_projects),
            "similar_projects": [
                {
                    "title": p.get("title", p.get("project_title", p.get("job_title", "Project"))),
                    "company": p.get("company", p.get("company_name", "N/A")),
                    "urls": _get_portfolio_urls(p)
                }
                for p in similar_projects
            ]
        }
    except Exception as e:
        logger.error(f"Error previewing prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_from_lab(request: GenerateFromLabRequest):
    """Generate proposal using custom instructions"""
    try:
        db, openai_service, retrieval_pipeline = get_services()
        
        job_data = {
            "job_title": request.job_title,
            "job_description": request.job_description,
            "company_name": request.company_name,
            "skills_required": request.skills_required
        }
        
        # Get similar projects
        similar_projects = []
        if request.include_similar_projects:
            try:
                # Fetch historical jobs from portfolio_items
                all_jobs = _get_portfolio_items_as_jobs(db)
                if all_jobs and retrieval_pipeline:
                    results = retrieval_pipeline.retrieve_for_proposal(
                        new_job_requirements=job_data,
                        all_jobs=all_jobs,
                        top_k=request.similar_projects_count
                    )
                    similar_projects = results.get("similar_projects", [])
            except Exception as e:
                logger.warning(f"Could not retrieve similar projects: {e}")
        
        # Build prompt
        prompt = build_lab_prompt(
            job_data=job_data,
            instructions=request.instructions,
            similar_projects=similar_projects,
            max_words=request.max_words
        )
        
        # Generate with OpenAI
        response = openai_service.client.chat.completions.create(
            model=openai_service.llm_model,
            messages=[
                {"role": "system", "content": "You are an expert freelance proposal writer. Follow all instructions precisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        proposal = response.choices[0].message.content.strip()
        word_count = len(proposal.split())
        
        return {
            "proposal": proposal,
            "word_count": word_count,
            "prompt_used": prompt,
            "similar_projects": [
                {
                    "title": p.get("title", p.get("project_title", p.get("job_title", "Project"))),
                    "company": p.get("company", p.get("company_name", "N/A")),
                    "urls": _get_portfolio_urls(p)
                }
                for p in similar_projects
            ]
        }
    except Exception as e:
        logger.error(f"Error generating proposal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/default-instructions")
async def get_default_instructions():
    """Get default section instructions"""
    return SectionInstructions().dict()

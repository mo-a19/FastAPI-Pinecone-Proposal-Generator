"""
Job Ingestion Service - Clean module for ingesting Upwork jobs.

This service handles:
- Parsing job postings from Upwork
- Converting to portfolio_items format
- Triggering embeddings for retrieval
- Tracking job history for analytics

Part of the new architecture (replaces old 5-chunk strategy).
"""
import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator

from app.infra.mongodb.repositories.portfolio_repo import PortfolioRepository, get_portfolio_repo
from app.services.embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


# ===================== INPUT SCHEMAS =====================

class JobIngestionRequest(BaseModel):
    """Input for ingesting a completed job/project."""
    
    # Required tenant context
    org_id: str = Field(..., description="Organization ID")
    profile_id: str = Field(..., description="Freelancer profile ID")
    
    # Project data (maps to portfolio_items)
    project_title: str = Field(..., min_length=2, max_length=200)
    job_description: str = Field(..., min_length=10, description="Original job posting")
    proposal_text: str = Field(..., min_length=10, description="Submitted proposal")
    skills: List[str] = Field(..., min_length=1)
    
    # Optional enrichment
    client_name: Optional[str] = None
    client_feedback: Optional[str] = None
    portfolio_url: Optional[str] = None
    industry: str = Field(default="general")
    hourly_rate: Optional[float] = None
    fixed_price: Optional[float] = None
    duration_days: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    @validator("skills", pre=True)
    def clean_skills(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return [s.strip() for s in v if s.strip()][:15]


class JobIngestionResult(BaseModel):
    """Result from job ingestion."""
    success: bool
    item_id: Optional[str] = None
    embedded: bool = False
    errors: List[str] = Field(default_factory=list)


# ===================== SERVICE =====================

class JobIngestionService:
    """
    Service for ingesting completed jobs into the portfolio system.
    
    Converts job history into portfolio_items for retrieval-augmented
    proposal generation.
    
    Usage:
        service = JobIngestionService()
        result = service.ingest(JobIngestionRequest(...))
    """
    
    def __init__(
        self,
        portfolio_repo: PortfolioRepository = None,
        embedding_service: EmbeddingService = None
    ):
        self.portfolio_repo = portfolio_repo or get_portfolio_repo()
        self.embedding_service = embedding_service or get_embedding_service()
    
    def ingest(
        self,
        request: JobIngestionRequest,
        auto_embed: bool = True
    ) -> JobIngestionResult:
        """
        Ingest a completed job as a portfolio item.
        
        Extracts deliverables and outcome from job description + proposal,
        then stores as a lean portfolio item for future retrieval.
        """
        errors = []
        
        try:
            # Extract deliverables from job description
            deliverables = self._extract_deliverables(
                request.job_description,
                request.project_title
            )
            
            # Extract outcome from proposal/feedback
            outcome = self._extract_outcome(
                request.proposal_text,
                request.client_feedback
            )
            
            # Calculate duration if dates provided
            duration = request.duration_days
            if not duration and request.start_date and request.end_date:
                duration = self._calculate_duration(
                    request.start_date,
                    request.end_date
                )
            
            # Create portfolio item
            result = self.portfolio_repo.create(
                org_id=request.org_id,
                profile_id=request.profile_id,
                project_title=request.project_title,
                deliverables=deliverables,
                skills=request.skills,
                outcome=outcome,
                portfolio_url=request.portfolio_url,
                industry=request.industry,
                client_feedback=request.client_feedback,
                duration_days=duration
            )
            
            item_id = result.get("item_id")
            if not item_id:
                return JobIngestionResult(
                    success=False,
                    errors=["Failed to create portfolio item"]
                )
            
            # Embed for retrieval
            embedded = False
            if auto_embed and self.embedding_service:
                try:
                    embed_result = self.embedding_service.embed_portfolio_item(item_id)
                    embedded = embed_result.get("success", False)
                except Exception as e:
                    logger.warning(f"Embedding failed for {item_id}: {e}")
                    errors.append(f"Embedding deferred: {str(e)}")
            
            logger.info(f"Ingested job: {request.project_title} -> {item_id}")
            
            return JobIngestionResult(
                success=True,
                item_id=item_id,
                embedded=embedded,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return JobIngestionResult(
                success=False,
                errors=[str(e)]
            )
    
    def ingest_bulk(
        self,
        requests: List[JobIngestionRequest],
        auto_embed: bool = True
    ) -> Dict[str, Any]:
        """Ingest multiple jobs."""
        results = []
        success_count = 0
        
        for req in requests:
            result = self.ingest(req, auto_embed=auto_embed)
            results.append(result.model_dump())
            if result.success:
                success_count += 1
        
        return {
            "total": len(requests),
            "succeeded": success_count,
            "failed": len(requests) - success_count,
            "results": results
        }
    
    def _extract_deliverables(
        self,
        job_description: str,
        title: str
    ) -> List[str]:
        """Extract key deliverables from job description."""
        deliverables = []
        
        # Use title as primary deliverable
        if title:
            deliverables.append(title)
        
        # Look for bullet points or numbered items
        lines = job_description.split("\n")
        for line in lines:
            line = line.strip()
            # Match bullets, dashes, numbers
            if re.match(r'^[-•*]\s*\w', line) or re.match(r'^\d+[.)]\s*\w', line):
                clean = re.sub(r'^[-•*\d.)\s]+', '', line).strip()
                if 10 < len(clean) < 100:
                    deliverables.append(clean)
        
        # Look for "deliverables:" section
        desc_lower = job_description.lower()
        if "deliverable" in desc_lower:
            idx = desc_lower.index("deliverable")
            section = job_description[idx:idx+500]
            for line in section.split("\n")[1:5]:
                line = line.strip()
                if line and len(line) > 10:
                    deliverables.append(line[:100])
        
        # Deduplicate and limit
        seen = set()
        unique = []
        for d in deliverables:
            if d.lower() not in seen:
                seen.add(d.lower())
                unique.append(d)
        
        return unique[:5]  # Max 5 deliverables
    
    def _extract_outcome(
        self,
        proposal: str,
        feedback: Optional[str]
    ) -> str:
        """Extract outcome from proposal or feedback."""
        # Prefer client feedback as proof of outcome
        if feedback and len(feedback) > 20:
            return feedback[:300]
        
        # Extract from proposal - look for results/outcome language
        outcome_patterns = [
            r"(?:result|outcome|deliver|achieve|accomplish)[:\s]+(.{30,200})",
            r"(?:you will|you'll|we will|we'll|I will|I'll)\s+(?:get|receive|have)\s+(.{30,200})",
        ]
        
        for pattern in outcome_patterns:
            match = re.search(pattern, proposal, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:200]
        
        # Fallback: first sentence of proposal
        sentences = re.split(r'[.!?]+', proposal)
        if sentences:
            return sentences[0].strip()[:200]
        
        return "Project completed successfully"
    
    def _calculate_duration(self, start: str, end: str) -> int:
        """Calculate duration in days from date strings."""
        try:
            # Try common date formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
                try:
                    start_dt = datetime.strptime(start, fmt)
                    end_dt = datetime.strptime(end, fmt)
                    days = (end_dt - start_dt).days
                    return max(1, days)
                except ValueError:
                    continue
        except:
            pass
        return None


# ===================== FACTORY =====================

_service_instance: JobIngestionService = None

def get_job_ingestion_service() -> JobIngestionService:
    """Get singleton job ingestion service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = JobIngestionService()
    return _service_instance

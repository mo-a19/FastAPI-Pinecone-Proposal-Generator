"""
Job Ingestion API Routes (v2)

Clean endpoints for job ingestion using the new portfolio-based architecture.
Works with multi-tenant organization model.

Endpoints:
- POST /api/jobs/ingest - Ingest a single job
- POST /api/jobs/ingest/bulk - Ingest multiple jobs
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List
from pydantic import BaseModel, Field

from app.middleware.auth import verify_api_key
from app.services.job_ingestion_service import (
    JobIngestionService,
    JobIngestionRequest,
    JobIngestionResult,
    get_job_ingestion_service
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ===================== REQUEST/RESPONSE MODELS =====================

class IngestJobRequest(BaseModel):
    """Request to ingest a completed job."""
    org_id: str = Field(..., description="Organization ID")
    profile_id: str = Field(..., description="Freelancer profile ID")
    project_title: str = Field(..., min_length=2, max_length=200)
    job_description: str = Field(..., min_length=10)
    proposal_text: str = Field(..., min_length=10)
    skills: List[str] = Field(..., min_length=1)
    
    # Optional
    client_name: str = None
    client_feedback: str = None
    portfolio_url: str = None
    industry: str = "general"
    duration_days: int = None
    start_date: str = None
    end_date: str = None


class IngestResponse(BaseModel):
    """Response from job ingestion."""
    success: bool
    item_id: str = None
    embedded: bool = False
    message: str = None


class BulkIngestResponse(BaseModel):
    """Response from bulk ingestion."""
    total: int
    succeeded: int
    failed: int


# ===================== ENDPOINTS =====================

@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_job(
    request: IngestJobRequest,
    auto_embed: bool = Query(True, description="Automatically embed for retrieval"),
    _: str = Depends(verify_api_key)
):
    """
    Ingest a completed job into the portfolio system.
    
    Converts job history into a portfolio item for RAG-based proposal generation.
    Extracts deliverables and outcomes automatically from job description and proposal.
    """
    try:
        service = get_job_ingestion_service()
        
        # Convert to service request
        ingestion_request = JobIngestionRequest(
            org_id=request.org_id,
            profile_id=request.profile_id,
            project_title=request.project_title,
            job_description=request.job_description,
            proposal_text=request.proposal_text,
            skills=request.skills,
            client_name=request.client_name,
            client_feedback=request.client_feedback,
            portfolio_url=request.portfolio_url,
            industry=request.industry,
            duration_days=request.duration_days,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        result = service.ingest(ingestion_request, auto_embed=auto_embed)
        
        if not result.success:
            raise HTTPException(400, result.errors[0] if result.errors else "Ingestion failed")
        
        return IngestResponse(
            success=True,
            item_id=result.item_id,
            embedded=result.embedded,
            message="Job ingested successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job ingestion error: {e}")
        raise HTTPException(500, str(e))


@router.post("/ingest/bulk", response_model=BulkIngestResponse)
async def ingest_jobs_bulk(
    requests: List[IngestJobRequest],
    auto_embed: bool = Query(True, description="Automatically embed for retrieval"),
    _: str = Depends(verify_api_key)
):
    """
    Ingest multiple jobs at once.
    
    Useful for importing job history in bulk.
    """
    try:
        service = get_job_ingestion_service()
        
        # Convert to service requests
        ingestion_requests = [
            JobIngestionRequest(
                org_id=r.org_id,
                profile_id=r.profile_id,
                project_title=r.project_title,
                job_description=r.job_description,
                proposal_text=r.proposal_text,
                skills=r.skills,
                client_name=r.client_name,
                client_feedback=r.client_feedback,
                portfolio_url=r.portfolio_url,
                industry=r.industry,
                duration_days=r.duration_days,
                start_date=r.start_date,
                end_date=r.end_date
            )
            for r in requests
        ]
        
        result = service.ingest_bulk(ingestion_requests, auto_embed=auto_embed)
        
        return BulkIngestResponse(
            total=result["total"],
            succeeded=result["succeeded"],
            failed=result["failed"]
        )
        
    except Exception as e:
        logger.error(f"Bulk ingestion error: {e}")
        raise HTTPException(500, str(e))

"""
Portfolio CRUD Routes

Manage portfolio items (past projects) for proposal generation.
Each item = 1 vector in Pinecone for semantic matching.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.middleware.auth import verify_api_key
from app.infra.mongodb.repositories.portfolio_repo import (
    PortfolioRepository,
    get_portfolio_repo
)
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


# ===================== REQUEST MODELS =====================

class CreatePortfolioItemRequest(BaseModel):
    """Create a new portfolio item"""
    org_id: str = Field(..., description="Organization ID")
    profile_id: str = Field(..., description="Profile this belongs to")
    project_title: str = Field(..., max_length=200, description="Project name")
    deliverables: List[str] = Field(..., min_length=1, description="What was delivered")
    skills: List[str] = Field(..., min_length=1, description="Tech stack used")
    outcome: str = Field(..., max_length=500, description="Key result achieved")
    portfolio_url: Optional[str] = Field(None, description="Link to live work")
    industry: Optional[str] = Field(None, description="Industry sector")
    client_feedback: Optional[str] = Field(None, max_length=1000, description="Testimonial")
    duration_days: Optional[int] = Field(None, ge=1, description="Project duration")


class UpdatePortfolioItemRequest(BaseModel):
    """Update portfolio item fields"""
    project_title: Optional[str] = Field(None, max_length=200)
    deliverables: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    outcome: Optional[str] = Field(None, max_length=500)
    portfolio_url: Optional[str] = None
    industry: Optional[str] = None
    client_feedback: Optional[str] = Field(None, max_length=1000)
    duration_days: Optional[int] = Field(None, ge=1)


class PortfolioItemResponse(BaseModel):
    """Single portfolio item response"""
    success: bool
    item: Dict[str, Any]


class PortfolioListResponse(BaseModel):
    """Multiple portfolio items response"""
    success: bool
    items: List[Dict[str, Any]]
    count: int


class EmbedResponse(BaseModel):
    """Embedding batch response"""
    success: bool
    embedded: int
    failed: int
    items: List[str]


# ===================== ENDPOINTS =====================

@router.post("", response_model=PortfolioItemResponse, status_code=201)
async def create_portfolio_item(
    request: CreatePortfolioItemRequest,
    embed: bool = Query(False, description="Auto-embed after create?"),
    _: str = Depends(verify_api_key)
):
    """
    Create a new portfolio item.
    
    Optional: Set embed=true to immediately generate embedding.
    """
    try:
        repo = get_portfolio_repo()
        
        result = repo.create(
            org_id=request.org_id,
            profile_id=request.profile_id,
            project_title=request.project_title,
            deliverables=request.deliverables,
            skills=request.skills,
            outcome=request.outcome,
            portfolio_url=request.portfolio_url,
            industry=request.industry,
            client_feedback=request.client_feedback,
            duration_days=request.duration_days
        )
        
        item = repo.get_by_item_id(result["item_id"])
        
        # Auto-embed if requested
        if embed:
            embedding_svc = get_embedding_service()
            embedding_svc.embed_portfolio_item(item, repo)
            item = repo.get_by_item_id(result["item_id"])  # Refresh
        
        item["_id"] = str(item.get("_id", ""))
        
        return PortfolioItemResponse(success=True, item=item)
        
    except Exception as e:
        logger.error(f"Error creating portfolio item: {e}")
        raise HTTPException(500, str(e))


@router.get("/{item_id}", response_model=PortfolioItemResponse)
async def get_portfolio_item(
    item_id: str,
    _: str = Depends(verify_api_key)
):
    """Get a specific portfolio item by ID."""
    try:
        repo = get_portfolio_repo()
        item = repo.get_by_item_id(item_id)
        
        if not item:
            raise HTTPException(404, "Portfolio item not found")
        
        item["_id"] = str(item.get("_id", ""))
        return PortfolioItemResponse(success=True, item=item)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching portfolio item: {e}")
        raise HTTPException(500, str(e))


@router.get("", response_model=PortfolioListResponse)
async def list_portfolio_items(
    org_id: str = Query(None, description="Filter by organization"),
    profile_id: str = Query(None, description="Filter by profile"),
    _: str = Depends(verify_api_key)
):
    """List portfolio items. Filter by org_id or profile_id."""
    try:
        repo = get_portfolio_repo()
        
        if profile_id:
            items = repo.list_by_profile(profile_id)
        elif org_id:
            items = repo.list_by_org(org_id)
        else:
            raise HTTPException(400, "Provide org_id or profile_id")
        
        return PortfolioListResponse(
            success=True,
            items=items,
            count=len(items)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing portfolio items: {e}")
        raise HTTPException(500, str(e))


@router.put("/{item_id}", response_model=PortfolioItemResponse)
async def update_portfolio_item(
    item_id: str,
    request: UpdatePortfolioItemRequest,
    embed: bool = Query(False, description="Re-embed after update?"),
    _: str = Depends(verify_api_key)
):
    """
    Update a portfolio item.
    
    Note: Updates mark item as needing re-embedding (is_embedded=False).
    Set embed=true to immediately re-generate embedding.
    """
    try:
        repo = get_portfolio_repo()
        
        existing = repo.get_by_item_id(item_id)
        if not existing:
            raise HTTPException(404, "Portfolio item not found")
        
        updates = {k: v for k, v in request.model_dump().items() if v is not None}
        
        if not updates:
            raise HTTPException(400, "No fields to update")
        
        repo.update(item_id, updates)
        
        item = repo.get_by_item_id(item_id)
        
        # Re-embed if requested
        if embed:
            embedding_svc = get_embedding_service()
            embedding_svc.embed_portfolio_item(item, repo)
            item = repo.get_by_item_id(item_id)  # Refresh
        
        item["_id"] = str(item.get("_id", ""))
        
        return PortfolioItemResponse(success=True, item=item)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating portfolio item: {e}")
        raise HTTPException(500, str(e))


@router.delete("/{item_id}")
async def delete_portfolio_item(
    item_id: str,
    _: str = Depends(verify_api_key)
):
    """Delete a portfolio item and its embedding."""
    try:
        repo = get_portfolio_repo()
        
        existing = repo.get_by_item_id(item_id)
        if not existing:
            raise HTTPException(404, "Portfolio item not found")
        
        # Delete from Pinecone if embedded
        if existing.get("is_embedded"):
            embedding_svc = get_embedding_service()
            embedding_svc.delete_embedding(item_id)
        
        # Delete from MongoDB
        repo.delete(item_id)
        
        return {"success": True, "message": "Portfolio item deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting portfolio item: {e}")
        raise HTTPException(500, str(e))


# ===================== EMBEDDING ENDPOINTS =====================

@router.post("/embed/{item_id}", response_model=PortfolioItemResponse)
async def embed_single_item(
    item_id: str,
    _: str = Depends(verify_api_key)
):
    """Generate embedding for a single portfolio item."""
    try:
        repo = get_portfolio_repo()
        item = repo.get_by_item_id(item_id)
        
        if not item:
            raise HTTPException(404, "Portfolio item not found")
        
        embedding_svc = get_embedding_service()
        result = embedding_svc.embed_portfolio_item(item, repo)
        
        if not result:
            raise HTTPException(500, "Failed to generate embedding")
        
        item = repo.get_by_item_id(item_id)
        item["_id"] = str(item.get("_id", ""))
        
        return PortfolioItemResponse(success=True, item=item)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error embedding item: {e}")
        raise HTTPException(500, str(e))


@router.post("/embed-pending", response_model=EmbedResponse)
async def embed_pending_items(
    org_id: str = Query(None, description="Filter by organization"),
    limit: int = Query(50, ge=1, le=200, description="Max items to embed"),
    _: str = Depends(verify_api_key)
):
    """Batch embed all pending portfolio items."""
    try:
        repo = get_portfolio_repo()
        embedding_svc = get_embedding_service()
        
        result = embedding_svc.embed_pending_items(repo, org_id=org_id, limit=limit)
        
        return EmbedResponse(
            success=True,
            embedded=result["embedded"],
            failed=result["failed"],
            items=result["items"]
        )
        
    except Exception as e:
        logger.error(f"Error batch embedding: {e}")
        raise HTTPException(500, str(e))

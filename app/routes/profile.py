"""
Profile Management Routes

CRUD operations for freelancer profiles with multi-tenant support.
Supports both individual freelancers (1 profile) and agencies (multiple profiles).
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.middleware.auth import verify_api_key
from app.infra.mongodb.repositories.profile_repo import (
    FreelancerProfileRepository, 
    ImportSource,
    get_freelancer_profile_repo
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/profiles", tags=["profiles"])


# ===================== REQUEST MODELS =====================

class CreateProfileRequest(BaseModel):
    """Create a new freelancer profile"""
    org_id: str = Field(..., description="Organization ID")
    name: str = Field(..., max_length=100, description="Freelancer name")
    title: str = Field(..., max_length=200, description="Professional title")
    bio: str = Field(..., max_length=2000, description="Professional summary")
    skills: List[str] = Field(..., min_length=1, description="List of skills")
    hourly_rate: Optional[float] = Field(None, ge=0, description="Hourly rate in USD")
    years_experience: Optional[int] = Field(None, ge=0, description="Years of experience")


class UpdateProfileRequest(BaseModel):
    """Update profile fields"""
    name: Optional[str] = Field(None, max_length=100)
    title: Optional[str] = Field(None, max_length=200)
    bio: Optional[str] = Field(None, max_length=2000)
    skills: Optional[List[str]] = None
    hourly_rate: Optional[float] = Field(None, ge=0)
    years_experience: Optional[int] = Field(None, ge=0)


class ProfileResponse(BaseModel):
    """Single profile response"""
    success: bool
    profile: Dict[str, Any]


class ProfileListResponse(BaseModel):
    """Multiple profiles response"""
    success: bool
    profiles: List[Dict[str, Any]]
    count: int


# ===================== ENDPOINTS =====================

@router.post("", response_model=ProfileResponse, status_code=201)
async def create_profile(
    request: CreateProfileRequest,
    _: str = Depends(verify_api_key)
):
    """
    Create a new freelancer profile.
    
    Individual accounts: limited to 1 profile.
    Agency accounts: limited by plan tier (3/10/unlimited).
    """
    try:
        repo = get_freelancer_profile_repo()
        
        result = repo.create(
            org_id=request.org_id,
            name=request.name,
            title=request.title,
            bio=request.bio,
            skills=request.skills,
            hourly_rate=request.hourly_rate,
            years_experience=request.years_experience,
            source=ImportSource.MANUAL
        )
        
        # Check if limit was reached
        if "error" in result:
            raise HTTPException(403, result["error"])
        
        # Fetch created profile
        profile = repo.get_by_profile_id(result["profile_id"])
        profile["_id"] = str(profile.get("_id", ""))
        
        return ProfileResponse(success=True, profile=profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating profile: {e}")
        raise HTTPException(500, str(e))


@router.get("/{profile_id}", response_model=ProfileResponse)
async def get_profile(
    profile_id: str,
    _: str = Depends(verify_api_key)
):
    """Get a specific profile by ID."""
    try:
        repo = get_freelancer_profile_repo()
        profile = repo.get_by_profile_id(profile_id)
        
        if not profile:
            raise HTTPException(404, "Profile not found")
        
        profile["_id"] = str(profile.get("_id", ""))
        return ProfileResponse(success=True, profile=profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching profile: {e}")
        raise HTTPException(500, str(e))


@router.get("", response_model=ProfileListResponse)
async def list_profiles(
    org_id: str = Query(..., description="Organization ID"),
    _: str = Depends(verify_api_key)
):
    """List all profiles for an organization."""
    try:
        repo = get_freelancer_profile_repo()
        profiles = repo.list_by_org(org_id)
        
        return ProfileListResponse(
            success=True,
            profiles=profiles,
            count=len(profiles)
        )
        
    except Exception as e:
        logger.error(f"Error listing profiles: {e}")
        raise HTTPException(500, str(e))


@router.put("/{profile_id}", response_model=ProfileResponse)
async def update_profile(
    profile_id: str,
    request: UpdateProfileRequest,
    _: str = Depends(verify_api_key)
):
    """Update an existing profile."""
    try:
        repo = get_freelancer_profile_repo()
        
        # Check exists
        existing = repo.get_by_profile_id(profile_id)
        if not existing:
            raise HTTPException(404, "Profile not found")
        
        # Build update dict (exclude None)
        updates = {k: v for k, v in request.model_dump().items() if v is not None}
        
        if not updates:
            raise HTTPException(400, "No fields to update")
        
        repo.update(profile_id, updates)
        
        # Return updated profile
        profile = repo.get_by_profile_id(profile_id)
        profile["_id"] = str(profile.get("_id", ""))
        
        return ProfileResponse(success=True, profile=profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(500, str(e))


@router.delete("/{profile_id}")
async def delete_profile(
    profile_id: str,
    _: str = Depends(verify_api_key)
):
    """Soft delete a profile (deactivate)."""
    try:
        repo = get_freelancer_profile_repo()
        
        existing = repo.get_by_profile_id(profile_id)
        if not existing:
            raise HTTPException(404, "Profile not found")
        
        repo.deactivate(profile_id)
        
        return {"success": True, "message": "Profile deactivated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting profile: {e}")
        raise HTTPException(500, str(e))


# ===================== IMPORT ENDPOINTS =====================

class UpworkImportRequest(BaseModel):
    """Import profile from Upwork data."""
    org_id: str = Field(..., description="Organization ID")
    name: str = Field(..., min_length=2, max_length=100)
    title: str = Field(..., min_length=2, max_length=200)
    overview: Optional[str] = Field(None, max_length=5000)
    hourly_rate: Optional[float] = Field(None, ge=0)
    skills: List[str] = Field(default_factory=list)
    job_success_score: Optional[int] = Field(None, ge=0, le=100)
    total_earnings: Optional[float] = Field(None, ge=0)
    total_jobs: Optional[int] = Field(None, ge=0)
    total_hours: Optional[float] = Field(None, ge=0)
    profile_url: Optional[str] = None


@router.post("/import/upwork", response_model=ProfileResponse, status_code=201)
async def import_upwork_profile(
    request: UpworkImportRequest,
    _: str = Depends(verify_api_key)
):
    """
    Import a profile from Upwork.
    
    User provides their Upwork profile data as JSON.
    This respects Upwork ToS by not scraping - user consents to share their own data.
    """
    from app.services.profile_import_service import ProfileImportService, UpworkProfileInput
    
    try:
        repo = get_freelancer_profile_repo()
        service = ProfileImportService(repo)
        
        # Convert to UpworkProfileInput
        profile_input = UpworkProfileInput(
            name=request.name,
            title=request.title,
            overview=request.overview,
            hourly_rate=request.hourly_rate,
            skills=request.skills,
            job_success_score=request.job_success_score,
            total_earnings=request.total_earnings,
            total_jobs=request.total_jobs,
            total_hours=request.total_hours,
            profile_url=request.profile_url
        )
        
        result = service.import_from_upwork(request.org_id, profile_input)
        
        if not result.success:
            raise HTTPException(400, result.errors[0] if result.errors else "Import failed")
        
        # Fetch created profile
        profile = repo.get_by_profile_id(result.profile_id)
        profile["_id"] = str(profile.get("_id", ""))
        
        return ProfileResponse(success=True, profile=profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upwork import error: {e}")
        raise HTTPException(500, str(e))

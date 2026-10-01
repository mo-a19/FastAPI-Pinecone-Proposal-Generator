"""
Profile Import Service - Import freelancer profiles from external sources.

Supports:
- UPWORK: Manual JSON input (paste profile data) or API integration
- LINKEDIN: Future support
- MANUAL: Direct form input

Note: Direct Upwork scraping violates their ToS. Use their API or manual input.
"""
import re
import logging
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, HttpUrl, validator

from app.domain.constants import ImportSource

logger = logging.getLogger(__name__)


# NOTE: ImportSource enum moved to app/domain/constants.py (single source of truth)


class UpworkProfileInput(BaseModel):
    """Schema for Upwork profile import (manual JSON paste)."""
    name: str = Field(..., min_length=2, max_length=100)
    title: str = Field(..., min_length=2, max_length=200)
    overview: Optional[str] = Field(None, max_length=5000)
    hourly_rate: Optional[float] = Field(None, ge=0)
    skills: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    job_success_score: Optional[int] = Field(None, ge=0, le=100)
    total_earnings: Optional[float] = Field(None, ge=0)
    total_jobs: Optional[int] = Field(None, ge=0)
    total_hours: Optional[float] = Field(None, ge=0)
    profile_url: Optional[str] = None
    
    @validator("skills", pre=True)
    def clean_skills(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return [s.strip() for s in v if s.strip()][:20]  # Max 20 skills
    
    @validator("hourly_rate", pre=True)
    def parse_rate(cls, v):
        if isinstance(v, str):
            nums = re.findall(r"[\d.]+", v)
            return float(nums[0]) if nums else None
        return v


class ProfileImportResult(BaseModel):
    """Result from profile import operation."""
    success: bool
    profile_id: Optional[str] = None
    source: ImportSource
    name: str
    errors: list[str] = Field(default_factory=list)


class ProfileImportService:
    """
    Service for importing freelancer profiles from external sources.
    
    Usage:
        service = ProfileImportService(profile_repo)
        result = service.import_from_upwork(org_id, upwork_data)
    """
    
    def __init__(self, profile_repo):
        self.profile_repo = profile_repo
    
    def import_from_upwork(
        self,
        org_id: str,
        profile_data: UpworkProfileInput
    ) -> ProfileImportResult:
        """
        Import a profile from Upwork structured data.
        
        Args:
            org_id: Organization to import into
            profile_data: Validated Upwork profile data
            
        Returns:
            ProfileImportResult with success status and profile_id
        """
        errors = []
        
        try:
            # Calculate years of experience from total hours (rough estimate)
            years_exp = None
            if profile_data.total_hours:
                # Assume 1500 hours/year average
                years_exp = int(profile_data.total_hours / 1500)
                years_exp = max(1, min(years_exp, 30))  # Clamp 1-30
            
            # Create profile
            result = self.profile_repo.create(
                org_id=org_id,
                name=profile_data.name,
                title=profile_data.title,
                bio=profile_data.overview or f"Professional {profile_data.title}",
                skills=profile_data.skills,
                hourly_rate=profile_data.hourly_rate,
                years_experience=years_exp,
                source="UPWORK",
                source_url=profile_data.profile_url
            )
            
            if "error" in result:
                return ProfileImportResult(
                    success=False,
                    source=ImportSource.UPWORK,
                    name=profile_data.name,
                    errors=[result["error"]]
                )
            
            logger.info(f"Imported Upwork profile: {profile_data.name} -> {result.get('profile_id')}")
            
            return ProfileImportResult(
                success=True,
                profile_id=result.get("profile_id"),
                source=ImportSource.UPWORK,
                name=profile_data.name
            )
            
        except Exception as e:
            logger.error(f"Upwork import failed for {profile_data.name}: {e}")
            errors.append(str(e))
            
            return ProfileImportResult(
                success=False,
                source=ImportSource.UPWORK,
                name=profile_data.name,
                errors=errors
            )
    
    def import_bulk(
        self,
        org_id: str,
        profiles: list[UpworkProfileInput],
        source: ImportSource = ImportSource.UPWORK
    ) -> dict:
        """
        Import multiple profiles at once.
        
        Returns:
            {"imported": int, "failed": int, "results": list[ProfileImportResult]}
        """
        results = []
        imported = 0
        failed = 0
        
        for profile in profiles:
            if source == ImportSource.UPWORK:
                result = self.import_from_upwork(org_id, profile)
            else:
                result = ProfileImportResult(
                    success=False,
                    source=source,
                    name=profile.name,
                    errors=[f"Source {source} not yet supported"]
                )
            
            results.append(result)
            if result.success:
                imported += 1
            else:
                failed += 1
        
        return {
            "imported": imported,
            "failed": failed,
            "results": [r.model_dump() for r in results]
        }


# Convenience function for parsing Upwork profile from raw text
def parse_upwork_profile_text(text: str) -> dict:
    """
    Parse a raw Upwork profile text (copy-pasted from profile page).
    
    This is a best-effort parser for common profile formats.
    Users should validate the output before importing.
    """
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    
    profile = {
        "name": "",
        "title": "",
        "overview": "",
        "skills": [],
        "hourly_rate": None,
    }
    
    # Simple heuristics
    for i, line in enumerate(lines):
        # First non-empty line is usually name
        if not profile["name"] and len(line) < 50 and not line.startswith("$"):
            profile["name"] = line
            continue
        
        # Lines with $ are likely rates
        if "$" in line and "/hr" in line.lower():
            rate_match = re.search(r"\$(\d+(?:\.\d+)?)", line)
            if rate_match:
                profile["hourly_rate"] = float(rate_match.group(1))
            continue
        
        # Short lines after name could be title
        if not profile["title"] and 10 < len(line) < 100:
            profile["title"] = line
            continue
        
        # Longer text blocks are overview
        if len(line) > 100:
            profile["overview"] = line
    
    return profile

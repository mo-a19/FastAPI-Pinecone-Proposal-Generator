"""
Job Preferences Repository - Upwork-like filtering for job selection.

Filters match Upwork's search filters:
https://www.upwork.com/nx/search/jobs/

Two filter types:
- Hard filters: Must match (excludes jobs)
- Soft preferences: Ranking boost (doesn't exclude)
"""
import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from app.infra.mongodb.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ExperienceLevel(str, Enum):
    """Upwork experience levels."""
    ENTRY = "entry"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class JobType(str, Enum):
    """Upwork job types."""
    HOURLY = "hourly"
    FIXED = "fixed"


class ProjectLength(str, Enum):
    """Upwork project length categories."""
    LESS_THAN_MONTH = "less_than_month"
    ONE_TO_THREE_MONTHS = "1_to_3_months"
    THREE_TO_SIX_MONTHS = "3_to_6_months"
    MORE_THAN_SIX_MONTHS = "more_than_6_months"


class HoursPerWeek(str, Enum):
    """Upwork hours per week categories."""
    LESS_THAN_30 = "less_than_30"
    MORE_THAN_30 = "more_than_30"


class JobPreferencesRepository(BaseRepository[Dict[str, Any]]):
    """
    Job filtering preferences per org/user.
    
    Mirrors Upwork's search filters for consistent UX.
    """
    
    collection_name = "job_preferences"
    
    def create_or_update(
        self,
        org_id: str,
        profile_id: str = None,  # None = org-wide default
        filters: Dict[str, Any] = None,
        preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create or update job preferences."""
        pref_id = f"pref_{uuid.uuid4().hex[:8]}"
        
        # Build filter document with Upwork-like structure
        doc = {
            "pref_id": pref_id,
            "org_id": org_id,
            "profile_id": profile_id,  # None = org default
            
            # === HARD FILTERS (Must match) ===
            "filters": {
                # Client filters
                "payment_verified": filters.get("payment_verified") if filters else None,
                "min_client_rating": filters.get("min_client_rating"),  # 1-5
                "min_client_spend": filters.get("min_client_spend"),  # USD total
                "min_client_hires": filters.get("min_client_hires"),  # Previous hires
                "client_country": filters.get("client_country"),  # List of countries
                
                # Job type filters
                "job_type": filters.get("job_type"),  # hourly, fixed, or None for both
                "experience_levels": filters.get("experience_levels", []),  # entry, intermediate, expert
                
                # Budget filters
                "min_budget": filters.get("min_budget"),  # Fixed price min
                "max_budget": filters.get("max_budget"),  # Fixed price max
                "min_hourly_rate": filters.get("min_hourly_rate"),
                "max_hourly_rate": filters.get("max_hourly_rate"),
                
                # Project scope filters
                "project_lengths": filters.get("project_lengths", []),  # List of ProjectLength
                "hours_per_week": filters.get("hours_per_week"),  # HoursPerWeek enum
                
                # Category filters
                "categories": filters.get("categories", []),  # Upwork categories
                "subcategories": filters.get("subcategories", []),
                
                # Competition filter
                "max_proposals": filters.get("max_proposals"),  # Skip if > N proposals already
                
                # Keyword exclusions
                "excluded_keywords": filters.get("excluded_keywords", []),
                "required_keywords": filters.get("required_keywords", []),
            } if filters else {},
            
            # === SOFT PREFERENCES (Ranking boost) ===
            "preferences": {
                "preferred_categories": preferences.get("preferred_categories", []) if preferences else [],
                "preferred_skills": preferences.get("preferred_skills", []) if preferences else [],
                "preferred_industries": preferences.get("preferred_industries", []) if preferences else [],
                "long_term_preferred": preferences.get("long_term_preferred", False) if preferences else False,
                "timezone_compatible": preferences.get("timezone_compatible") if preferences else None,
            } if preferences else {},
            
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Upsert - update if exists for this org/profile combo
        existing = self.find_one({
            "org_id": org_id,
            "profile_id": profile_id
        })
        
        if existing:
            doc.pop("pref_id")
            doc.pop("created_at")
            result = self.collection.update_one(
                {"org_id": org_id, "profile_id": profile_id},
                {"$set": doc}
            )
            logger.info(f"Updated job preferences for org {org_id}, profile {profile_id}")
            return {"pref_id": existing["pref_id"], "updated": True}
        else:
            db_id = self.insert_one(doc)
            logger.info(f"Created job preferences: {pref_id}")
            return {"pref_id": pref_id, "db_id": db_id, "updated": False}
    
    def get_for_profile(self, org_id: str, profile_id: str = None) -> Optional[Dict[str, Any]]:
        """Get preferences for profile, falling back to org default."""
        # Try profile-specific first
        if profile_id:
            prefs = self.find_one({"org_id": org_id, "profile_id": profile_id, "is_active": True})
            if prefs:
                return prefs
        # Fall back to org default
        return self.find_one({"org_id": org_id, "profile_id": None, "is_active": True})
    
    def list_by_org(self, org_id: str) -> List[Dict[str, Any]]:
        """List all preference sets for an org."""
        results = list(self.collection.find({"org_id": org_id, "is_active": True}))
        for r in results:
            r["_id"] = str(r["_id"])
        return results
    
    def matches_job(self, prefs: Dict[str, Any], job: Dict[str, Any]) -> tuple[bool, float]:
        """
        Check if job matches filters and calculate preference score.
        
        Returns:
            (passes_filters: bool, preference_score: float 0-1)
        """
        filters = prefs.get("filters", {})
        preferences = prefs.get("preferences", {})
        
        # === Check hard filters ===
        
        # Payment verified
        if filters.get("payment_verified") and not job.get("payment_verified"):
            return False, 0.0
        
        # Min budget
        if filters.get("min_budget") and job.get("budget", 0) < filters["min_budget"]:
            return False, 0.0
        
        # Max budget
        if filters.get("max_budget") and job.get("budget", float("inf")) > filters["max_budget"]:
            return False, 0.0
        
        # Min hourly rate
        if filters.get("min_hourly_rate") and job.get("hourly_rate", 0) < filters["min_hourly_rate"]:
            return False, 0.0
        
        # Job type
        if filters.get("job_type") and job.get("job_type") != filters["job_type"]:
            return False, 0.0
        
        # Experience level
        exp_levels = filters.get("experience_levels", [])
        if exp_levels and job.get("experience_level") not in exp_levels:
            return False, 0.0
        
        # Categories
        categories = filters.get("categories", [])
        if categories and job.get("category") not in categories:
            return False, 0.0
        
        # Max proposals (competition)
        if filters.get("max_proposals") and job.get("proposals_count", 0) > filters["max_proposals"]:
            return False, 0.0
        
        # Excluded keywords
        excluded = filters.get("excluded_keywords", [])
        job_text = f"{job.get('title', '')} {job.get('description', '')}".lower()
        for kw in excluded:
            if kw.lower() in job_text:
                return False, 0.0
        
        # Required keywords
        required = filters.get("required_keywords", [])
        for kw in required:
            if kw.lower() not in job_text:
                return False, 0.0
        
        # Client filters
        if filters.get("min_client_rating") and job.get("client_rating", 5) < filters["min_client_rating"]:
            return False, 0.0
        
        if filters.get("min_client_hires") and job.get("client_hires", 0) < filters["min_client_hires"]:
            return False, 0.0
        
        # === Calculate preference score ===
        score = 0.5  # Base score
        
        # Preferred categories bonus
        if job.get("category") in preferences.get("preferred_categories", []):
            score += 0.15
        
        # Skill match bonus
        job_skills = set(s.lower() for s in job.get("skills", []))
        pref_skills = set(s.lower() for s in preferences.get("preferred_skills", []))
        if job_skills & pref_skills:
            score += 0.1 * len(job_skills & pref_skills)
        
        # Industry match bonus
        if job.get("industry") in preferences.get("preferred_industries", []):
            score += 0.1
        
        # Long-term bonus
        if preferences.get("long_term_preferred") and job.get("project_length") in ["3_to_6_months", "more_than_6_months"]:
            score += 0.1
        
        return True, min(score, 1.0)
    
    def delete(self, pref_id: str) -> bool:
        """Soft delete preferences."""
        result = self.collection.update_one(
            {"pref_id": pref_id},
            {"$set": {"is_active": False, "deleted_at": datetime.utcnow()}}
        )
        return result.modified_count > 0


# Singleton
_job_prefs_repo: Optional[JobPreferencesRepository] = None

def get_job_prefs_repo() -> JobPreferencesRepository:
    global _job_prefs_repo
    if _job_prefs_repo is None:
        _job_prefs_repo = JobPreferencesRepository()
    return _job_prefs_repo

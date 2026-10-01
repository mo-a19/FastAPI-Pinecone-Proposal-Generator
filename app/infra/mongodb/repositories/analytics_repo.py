"""
Analytics Repository

Handles all operations for analytics, statistics, skills, and profile collections.
Consolidates reporting and metrics operations from db.py.
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pymongo import DESCENDING, ASCENDING

from app.infra.mongodb.base_repository import BaseRepository
from app.infra.mongodb.connection import get_database

logger = logging.getLogger(__name__)


class AnalyticsRepository:
    """
    Repository for analytics and statistics.
    
    Unlike other repositories, this queries across multiple collections
    so it doesn't extend BaseRepository directly.
    """
    
    def __init__(self):
        self.db = get_database()
    
    def _build_date_query(
        self, 
        since: Optional[datetime], 
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build MongoDB query with optional date and source filters."""
        query = {}
        if since:
            query["sent_at"] = {"$gte": since}
        if source:
            query["source"] = source
        return query
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics across all collections."""
        try:
            return {
                "portfolio_items_count": self.db["portfolio_items"].count_documents({}),
                "total_feedback": self.db["feedback_data"].count_documents({}),
                "sent_proposals": self.db["sent_proposals"].count_documents({}),
                "cached_embeddings": self.db["embedding_cache"].count_documents({})
            }
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}
    
    def get_industry_statistics(self) -> Dict[str, int]:
        """Get portfolio statistics by industry."""
        try:
            pipeline = [
                {"$group": {"_id": "$industry", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            results = list(self.db["portfolio_items"].aggregate(pipeline))
            return {item["_id"]: item["count"] for item in results}
        except Exception as e:
            logger.error(f"Error getting industry statistics: {e}")
            return {}
    
    def get_analytics_summary(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get dashboard summary statistics."""
        try:
            base_query = self._build_date_query(since)
            col = self.db["sent_proposals"]
            
            total = col.count_documents(base_query)
            ai_count = col.count_documents({**base_query, "source": "ai_generated"})
            manual_count = col.count_documents({**base_query, "source": "manual"})
            hired = col.count_documents({**base_query, "outcome": "hired"})
            viewed = col.count_documents({**base_query, "outcome": {"$in": ["viewed", "hired"]}})
            discussions = col.count_documents({**base_query, "discussion_initiated": True})
            
            ai_hired = col.count_documents({**base_query, "source": "ai_generated", "outcome": "hired"})
            manual_hired = col.count_documents({**base_query, "source": "manual", "outcome": "hired"})
            
            return {
                "total_proposals": total,
                "ai_generated_count": ai_count,
                "manual_count": manual_count,
                "total_hired": hired,
                "total_viewed": viewed,
                "overall_hire_rate": round(hired / total * 100, 1) if total else 0,
                "ai_hire_rate": round(ai_hired / ai_count * 100, 1) if ai_count else 0,
                "manual_hire_rate": round(manual_hired / manual_count * 100, 1) if manual_count else 0,
                "message_market_fit": round(discussions / viewed * 100, 1) if viewed else 0
            }
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {}
    
    def get_conversion_funnel(
        self, 
        since: Optional[datetime] = None, 
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get conversion funnel: Sent → Viewed → Discussed → Hired."""
        try:
            query = self._build_date_query(since, source)
            col = self.db["sent_proposals"]
            
            sent = col.count_documents(query)
            viewed = col.count_documents({**query, "outcome": {"$in": ["viewed", "hired"]}})
            discussed = col.count_documents({**query, "discussion_initiated": True})
            hired = col.count_documents({**query, "outcome": "hired"})
            
            funnel = [
                {"stage": "Sent", "count": sent, "percentage": 100.0},
                {"stage": "Viewed", "count": viewed, "percentage": round(viewed / sent * 100, 1) if sent else 0},
                {"stage": "Discussed", "count": discussed, "percentage": round(discussed / sent * 100, 1) if sent else 0},
                {"stage": "Hired", "count": hired, "percentage": round(hired / sent * 100, 1) if sent else 0}
            ]
            return {"funnel": funnel, "total_sent": sent}
        except Exception as e:
            logger.error(f"Error getting funnel: {e}")
            return {"funnel": [], "total_sent": 0}
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get proposal conversion statistics for Message-Market Fit."""
        try:
            col = self.db["sent_proposals"]
            total = col.count_documents({})
            viewed = col.count_documents({"outcome": {"$in": ["viewed", "hired"]}})
            hired = col.count_documents({"outcome": "hired"})
            discussions = col.count_documents({"discussion_initiated": True})
            rejected = col.count_documents({"outcome": "rejected"})
            
            return {
                "total_sent": total,
                "total_viewed": viewed,
                "total_hired": hired,
                "total_discussions": discussions,
                "total_rejected": rejected,
                "view_rate": round((viewed / total * 100), 1) if total > 0 else 0,
                "hire_rate": round((hired / total * 100), 1) if total > 0 else 0,
                "discussion_rate": round((discussions / total * 100), 1) if total > 0 else 0,
                "message_market_fit": round((discussions / viewed * 100), 1) if viewed > 0 else 0,
                "view_to_hire_rate": round((hired / viewed * 100), 1) if viewed > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting conversion stats: {e}")
            return {}
    
    def get_source_comparison(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Compare AI-generated vs manual proposals."""
        try:
            col = self.db["sent_proposals"]
            
            def get_stats(source: str) -> Dict[str, Any]:
                query = self._build_date_query(since, source)
                total = col.count_documents(query)
                hired = col.count_documents({**query, "outcome": "hired"})
                viewed = col.count_documents({**query, "outcome": {"$in": ["viewed", "hired"]}})
                return {
                    "total": total,
                    "hired": hired,
                    "viewed": viewed,
                    "hire_rate": round(hired / total * 100, 1) if total else 0,
                    "view_rate": round(viewed / total * 100, 1) if total else 0
                }
            
            ai = get_stats("ai_generated")
            manual = get_stats("manual")
            
            effectiveness = 1.0
            if manual["hire_rate"] > 0:
                effectiveness = round(ai["hire_rate"] / manual["hire_rate"], 2)
            elif ai["hire_rate"] > 0:
                effectiveness = float("inf")
            
            return {"ai_generated": ai, "manual": manual, "ai_effectiveness": effectiveness}
        except Exception as e:
            logger.error(f"Error getting comparison: {e}")
            return {"ai_generated": {}, "manual": {}, "ai_effectiveness": 1.0}
    
    def get_combined_funnel(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get combined funnel data for AI-generated and Manual proposals."""
        try:
            sent_col = self.db["sent_proposals"]
            proposals_col = self.db["proposals"]
            
            def get_funnel_for_source(source: str) -> Dict[str, Any]:
                query = self._build_date_query(since, source)
                
                sent = sent_col.count_documents(query)
                viewed = sent_col.count_documents({**query, "outcome": {"$in": ["viewed", "hired"]}})
                discussed = sent_col.count_documents({**query, "discussion_initiated": True})
                hired = sent_col.count_documents({**query, "outcome": "hired"})
                
                generated = 0
                if source == "ai_generated":
                    gen_query = {"created_at": {"$gte": since}} if since else {}
                    generated = proposals_col.count_documents(gen_query)
                else:
                    generated = sent
                
                base = sent if sent > 0 else 1
                
                return {
                    "generated": generated,
                    "sent": sent,
                    "viewed": viewed,
                    "discussed": discussed,
                    "hired": hired,
                    "rates": {
                        "sent_rate": round(sent / generated * 100, 1) if generated > 0 else 0,
                        "view_rate": round(viewed / base * 100, 1),
                        "discuss_rate": round(discussed / base * 100, 1),
                        "hire_rate": round(hired / base * 100, 1)
                    }
                }
            
            ai_data = get_funnel_for_source("ai_generated")
            manual_data = get_funnel_for_source("manual")
            
            total_generated = ai_data["generated"] + manual_data["generated"]
            total_sent = ai_data["sent"] + manual_data["sent"]
            total_viewed = ai_data["viewed"] + manual_data["viewed"]
            total_discussed = ai_data["discussed"] + manual_data["discussed"]
            total_hired = ai_data["hired"] + manual_data["hired"]
            
            return {
                "ai": ai_data,
                "manual": manual_data,
                "totals": {
                    "generated": total_generated,
                    "sent": total_sent,
                    "viewed": total_viewed,
                    "discussed": total_discussed,
                    "hired": total_hired
                },
                "ai_share": round(ai_data["sent"] / total_sent * 100, 1) if total_sent > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting combined funnel: {e}")
            return {
                "ai": {"generated": 0, "sent": 0, "viewed": 0, "discussed": 0, "hired": 0, "rates": {}},
                "manual": {"generated": 0, "sent": 0, "viewed": 0, "discussed": 0, "hired": 0, "rates": {}},
                "totals": {"generated": 0, "sent": 0, "viewed": 0, "discussed": 0, "hired": 0},
                "ai_share": 0
            }
    
    def get_proposal_trends(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get daily proposal trends for charts."""
        try:
            if not since:
                since = datetime.utcnow() - timedelta(days=30)
            
            pipeline = [
                {"$match": {"sent_at": {"$gte": since}}},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$sent_at"}},
                    "sent": {"$sum": 1},
                    "hired": {"$sum": {"$cond": [{"$eq": ["$outcome", "hired"]}, 1, 0]}}
                }},
                {"$sort": {"_id": 1}}
            ]
            
            results = list(self.db["sent_proposals"].aggregate(pipeline))
            return [
                {
                    "date": r["_id"],
                    "sent": r["sent"],
                    "hired": r["hired"],
                    "hire_rate": round(r["hired"] / r["sent"] * 100, 1) if r["sent"] else 0
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error getting trends: {e}")
            return []
    
    def get_skills_performance(
        self, 
        since: Optional[datetime] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get performance breakdown by skill."""
        try:
            match_stage = {"$match": {"sent_at": {"$gte": since}}} if since else {"$match": {}}
            
            pipeline = [
                match_stage,
                {"$unwind": "$skills_required"},
                {"$group": {
                    "_id": "$skills_required",
                    "total": {"$sum": 1},
                    "hired": {"$sum": {"$cond": [{"$eq": ["$outcome", "hired"]}, 1, 0]}}
                }},
                {"$match": {"total": {"$gte": 2}}},
                {"$sort": {"hired": -1, "total": -1}},
                {"$limit": limit}
            ]
            
            results = list(self.db["sent_proposals"].aggregate(pipeline))
            return [
                {
                    "skill": r["_id"],
                    "total": r["total"],
                    "hired": r["hired"],
                    "hire_rate": round(r["hired"] / r["total"] * 100, 1) if r["total"] else 0
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error getting skills performance: {e}")
            return []
    
    def get_industry_performance(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get performance breakdown by industry."""
        try:
            match_stage = {"$match": {"sent_at": {"$gte": since}}} if since else {"$match": {}}
            
            pipeline = [
                match_stage,
                {"$group": {
                    "_id": "$industry",
                    "total": {"$sum": 1},
                    "hired": {"$sum": {"$cond": [{"$eq": ["$outcome", "hired"]}, 1, 0]}},
                    "viewed": {"$sum": {"$cond": [{"$in": ["$outcome", ["viewed", "hired"]]}, 1, 0]}}
                }},
                {"$sort": {"total": -1}}
            ]
            
            results = list(self.db["sent_proposals"].aggregate(pipeline))
            return [
                {
                    "industry": r["_id"] or "general",
                    "total": r["total"],
                    "hired": r["hired"],
                    "viewed": r["viewed"],
                    "hire_rate": round(r["hired"] / r["total"] * 100, 1) if r["total"] else 0
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error getting industry performance: {e}")
            return []


class SkillRepository(BaseRepository[Dict[str, Any]]):
    """Repository for skills and skill embeddings."""
    
    collection_name = "skills"
    
    def save_skills(self, contract_id: str, skills: List[str]) -> List[str]:
        """
        Save skills from a job to skills collection.
        
        Args:
            contract_id: Contract ID for reference
            skills: List of skill names
            
        Returns:
            List of skill IDs saved
        """
        try:
            skill_ids = []
            
            for skill in skills:
                if not skill or not isinstance(skill, str):
                    continue
                
                skill_lower = skill.strip().lower()
                
                skill_doc = {
                    "skill_name": skill,
                    "skill_name_lower": skill_lower,
                    "contracts": [contract_id],
                    "frequency": 1,
                    "last_used": datetime.utcnow(),
                    "created_at": datetime.utcnow()
                }
                
                existing = self.collection.find_one({"skill_name_lower": skill_lower})
                
                if existing:
                    self.collection.update_one(
                        {"_id": existing["_id"]},
                        {
                            "$inc": {"frequency": 1},
                            "$addToSet": {"contracts": contract_id},
                            "$set": {"last_used": datetime.utcnow()}
                        }
                    )
                    skill_ids.append(str(existing["_id"]))
                else:
                    result = self.collection.insert_one(skill_doc)
                    skill_ids.append(str(result.inserted_id))
            
            if skill_ids:
                logger.info(f"Saved {len(skill_ids)} skills for contract {contract_id}")
            
            return skill_ids
        except Exception as e:
            logger.error(f"Error saving skills: {e}")
            return []
    
    def get_all_skills(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all unique skills sorted by frequency."""
        try:
            skills = list(
                self.collection
                .find()
                .sort("frequency", DESCENDING)
                .limit(limit)
            )
            for s in skills:
                s["_id"] = str(s["_id"])
            return skills
        except Exception as e:
            logger.error(f"Error retrieving skills: {e}")
            return []
    
    def get_by_contract(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get skills associated with a specific contract."""
        try:
            skills = list(
                self.collection
                .find({"contracts": contract_id})
                .sort("frequency", DESCENDING)
            )
            for s in skills:
                s["_id"] = str(s["_id"])
            return skills
        except Exception as e:
            logger.error(f"Error retrieving skills for contract: {e}")
            return []
    
    def search_skills(self, search_term: str) -> List[Dict[str, Any]]:
        """Search for skills by name (case-insensitive)."""
        try:
            search_lower = search_term.strip().lower()
            skills = list(
                self.collection
                .find({"skill_name_lower": {"$regex": search_lower, "$options": "i"}})
                .sort("frequency", DESCENDING)
            )
            for s in skills:
                s["_id"] = str(s["_id"])
            return skills
        except Exception as e:
            logger.error(f"Error searching skills: {e}")
            return []


class SkillEmbeddingRepository(BaseRepository[Dict[str, Any]]):
    """Repository for skill embeddings."""
    
    collection_name = "skill_embeddings"
    
    def save_embedding(self, skill_name: str, embedding: List[float], model: str) -> str:
        """Save embedding for a skill."""
        try:
            skill_embedding = {
                "skill_name": skill_name,
                "skill_name_lower": skill_name.strip().lower(),
                "embedding": embedding,
                "embedding_model": model,
                "created_at": datetime.utcnow()
            }
            
            result = self.collection.insert_one(skill_embedding)
            logger.info(f"Saved embedding for skill: {skill_name}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving skill embedding: {e}")
            return ""
    
    def get_embedding(self, skill_name: str) -> Optional[List[float]]:
        """Get embedding for a skill."""
        try:
            skill_lower = skill_name.strip().lower()
            doc = self.collection.find_one({"skill_name_lower": skill_lower})
            if doc:
                return doc.get("embedding")
            return None
        except Exception as e:
            logger.error(f"Error retrieving skill embedding: {e}")
            return None


class ProfileRepository(BaseRepository[Dict[str, Any]]):
    """Repository for user profile (singleton per instance)."""
    
    collection_name = "user_profile"
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile (creates default if not exists)."""
        try:
            profile = self.collection.find_one({"user_id": "default"})
            if not profile:
                profile = {
                    "user_id": "default",
                    "name": "",
                    "email": "",
                    "upwork_url": "",
                    "bio": "",
                    "hourly_rate": 0,
                    "skills": [],
                    "timezone": "UTC",
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                self.collection.insert_one(profile)
            profile["_id"] = str(profile["_id"])
            return profile
        except Exception as e:
            logger.error(f"Error getting profile: {e}")
            return {}
    
    def update_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile."""
        try:
            data["updated_at"] = datetime.utcnow()
            self.collection.update_one(
                {"user_id": "default"},
                {"$set": data},
                upsert=True
            )
            return self.get_profile()
        except Exception as e:
            logger.error(f"Error updating profile: {e}")
            raise


class AdminRepository(BaseRepository[Dict[str, Any]]):
    """Repository for admin operations (activity logging, API keys)."""
    
    collection_name = "activity_log"
    
    def __init__(self):
        super().__init__()
        self.db = get_database()
    
    def log_activity(
        self,
        user_name: str,
        action: str,
        target: str = None,
        details: dict = None,
        ip: str = None
    ):
        """Log an activity for audit trail."""
        try:
            self.collection.insert_one({
                "user_name": user_name,
                "action": action,
                "target": target,
                "details": details,
                "ip": ip,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            logger.error(f"Error logging activity: {e}")
    
    def init_admin_collections(self):
        """Initialize admin collections and indexes."""
        try:
            # API Keys collection
            self.db["api_keys"].create_index([("key_hash", ASCENDING)], unique=True)
            self.db["api_keys"].create_index([("name", ASCENDING)])
            self.db["api_keys"].create_index([("is_active", ASCENDING)])
            self.db["api_keys"].create_index([("created_at", DESCENDING)])
            
            # Activity Log collection
            self.db["activity_log"].create_index([("timestamp", DESCENDING)])
            self.db["activity_log"].create_index([("key_prefix", ASCENDING)])
            self.db["activity_log"].create_index([("action", ASCENDING)])
            
            logger.info("Admin collections initialized")
        except Exception as e:
            logger.error(f"Error initializing admin collections: {e}")


# Singleton instances
_analytics_repo: Optional[AnalyticsRepository] = None
_skill_repo: Optional[SkillRepository] = None
_skill_embedding_repo: Optional[SkillEmbeddingRepository] = None
_profile_repo: Optional[ProfileRepository] = None
_admin_repo: Optional[AdminRepository] = None


def get_analytics_repo() -> AnalyticsRepository:
    """Get singleton AnalyticsRepository instance."""
    global _analytics_repo
    if _analytics_repo is None:
        _analytics_repo = AnalyticsRepository()
    return _analytics_repo


def get_skill_repo() -> SkillRepository:
    """Get singleton SkillRepository instance."""
    global _skill_repo
    if _skill_repo is None:
        _skill_repo = SkillRepository()
    return _skill_repo


def get_skill_embedding_repo() -> SkillEmbeddingRepository:
    """Get singleton SkillEmbeddingRepository instance."""
    global _skill_embedding_repo
    if _skill_embedding_repo is None:
        _skill_embedding_repo = SkillEmbeddingRepository()
    return _skill_embedding_repo


def get_profile_repo() -> ProfileRepository:
    """Get singleton ProfileRepository instance."""
    global _profile_repo
    if _profile_repo is None:
        _profile_repo = ProfileRepository()
    return _profile_repo


def get_admin_repo() -> AdminRepository:
    """Get singleton AdminRepository instance."""
    global _admin_repo
    if _admin_repo is None:
        _admin_repo = AdminRepository()
    return _admin_repo

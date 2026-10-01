"""
Database Connection Manager

SINGLE RESPONSIBILITY: MongoDB connection management only.

For CRUD operations, use the repository pattern:
- app/infra/mongodb/repositories/training_repo.py
- app/infra/mongodb/repositories/proposal_repo.py
- app/infra/mongodb/repositories/portfolio_repo.py
- app/infra/mongodb/repositories/analytics_repo.py

This file was cleaned from 1391 lines to ~250 lines by removing
duplicated CRUD methods that exist in repositories.
"""
import logging
from typing import Optional
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from contextlib import contextmanager
import certifi

from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    MongoDB connection management.
    
    ONLY handles:
    - Connection lifecycle
    - Collection initialization
    - Index creation
    
    For data operations, use repositories:
        from app.infra.mongodb.repositories import get_training_repo
        repo = get_training_repo()
        repo.insert_training_data(...)
    """
    
    def __init__(self, connection_string: str = None, db_name: str = None):
        """Initialize MongoDB connection."""
        self.connection_string = connection_string or settings.MONGODB_URI
        self.db_name = db_name or settings.MONGODB_DB_NAME
        self.client: Optional[MongoClient] = None
        self.db = None
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection with SSL configuration."""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=15000,
                socketTimeoutMS=15000,
                retryWrites=True,
                maxPoolSize=50,
                minPoolSize=10,
                tls=True,
                tlsCAFile=certifi.where()
            )
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            
            logger.info(f"Connected to MongoDB: {self.db_name}")
            self._init_collections()
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    def _init_collections(self):
        """Initialize collections and create indexes."""
        try:
            collections = {
                "training_data": "Raw job data for training",
                "chunks": "Smart chunks from training data",
                "embeddings": "Embedding metadata",
                "proposals": "Generated proposals",
                "feedback_data": "Client feedback analysis",
                "embedding_cache": "Embeddings cache",
                "skills": "Skills with frequency tracking",
                "skill_embeddings": "Skill vectors",
                "sent_proposals": "Sent proposals with outcomes",
                "user_profile": "User profile data",
                "portfolio_items": "Lean portfolio items (new)",
                "organizations": "Multi-tenant orgs",
                "users": "Multi-tenant users",
                "freelancer_profiles": "Freelancer profiles",
                "job_preferences": "Job filters",
            }
            
            for collection_name in collections.keys():
                if collection_name not in self.db.list_collection_names():
                    self.db.create_collection(collection_name)
                    logger.info(f"Created collection: {collection_name}")
            
            self._create_indexes()
        except Exception as e:
            logger.error(f"Error initializing collections: {str(e)}")
            raise
    
    def _create_indexes(self):
        """Create indexes for all collections."""
        try:
            # Training Data
            self.db["training_data"].create_index([("contract_id", ASCENDING)], unique=True)
            self.db["training_data"].create_index([("company_name", ASCENDING)])
            self.db["training_data"].create_index([("industry", ASCENDING)])
            self.db["training_data"].create_index([("job_title", ASCENDING)])
            self.db["training_data"].create_index([("created_at", DESCENDING)])
            self.db["training_data"].create_index([("skills_required", ASCENDING)])
            self.db["training_data"].create_index([("project_status", ASCENDING)])
            
            try:
                self.db["training_data"].create_index([
                    ("job_description", TEXT),
                    ("your_proposal_text", TEXT),
                    ("client_feedback_text", TEXT)
                ])
            except Exception as e:
                logger.debug(f"Text index may already exist: {e}")
            
            # Chunks
            self.db["chunks"].create_index([("contract_id", ASCENDING)])
            self.db["chunks"].create_index([("chunk_id", ASCENDING)], unique=True)
            self.db["chunks"].create_index([("chunk_type", ASCENDING)])
            self.db["chunks"].create_index([("priority", DESCENDING)])
            self.db["chunks"].create_index([("industry", ASCENDING)])
            self.db["chunks"].create_index([("created_at", DESCENDING)])
            
            # Embeddings
            self.db["embeddings"].create_index([("chunk_id", ASCENDING)], unique=True)
            self.db["embeddings"].create_index([("contract_id", ASCENDING)])
            self.db["embeddings"].create_index([("created_at", DESCENDING)])
            
            # Proposals
            self.db["proposals"].create_index([("contract_id", ASCENDING)])
            self.db["proposals"].create_index([("created_at", DESCENDING)])
            self.db["proposals"].create_index([("company_name", ASCENDING)])
            
            # Sent Proposals
            self.db["sent_proposals"].create_index([("proposal_id", ASCENDING)], unique=True)
            self.db["sent_proposals"].create_index([("outcome", ASCENDING)])
            self.db["sent_proposals"].create_index([("sent_at", DESCENDING)])
            
            # Skills
            self.db["skills"].create_index([("skill_name_lower", ASCENDING)], unique=True)
            self.db["skills"].create_index([("frequency", DESCENDING)])
            
            # Portfolio Items (new lean schema)
            self.db["portfolio_items"].create_index([("item_id", ASCENDING)], unique=True)
            self.db["portfolio_items"].create_index([("org_id", ASCENDING)])
            self.db["portfolio_items"].create_index([("profile_id", ASCENDING)])
            self.db["portfolio_items"].create_index([("is_embedded", ASCENDING)])
            
            # Organizations
            self.db["organizations"].create_index([("org_id", ASCENDING)], unique=True)
            self.db["organizations"].create_index([("slug", ASCENDING)], unique=True)
            
            # Users
            self.db["users"].create_index([("user_id", ASCENDING)], unique=True)
            self.db["users"].create_index([("org_id", ASCENDING)])
            self.db["users"].create_index([("email", ASCENDING)])
            
            # Freelancer Profiles
            self.db["freelancer_profiles"].create_index([("profile_id", ASCENDING)], unique=True)
            self.db["freelancer_profiles"].create_index([("org_id", ASCENDING)])
            
            # User Profile (legacy singleton)
            self.db["user_profile"].create_index([("user_id", ASCENDING)], unique=True)
            
            # Job Requirements Cache (auto-expires in 2 days)
            self.db["job_requirements_cache"].create_index([("cache_key", ASCENDING)], unique=True)
            self.db["job_requirements_cache"].create_index([("updated_at", ASCENDING)], expireAfterSeconds=172800)
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
    
    def init_admin_collections(self):
        """Initialize admin collections and indexes."""
        try:
            self.db["api_keys"].create_index([("key_hash", ASCENDING)], unique=True)
            self.db["api_keys"].create_index([("name", ASCENDING)])
            self.db["api_keys"].create_index([("is_active", ASCENDING)])
            self.db["api_keys"].create_index([("created_at", DESCENDING)])
            
            self.db["activity_log"].create_index([("timestamp", DESCENDING)])
            self.db["activity_log"].create_index([("key_prefix", ASCENDING)])
            self.db["activity_log"].create_index([("action", ASCENDING)])
            
            logger.info("Admin collections initialized")
        except Exception as e:
            logger.error(f"Error initializing admin collections: {e}")
    
    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    def close(self):
        """Alias for disconnect."""
        self.disconnect()
    
    def connect(self):
        """Re-establish connection."""
        self._connect()
    
    def get_collection(self, collection_name: str):
        """Get a collection by name."""
        return self.db[collection_name]
    
    @contextmanager
    def session(self):
        """Context manager for database sessions."""
        session = self.client.start_session()
        try:
            yield session
        finally:
            session.end_session()
    
    # =========================================================================
    # LEGACY SHIMS - Delegate to repositories
    # These methods exist for backward compatibility with deprecated code:
    # - app/utils/job_data_processor.py
    # - app/routes/job_data_ingestion.py
    # 
    # For new code, use repositories directly:
    #   from app.infra.mongodb.repositories import get_training_repo
    # =========================================================================
    
    def insert_training_data(self, job_data: dict) -> dict:
        """LEGACY: Use TrainingDataRepository.insert() instead."""
        from app.infra.mongodb.repositories.training_repo import get_training_repo
        return get_training_repo().insert(job_data)
    
    def get_training_data(self, contract_id: str) -> Optional[dict]:
        """LEGACY: Use TrainingDataRepository.get_by_contract_id() instead."""
        from app.infra.mongodb.repositories.training_repo import get_training_repo
        return get_training_repo().get_by_contract_id(contract_id)
    
    def get_all_training_data(self, skip: int = 0, limit: int = 100, filters: dict = None) -> list:
        """LEGACY: Use TrainingDataRepository.get_all() instead."""
        from app.infra.mongodb.repositories.training_repo import get_training_repo
        return get_training_repo().get_all(skip=skip, limit=limit, filters=filters)
    
    def update_training_data(self, contract_id: str, update_data: dict) -> bool:
        """LEGACY: Use TrainingDataRepository.update() instead."""
        from app.infra.mongodb.repositories.training_repo import get_training_repo
        return get_training_repo().update({"contract_id": contract_id}, update_data)
    
    def get_chunks_by_contract(self, contract_id: str) -> list:
        """LEGACY: Use ChunkRepository.get_by_contract() instead."""
        from app.infra.mongodb.repositories.training_repo import get_chunk_repo
        return get_chunk_repo().get_by_contract(contract_id)
    
    def insert_chunks(self, chunks: list, contract_id: str) -> tuple:
        """LEGACY: Use ChunkRepository.insert_batch() instead."""
        from app.infra.mongodb.repositories.training_repo import get_chunk_repo
        return get_chunk_repo().insert_batch(chunks, contract_id)
    
    def get_chunks_pending_embedding(self, limit: int = 100) -> list:
        """LEGACY: Use ChunkRepository.get_pending_embeddings() instead."""
        from app.infra.mongodb.repositories.training_repo import get_chunk_repo
        return get_chunk_repo().get_pending_embeddings(limit=limit)
    
    def update_chunk_embedding_status(self, chunk_id: str, embedded: bool = True, pinecone_id: str = None):
        """LEGACY: Use ChunkRepository.update_embedding_status() instead."""
        from app.infra.mongodb.repositories.training_repo import get_chunk_repo
        status = "embedded" if embedded else "pending"
        return get_chunk_repo().update_embedding_status(chunk_id, status, pinecone_id)
    
    def insert_embeddings_batch(self, records: list) -> int:
        """LEGACY: Use EmbeddingRepository.insert_batch() instead."""
        from app.infra.mongodb.repositories.training_repo import get_embedding_repo
        return get_embedding_repo().insert_batch(records)
    
    def insert_feedback(self, feedback_data: dict) -> str:
        """LEGACY: Use FeedbackRepository.insert_feedback() instead."""
        from app.infra.mongodb.repositories.proposal_repo import get_feedback_repo
        return get_feedback_repo().insert_feedback(feedback_data)
    
    def save_proposal(self, proposal_data: dict) -> str:
        """LEGACY: Use ProposalRepository.save_proposal() instead."""
        from app.infra.mongodb.repositories.proposal_repo import get_proposal_repo
        return get_proposal_repo().save_proposal(proposal_data)
    
    def save_skills(self, contract_id: str, skills: list) -> list:
        """LEGACY: Direct insert to skills collection."""
        skill_ids = []
        for skill in skills:
            skill_lower = skill.lower().strip()
            result = self.db["skills"].update_one(
                {"skill_name_lower": skill_lower},
                {"$set": {"skill_name": skill, "skill_name_lower": skill_lower},
                 "$inc": {"frequency": 1},
                 "$addToSet": {"contracts": contract_id}},
                upsert=True
            )
            skill_ids.append(str(result.upserted_id) if result.upserted_id else skill_lower)
        return skill_ids
    
    # =========================================================================
    # LEGACY ANALYTICS SHIMS - Delegate to AnalyticsRepository
    # Used by app/routes/analytics.py
    # =========================================================================
    
    def get_analytics_summary(self, since=None):
        """LEGACY: Use AnalyticsRepository.get_analytics_summary() instead."""
        from app.infra.mongodb.repositories.analytics_repo import get_analytics_repo
        return get_analytics_repo().get_analytics_summary(since)
    
    def get_conversion_funnel(self, since=None, source=None):
        """LEGACY: Use AnalyticsRepository.get_conversion_funnel() instead."""
        from app.infra.mongodb.repositories.analytics_repo import get_analytics_repo
        return get_analytics_repo().get_conversion_funnel(since, source)
    
    def get_combined_funnel(self, since=None):
        """LEGACY: Use AnalyticsRepository.get_combined_funnel() instead."""
        from app.infra.mongodb.repositories.analytics_repo import get_analytics_repo
        return get_analytics_repo().get_combined_funnel(since)
    
    def get_source_comparison(self, since=None):
        """LEGACY: Use AnalyticsRepository.get_source_comparison() instead."""
        from app.infra.mongodb.repositories.analytics_repo import get_analytics_repo
        return get_analytics_repo().get_source_comparison(since)
    
    def get_proposal_trends(self, since=None):
        """LEGACY: Use AnalyticsRepository.get_proposal_trends() instead."""
        from app.infra.mongodb.repositories.analytics_repo import get_analytics_repo
        return get_analytics_repo().get_proposal_trends(since)
    
    def get_skills_performance(self, since=None, limit=10):
        """LEGACY: Use AnalyticsRepository.get_skills_performance() instead."""
        from app.infra.mongodb.repositories.analytics_repo import get_analytics_repo
        return get_analytics_repo().get_skills_performance(since, limit)
    
    def get_industry_performance(self, since=None):
        """LEGACY: Use AnalyticsRepository.get_industry_performance() instead."""
        from app.infra.mongodb.repositories.analytics_repo import get_analytics_repo
        return get_analytics_repo().get_industry_performance(since)


# =============================================================================
# SINGLETON PATTERN
# =============================================================================

_db_manager: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Get or create database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.init_admin_collections()
    return _db_manager


def close_db():
    """Close database connection."""
    global _db_manager
    if _db_manager:
        _db_manager.disconnect()
        _db_manager = None

"""
Training Data Repository

Handles all operations for training data, chunks, and embeddings collections.
Consolidates related operations that were spread across db.py.
"""
import logging
import uuid
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pymongo import DESCENDING
from pymongo.errors import DuplicateKeyError

from app.infra.mongodb.base_repository import BaseRepository
from app.infra.mongodb.connection import get_collection

logger = logging.getLogger(__name__)


class TrainingDataRepository(BaseRepository[Dict[str, Any]]):
    """Repository for training data (jobs/projects)."""
    
    collection_name = "training_data"
    
    def insert_training_data(self, job_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Insert a new training data record with auto-generated contract_id.
        
        Args:
            job_data: Job information from user
            
        Returns:
            Dict with db_id and contract_id
        """
        try:
            # Generate unique contract_id if not provided
            if "contract_id" not in job_data or not job_data["contract_id"]:
                short_id = uuid.uuid4().hex[:8]
                job_data["contract_id"] = f"job_{short_id}"
            
            job_data["created_at"] = datetime.utcnow()
            job_data["updated_at"] = datetime.utcnow()
            
            # Ensure required fields
            job_data.setdefault("industry", "general")
            job_data.setdefault("project_status", "completed")
            job_data.setdefault("urgent_adhoc", False)
            
            result = self.collection.insert_one(job_data)
            logger.info(f"Inserted training data: {job_data['contract_id']}")
            
            return {
                "db_id": str(result.inserted_id),
                "contract_id": job_data["contract_id"]
            }
        except DuplicateKeyError:
            logger.error(f"Duplicate contract_id: {job_data.get('contract_id')}")
            raise
    
    def get_by_contract_id(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """Get training data by contract_id."""
        return self.find_one({"contract_id": contract_id})
    
    def get_all(
        self, 
        skip: int = 0, 
        limit: int = 50, 
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Get all training data with pagination and filtering."""
        try:
            filter_query = filters or {}
            results = list(
                self.collection
                .find(filter_query)
                .sort("created_at", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            for r in results:
                r["_id"] = str(r["_id"])
            return results
        except Exception as e:
            logger.error(f"Error retrieving training data: {e}")
            return []
    
    def update(self, contract_id: str, update_data: Dict[str, Any]) -> bool:
        """Update training data by contract_id."""
        try:
            update_data["updated_at"] = datetime.utcnow()
            result = self.collection.update_one(
                {"contract_id": contract_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating training data: {e}")
            raise
    
    def get_count(self) -> int:
        """Get count of training data records."""
        return self.count({})


class ChunkRepository(BaseRepository[Dict[str, Any]]):
    """Repository for chunked training data."""
    
    collection_name = "chunks"
    
    def insert_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        contract_id: str
    ) -> Tuple[int, List[str]]:
        """
        Insert smart chunks created from training data.
        
        Args:
            chunks: List of chunks from DataChunker
            contract_id: Contract ID for reference
            
        Returns:
            Tuple of (total_chunks_inserted, list_of_chunk_ids)
        """
        try:
            chunk_ids = []
            insert_docs = []
            
            for idx, chunk in enumerate(chunks):
                # Support both old and new chunk formats
                chunk_text = chunk.get("text") or chunk.get("content", "")
                chunk_type = chunk.get("chunk_type") or chunk.get("type", "text")
                
                if not chunk_text or not chunk_text.strip():
                    logger.warning(f"Skipping empty chunk {idx} for {contract_id}")
                    continue
                
                chunk_id = f"{contract_id}_{chunk_type}_{idx}"
                metadata = chunk.get("metadata", {})
                
                chunk_doc = {
                    "chunk_id": chunk_id,
                    "contract_id": contract_id,
                    "content": chunk_text,
                    "chunk_type": chunk_type,
                    "priority": chunk.get("priority", 1.0),
                    "length": chunk.get("length", len(chunk_text)),
                    "task_type": metadata.get("task_type", "").lower(),
                    "industry": metadata.get("industry", "general").lower(),
                    "skills_required": metadata.get("skills", []),
                    "company_name": metadata.get("company_name", ""),
                    "job_title": metadata.get("job_title", ""),
                    "urgency": metadata.get("urgency", "normal"),
                    "task_complexity": metadata.get("task_complexity", "medium"),
                    "is_completed": metadata.get("is_completed", False),
                    "duration_days": metadata.get("duration_days"),
                    "project_status": chunk.get("project_status", ""),
                    "deliverables": chunk.get("deliverables", []),
                    "outcome": chunk.get("outcome"),
                    "created_at": datetime.utcnow(),
                    "embedding_status": "pending"
                }
                
                insert_docs.append(chunk_doc)
                chunk_ids.append(chunk_id)
            
            if insert_docs:
                self.collection.insert_many(insert_docs, ordered=False)
                logger.info(f"Inserted {len(insert_docs)} chunks for {contract_id}")
            
            return len(insert_docs), chunk_ids
        except Exception as e:
            logger.error(f"Error inserting chunks: {e}")
            raise
    
    def get_by_contract(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a contract, sorted by priority."""
        try:
            results = list(
                self.collection
                .find({"contract_id": contract_id})
                .sort("priority", DESCENDING)
            )
            for r in results:
                r["_id"] = str(r["_id"])
            return results
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def update_embedding_status(
        self, 
        chunk_id: str, 
        status: str, 
        pinecone_vector_id: str = None
    ) -> bool:
        """Update chunk's embedding status."""
        try:
            update_doc = {
                "embedding_status": status,
                "embedding_updated_at": datetime.utcnow()
            }
            if pinecone_vector_id:
                update_doc["pinecone_vector_id"] = pinecone_vector_id
            
            result = self.collection.update_one(
                {"chunk_id": chunk_id},
                {"$set": update_doc}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating chunk status: {e}")
            return False
    
    def get_pending_embeddings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get chunks that still need to be embedded."""
        try:
            results = list(
                self.collection
                .find({"embedding_status": "pending"})
                .sort("priority", DESCENDING)
                .limit(limit)
            )
            for r in results:
                r["_id"] = str(r["_id"])
            return results
        except Exception as e:
            logger.error(f"Error retrieving pending chunks: {e}")
            return []


class EmbeddingRepository(BaseRepository[Dict[str, Any]]):
    """Repository for embeddings and embedding cache."""
    
    collection_name = "embeddings"
    
    def insert_embedding(self, embedding_data: Dict[str, Any]) -> str:
        """Insert a single embedding record."""
        try:
            embedding_data["created_at"] = datetime.utcnow()
            result = self.collection.insert_one(embedding_data)
            logger.info(f"Inserted embedding for chunk: {embedding_data.get('chunk_id')}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting embedding: {e}")
            raise
    
    def insert_batch(self, embeddings_data: List[Dict[str, Any]]) -> int:
        """Batch insert embeddings."""
        try:
            for emb in embeddings_data:
                emb["created_at"] = datetime.utcnow()
            result = self.collection.insert_many(embeddings_data, ordered=False)
            logger.info(f"Inserted {len(result.inserted_ids)} embeddings in batch")
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Error inserting embeddings batch: {e}")
            raise
    
    def get_by_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding record for a chunk."""
        result = self.find_one({"chunk_id": chunk_id})
        return result
    
    def get_by_contract(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get all embeddings for a contract."""
        try:
            results = list(self.collection.find({"contract_id": contract_id}))
            for r in results:
                r["_id"] = str(r["_id"])
            return results
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {e}")
            return []


class EmbeddingCacheRepository(BaseRepository[Dict[str, Any]]):
    """Repository for cached embeddings (fast lookup by text hash)."""
    
    collection_name = "embedding_cache"
    
    def cache_embedding(
        self, 
        text: str, 
        embedding: List[float], 
        model: str
    ) -> bool:
        """Cache embedding for quick retrieval."""
        try:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            self.collection.update_one(
                {"text_hash": text_hash, "model": model},
                {
                    "$set": {
                        "text_hash": text_hash,
                        "text": text[:500],
                        "embedding": embedding,
                        "model": model,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error caching embedding: {e}")
            return False
    
    def get_cached_embedding(
        self, 
        text: str, 
        model: str
    ) -> Optional[List[float]]:
        """Get cached embedding if available."""
        try:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            result = self.collection.find_one({
                "text_hash": text_hash,
                "model": model
            })
            if result:
                logger.debug("Found cached embedding")
                return result.get("embedding")
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached embedding: {e}")
            return None


# Singleton instances for convenience
_training_repo: Optional[TrainingDataRepository] = None
_chunk_repo: Optional[ChunkRepository] = None
_embedding_repo: Optional[EmbeddingRepository] = None
_cache_repo: Optional[EmbeddingCacheRepository] = None


def get_training_repo() -> TrainingDataRepository:
    """Get singleton TrainingDataRepository instance."""
    global _training_repo
    if _training_repo is None:
        _training_repo = TrainingDataRepository()
    return _training_repo


def get_chunk_repo() -> ChunkRepository:
    """Get singleton ChunkRepository instance."""
    global _chunk_repo
    if _chunk_repo is None:
        _chunk_repo = ChunkRepository()
    return _chunk_repo


def get_embedding_repo() -> EmbeddingRepository:
    """Get singleton EmbeddingRepository instance."""
    global _embedding_repo
    if _embedding_repo is None:
        _embedding_repo = EmbeddingRepository()
    return _embedding_repo


def get_cache_repo() -> EmbeddingCacheRepository:
    """Get singleton EmbeddingCacheRepository instance."""
    global _cache_repo
    if _cache_repo is None:
        _cache_repo = EmbeddingCacheRepository()
    return _cache_repo

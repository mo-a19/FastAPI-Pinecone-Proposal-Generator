"""
Training Service

Business logic for training data ingestion, including:
- Processing job data uploads
- Creating chunks for semantic search
- Managing embeddings
- Pinecone vector synchronization
"""
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TrainingDataInput:
    """Input for training data upload."""
    company_name: str
    job_title: str
    job_description: str
    skills_required: List[str]
    proposal_submitted: str
    industry: Optional[str] = "general"
    task_type: Optional[str] = "other"
    project_status: str = "completed"
    portfolio_urls: List[str] = field(default_factory=list)
    client_feedback: Optional[str] = None
    client_feedback_url: Optional[str] = None
    contract_id: Optional[str] = None


@dataclass
class TrainingResult:
    """Result of training data ingestion."""
    success: bool
    contract_id: Optional[str] = None
    db_id: Optional[str] = None
    chunks_created: int = 0
    embeddings_created: int = 0
    vectors_upserted: int = 0
    error_message: Optional[str] = None


class TrainingService:
    """
    Service for training data ingestion and processing.
    
    Orchestrates:
    - TrainingDataRepository for storage
    - ChunkRepository for semantic chunks
    - EmbeddingRepository for embeddings
    - PineconeService for vector storage
    """
    
    def __init__(
        self,
        db_manager=None,
        openai_service=None,
        chunker=None,
        pinecone_service=None,
        feedback_processor=None
    ):
        """
        Initialize with dependencies.
        
        Args:
            db_manager: Database manager instance
            openai_service: OpenAI service for embeddings
            chunker: Chunk processor for creating semantic chunks
            pinecone_service: Pinecone service for vector storage
            feedback_processor: Processor for client feedback
        """
        self.db = db_manager
        self.openai_service = openai_service
        self.chunker = chunker
        self.pinecone_service = pinecone_service
        self.feedback_processor = feedback_processor
    
    def ingest_training_data(self, data: TrainingDataInput) -> TrainingResult:
        """
        Complete training data ingestion pipeline.
        
        Steps:
        1. Store raw job data in MongoDB
        2. Create smart chunks (metadata, proposal, description, feedback)
        3. Generate embeddings for chunks
        4. Store in Pinecone for semantic search
        
        Args:
            data: TrainingDataInput with job details
            
        Returns:
            TrainingResult with processing statistics
        """
        try:
            logger.info(f"[TrainingService] Starting ingestion for {data.company_name} - {data.job_title}")
            
            # Step 1: Store raw job data
            job_doc, contract_id = self._store_job_data(data)
            
            # Step 2: Create smart chunks
            chunks, chunk_ids = self._create_chunks(job_doc, contract_id)
            
            # Step 3: Generate and store embeddings
            embeddings_count = self._create_embeddings(chunks, contract_id)
            
            # Step 4: Upsert to Pinecone
            vectors_count = self._upsert_to_pinecone(chunks, contract_id)
            
            # Step 5: Process feedback if provided
            if data.client_feedback:
                self._process_feedback(data.client_feedback, contract_id)
            
            logger.info(f"[TrainingService] Completed ingestion: {contract_id}")
            
            return TrainingResult(
                success=True,
                contract_id=contract_id,
                db_id=job_doc.get("db_id"),
                chunks_created=len(chunks),
                embeddings_created=embeddings_count,
                vectors_upserted=vectors_count
            )
            
        except Exception as e:
            logger.error(f"[TrainingService] Error ingesting training data: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    def _store_job_data(self, data: TrainingDataInput) -> Tuple[Dict[str, Any], str]:
        """Store raw job data in MongoDB."""
        from app.infra.mongodb.repositories import get_training_repo
        
        job_data = {
            "company": data.company_name,
            "job_title": data.job_title,
            "job_description": data.job_description,
            "skills_required": data.skills_required,
            "proposal_submitted": data.proposal_submitted,
            "industry": data.industry or "general",
            "task_type": data.task_type or "other",
            "project_status": data.project_status,
            "portfolio_urls": data.portfolio_urls,
            "client_feedback": data.client_feedback,
            "client_feedback_url": data.client_feedback_url
        }
        
        if data.contract_id:
            job_data["contract_id"] = data.contract_id
        
        repo = get_training_repo()
        result = repo.insert_training_data(job_data)
        
        return result, result["contract_id"]
    
    def _create_chunks(
        self, 
        job_doc: Dict[str, Any], 
        contract_id: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Create smart chunks from job data."""
        if not self.chunker:
            logger.warning("No chunker available, skipping chunk creation")
            return [], []
        
        # Get the full job document for chunking
        from app.infra.mongodb.repositories import get_training_repo
        repo = get_training_repo()
        full_job = repo.get_by_contract_id(contract_id)
        
        if not full_job:
            return [], []
        
        # Create chunks using advanced chunker
        chunks = self.chunker.chunk_document(full_job)
        
        # Store chunks in MongoDB
        from app.infra.mongodb.repositories import get_chunk_repo
        chunk_repo = get_chunk_repo()
        count, chunk_ids = chunk_repo.insert_chunks(chunks, contract_id)
        
        logger.info(f"[TrainingService] Created {count} chunks for {contract_id}")
        
        return chunks, chunk_ids
    
    def _create_embeddings(
        self, 
        chunks: List[Dict[str, Any]], 
        contract_id: str
    ) -> int:
        """Generate embeddings for chunks."""
        if not self.openai_service or not chunks:
            return 0
        
        from app.infra.mongodb.repositories import get_embedding_repo, get_chunk_repo
        
        embedding_repo = get_embedding_repo()
        chunk_repo = get_chunk_repo()
        embeddings_created = 0
        
        for chunk in chunks:
            try:
                chunk_id = chunk.get("chunk_id")
                content = chunk.get("text") or chunk.get("content", "")
                
                if not content:
                    continue
                
                # Generate embedding
                embedding = self.openai_service.get_embedding(content)
                
                if embedding:
                    # Store embedding
                    embedding_repo.insert_embedding({
                        "chunk_id": chunk_id,
                        "contract_id": contract_id,
                        "embedding": embedding,
                        "model": settings.OPENAI_EMBEDDING_MODEL
                    })
                    
                    # Update chunk status
                    chunk_repo.update_embedding_status(chunk_id, "completed")
                    embeddings_created += 1
                    
            except Exception as e:
                logger.error(f"Error creating embedding for chunk: {e}")
        
        logger.info(f"[TrainingService] Created {embeddings_created} embeddings")
        return embeddings_created
    
    def _upsert_to_pinecone(
        self, 
        chunks: List[Dict[str, Any]], 
        contract_id: str
    ) -> int:
        """Upsert chunk embeddings to Pinecone."""
        if not self.pinecone_service or not chunks:
            return 0
        
        try:
            vectors = []
            
            for chunk in chunks:
                embedding = chunk.get("embedding")
                if not embedding:
                    # Try to get from database
                    from app.infra.mongodb.repositories import get_embedding_repo
                    repo = get_embedding_repo()
                    emb_doc = repo.get_by_chunk(chunk.get("chunk_id"))
                    embedding = emb_doc.get("embedding") if emb_doc else None
                
                if not embedding:
                    continue
                
                metadata = {
                    "contract_id": contract_id,
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "industry": chunk.get("metadata", {}).get("industry", "general"),
                    "task_type": chunk.get("metadata", {}).get("task_type", "other"),
                    "skills": chunk.get("metadata", {}).get("skills", [])
                }
                
                vectors.append({
                    "id": chunk.get("chunk_id"),
                    "values": embedding,
                    "metadata": metadata
                })
            
            if vectors:
                self.pinecone_service.upsert_vectors(vectors)
                logger.info(f"[TrainingService] Upserted {len(vectors)} vectors to Pinecone")
                return len(vectors)
                
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {e}")
        
        return 0
    
    def _process_feedback(self, feedback_text: str, contract_id: str):
        """Process and store client feedback."""
        if not self.feedback_processor:
            return
        
        try:
            from app.infra.mongodb.repositories import get_feedback_repo
            repo = get_feedback_repo()
            
            # Process feedback with AI
            processed = self.feedback_processor.process(feedback_text)
            
            repo.insert_feedback({
                "contract_id": contract_id,
                "raw_feedback": feedback_text,
                "processed": processed,
                "sentiment": processed.get("sentiment", "neutral")
            })
            
            logger.info(f"[TrainingService] Processed feedback for {contract_id}")
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
    
    def get_training_data(
        self, 
        contract_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve training data.
        
        Args:
            contract_id: Optional specific contract to retrieve
            skip: Pagination offset
            limit: Maximum records to return
            filters: Optional query filters
            
        Returns:
            List of training data records
        """
        from app.infra.mongodb.repositories import get_training_repo
        repo = get_training_repo()
        
        if contract_id:
            result = repo.get_by_contract_id(contract_id)
            return [result] if result else []
        
        return repo.get_all(skip=skip, limit=limit, filters=filters)
    
    def delete_training_data(self, contract_ids: List[str]) -> Dict[str, int]:
        """
        Delete training data and related records.
        
        Args:
            contract_ids: List of contract IDs to delete
            
        Returns:
            Dict with deletion counts
        """
        deleted_counts = {
            "training_data": 0,
            "chunks": 0,
            "embeddings": 0,
            "vectors": 0
        }
        
        for contract_id in contract_ids:
            try:
                # Delete from repositories
                from app.infra.mongodb.repositories import (
                    get_training_repo, get_chunk_repo, get_embedding_repo
                )
                
                # Note: Would need to add delete methods to repos
                # For now, using direct db access
                if self.db:
                    self.db.db.training_data.delete_one({"contract_id": contract_id})
                    deleted_counts["training_data"] += 1
                    
                    chunk_result = self.db.db.chunks.delete_many({"contract_id": contract_id})
                    deleted_counts["chunks"] += chunk_result.deleted_count
                    
                    emb_result = self.db.db.embeddings.delete_many({"contract_id": contract_id})
                    deleted_counts["embeddings"] += emb_result.deleted_count
                
                # Delete from Pinecone
                if self.pinecone_service:
                    self.pinecone_service.delete_by_contract(contract_id)
                    deleted_counts["vectors"] += 1
                    
            except Exception as e:
                logger.error(f"Error deleting {contract_id}: {e}")
        
        return deleted_counts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training data statistics."""
        from app.infra.mongodb.repositories import get_analytics_repo
        
        repo = get_analytics_repo()
        return repo.get_database_statistics()

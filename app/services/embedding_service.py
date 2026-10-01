"""
Embedding Service - Single vector per portfolio item.

Handles:
- Generating embeddings via OpenAI
- Upserting to Pinecone with org-scoped metadata
- Batch processing for pending items
"""
import logging
from typing import List, Dict, Any, Optional

from app.config import settings
from app.utils.openai_service import OpenAIService
from app.utils.pinecone_service import PineconeService
from app.infra.mongodb.repositories.portfolio_repo import PortfolioRepository

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Single-vector embedding for portfolio items."""
    
    def __init__(
        self,
        openai_service: OpenAIService = None,
        pinecone_service: PineconeService = None
    ):
        self.openai = openai_service or OpenAIService(api_key=settings.OPENAI_API_KEY)
        self.pinecone = pinecone_service
        self._init_pinecone()
    
    def _init_pinecone(self):
        """Lazy init Pinecone if not provided."""
        if not self.pinecone and settings.PINECONE_API_KEY:
            try:
                self.pinecone = PineconeService(
                    api_key=settings.PINECONE_API_KEY,
                    environment=settings.PINECONE_ENVIRONMENT,
                    index_name=settings.PINECONE_INDEX_NAME
                )
            except Exception as e:
                logger.error(f"Failed to init Pinecone: {e}")
    
    def embed_portfolio_item(
        self,
        item: Dict[str, Any],
        portfolio_repo: PortfolioRepository
    ) -> Optional[str]:
        """
        Embed a single portfolio item and upsert to Pinecone.
        
        Returns:
            embedding_id on success, None on failure
        """
        if not self.pinecone:
            logger.warning("Pinecone not configured, skipping embedding")
            return None
        
        item_id = item.get("item_id")
        if not item_id:
            logger.error("Item missing item_id")
            return None
        
        try:
            # Build embedding text
            text = portfolio_repo.build_embedding_text(item)
            
            # Generate embedding
            embedding = self.openai.get_embedding(text)
            
            # Build metadata for filtered search
            metadata = {
                "org_id": item.get("org_id", ""),
                "profile_id": item.get("profile_id", ""),
                "project_title": item.get("project_title", ""),
                "skills": ", ".join(item.get("skills", [])),
                "industry": item.get("industry", "general"),
                "has_portfolio_url": bool(item.get("portfolio_url")),
                "type": "portfolio"
            }
            
            # Upsert to Pinecone (item_id = vector_id)
            vectors = [(item_id, embedding, metadata)]
            self.pinecone.upsert_vectors(vectors)
            
            # Mark as embedded in MongoDB
            portfolio_repo.mark_embedded(item_id, item_id)
            
            logger.info(f"Embedded portfolio item: {item_id}")
            return item_id
            
        except Exception as e:
            logger.error(f"Failed to embed {item_id}: {e}")
            return None
    
    def embed_pending_items(
        self,
        portfolio_repo: PortfolioRepository,
        org_id: str = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Batch embed all pending portfolio items.
        
        Returns:
            {"embedded": int, "failed": int, "items": [...]}
        """
        pending = portfolio_repo.list_pending_embedding(org_id=org_id, limit=limit)
        
        if not pending:
            return {"embedded": 0, "failed": 0, "items": []}
        
        embedded = []
        failed = []
        
        for item in pending:
            result = self.embed_portfolio_item(item, portfolio_repo)
            if result:
                embedded.append(item.get("item_id"))
            else:
                failed.append(item.get("item_id"))
        
        return {
            "embedded": len(embedded),
            "failed": len(failed),
            "items": embedded
        }
    
    def delete_embedding(self, item_id: str) -> bool:
        """Delete vector from Pinecone when portfolio item is deleted."""
        if not self.pinecone:
            return True
        
        try:
            self.pinecone.delete_vectors([item_id])
            logger.info(f"Deleted embedding: {item_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete embedding {item_id}: {e}")
            return False


# Singleton
_embedding_service: Optional[EmbeddingService] = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

"""
Base Repository Pattern

Abstract base class for all MongoDB repositories.
Provides common CRUD operations and query helpers.
"""
import logging
from typing import Optional, List, Dict, Any, TypeVar, Generic
from datetime import datetime
from bson.objectid import ObjectId
from pymongo.collection import Collection

from app.infra.mongodb.connection import get_collection

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Dict[str, Any])


class BaseRepository(Generic[T]):
    """
    Base repository with common CRUD operations.
    
    Subclasses should set collection_name class attribute.
    """
    
    collection_name: str = None  # Override in subclass
    
    def __init__(self):
        if not self.collection_name:
            raise ValueError(f"collection_name must be set in {self.__class__.__name__}")
    
    @property
    def collection(self) -> Collection:
        """Get the MongoDB collection."""
        return get_collection(self.collection_name)
    
    def insert_one(self, document: Dict[str, Any]) -> str:
        """
        Insert a single document.
        
        Args:
            document: Document to insert
            
        Returns:
            Inserted document ID as string
        """
        document["created_at"] = datetime.utcnow()
        result = self.collection.insert_one(document)
        return str(result.inserted_id)
    
    def insert_many(self, documents: List[Dict[str, Any]]) -> int:
        """
        Insert multiple documents.
        
        Args:
            documents: List of documents to insert
            
        Returns:
            Number of inserted documents
        """
        now = datetime.utcnow()
        for doc in documents:
            doc["created_at"] = now
        result = self.collection.insert_many(documents)
        return len(result.inserted_ids)
    
    def find_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Find document by MongoDB ObjectId.
        
        Args:
            doc_id: Document ID string
            
        Returns:
            Document or None
        """
        try:
            result = self.collection.find_one({"_id": ObjectId(doc_id)})
            if result:
                result["_id"] = str(result["_id"])
            return result
        except Exception as e:
            logger.error(f"Error finding document by id: {e}")
            return None
    
    def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single document matching query.
        
        Args:
            query: MongoDB query dict
            
        Returns:
            Document or None
        """
        result = self.collection.find_one(query)
        if result:
            result["_id"] = str(result["_id"])
        return result
    
    def find_many(
        self,
        query: Dict[str, Any] = None,
        skip: int = 0,
        limit: int = 50,
        sort: List[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Find multiple documents.
        
        Args:
            query: MongoDB query dict
            skip: Number of documents to skip
            limit: Maximum documents to return
            sort: List of (field, direction) tuples
            
        Returns:
            List of documents
        """
        cursor = self.collection.find(query or {})
        
        if sort:
            cursor = cursor.sort(sort)
        
        cursor = cursor.skip(skip).limit(limit)
        
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        
        return results
    
    def update_one(
        self,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False
    ) -> bool:
        """
        Update a single document.
        
        Args:
            query: Query to find document
            update: Update operations
            upsert: Create if not exists
            
        Returns:
            True if document was modified
        """
        if "$set" not in update and "$unset" not in update:
            update = {"$set": update}
        
        update.setdefault("$set", {})["updated_at"] = datetime.utcnow()
        
        result = self.collection.update_one(query, update, upsert=upsert)
        return result.modified_count > 0 or result.upserted_id is not None
    
    def delete_one(self, query: Dict[str, Any]) -> bool:
        """
        Delete a single document.
        
        Args:
            query: Query to find document
            
        Returns:
            True if document was deleted
        """
        result = self.collection.delete_one(query)
        return result.deleted_count > 0
    
    def count(self, query: Dict[str, Any] = None) -> int:
        """
        Count documents matching query.
        
        Args:
            query: MongoDB query dict
            
        Returns:
            Document count
        """
        return self.collection.count_documents(query or {})
    
    def exists(self, query: Dict[str, Any]) -> bool:
        """
        Check if any document matches query.
        
        Args:
            query: MongoDB query dict
            
        Returns:
            True if at least one document exists
        """
        return self.collection.find_one(query) is not None
    
    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run aggregation pipeline.
        
        Args:
            pipeline: MongoDB aggregation pipeline
            
        Returns:
            List of results
        """
        results = list(self.collection.aggregate(pipeline))
        for doc in results:
            if "_id" in doc and isinstance(doc["_id"], ObjectId):
                doc["_id"] = str(doc["_id"])
        return results

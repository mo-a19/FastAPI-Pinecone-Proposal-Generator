"""
Proposal Repository

Handles all operations for proposals and sent_proposals collections.
Consolidates proposal-related operations from db.py.
"""
import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson.objectid import ObjectId
from pymongo import DESCENDING

from app.infra.mongodb.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ProposalRepository(BaseRepository[Dict[str, Any]]):
    """Repository for generated proposals (drafts)."""
    
    collection_name = "proposals"
    
    def save_proposal(self, proposal_data: Dict[str, Any]) -> str:
        """
        Save a generated proposal draft.
        
        Args:
            proposal_data: Proposal information
            
        Returns:
            Inserted document ID
        """
        try:
            proposal_data["created_at"] = datetime.utcnow()
            result = self.collection.insert_one(proposal_data)
            logger.info(f"Saved proposal for contract: {proposal_data.get('contract_id')}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving proposal: {e}")
            raise
    
    def get_by_id(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get a proposal by MongoDB ID."""
        return self.find_by_id(proposal_id)
    
    def get_by_contract(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get all proposals for a contract."""
        try:
            results = list(
                self.collection
                .find({"contract_id": contract_id})
                .sort("created_at", DESCENDING)
            )
            for r in results:
                r["_id"] = str(r["_id"])
            return results
        except Exception as e:
            logger.error(f"Error retrieving proposals: {e}")
            return []


class SentProposalRepository(BaseRepository[Dict[str, Any]]):
    """Repository for sent proposals with outcome tracking."""
    
    collection_name = "sent_proposals"
    
    def save_sent_proposal(self, proposal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a proposal that was sent to a client.
        Tracks outcomes: sent → viewed → hired
        
        Args:
            proposal_data: Contains proposal text, job details, and metadata
            
        Returns:
            Inserted document with proposal_id
        """
        try:
            proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
            
            document = {
                "proposal_id": proposal_id,
                "job_title": proposal_data.get("job_title", ""),
                "proposal_text": proposal_data.get("proposal_text", ""),
                "skills_required": proposal_data.get("skills_required", []),
                "word_count": proposal_data.get("word_count", 0),
                "source": proposal_data.get("source", "ai_generated"),
                "outcome": "sent",
                "sent_at": datetime.utcnow(),
                "viewed_at": None,
                "hired_at": None,
                "outcome_updated_at": None,
                "discussion_initiated": False,
                "rejection_reason": None,
                "created_at": datetime.utcnow()
            }
            
            result = self.collection.insert_one(document)
            logger.info(f"Saved sent proposal: {proposal_id}")
            
            return {
                "proposal_id": proposal_id,
                "db_id": str(result.inserted_id),
                "outcome": "sent",
                "sent_at": document["sent_at"].isoformat()
            }
        except Exception as e:
            logger.error(f"Error saving sent proposal: {e}")
            raise
    
    def update_outcome(
        self,
        proposal_id: str,
        outcome: str,
        discussion_initiated: bool = False,
        rejection_reason: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update the outcome of a sent proposal.
        
        Args:
            proposal_id: Unique proposal ID
            outcome: 'viewed', 'hired', or 'rejected'
            discussion_initiated: Whether client started a chat
            rejection_reason: Optional reason if rejected
            
        Returns:
            Updated document or None
        """
        try:
            valid_outcomes = ["sent", "viewed", "hired", "rejected"]
            if outcome not in valid_outcomes:
                raise ValueError(f"Invalid outcome: {outcome}. Must be one of {valid_outcomes}")
            
            update_data = {
                "outcome": outcome,
                "outcome_updated_at": datetime.utcnow()
            }
            
            if outcome == "viewed":
                update_data["viewed_at"] = datetime.utcnow()
                update_data["discussion_initiated"] = discussion_initiated
            elif outcome == "hired":
                update_data["hired_at"] = datetime.utcnow()
                update_data["viewed_at"] = update_data.get("viewed_at") or datetime.utcnow()
                update_data["discussion_initiated"] = True
            elif outcome == "rejected":
                update_data["rejection_reason"] = rejection_reason
            
            result = self.collection.find_one_and_update(
                {"proposal_id": proposal_id},
                {"$set": update_data},
                return_document=True
            )
            
            if result:
                logger.info(f"Updated proposal {proposal_id} outcome to: {outcome}")
                return {
                    "proposal_id": proposal_id,
                    "outcome": outcome,
                    "discussion_initiated": result.get("discussion_initiated", False),
                    "rejection_reason": result.get("rejection_reason"),
                    "updated_at": update_data["outcome_updated_at"].isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Error updating proposal outcome: {e}")
            raise
    
    def get_sent_proposals(
        self,
        skip: int = 0,
        limit: int = 50,
        outcome_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get sent proposals with optional filtering by outcome.
        
        Args:
            skip: Number to skip (pagination)
            limit: Maximum to return
            outcome_filter: Filter by outcome (sent, viewed, hired, rejected)
            
        Returns:
            List of sent proposals
        """
        try:
            query = {}
            if outcome_filter:
                query["outcome"] = outcome_filter
            
            results = list(
                self.collection
                .find(query)
                .sort("sent_at", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            
            for r in results:
                r["_id"] = str(r["_id"])
            
            return results
        except Exception as e:
            logger.error(f"Error getting sent proposals: {e}")
            return []
    
    def get_by_proposal_id(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get a single sent proposal by proposal_id."""
        result = self.find_one({"proposal_id": proposal_id})
        return result
    
    def get_effective_proposals(
        self,
        min_outcome: str = "viewed",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get proposals that were at least viewed (validates Hook effectiveness).
        Used for AI learning - these proposals have proven message-market fit.
        
        Args:
            min_outcome: Minimum outcome to include ('viewed' or 'hired')
            limit: Maximum to return
            
        Returns:
            List of effective proposals
        """
        try:
            if min_outcome == "hired":
                query = {"outcome": "hired"}
            else:
                query = {"outcome": {"$in": ["viewed", "hired"]}}
            
            results = list(
                self.collection
                .find(query)
                .sort([("outcome", DESCENDING), ("sent_at", DESCENDING)])
                .limit(limit)
            )
            
            for r in results:
                r["_id"] = str(r["_id"])
            
            return results
        except Exception as e:
            logger.error(f"Error getting effective proposals: {e}")
            return []
    
    def delete_all(self) -> int:
        """
        Delete all sent proposals (for testing/cleanup).
        
        Returns:
            Number of deleted documents
        """
        try:
            result = self.collection.delete_many({})
            logger.info(f"Deleted {result.deleted_count} sent proposals")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting all sent proposals: {e}")
            raise


class FeedbackRepository(BaseRepository[Dict[str, Any]]):
    """Repository for client feedback data."""
    
    collection_name = "feedback_data"
    
    def insert_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """
        Insert client feedback (from image OCR or text).
        
        Args:
            feedback_data: Feedback information with extracted text
            
        Returns:
            Inserted document ID
        """
        try:
            feedback_data["created_at"] = datetime.utcnow()
            result = self.collection.insert_one(feedback_data)
            logger.info(f"Inserted feedback for contract: {feedback_data.get('contract_id')}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting feedback: {e}")
            raise
    
    def get_by_contract(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a contract."""
        try:
            results = list(
                self.collection
                .find({"contract_id": contract_id})
                .sort("created_at", DESCENDING)
            )
            for r in results:
                r["_id"] = str(r["_id"])
            return results
        except Exception as e:
            logger.error(f"Error retrieving feedback: {e}")
            return []


# Singleton instances
_proposal_repo: Optional[ProposalRepository] = None
_sent_proposal_repo: Optional[SentProposalRepository] = None
_feedback_repo: Optional[FeedbackRepository] = None


def get_proposal_repo() -> ProposalRepository:
    """Get singleton ProposalRepository instance."""
    global _proposal_repo
    if _proposal_repo is None:
        _proposal_repo = ProposalRepository()
    return _proposal_repo


def get_sent_proposal_repo() -> SentProposalRepository:
    """Get singleton SentProposalRepository instance."""
    global _sent_proposal_repo
    if _sent_proposal_repo is None:
        _sent_proposal_repo = SentProposalRepository()
    return _sent_proposal_repo


def get_feedback_repo() -> FeedbackRepository:
    """Get singleton FeedbackRepository instance."""
    global _feedback_repo
    if _feedback_repo is None:
        _feedback_repo = FeedbackRepository()
    return _feedback_repo

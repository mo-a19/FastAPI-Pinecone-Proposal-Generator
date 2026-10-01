"""
Feedback Processing Utility

Handles both text and image feedback:
- Text feedback: Direct input from users
- Image feedback: Extract text using GPT-4 Vision OCR, then process
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FeedbackProcessor:
    """Process and enhance client feedback"""
    
    def __init__(self, openai_service):
        """
        Initialize feedback processor
        
        Args:
            openai_service: OpenAI service instance for OCR and analysis
        """
        self.openai_service = openai_service
    
    def process_feedback(
        self,
        feedback_content: str,
        feedback_type: str = "text",
        image_path: Optional[str] = None,
        is_url: bool = False
    ) -> Dict[str, Any]:
        """
        Process feedback - extract text if image, then return structured data
        
        Args:
            feedback_content: Text content or image path/URL
            feedback_type: "text" or "image"
            image_path: Path/URL to image if feedback_type is "image"
            is_url: True if image_path is URL, False if file path
            
        Returns:
            Dictionary with:
            - extracted_text: The actual feedback text
            - feedback_type: Type of feedback
            - source: Original source (path or URL)
            - processed_at: Timestamp
        """
        try:
            extracted_text = feedback_content
            
            # If image feedback, extract text using OCR
            if feedback_type == "image" and image_path:
                logger.info(f"Processing image feedback: {image_path[:50]}...")
                extracted_text = self.openai_service.extract_text_from_image(
                    image_source=image_path,
                    is_url=is_url
                )
                logger.info(f"Successfully extracted {len(extracted_text)} characters from image")
            
            return {
                "extracted_text": extracted_text,
                "feedback_type": feedback_type,
                "source": image_path if image_path else "direct_input",
                "is_url": is_url,
                "processed_at": datetime.utcnow().isoformat(),
                "text_length": len(extracted_text),
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {
                "extracted_text": feedback_content,
                "feedback_type": feedback_type,
                "source": image_path if image_path else "direct_input",
                "processed_at": datetime.utcnow().isoformat(),
                "success": False,
                "error": str(e)
            }
    
    def prepare_feedback_for_chunking(
        self,
        extracted_text: str,
        contract_id: str,
        company_name: str,
        industry: str = "general",
        skills_required: list = None
    ) -> Dict[str, Any]:
        """
        Prepare processed feedback for later chunking
        
        This adds metadata needed for embedding and retrieval
        
        Args:
            extracted_text: The extracted feedback text
            contract_id: Associated contract ID
            company_name: Company name for context
            industry: Industry for filtering
            skills_required: Skills list for context
            
        Returns:
            Dictionary ready for insertion into feedback collection
        """
        return {
            "contract_id": contract_id,
            "feedback_text": extracted_text,
            "company_name": company_name,
            "industry": industry,
            "skills_required": skills_required or [],
            "text_length": len(extracted_text),
            "created_at": datetime.utcnow(),
            "processed": True
        }


def merge_feedback_into_training_data(
    training_data: Dict[str, Any],
    feedback_text: str
) -> Dict[str, Any]:
    """
    Merge processed feedback into training data for chunking
    
    Args:
        training_data: Original training data
        feedback_text: Processed feedback text
        
    Returns:
        Training data with updated feedback
    """
    training_data["client_feedback"] = feedback_text
    training_data["feedback_processed_at"] = datetime.utcnow()
    training_data["has_feedback"] = True
    
    logger.info(f"Merged feedback into training data for {training_data.get('contract_id')}")
    
    return training_data

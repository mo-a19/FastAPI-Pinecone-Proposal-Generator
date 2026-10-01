"""
Portfolio Training Data Routes

API endpoints for managing portfolio entries used for proposal generation:
- Add portfolio entry (POST /portfolio)
- List entries (GET /list)
- Get entry detail (GET /{contract_id})
- Update entry (PUT /update/{contract_id})
- Delete entry (DELETE /delete/{contract_id})
- OCR text extraction (POST /extract-ocr)
- Statistics (GET /stats/overview, /stats/filter-options)

Authentication:
- All endpoints require API key via X-API-Key header
- Use verify_api_key dependency for protected routes
"""
import logging
import base64
from fastapi import APIRouter, HTTPException, Query, status, UploadFile, File, Depends
from typing import List, Optional
import io
from datetime import datetime
from PIL import Image

from app.middleware.auth import verify_api_key
from app.models.job_data_schema import (
    UpdateJobDataRequest,
    JobDataResponse,
    JobDataDetailResponse,
    UploadResponse,
    ListResponse,
    DeleteResponse,
    ErrorResponse,
    JobStatisticsResponse,
    PortfolioEntryRequest,
)
from app.utils.job_data_processor import JobDataProcessor
from app.db import get_db
from app.infra.mongodb.repositories.training_repo import get_training_repo, get_chunk_repo
from app.utils.openai_service import OpenAIService
# advanced_chunker.py deleted - use stub from job_data_processor
from app.utils.job_data_processor import _StubChunker as AdvancedChunkProcessor
from app.utils.feedback_processor import FeedbackProcessor
from app.utils.pinecone_service import PineconeService
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(tags=["training-data"])

# Service instances (lazy loaded)
_processor: Optional[JobDataProcessor] = None
_pinecone_service: Optional[PineconeService] = None


def get_pinecone_service() -> Optional[PineconeService]:
    """Get or initialize Pinecone service"""
    global _pinecone_service
    if _pinecone_service is None:
        try:
            if settings.PINECONE_API_KEY:
                _pinecone_service = PineconeService(
                    api_key=settings.PINECONE_API_KEY,
                    environment=settings.PINECONE_ENVIRONMENT,
                    index_name=settings.PINECONE_INDEX_NAME,
                    namespace=settings.PINECONE_NAMESPACE,
                    dimension=settings.PINECONE_DIMENSION
                )
                logger.info("Pinecone service initialized successfully")
            else:
                logger.warning("PINECONE_API_KEY not configured, Pinecone features disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone service: {str(e)}")
    return _pinecone_service


def get_processor() -> JobDataProcessor:
    """Get or initialize job data processor"""
    global _processor
    if _processor is None:
        db = get_db()
        openai_service = OpenAIService(
            api_key=settings.OPENAI_API_KEY,
            embedding_model=settings.OPENAI_EMBEDDING_MODEL,
            llm_model=settings.OPENAI_LLM_MODEL,
            vision_model=settings.OPENAI_VISION_MODEL
        )
        chunker = AdvancedChunkProcessor()
        feedback_processor = FeedbackProcessor(openai_service)
        pinecone_service = get_pinecone_service()
        
        _processor = JobDataProcessor(
            db=db,
            openai_service=openai_service,
            chunker=chunker,
            feedback_processor=feedback_processor,
            pinecone_service=pinecone_service
        )
    return _processor


# ===================== PORTFOLIO ENTRY (NEW - PRIMARY) =====================

@router.post(
    "/portfolio",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add portfolio entry (Recommended)",
    responses={
        201: {"description": "Portfolio entry added successfully"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def add_portfolio_entry(
    entry: PortfolioEntryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Add a portfolio entry with clean, minimal fields.
    
    This is the RECOMMENDED endpoint for adding training data.
    Only requires core fields - everything else is auto-generated.
    
    **Required Fields:**
    - `client_name`: Company/client name
    - `skills`: Technologies used (array)
    - `deliverables`: What you built (array)
    
    **Optional Fields:**
    - `industry`: SaaS, E-commerce, etc.
    - `platform`: WordPress, Shopify, React, etc.
    - `portfolio_urls`: Live links to show work
    - `outcome`: {stats, loom_url} - measurable results
    - `client_feedback`: {text, url} - testimonial
    """
    try:
        processor = get_processor()
        
        # Convert to legacy format for backward compatibility
        job_dict = entry.to_legacy_format()
        
        logger.info(f"Adding portfolio entry for: {entry.client_name}")
        
        # Execute pipeline
        pipeline_result = processor.process_complete_pipeline(job_dict, save_to_pinecone=True)
        contract_id = pipeline_result["contract_id"]
        
        # Get stored data for response
        stored_job = processor.get_job_with_chunks(contract_id)
        
        return UploadResponse(
            status="success",
            db_id=str(stored_job.get("_id", "")),
            contract_id=contract_id,
            message=f"Portfolio entry for '{entry.client_name}' added successfully",
            data=JobDataResponse(
                db_id=str(stored_job.get("_id", "")),
                contract_id=contract_id,
                company_name=stored_job.get("company_name", entry.client_name),
                job_title=stored_job.get("job_title", f"Project for {entry.client_name}"),
                industry=stored_job.get("industry"),
                skills_required=stored_job.get("skills_required", entry.skills),
                task_type="portfolio",
                platform=stored_job.get("platform"),
                project_status="completed",
                urgent_adhoc=False,
                start_date=None,
                end_date=None,
                portfolio_url=entry.portfolio_urls[0] if entry.portfolio_urls else None,
                is_portfolio_entry=True,
                created_at=stored_job.get("created_at", datetime.utcnow()),
                updated_at=stored_job.get("updated_at")
            )
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Portfolio entry failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add portfolio entry: {str(e)}")


# ===================== RETRIEVE ENDPOINTS =====================

@router.get(
    "/list",
    response_model=ListResponse,
    summary="List all portfolio entries",
    responses={
        200: {"description": "Entries retrieved"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def list_job_data(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    status: Optional[str] = Query(None, description="Filter by project status"),
    company: Optional[str] = Query(None, description="Search by company name"),
    api_key: str = Depends(verify_api_key)
):
    """
    List all job training data with optional filtering and pagination
    
    **Query Parameters:**
    - `skip`: Pagination offset (default: 0)
    - `limit`: Items per page (default: 50, max: 100)
    - `industry`: Filter by industry (optional)
    - `status`: Filter by project status (optional)
    - `company`: Search by company name (optional)
    
    **Returns:**
    - List of job summaries
    - Total count
    - Pagination info
    """
    try:
        db = get_db()
        collection = db.db["portfolio_items"]
        
        # Build filter for portfolio_items schema
        filters = {}
        if industry:
            filters["industry"] = industry
        if company:
            filters["project_title"] = {"$regex": company, "$options": "i"}
        
        # Get items from portfolio_items collection
        cursor = collection.find(filters).sort("created_at", -1).skip(skip).limit(limit)
        jobs = list(cursor)
        total = collection.count_documents(filters)
        
        # Map portfolio_items schema to legacy JobDataResponse format
        items = [
            JobDataResponse(
                db_id=str(job["_id"]),
                contract_id=job.get("item_id", str(job["_id"])),
                company_name=job.get("project_title", ""),
                job_title=job.get("project_title", ""),
                industry=job.get("industry", "general"),
                skills_required=job.get("skills", []),
                task_type=job.get("industry"),  # Use industry as task_type
                platform=job.get("platform"),
                project_status="completed",
                urgent_adhoc=False,
                start_date=None,
                end_date=None,
                portfolio_url=job.get("portfolio_url"),
                is_portfolio_entry=True,
                created_at=job.get("created_at", datetime.utcnow()).isoformat() if isinstance(job.get("created_at"), datetime) else str(job.get("created_at", datetime.utcnow().isoformat())),
                updated_at=job.get("updated_at").isoformat() if isinstance(job.get("updated_at"), datetime) else job.get("updated_at")
            )
            for job in jobs
        ]
        
        return ListResponse(
            status="success",
            total=total,
            count=len(items),
            skip=skip,
            limit=limit,
            items=items
        )
    
    except Exception as e:
        logger.error(f"Error listing job data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve jobs")


@router.get(
    "/{contract_id}",
    response_model=JobDataDetailResponse,
    summary="Get entry by contract ID",
    responses={
        200: {"description": "Entry retrieved"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Entry not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_job_data(
    contract_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Retrieve detailed job data by contract ID
    
    **Returns:**
    - Complete job information
    - Chunk statistics (total chunks and embedded chunks)
    - All metadata and feedback
    """
    try:
        db = get_db()
        collection = db.db["portfolio_items"]
        
        # Search by item_id (new schema) or _id as fallback
        job_data = collection.find_one({"item_id": contract_id})
        if not job_data:
            # Try by ObjectId if it looks like one
            from bson import ObjectId
            try:
                job_data = collection.find_one({"_id": ObjectId(contract_id)})
            except:
                pass
        
        if not job_data:
            raise HTTPException(status_code=404, detail=f"Job not found: {contract_id}")
        
        # Map portfolio_items schema to legacy response format
        return JobDataDetailResponse(
            db_id=str(job_data["_id"]),
            contract_id=job_data.get("item_id", str(job_data["_id"])),
            company_name=job_data.get("project_title", ""),
            job_title=job_data.get("project_title", ""),
            job_description="",  # Not in new schema
            your_proposal_text="",  # Not in new schema
            skills_required=job_data.get("skills", []),
            industry=job_data.get("industry", "general"),
            project_status="completed",
            start_date=None,
            end_date=None,
            portfolio_url=job_data.get("portfolio_url"),
            portfolio_urls=[job_data.get("portfolio_url")] if job_data.get("portfolio_url") else [],
            temporary_link=None,
            client_feedback_url=job_data.get("client_feedback", {}).get("url") if isinstance(job_data.get("client_feedback"), dict) else None,
            client_feedback_text=job_data.get("client_feedback", {}).get("text") if isinstance(job_data.get("client_feedback"), dict) else job_data.get("client_feedback"),
            task_type=job_data.get("industry"),
            platform=job_data.get("platform"),
            urgent_adhoc=False,
            is_portfolio_entry=True,
            deliverables=job_data.get("deliverables", []),
            outcome=job_data.get("outcome"),
            created_at=job_data.get("created_at", datetime.utcnow()).isoformat() if isinstance(job_data.get("created_at"), datetime) else str(job_data.get("created_at", "")),
            updated_at=job_data.get("updated_at").isoformat() if isinstance(job_data.get("updated_at"), datetime) else job_data.get("updated_at"),
            chunks_count=1,
            embedded_chunks_count=1
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job data")


# ===================== UPDATE ENDPOINTS =====================

@router.put(
    "/update/{contract_id}",
    response_model=UploadResponse,
    summary="Update entry by contract ID",
    responses={
        200: {"description": "Entry updated successfully"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Entry not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def update_job_data(
    contract_id: str,
    update_data: UpdateJobDataRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Update existing job data entry.
    
    **Path Parameters:**
    - `contract_id`: Contract ID to update
    
    **Updatable Fields:**
    - company_name, job_title, job_description
    - your_proposal_text, skills_required, industry
    - platform, task_type, project_status
    - portfolio_urls, client_feedback_text/url
    - And all other job metadata
    
    **Note:** Chunks and embeddings will be regenerated if content changes.
    """
    try:
        db = get_db()
        collection = db.db["portfolio_items"]
        
        # Check if job exists by item_id
        existing = collection.find_one({"item_id": contract_id})
        if not existing:
            raise HTTPException(status_code=404, detail=f"Job not found: {contract_id}")
        
        # Build update dict from request (only non-None values)
        request_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}
        
        if not request_dict:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        # Map legacy field names to new portfolio_items schema
        update_dict = {}
        
        # Map company_name -> project_title
        if "company_name" in request_dict:
            update_dict["project_title"] = request_dict["company_name"]
        
        # Map skills_required -> skills
        if "skills_required" in request_dict:
            update_dict["skills"] = request_dict["skills_required"]
        
        # Direct mappings
        if "industry" in request_dict:
            update_dict["industry"] = request_dict["industry"]
        if "platform" in request_dict:
            update_dict["platform"] = request_dict["platform"]
        if "deliverables" in request_dict:
            update_dict["deliverables"] = request_dict["deliverables"]
        
        # Handle portfolio_urls -> portfolio_url (take first)
        if "portfolio_urls" in request_dict and request_dict["portfolio_urls"]:
            update_dict["portfolio_url"] = request_dict["portfolio_urls"][0] if request_dict["portfolio_urls"] else None
        
        # Handle outcome
        if "outcome" in request_dict and request_dict["outcome"] is not None:
            if hasattr(request_dict["outcome"], "model_dump"):
                update_dict["outcome"] = request_dict["outcome"].model_dump()
            else:
                update_dict["outcome"] = request_dict["outcome"]
        
        # Handle client feedback
        if "client_feedback_text" in request_dict or "client_feedback_url" in request_dict:
            update_dict["client_feedback"] = {
                "text": request_dict.get("client_feedback_text"),
                "url": str(request_dict["client_feedback_url"]) if request_dict.get("client_feedback_url") else None
            }
        
        # Add updated timestamp
        update_dict["updated_at"] = datetime.utcnow()
        
        logger.info(f"Updating {contract_id} with: {list(update_dict.keys())}")
        
        # Update in database
        result = collection.update_one(
            {"item_id": contract_id},
            {"$set": update_dict}
        )
        
        if result.modified_count == 0:
            logger.warning(f"No fields modified for {contract_id}")
        
        # Check if content changed - if so, regenerate Pinecone embedding
        content_fields = ["skills", "deliverables", "project_title", "industry"]
        content_changed = any(f in update_dict for f in content_fields)
        
        if content_changed:
            logger.info(f"Content changed for {contract_id}, regenerating Pinecone vector...")
            
            pinecone_service = get_pinecone_service()
            if pinecone_service:
                try:
                    # Delete old vector
                    pinecone_service.delete_by_contract(contract_id)
                    
                    # Get updated item and re-embed
                    updated_item = collection.find_one({"item_id": contract_id})
                    if updated_item:
                        from app.utils.openai_service import OpenAIService
                        from app.config import settings
                        import os
                        
                        api_key_openai = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
                        openai_service = OpenAIService(api_key_openai)
                        
                        # Build embedding text
                        embed_text = f"{updated_item.get('project_title', '')} {' '.join(updated_item.get('skills', []))} {' '.join(updated_item.get('deliverables', []))} {updated_item.get('industry', '')}"
                        embedding = openai_service.get_embedding(embed_text)
                        
                        if embedding:
                            metadata = {
                                "item_id": contract_id,
                                "project_title": updated_item.get("project_title", ""),
                                "skills": updated_item.get("skills", []),
                                "industry": updated_item.get("industry", ""),
                                "deliverables": updated_item.get("deliverables", [])
                            }
                            pinecone_service.upsert_vectors([(contract_id, embedding, metadata)])
                            logger.info(f"✓ Re-embedded {contract_id} to Pinecone")
                except Exception as e:
                    logger.warning(f"Error updating Pinecone: {str(e)}")
        
        # Get updated item for response
        stored_job = collection.find_one({"item_id": contract_id})
        
        response_data = JobDataResponse(
            db_id=str(stored_job["_id"]),
            contract_id=contract_id,
            company_name=stored_job.get("project_title", ""),
            job_title=stored_job.get("project_title", ""),
            industry=stored_job.get("industry"),
            skills_required=stored_job.get("skills", []),
            task_type=stored_job.get("industry"),
            platform=stored_job.get("platform"),
            project_status="completed",
            urgent_adhoc=False,
            start_date=None,
            end_date=None,
            portfolio_url=stored_job.get("portfolio_url"),
            is_portfolio_entry=True,
            created_at=stored_job.get("created_at").isoformat() if isinstance(stored_job.get("created_at"), datetime) else str(stored_job.get("created_at", "")),
            updated_at=stored_job.get("updated_at").isoformat() if isinstance(stored_job.get("updated_at"), datetime) else str(stored_job.get("updated_at", ""))
        )
        
        logger.info(f"✅ Updated job: {contract_id}")
        
        return UploadResponse(
            status="success",
            db_id=str(stored_job["_id"]),
            contract_id=contract_id,
            message=f"Job updated successfully" + (" (re-embedded)" if content_changed else ""),
            data=response_data
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating job data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update job data")


# ===================== DELETE ENDPOINTS =====================

@router.delete(
    "/delete/{contract_id}",
    response_model=DeleteResponse,
    summary="Delete entry by contract ID",
    responses={
        200: {"description": "Entry deleted"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Entry not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def delete_job_data(
    contract_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Delete job data and all associated chunks
    
    **Path Parameters:**
    - `contract_id`: Contract ID to delete
    
    **Cascading Deletes:**
    - Portfolio item record
    - Pinecone vector
    
    **Returns:**
    - Count of deleted items
    """
    try:
        db = get_db()
        collection = db.db["portfolio_items"]
        
        # Check if exists
        existing = collection.find_one({"item_id": contract_id})
        if not existing:
            raise HTTPException(status_code=404, detail=f"Contract not found: {contract_id}")
        
        # Delete from Pinecone first
        pinecone_service = get_pinecone_service()
        if pinecone_service:
            try:
                pinecone_service.delete_by_contract(contract_id)
                logger.info(f"Deleted Pinecone vector for {contract_id}")
            except Exception as e:
                logger.warning(f"Error deleting from Pinecone: {str(e)}")
        
        # Delete from MongoDB
        result = collection.delete_one({"item_id": contract_id})
        jobs_deleted = result.deleted_count
        
        return DeleteResponse(
            status="success",
            deleted_count=jobs_deleted,
            chunks_deleted=0,
            message=f"Deleted portfolio item and Pinecone vector"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete job data")


# ===================== OCR ENDPOINTS =====================

@router.post(
    "/extract-ocr",
    summary="Extract text from image using OCR (GPT-4 Vision)",
    responses={
        200: {"description": "Text extracted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid image or no image provided"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "OCR processing failed"}
    }
)
async def extract_text_from_feedback_image(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Extract text from feedback image using GPT-4 Vision OCR
    
    This endpoint accepts an image file and uses GPT-4 Vision to extract
    all text content. Perfect for processing client feedback screenshots,
    reviews, or handwritten notes.
    
    **Request:**
    - Multipart form with 'file' field containing image
    
    **Supported formats:** PNG, JPG, JPEG, GIF, WebP
    
    **Returns:**
    - extracted_text: The text content found in the image
    - success: Whether extraction was successful
    - message: Status message
    """
    try:
        # Validate file type
        allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Allowed: PNG, JPG, GIF, WebP"
            )
        
        # Validate file size (max 5MB)
        file_content = await file.read()
        if len(file_content) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 5MB limit"
            )
        
        # Validate image format
        try:
            image = Image.open(io.BytesIO(file_content))
            image.verify()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        logger.info(f"🔍 Starting OCR text extraction for file: {file.filename}")
        
        # Get OpenAI service
        processor = get_processor()
        openai_service = processor.openai_service
        
        # Convert image to base64 for Vision API
        import base64
        image_data = base64.standard_b64encode(file_content).decode("utf-8")
        
        # Determine media type
        media_type = file.content_type if file.content_type in allowed_types else "image/jpeg"
        
        # Create data URL
        data_url = f"data:{media_type};base64,{image_data}"
        
        # Extract text using Vision API via base64 data URL
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": data_url,
                "detail": "high"
            }
        }
        
        response = openai_service.client.chat.completions.create(
            model=openai_service.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {
                            "type": "text",
                            "text": """Please extract ALL text from this image. 

Extract:
1. All written text
2. All typed text
3. All labels and headers
4. All numbers and dates
5. Any feedback or review content

Format the output clearly, maintaining the structure as much as possible. 
If this is a review/feedback screenshot, extract the complete feedback text."""
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        
        extracted_text = response.choices[0].message.content
        
        logger.info(f"✅ OCR extraction completed for {file.filename} - Extracted {len(extracted_text)} characters")
        
        return {
            "success": True,
            "message": "Text extracted successfully from image",
            "extracted_text": extracted_text,
            "filename": file.filename,
            "character_count": len(extracted_text)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during OCR extraction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"OCR extraction failed: {str(e)}"
        )


# ===================== STATISTICS ENDPOINTS =====================

@router.get(
    "/stats/overview",
    response_model=JobStatisticsResponse,
    summary="Get portfolio statistics",
    responses={
        200: {"description": "Statistics retrieved"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_statistics(
    api_key: str = Depends(verify_api_key)
):
    """
    Get statistics about stored job data and chunks
    
    **Returns:**
    - Total job count
    - Chunk statistics
    - Breakdown by industry
    - Breakdown by project status
    """
    try:
        processor = get_processor()
        stats = processor.get_statistics()
        
        return JobStatisticsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.get(
    "/stats/filter-options",
    summary="Get unique filter options from database",
    responses={
        200: {"description": "Filter options retrieved"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_filter_options(
    api_key: str = Depends(verify_api_key)
):
    """
    Get unique task types and platforms from the database for filter dropdowns.
    
    **Returns:**
    - task_types: List of unique task types from database
    - platforms: List of unique platforms from database
    """
    try:
        db = get_db()
        
        # Get all portfolio items
        all_jobs = list(db.db["portfolio_items"].find({}, {"industry": 1, "platform": 1, "_id": 0}))
        
        # Extract unique industries as task types (migration unified these)
        task_types = sorted(list(set(
            job.get("industry") 
            for job in all_jobs 
            if job.get("industry")
        )))
        
        # Extract unique platforms (non-null)
        platforms = sorted(list(set(
            job.get("platform") 
            for job in all_jobs 
            if job.get("platform")
        )))
        
        return {
            "task_types": task_types,
            "platforms": platforms
        }
    
    except Exception as e:
        logger.error(f"Error getting filter options: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve filter options")



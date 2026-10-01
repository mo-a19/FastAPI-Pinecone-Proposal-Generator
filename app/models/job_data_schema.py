"""
Pydantic schemas for job data ingestion and validation

Defines request/response models for:
- Job data upload
- Chunk retrieval
- Query responses
- Error handling
"""
from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


# ===================== ENUMS =====================

class ProjectStatus(str, Enum):
    """Project status enumeration"""
    COMPLETED = "completed"
    ONGOING = "ongoing"
    PENDING = "pending"


class TaskType(str, Enum):
    """Task type enumeration for project classification"""
    WEBSITE = "website"
    MOBILE_APP = "mobile_app"
    BACKEND_API = "backend_api"
    FRONTEND = "frontend"
    FULL_STACK = "full_stack"
    CONSULTATION = "consultation"
    MAINTENANCE = "maintenance"
    CONTENT = "content"
    DESIGN = "design"
    OTHER = "other"


class ChunkType(str, Enum):
    """Chunk type enumeration"""
    METADATA = "metadata"
    PROPOSAL = "proposal"
    DESCRIPTION = "description"
    FEEDBACK = "feedback"
    SUMMARY = "summary"


# ===================== OUTCOME SCHEMA =====================

class OutcomeData(BaseModel):
    """
    Structured outcome for portfolio entries.
    
    Contains measurable results + optional video proof.
    This is critical for winning proposals - shows concrete results.
    """
    stats: Optional[str] = Field(
        None,
        max_length=500,
        description="Metrics/stats achieved (e.g., '4.2% conversion rate, 1.9s mobile load time, +22% AOV')"
    )
    loom_url: Optional[str] = Field(
        None,
        description="Loom video URL showing the work/results as proof"
    )
    
    @validator("loom_url", pre=True)
    def validate_loom_url(cls, v):
        """Validate Loom URL format"""
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        if isinstance(v, str):
            v = v.strip()
            # Accept loom.com URLs or empty
            if v and 'loom.com' not in v.lower() and not v.startswith('http'):
                return None
        return v
    
    def to_display_string(self) -> str:
        """Convert to display string for prompts"""
        parts = []
        if self.stats:
            parts.append(self.stats)
        if self.loom_url:
            parts.append(f"Video: {self.loom_url}")
        return " | ".join(parts) if parts else ""
    
    class Config:
        json_schema_extra = {
            "example": {
                "stats": "4.2% conversion (from 1.8%), 1.9s mobile load, +22% AOV",
                "loom_url": "https://www.loom.com/share/abc123"
            }
        }


# ===================== REQUEST SCHEMAS =====================

class SkillsValidator:
    """Validator for skills list"""
    
    @staticmethod
    def validate_skills(skills: List[str]) -> List[str]:
        """Ensure skills are non-empty strings"""
        if not skills:
            raise ValueError("Skills list cannot be empty")
        return [s.strip() for s in skills if s.strip()]


# ===================== CLIENT FEEDBACK SCHEMA =====================

class ClientFeedback(BaseModel):
    """
    Client feedback with optional testimonial text and review URL.
    """
    text: Optional[str] = Field(
        None,
        max_length=1000,
        description="Client testimonial/feedback quote"
    )
    url: Optional[str] = Field(
        None,
        description="Link to review (Upwork profile, Google review, etc.)"
    )
    
    @validator("url", pre=True)
    def validate_url(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v.strip() if isinstance(v, str) else v
    
    def to_display_string(self) -> str:
        """Convert to display string for prompts"""
        parts = []
        if self.text:
            parts.append(f'"{self.text}"')
        if self.url:
            parts.append(f"Review: {self.url}")
        return " | ".join(parts) if parts else ""
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Excellent work! Delivered ahead of schedule.",
                "url": "https://upwork.com/freelancers/~abc123"
            }
        }


# ===================== PORTFOLIO ENTRY SCHEMA (PRIMARY) =====================

class PortfolioEntryRequest(BaseModel):
    """
    Clean portfolio entry schema - the PRIMARY way to add training data.
    
    7 core fields:
    - client_name: Who you worked for
    - industry, platform: Categorization
    - skills, portfolio_urls: Proof of work
    - deliverables, outcome: What you built + results
    - client_feedback: Testimonial (text + URL)
    """
    # Client/Company name - who you worked for
    client_name: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Company or client name (who you built this for)"
    )
    
    # Categorization
    industry: Optional[str] = Field(
        None,
        max_length=100,
        description="Industry (SaaS, E-commerce, Healthcare, etc.)"
    )
    platform: Optional[str] = Field(
        None,
        max_length=50,
        description="Platform/tech (WordPress, Shopify, React, etc.)"
    )
    
    # Skills & proof
    skills: List[str] = Field(
        ...,
        min_items=1,
        description="Skills/technologies used"
    )
    portfolio_urls: Optional[List[str]] = Field(
        default_factory=list,
        description="Live links to show the work"
    )
    
    # What you delivered + results
    deliverables: List[str] = Field(
        ...,
        min_items=1,
        description="What you actually built/delivered"
    )
    outcome: Optional[OutcomeData] = Field(
        None,
        description="Results achieved (stats + optional Loom video)"
    )
    
    # Client testimonial with text and/or URL
    client_feedback: Optional[ClientFeedback] = Field(
        None,
        description="Client testimonial (text quote and/or review URL)"
    )
    
    @validator("skills", pre=True)
    def validate_skills(cls, v):
        if isinstance(v, str):
            import json
            try:
                v = json.loads(v)
            except:
                v = [s.strip() for s in v.split(",") if s.strip()]
        if not v:
            raise ValueError("At least one skill is required")
        return [s.strip() for s in v if s.strip()][:15]
    
    @validator("deliverables", pre=True)
    def validate_deliverables(cls, v):
        if isinstance(v, str):
            import json
            try:
                v = json.loads(v)
            except:
                v = [d.strip() for d in v.split(",") if d.strip()]
        if not v:
            raise ValueError("At least one deliverable is required")
        return [d.strip() for d in v if d.strip()][:10]
    
    @validator("portfolio_urls", pre=True)
    def validate_urls(cls, v):
        if not v:
            return []
        if isinstance(v, str):
            import json
            try:
                v = json.loads(v)
            except:
                v = [v] if v.strip() else []
        return [u.strip() for u in v if u and isinstance(u, str) and u.strip()][:5]
    
    @validator("outcome", pre=True)
    def parse_outcome(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return OutcomeData(**v)
        return v
    
    @validator("client_feedback", pre=True)
    def parse_feedback(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            # Legacy: just text string
            return ClientFeedback(text=v) if v.strip() else None
        if isinstance(v, dict):
            return ClientFeedback(**v)
        return v
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert to legacy JobDataUploadRequest format for backward compatibility.
        Auto-generates required fields that are no longer in the UI.
        """
        import time
        timestamp = int(time.time())
        skills_str = "-".join(self.skills[:3])[:30]
        
        # Extract feedback text and URL
        feedback_text = self.client_feedback.text if self.client_feedback else None
        feedback_url = self.client_feedback.url if self.client_feedback else None
        
        return {
            # Use client_name as company_name for display
            "company_name": self.client_name,
            "job_title": f"Project for {self.client_name}",
            "job_description": f"Built for {self.client_name}. Skills: {', '.join(self.skills)}. Deliverables: {', '.join(self.deliverables)}.",
            "your_proposal_text": "Portfolio entry.",
            
            # Core fields
            "skills_required": self.skills,
            "industry": self.industry,
            "platform": self.platform or "External",
            "portfolio_urls": self.portfolio_urls,
            "deliverables": self.deliverables,
            "outcome": self.outcome.model_dump() if self.outcome else None,
            "client_feedback_text": feedback_text,
            "client_feedback_url": feedback_url,
            
            # Auto-generated defaults
            "task_type": "portfolio",
            "project_status": "completed",
            "is_portfolio_entry": True,
            "urgent_adhoc": False,
            "start_date": None,
            "end_date": None,
            "temporary_link": None,
            "is_portfolio_entry": True,
        }
    
    class Config:
        json_schema_extra = {
            "example": {
                "client_name": "TechCorp Inc",
                "industry": "E-commerce",
                "platform": "Shopify",
                "skills": ["Shopify", "Liquid", "JavaScript"],
                "portfolio_urls": ["https://store.example.com"],
                "deliverables": ["Custom theme", "Product pages", "Checkout optimization"],
                "outcome": {
                    "stats": "4.2% conversion rate, 1.9s mobile load",
                    "loom_url": "https://loom.com/share/abc123"
                },
                "client_feedback": {
                    "text": "Excellent work! Delivered ahead of schedule.",
                    "url": "https://upwork.com/freelancers/~abc123"
                }
            }
        }


# ===================== LEGACY SCHEMA (Backward Compatibility) =====================

class JobDataUploadRequest(BaseModel):
    """
    LEGACY: Full request model for backward compatibility.
    
    New submissions should use PortfolioEntryRequest instead.
    This schema is kept for existing training data and API compatibility.
    """
    
    contract_id: Optional[str] = Field(
        None,
        description="Unique contract ID. Auto-generated if not provided"
    )
    company_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the company"
    )
    job_title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Job title/position"
    )
    job_description: str = Field(
        ...,
        min_length=10,
        description="Full job description"
    )
    your_proposal_text: str = Field(
        ...,
        min_length=10,
        description="Your proposal that was submitted"
    )
    skills_required: List[str] = Field(
        ...,
        min_items=1,
        description="List of required skills"
    )
    industry: Optional[str] = Field(
        None,
        max_length=100,
        description="Industry of the job (optional)"
    )
    start_date: Optional[str] = Field(
        None,
        description="Project start date (ISO format)"
    )
    end_date: Optional[str] = Field(
        None,
        description="Project end date (ISO format)"
    )
    portfolio_urls: Optional[List[str]] = Field(
        default_factory=list,
        description="List of portfolio or past work URLs"
    )
    temporary_link: Optional[str] = Field(
        None,
        description="Temporary link/URL for additional resources"
    )
    client_feedback_url: Optional[HttpUrl] = Field(
        None,
        description="URL to client feedback (optional for portfolio entries)"
    )
    client_feedback_text: Optional[str] = Field(
        None,
        description="Client feedback text (optional, extracted from URL)"
    )
    project_status: str = Field(
        "completed",
        description="Status of the project: completed, ongoing, cancelled"
    )
    task_type: str = Field(
        "other",
        description="Type of task/project"
    )
    other_task_type: Optional[str] = Field(
        None,
        description="Custom task type if task_type is 'other'"
    )
    platform: Optional[str] = Field(
        None,
        description="Platform/technology (WordPress, Shopify, React, etc.)"
    )
    urgent_adhoc: bool = Field(
        False,
        description="Whether this was an urgent/adhoc project"
    )
    is_portfolio_entry: bool = Field(
        False,
        description="Whether this is a portfolio entry (vs standard Upwork training entry)"
    )
    
    # ===== DELIVERABLES & OUTCOMES (Critical for matching) =====
    deliverables: Optional[List[str]] = Field(
        default_factory=list,
        description="What was actually built/delivered (e.g., 'Custom dashboard', 'Payment integration')"
    )
    # Structured outcome with stats + video proof
    outcome: Optional[OutcomeData] = Field(
        None,
        description="Structured outcome with metrics and optional Loom video proof"
    )
    
    @validator("skills_required", pre=True)
    def validate_skills(cls, v):
        """Validate and clean skills list"""
        if isinstance(v, str):
            import json
            try:
                v = json.loads(v)
            except:
                v = [v] if v else []
        if not v:
            raise ValueError("Skills list cannot be empty")
        return SkillsValidator.validate_skills(v)
    
    @validator("outcome", pre=True)
    def parse_outcome(cls, v):
        """Parse outcome from dict or OutcomeData"""
        if v is None:
            return None
        if isinstance(v, dict):
            return OutcomeData(**v)
        return v
    
    @validator("company_name", "job_title", "industry")
    def strip_whitespace(cls, v):
        """Strip whitespace from string fields"""
        if isinstance(v, str):
            return v.strip()
        return v
    
    @validator("portfolio_urls", pre=True)
    def validate_portfolio_urls(cls, v):
        """Validate portfolio URLs"""
        if not v:
            return []
        if isinstance(v, str):
            import json
            try:
                v = json.loads(v)
            except:
                v = [v] if v else []
        if not isinstance(v, list):
            return []
        return [url.strip() for url in v if url and isinstance(url, str) and url.strip()]
    
    @validator("deliverables", pre=True)
    def validate_deliverables(cls, v):
        """Validate deliverables list"""
        if not v:
            return []
        if isinstance(v, str):
            import json
            try:
                v = json.loads(v)
            except:
                v = [v] if v else []
        if not isinstance(v, list):
            return []
        return [d.strip() for d in v if d and isinstance(d, str) and d.strip()]
    
    @validator("task_type", pre=True)
    def validate_task_type(cls, v):
        """Validate task type"""
        if isinstance(v, str):
            v = v.strip().lower()
            return v if v else "other"
        return v or "other"
    
    @validator("client_feedback_text", pre=True)
    def validate_client_feedback_text(cls, v):
        """Ensure client_feedback_text is properly handled"""
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        if isinstance(v, str):
            return v.strip()
        return v
    
    @validator("client_feedback_url", pre=True)
    def validate_client_feedback_url(cls, v):
        """Ensure empty strings are converted to None for HttpUrl validation"""
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v
    
    @validator("start_date", "end_date", pre=True)
    def validate_dates(cls, v):
        """Handle optional date fields"""
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "company_name": "TechCorp Inc",
                "job_title": "Senior Backend Engineer",
                "job_description": "We're looking for an experienced backend engineer...",
                "your_proposal_text": "I'm excited to work on this project...",
                "skills_required": ["Python", "FastAPI", "PostgreSQL"],
                "industry": "Technology",
                "task_type": "backend_api",
                "project_status": "completed",
                "urgent_adhoc": False,
                "start_date": "2024-12-01",
                "end_date": "2024-12-15",
                "portfolio_urls": ["https://github.com/yourname", "https://yourportfolio.com"],
                "client_feedback_url": "https://upwork.com/reviews/feedback-123",
                "client_feedback_text": "Great work! Very responsive."
            }
        }


class UpdateJobDataRequest(BaseModel):
    """Request model for updating job data"""
    
    company_name: Optional[str] = Field(None, min_length=1, max_length=255)
    job_title: Optional[str] = Field(None, min_length=1, max_length=255)
    job_description: Optional[str] = Field(None, min_length=10)
    your_proposal_text: Optional[str] = Field(None, min_length=10)
    skills_required: Optional[List[str]] = Field(None, min_items=1)
    industry: Optional[str] = Field(None, max_length=100)
    client_feedback_url: Optional[HttpUrl] = Field(None, description="URL to client feedback")
    client_feedback_text: Optional[str] = Field(None, description="Client feedback text")
    start_date: Optional[str] = Field(None, description="Project start date (ISO format)")
    end_date: Optional[str] = Field(None, description="Project end date (ISO format)")
    portfolio_urls: Optional[List[str]] = Field(None, description="List of portfolio URLs")
    project_status: Optional[str] = Field(None)
    task_type: Optional[str] = Field(None, description="Type of task/project")
    other_task_type: Optional[str] = Field(None, description="Custom task type")
    platform: Optional[str] = Field(None, description="Platform/technology (WordPress, Shopify, React, etc.)")
    urgent_adhoc: Optional[bool] = Field(None)
    is_portfolio_entry: Optional[bool] = Field(None, description="Whether this is a portfolio entry")
    
    # ===== DELIVERABLES & OUTCOMES =====
    deliverables: Optional[List[str]] = Field(None, description="What was built/delivered")
    outcome: Optional[OutcomeData] = Field(None, description="Structured outcome with stats + Loom proof")
    
    @validator("skills_required", pre=True, always=True)
    def validate_skills(cls, v):
        """Validate and clean skills list if provided"""
        if v is None:
            return None
        return SkillsValidator.validate_skills(v)
    
    @validator("outcome", pre=True)
    def parse_outcome(cls, v):
        """Parse outcome from dict or OutcomeData"""
        if v is None:
            return None
        if isinstance(v, dict):
            return OutcomeData(**v)
        return v
    
    class Config:
        use_enum_values = True


class DeleteJobsRequest(BaseModel):
    """Request model for bulk deletion"""
    
    contract_ids: List[str] = Field(
        ...,
        min_items=1,
        description="List of contract IDs to delete"
    )


# ===================== RESPONSE SCHEMAS =====================

class JobDataResponse(BaseModel):
    """Response model for job data - includes key fields for training"""
    
    db_id: str = Field(..., description="MongoDB document ID")
    contract_id: str = Field(..., description="Unique contract ID")
    company_name: str = Field(..., description="Company name")
    job_title: str = Field(..., description="Job title/position")
    industry: Optional[str] = Field(None, description="Industry sector")
    skills_required: Optional[List[str]] = Field(None, description="Required skills")
    task_type: Optional[str] = Field(None, description="Type of task/project")
    platform: Optional[str] = Field(None, description="Platform/technology")
    project_status: str = Field(..., description="Project status")
    urgent_adhoc: Optional[bool] = Field(None, description="Is urgent/adhoc project")
    start_date: Optional[str] = Field(None, description="Project start date")
    end_date: Optional[str] = Field(None, description="Project end date")
    portfolio_url: Optional[str] = Field(None, description="Portfolio URL")
    is_portfolio_entry: Optional[bool] = Field(None, description="Whether this is a portfolio entry")
    created_at: datetime = Field(..., description="Created timestamp")
    updated_at: Optional[datetime] = Field(None, description="Updated timestamp")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "db_id": "507f1f77bcf86cd799439011",
                "contract_id": "550e8400-e29b-41d4-a716-446655440000",
                "company_name": "TechCorp Inc",
                "job_title": "Senior Backend Engineer",
                "industry": "Technology",
                "skills_required": ["Python", "FastAPI", "PostgreSQL"],
                "task_type": "backend_api",
                "platform": "Python",
                "project_status": "completed",
                "urgent_adhoc": False,
                "start_date": "2025-12-01",
                "end_date": "2025-12-15",
                "portfolio_url": "https://github.com/yourname",
                "created_at": "2025-12-03T10:30:00Z",
                "updated_at": "2025-12-03T10:30:00Z"
            }
        }


class JobDataDetailResponse(JobDataResponse):
    """Detailed response model for job data"""
    
    job_description: str
    your_proposal_text: str
    skills_required: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    portfolio_url: Optional[str] = None
    portfolio_urls: Optional[List[str]] = Field(None, description="List of portfolio URLs")
    temporary_link: Optional[str] = Field(None, description="Temporary link/URL for additional resources")
    client_feedback_url: Optional[str] = None
    client_feedback_text: Optional[str] = None
    task_type: Optional[str] = Field(None, description="Type of task/project")
    platform: Optional[str] = Field(None, description="Platform/technology")
    urgent_adhoc: bool
    is_portfolio_entry: bool = Field(False, description="Whether this is a portfolio entry")
    deliverables: Optional[List[str]] = Field(None, description="What was built/delivered")
    outcome: Optional[OutcomeData] = Field(None, description="Structured outcome with stats + Loom proof")
    chunks_count: Optional[int] = Field(None, description="Number of chunks created")
    embedded_chunks_count: Optional[int] = Field(None, description="Number of embedded chunks")


class ChunkResponse(BaseModel):
    """Response model for chunk data"""
    
    chunk_id: str
    contract_id: str
    content: str
    chunk_type: str
    priority: float
    length: int
    industry: str
    skills_required: List[str]
    company_name: str
    project_status: str
    embedding_status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Response for successful upload"""
    
    status: str = Field("success", description="Operation status")
    db_id: str = Field(..., description="MongoDB document ID")
    contract_id: str = Field(..., description="Unique contract ID")
    message: str = Field(..., description="Success message")
    data: JobDataResponse
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "db_id": "507f1f77bcf86cd799439011",
                "contract_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "Job data uploaded successfully",
                "data": {
                    "db_id": "507f1f77bcf86cd799439011",
                    "contract_id": "550e8400-e29b-41d4-a716-446655440000",
                    "company_name": "TechCorp Inc",
                    "job_title": "Senior Backend Engineer",
                    "industry": "Technology",
                    "project_status": "completed",
                    "created_at": "2025-12-03T10:30:00Z"
                }
            }
        }


class ListResponse(BaseModel):
    """Response for list operations"""
    
    status: str = Field("success")
    total: int = Field(..., description="Total count")
    count: int = Field(..., description="Items returned")
    skip: int = Field(..., description="Items skipped")
    limit: int = Field(..., description="Limit applied")
    items: List[Any] = Field(..., description="List of items")


class DeleteResponse(BaseModel):
    """Response for deletion"""
    
    status: str = Field("success")
    deleted_count: int = Field(..., description="Number of deleted items")
    chunks_deleted: int = Field(..., description="Number of associated chunks deleted")
    message: str = Field(..., description="Success message")


class ErrorResponse(BaseModel):
    """Response for errors"""
    
    status: str = Field("error")
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Additional details")


# ===================== STATISTICS SCHEMAS =====================

class JobStatisticsResponse(BaseModel):
    """Statistics about job data - optimized for proposal generation"""
    
    total_jobs: int = Field(..., description="Total job records")
    total_chunks: int = Field(0, description="Total chunks created")
    chunks_embedded: int = Field(0, description="Chunks with embeddings")
    chunks_pending: int = Field(0, description="Chunks pending embedding")
    by_industry: Dict[str, int] = Field(default_factory=dict, description="Jobs by industry")
    by_status: Dict[str, int] = Field(default_factory=dict, description="Jobs by project status")
    
    # New proposal-focused metrics
    avg_proposal_length: int = Field(0, description="Average proposal length in characters")
    completion_rate: float = Field(0.0, description="Percentage of completed projects")
    success_rate: float = Field(0.0, description="Percentage of successful proposals")
    avg_satisfaction_score: float = Field(0.0, description="Average client satisfaction score")
    top_skills: List[str] = Field(default_factory=list, description="Top used skills")
    by_task_type: Dict[str, int] = Field(default_factory=dict, description="Jobs by task type")

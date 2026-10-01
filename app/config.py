"""
Application Configuration
Load settings from environment variables with validation
"""
import os
from typing import Optional

class Settings:
    """Application configuration from environment variables"""
    
    # MongoDB Configuration
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "proposal_generator")
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX", "proposal-engine")
    PINECONE_HOST: str = os.getenv("PINECONE_HOST", "")  # e.g., https://<index-name>-<id>.svc.<region>.pinecone.io
    PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "portfolio")
    PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", "3072"))  # Must match text-embedding-3-large dimensions
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
    OPENAI_VISION_MODEL: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
    
    # Application Configuration
    APP_NAME: str = "AI Proposal Generator"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MIN_CHUNK_SIZE: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "2048"))
    
    # Embedding Configuration
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
    EMBEDDING_CACHE_ENABLED: bool = os.getenv("EMBEDDING_CACHE_ENABLED", "True").lower() == "true"
    
    # Proposal Generation Configuration
    PROPOSAL_TOP_K: int = int(os.getenv("PROPOSAL_TOP_K", "5"))
    MIN_SIMILARITY_SCORE: float = float(os.getenv("MIN_SIMILARITY_SCORE", "0.5"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Rate Limiting
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "30/minute")  # Format: "count/period"
    
    # CORS - comma-separated origins
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "https://genproposals.com,https://www.genproposals.com,http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5500,http://127.0.0.1:5501,http://localhost:5500,http://localhost:5501")
    
    # Admin API Key for protected endpoints
    ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "")
    
    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    ALLOWED_IMAGE_FORMATS: tuple = ("png", "jpg", "jpeg", "gif", "webp")

# Initialize settings
settings = Settings()

def validate_settings() -> bool:
    """
    Validate that all required settings are configured
    
    Returns:
        True if all required settings are present
        
    Raises:
        ValueError: If required settings are missing
    """
    required_keys = {
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        "PINECONE_API_KEY": settings.PINECONE_API_KEY,
        "MONGODB_URI": settings.MONGODB_URI,
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
    
    return True

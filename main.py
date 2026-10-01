from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

from app.config import settings
from app.routes import job_data_router, jobs_router, data_entry_router
from app.routes.proposals import router as proposals_router
from app.routes.profile import router as profile_router
from app.routes.portfolio import router as portfolio_router
from app.routes.analytics import router as analytics_router
from app.routes.admin import router as admin_router
from app.routes.lab import router as lab_router
from app.middleware.auth import verify_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])

app = FastAPI(
    title="AI Proposal Generator Backend",
    version="1.0.0",
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - parse origins from config
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins and origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "AI Proposal Generator Backend",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/auth/verify")
async def verify_api_key_endpoint(auth_result: dict = Depends(verify_api_key)):
    """Verify API key and return role/permissions with tenant context"""
    role = auth_result.get("role", "user")
    return {
        "valid": True,
        "role": role,
        "permissions": auth_result.get("permissions", []),
        "is_admin": role in ("admin", "super_admin"),
        "is_super_admin": role == "super_admin",
        # Multi-tenant context
        "org_id": auth_result.get("org_id"),
        "user_id": auth_result.get("user_id"),
        "name": auth_result.get("name"),
    }


app.include_router(job_data_router, prefix="/api/job-data")  # LEGACY
app.include_router(jobs_router)  # New job ingestion: /api/jobs
app.include_router(proposals_router)
app.include_router(profile_router)
app.include_router(portfolio_router)
app.include_router(analytics_router)
app.include_router(admin_router)
app.include_router(lab_router)
app.include_router(data_entry_router, prefix="/api")  # Manual data entry: /api/data-entries


@app.on_event("startup")
async def warmup_proposal_corpus():
    """
    Warm up the past-proposal corpus in a background thread so the first
    /proposals/similar-job request is instant in production.
    Failures are non-fatal: the endpoint will simply load lazily on demand.
    """
    import threading
    def _warm():
        try:
            from app.services.proposal_corpus_service import ProposalCorpusService
            from app.utils.openai_service import OpenAIService
            svc = ProposalCorpusService.get_instance(
                OpenAIService(api_key=settings.OPENAI_API_KEY)
            )
            svc.ensure_loaded()
            logger.info("[Startup] Proposal corpus warmed up")
        except Exception as e:
            logger.warning(f"[Startup] Proposal corpus warmup skipped: {e}")
    threading.Thread(target=_warm, daemon=True, name="corpus-warmup").start()

# Serve frontend static files - mount AFTER API routes to avoid conflicts
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/frontend", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )

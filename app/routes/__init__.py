"""
Routes package - exports all API routers

LEGACY NOTICE:
- job_data_router: Old 5-chunk strategy, deprecated. Use jobs_router instead.
- jobs_router: New clean job ingestion API
"""
from app.routes.job_data_ingestion import router as job_data_router  # LEGACY
from app.routes.jobs import router as jobs_router
from app.routes.data_entry import router as data_entry_router

__all__ = ["job_data_router", "jobs_router", "data_entry_router"]

"""
Analytics Dashboard Routes

Provides conversion funnel, AI vs Manual comparison, and trend data.
All endpoints require authentication.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum

from app.middleware.auth import verify_api_key
from app.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ===================== ENUMS & MODELS =====================

class DateRange(str, Enum):
    WEEK = "7d"
    MONTH = "30d"
    QUARTER = "90d"
    ALL = "all"


class FunnelStage(BaseModel):
    """Single funnel stage data"""
    stage: str
    count: int
    percentage: float


class FunnelResponse(BaseModel):
    """Conversion funnel response"""
    success: bool
    funnel: List[FunnelStage]
    total_sent: int


class ComparisonResponse(BaseModel):
    """AI vs Manual comparison"""
    success: bool
    ai_generated: Dict[str, Any]
    manual: Dict[str, Any]
    ai_effectiveness: float = Field(description="How much more effective AI is (1.5 = 50% better)")


class SummaryResponse(BaseModel):
    """Dashboard summary stats"""
    success: bool
    total_proposals: int
    ai_generated_count: int
    manual_count: int
    total_hired: int
    overall_hire_rate: float
    ai_hire_rate: float
    manual_hire_rate: float
    total_viewed: int
    message_market_fit: float


class TrendPoint(BaseModel):
    """Single trend data point"""
    date: str
    sent: int
    hired: int
    hire_rate: float


class TrendsResponse(BaseModel):
    """Trends over time"""
    success: bool
    period: str
    data: List[TrendPoint]


class SkillPerformance(BaseModel):
    """Skill performance data"""
    skill: str
    total: int
    hired: int
    hire_rate: float


class SkillsResponse(BaseModel):
    """Skills performance response"""
    success: bool
    skills: List[SkillPerformance]


# ===================== HELPER =====================

def _get_date_filter(range: DateRange) -> Optional[datetime]:
    """Convert date range to datetime filter"""
    if range == DateRange.ALL:
        return None
    days = {"7d": 7, "30d": 30, "90d": 90}
    return datetime.utcnow() - timedelta(days=days[range.value])


# ===================== ENDPOINTS =====================

@router.get("/summary", response_model=SummaryResponse)
async def get_summary(
    range: DateRange = Query(DateRange.ALL, description="Date range filter"),
    api_key: str = Depends(verify_api_key)
):
    """
    Get dashboard summary statistics.
    
    Returns total proposals, AI vs manual split, and conversion rates.
    """
    try:
        db = get_db()
        stats = db.get_analytics_summary(_get_date_filter(range))
        return SummaryResponse(success=True, **stats)
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        raise HTTPException(500, str(e))


@router.get("/funnel", response_model=FunnelResponse)
async def get_funnel(
    range: DateRange = Query(DateRange.ALL),
    source: Optional[str] = Query(None, description="Filter by source: ai_generated or manual"),
    api_key: str = Depends(verify_api_key)
):
    """
    Get conversion funnel data.
    
    Stages: Sent → Viewed → Discussed → Hired
    """
    try:
        db = get_db()
        funnel_data = db.get_conversion_funnel(_get_date_filter(range), source)
        return FunnelResponse(success=True, **funnel_data)
    except Exception as e:
        logger.error(f"Error getting funnel: {e}")
        raise HTTPException(500, str(e))


class FunnelSourceData(BaseModel):
    """Funnel data for a single source"""
    generated: int
    sent: int
    viewed: int
    discussed: int
    hired: int
    rates: Dict[str, float]


class CombinedFunnelResponse(BaseModel):
    """Combined funnel response with AI and Manual data"""
    success: bool
    ai: FunnelSourceData
    manual: FunnelSourceData
    totals: Dict[str, int]
    ai_share: float = Field(description="Percentage of sent proposals that are AI-generated")


@router.get("/funnel-combined", response_model=CombinedFunnelResponse)
async def get_combined_funnel(
    range: DateRange = Query(DateRange.ALL),
    api_key: str = Depends(verify_api_key)
):
    """
    Get combined conversion funnel for AI-generated and Manual proposals.
    
    Returns full flow: Generated → Sent → Viewed → Discussed → Hired
    Both AI and Manual data in a single response for unified visualization.
    """
    try:
        db = get_db()
        funnel_data = db.get_combined_funnel(_get_date_filter(range))
        return CombinedFunnelResponse(success=True, **funnel_data)
    except Exception as e:
        logger.error(f"Error getting combined funnel: {e}")
        raise HTTPException(500, str(e))


@router.get("/comparison", response_model=ComparisonResponse)
async def get_comparison(
    range: DateRange = Query(DateRange.ALL),
    api_key: str = Depends(verify_api_key)
):
    """
    Compare AI-generated vs manually written proposals.
    
    Shows conversion rates and calculates AI effectiveness multiplier.
    """
    try:
        db = get_db()
        comparison = db.get_source_comparison(_get_date_filter(range))
        return ComparisonResponse(success=True, **comparison)
    except Exception as e:
        logger.error(f"Error getting comparison: {e}")
        raise HTTPException(500, str(e))


@router.get("/trends", response_model=TrendsResponse)
async def get_trends(
    range: DateRange = Query(DateRange.MONTH),
    api_key: str = Depends(verify_api_key)
):
    """
    Get proposal trends over time.
    
    Returns daily/weekly aggregated data for charts.
    """
    try:
        db = get_db()
        trends = db.get_proposal_trends(_get_date_filter(range))
        return TrendsResponse(success=True, period=range.value, data=trends)
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(500, str(e))


@router.get("/skills", response_model=SkillsResponse)
async def get_skills_performance(
    range: DateRange = Query(DateRange.ALL),
    limit: int = Query(10, ge=1, le=50),
    api_key: str = Depends(verify_api_key)
):
    """
    Get performance by skill.
    
    Shows which skills have the highest hire rates.
    """
    try:
        db = get_db()
        skills = db.get_skills_performance(_get_date_filter(range), limit)
        return SkillsResponse(success=True, skills=skills)
    except Exception as e:
        logger.error(f"Error getting skills performance: {e}")
        raise HTTPException(500, str(e))


@router.get("/industry", response_model=Dict[str, Any])
async def get_industry_breakdown(
    range: DateRange = Query(DateRange.ALL),
    api_key: str = Depends(verify_api_key)
):
    """
    Get performance breakdown by industry.
    """
    try:
        db = get_db()
        data = db.get_industry_performance(_get_date_filter(range))
        return {"success": True, "industries": data}
    except Exception as e:
        logger.error(f"Error getting industry breakdown: {e}")
        raise HTTPException(500, str(e))

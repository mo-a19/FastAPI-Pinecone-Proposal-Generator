"""
Analytics Service

Business logic for proposal analytics, including:
- Conversion funnel analysis
- Source comparison (AI vs Manual)
- Performance trends
- Skills and industry breakdowns
"""
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ConversionFunnel:
    """Conversion funnel data."""
    sent: int = 0
    viewed: int = 0
    discussed: int = 0
    hired: int = 0
    
    @property
    def view_rate(self) -> float:
        return round(self.viewed / self.sent * 100, 1) if self.sent else 0
    
    @property
    def discuss_rate(self) -> float:
        return round(self.discussed / self.sent * 100, 1) if self.sent else 0
    
    @property
    def hire_rate(self) -> float:
        return round(self.hired / self.sent * 100, 1) if self.sent else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sent": self.sent,
            "viewed": self.viewed,
            "discussed": self.discussed,
            "hired": self.hired,
            "view_rate": self.view_rate,
            "discuss_rate": self.discuss_rate,
            "hire_rate": self.hire_rate
        }


@dataclass
class SourceComparison:
    """AI vs Manual proposal comparison."""
    ai_total: int = 0
    ai_hired: int = 0
    ai_viewed: int = 0
    manual_total: int = 0
    manual_hired: int = 0
    manual_viewed: int = 0
    
    @property
    def ai_hire_rate(self) -> float:
        return round(self.ai_hired / self.ai_total * 100, 1) if self.ai_total else 0
    
    @property
    def manual_hire_rate(self) -> float:
        return round(self.manual_hired / self.manual_total * 100, 1) if self.manual_total else 0
    
    @property
    def ai_effectiveness(self) -> float:
        """How much better AI performs vs manual."""
        if self.manual_hire_rate > 0:
            return round(self.ai_hire_rate / self.manual_hire_rate, 2)
        elif self.ai_hire_rate > 0:
            return float("inf")
        return 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ai_generated": {
                "total": self.ai_total,
                "hired": self.ai_hired,
                "viewed": self.ai_viewed,
                "hire_rate": self.ai_hire_rate,
                "view_rate": round(self.ai_viewed / self.ai_total * 100, 1) if self.ai_total else 0
            },
            "manual": {
                "total": self.manual_total,
                "hired": self.manual_hired,
                "viewed": self.manual_viewed,
                "hire_rate": self.manual_hire_rate,
                "view_rate": round(self.manual_viewed / self.manual_total * 100, 1) if self.manual_total else 0
            },
            "ai_effectiveness": self.ai_effectiveness
        }


class AnalyticsService:
    """
    Service for proposal analytics and insights.
    
    Provides:
    - Dashboard summary statistics
    - Conversion funnel analysis
    - Source comparison (AI vs Manual)
    - Performance trends over time
    - Skills and industry breakdowns
    """
    
    def __init__(self, analytics_repo=None):
        """
        Initialize with dependencies.
        
        Args:
            analytics_repo: AnalyticsRepository instance
        """
        self.repo = analytics_repo
    
    def _get_repo(self):
        """Lazy load analytics repository."""
        if not self.repo:
            from app.infra.mongodb.repositories import get_analytics_repo
            self.repo = get_analytics_repo()
        return self.repo
    
    def get_dashboard_summary(
        self, 
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get dashboard summary statistics.
        
        Args:
            since: Optional start date filter
            
        Returns:
            Summary with totals, rates, and comparisons
        """
        try:
            repo = self._get_repo()
            return repo.get_analytics_summary(since)
        except Exception as e:
            logger.error(f"Error getting dashboard summary: {e}")
            return {}
    
    def get_conversion_funnel(
        self,
        since: Optional[datetime] = None,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get conversion funnel: Sent → Viewed → Discussed → Hired.
        
        Args:
            since: Optional start date filter
            source: Optional filter by 'ai_generated' or 'manual'
            
        Returns:
            Funnel stages with counts and percentages
        """
        try:
            repo = self._get_repo()
            return repo.get_conversion_funnel(since, source)
        except Exception as e:
            logger.error(f"Error getting conversion funnel: {e}")
            return {"funnel": [], "total_sent": 0}
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """
        Get proposal conversion statistics.
        
        Calculates:
        - View rate
        - Hire rate
        - Discussion rate
        - Message-Market Fit (discussions / viewed)
        - View-to-Hire rate
        
        Returns:
            Conversion statistics dict
        """
        try:
            repo = self._get_repo()
            return repo.get_conversion_stats()
        except Exception as e:
            logger.error(f"Error getting conversion stats: {e}")
            return {}
    
    def get_source_comparison(
        self,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Compare AI-generated vs manual proposals.
        
        Args:
            since: Optional start date filter
            
        Returns:
            Comparison with totals, rates, and AI effectiveness multiplier
        """
        try:
            repo = self._get_repo()
            return repo.get_source_comparison(since)
        except Exception as e:
            logger.error(f"Error getting source comparison: {e}")
            return {"ai_generated": {}, "manual": {}, "ai_effectiveness": 1.0}
    
    def get_combined_funnel(
        self,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get combined funnel for AI and Manual proposals.
        
        Shows full flow: Generated → Sent → Viewed → Discussed → Hired
        
        Args:
            since: Optional start date filter
            
        Returns:
            Combined funnel with AI and Manual breakdown
        """
        try:
            repo = self._get_repo()
            return repo.get_combined_funnel(since)
        except Exception as e:
            logger.error(f"Error getting combined funnel: {e}")
            return {
                "ai": {"generated": 0, "sent": 0, "viewed": 0, "discussed": 0, "hired": 0, "rates": {}},
                "manual": {"generated": 0, "sent": 0, "viewed": 0, "discussed": 0, "hired": 0, "rates": {}},
                "totals": {"generated": 0, "sent": 0, "viewed": 0, "discussed": 0, "hired": 0},
                "ai_share": 0
            }
    
    def get_proposal_trends(
        self,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get daily proposal trends for charts.
        
        Args:
            days: Number of days to include (default 30)
            
        Returns:
            List of daily stats with date, sent, hired, hire_rate
        """
        try:
            since = datetime.utcnow() - timedelta(days=days)
            repo = self._get_repo()
            return repo.get_proposal_trends(since)
        except Exception as e:
            logger.error(f"Error getting proposal trends: {e}")
            return []
    
    def get_skills_performance(
        self,
        since: Optional[datetime] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get performance breakdown by skill.
        
        Args:
            since: Optional start date filter
            limit: Maximum skills to return
            
        Returns:
            List of skills with total, hired, and hire_rate
        """
        try:
            repo = self._get_repo()
            return repo.get_skills_performance(since, limit)
        except Exception as e:
            logger.error(f"Error getting skills performance: {e}")
            return []
    
    def get_industry_performance(
        self,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get performance breakdown by industry.
        
        Args:
            since: Optional start date filter
            
        Returns:
            List of industries with total, hired, viewed, and hire_rate
        """
        try:
            repo = self._get_repo()
            return repo.get_industry_performance(since)
        except Exception as e:
            logger.error(f"Error getting industry performance: {e}")
            return []
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        
        Returns:
            Statistics across all collections
        """
        try:
            repo = self._get_repo()
            return repo.get_database_statistics()
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}
    
    def calculate_message_market_fit(
        self,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate Message-Market Fit score.
        
        MMF = (Discussions / Viewed) * 100
        
        This metric shows how well proposals resonate with clients.
        High MMF means the hook and approach are working.
        
        Args:
            since: Optional start date filter
            
        Returns:
            MMF score with breakdown
        """
        try:
            stats = self.get_conversion_stats()
            
            viewed = stats.get("total_viewed", 0)
            discussions = stats.get("total_discussions", 0)
            hired = stats.get("total_hired", 0)
            
            mmf = round((discussions / viewed * 100), 1) if viewed > 0 else 0
            view_to_hire = round((hired / viewed * 100), 1) if viewed > 0 else 0
            
            # Interpret the score
            if mmf >= 50:
                interpretation = "Excellent - proposals strongly resonate"
            elif mmf >= 30:
                interpretation = "Good - proposals engaging clients"
            elif mmf >= 15:
                interpretation = "Fair - room for improvement"
            else:
                interpretation = "Needs work - review hook and approach"
            
            return {
                "message_market_fit": mmf,
                "view_to_hire_rate": view_to_hire,
                "total_viewed": viewed,
                "total_discussions": discussions,
                "total_hired": hired,
                "interpretation": interpretation,
                "recommendation": self._get_mmf_recommendation(mmf, view_to_hire)
            }
            
        except Exception as e:
            logger.error(f"Error calculating MMF: {e}")
            return {"message_market_fit": 0, "error": str(e)}
    
    def _get_mmf_recommendation(self, mmf: float, view_to_hire: float) -> str:
        """Get recommendation based on MMF and conversion rates."""
        if mmf < 15:
            return "Focus on improving opening hook and value proposition"
        elif mmf >= 30 and view_to_hire < 20:
            return "Hook is working but closing needs improvement"
        elif mmf >= 50 and view_to_hire >= 30:
            return "Proposals are highly effective - maintain current approach"
        else:
            return "Continue testing different approaches"
    
    def get_insights_summary(
        self,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive insights summary.
        
        Combines all analytics into actionable insights.
        
        Args:
            since: Optional start date filter
            
        Returns:
            Complete insights with recommendations
        """
        try:
            summary = self.get_dashboard_summary(since)
            mmf = self.calculate_message_market_fit(since)
            source_comp = self.get_source_comparison(since)
            top_skills = self.get_skills_performance(since, limit=5)
            top_industries = self.get_industry_performance(since)[:5]
            
            return {
                "summary": summary,
                "message_market_fit": mmf,
                "source_comparison": source_comp,
                "top_performing_skills": top_skills,
                "top_performing_industries": top_industries,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting insights summary: {e}")
            return {"error": str(e)}

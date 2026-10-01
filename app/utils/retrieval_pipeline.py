"""
Multi-Layer Retrieval Pipeline for Proposal Generation

Implements 3-stage intelligent retrieval:
1. METADATA FILTERING (Fast) - Filter by industry, skills, urgency, etc.
2. SEMANTIC SEARCH (Smart) - Find similar projects using embeddings
3. FEEDBACK ANALYSIS (Learning) - Extract patterns from top projects

Returns ranked similar projects with actionable insights for proposal generation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

# Import centralized constants
from app.domain.constants import (
    PLATFORM_KEYWORDS,
    AI_ML_KEYWORDS,
    COMPLEXITY_LEVELS,
)

# Type hint for JobRequirements (avoid circular import)
if TYPE_CHECKING:
    from app.services.job_requirements_service import JobRequirements

logger = logging.getLogger(__name__)


@dataclass
class FilterCriteria:
    """Metadata filter criteria for Stage 1"""
    industries: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    task_type: Optional[str] = None
    min_complexity: Optional[str] = None  # low, medium, high
    max_complexity: Optional[str] = None
    min_duration_days: Optional[int] = None
    max_duration_days: Optional[int] = None
    min_effectiveness: Optional[float] = None  # 0-1
    min_satisfaction: Optional[float] = None  # 1-5
    completed_only: bool = False
    with_feedback: bool = False
    platform: Optional[str] = None  # wordpress, shopify, woocommerce, etc.
    urgent_adhoc: Optional[bool] = None  # Prioritize urgent/adhoc projects when job is urgent


class RetrievalPipeline:
    """Multi-stage retrieval for proposal generation"""
    
    # NOTE: Constants moved to app/domain/constants.py
    # Use PLATFORM_KEYWORDS, AI_ML_KEYWORDS, COMPLEXITY_LEVELS directly (imported at module level)

    def __init__(self, db=None, pinecone_service=None):
        """
        Initialize retrieval pipeline.
        
        Args:
            db: Database instance for job data access
            pinecone_service: Pinecone service for semantic search
        """
        self.db = db
        self.pinecone_service = pinecone_service
        logger.info("RetrievalPipeline initialized")

    def retrieve_for_proposal(
        self,
        new_job_requirements: Dict[str, Any],
        all_jobs: List[Dict[str, Any]],
        top_k: int = 5,
        use_semantic_search: bool = True,
        job_requirements: Optional["JobRequirements"] = None
    ) -> Dict[str, Any]:
        """
        Complete retrieval pipeline for a new job requiring a proposal.
        
        Args:
            new_job_requirements: New job data to generate proposal for
            all_jobs: All historical jobs in the system
            top_k: Number of similar projects to return
            use_semantic_search: Whether to use semantic search (requires embeddings)
            job_requirements: Extracted structured requirements (Issue #24 enhancement)
            
        Returns:
            Comprehensive retrieval result with context for proposal generation
        """
        job_id = new_job_requirements.get("contract_id", "unknown")
        logger.info(f"[Retrieval] Starting pipeline for {job_id}")
        
        # Log extracted requirements if available (Issue #24)
        if job_requirements:
            logger.info(f"  → Using extracted requirements: deliverables={len(job_requirements.specific_deliverables)}, "
                       f"problems={len(job_requirements.problems_mentioned)}, tone={job_requirements.client_tone}")

        # STAGE 1: METADATA FILTERING
        logger.debug(f"[Stage 1] Filtering by metadata...")
        filter_criteria = self._build_filter_criteria(new_job_requirements, job_requirements)
        filtered_jobs = self._metadata_filter(all_jobs, filter_criteria)
        logger.info(f"  → Filtered: {len(all_jobs)} jobs → {len(filtered_jobs)} matches")

        if not filtered_jobs:
            logger.warning(f"  ⚠ No jobs match filter criteria, using all jobs")
            filtered_jobs = all_jobs[:20]  # Fallback to top 20

        # STAGE 2: SEMANTIC SEARCH (enhanced with JobRequirements - Issue #24)
        logger.debug(f"[Stage 2] Semantic similarity search...")
        ranked_projects = self._semantic_rank(
            new_job_requirements,
            filtered_jobs,
            use_semantic_search=use_semantic_search,
            job_requirements=job_requirements
        )
        ranked_projects = ranked_projects[:top_k]
        logger.info(f"  → Ranked: {len(ranked_projects)} top similar projects")

        # STAGE 3: FEEDBACK ANALYSIS
        logger.debug(f"[Stage 3] Analyzing feedback patterns...")
        insights = self._extract_insights(ranked_projects, new_job_requirements)
        logger.info(f"  → Extracted {len(insights['success_patterns'])} success patterns")

        result = {
            "query_job": {
                "contract_id": job_id,
                "title": new_job_requirements.get("job_title"),
                "industry": new_job_requirements.get("industry"),
                "skills": new_job_requirements.get("skills_required", []),
            },
            "stage1_filtered_count": len(filtered_jobs),
            "similar_projects": [
                {
                    "contract_id": p.get("contract_id"),
                    "company": p.get("company_name"),
                    "company_name": p.get("company_name"),
                    "title": p.get("job_title"),
                    "task_type": p.get("task_type"),
                    "industry": p.get("industry"),
                    "skills": p.get("skills_required", []),
                    "deliverables": p.get("deliverables", []),
                    "outcome": p.get("outcome"),  # Structured: {stats, loom_url}
                    "outcomes": p.get("outcomes", ""),  # Legacy fallback for unmigrated data
                    "similarity_score": score,
                    "portfolio_urls": p.get("portfolio_urls", []),
                    "client_feedback_url": p.get("client_feedback_url"),
                }
                for p, score in ranked_projects
            ],
            "insights": insights,
            "proposal_context": self._build_proposal_context(ranked_projects, new_job_requirements)
        }

        logger.info(f"✓ [Retrieval] Complete: {len(ranked_projects)} projects, "
                   f"{len(insights['success_patterns'])} patterns, "
                   f"{len(insights['winning_sections'])} sections")

        return result

    def _build_filter_criteria(
        self,
        job_data: Dict[str, Any],
        job_requirements: Optional["JobRequirements"] = None
    ) -> FilterCriteria:
        """
        Build intelligent filter criteria from new job requirements.
        
        Prioritizes projects with feedback and proven effectiveness while
        keeping filters flexible to find relevant matches.
        
        CRITICAL: Now uses LLM-based semantic industry detection when needed,
        and extracts CLIENT INTENTS (what they actually want done).
        
        ENHANCED (Issue #24): Uses JobRequirements for better filtering when available.
        """
        from app.utils.metadata_extractor import MetadataExtractor
        
        skills = job_data.get("skills_required", [])
        industry = job_data.get("industry", "")
        task_type = job_data.get("task_type", "")
        complexity = job_data.get("task_complexity", "medium")
        urgency = job_data.get("urgent_adhoc", False)
        
        # ENHANCED: Use extracted requirements if available (Issue #24)
        if job_requirements:
            # Use extracted tech stack as additional skills
            if job_requirements.tech_stack_mentioned:
                skills = list(set(skills + job_requirements.tech_stack_mentioned))
            # Use inferred industry if job_data doesn't have one
            if not industry and job_requirements.inferred_industry:
                industry = job_requirements.inferred_industry
            # Use extracted complexity
            if job_requirements.complexity_level:
                complexity = job_requirements.complexity_level
            # Urgent tone means urgent job
            if job_requirements.client_tone == "urgent":
                urgency = True
        
        # CRITICAL: Detect platform from job description and skills
        platform = self._detect_platform(job_data)
        if platform:
            logger.info(f"  → Detected platform: {platform.upper()} - will prioritize {platform} projects")

        # CRITICAL: Extract client intents (what they actually want done)
        client_intents = MetadataExtractor.extract_client_intents(job_data)
        if client_intents:
            logger.info(f"  → Detected client intents: {client_intents[:3]}")
            # Store intents in job_data for later use in ranking
            job_data["client_intents"] = client_intents

        # Log urgency detection
        if urgency:
            logger.info(f"  → URGENT/AD-HOC project detected - will prioritize similar urgent projects")

        # Build flexible criteria (not overly restrictive)
        # We want to find projects with feedback first, then broaden if needed
        return FilterCriteria(
            industries=[industry] if industry else None,
            skills=skills if skills else None,
            task_type=task_type if task_type else None,
            with_feedback=False,  # Don't require feedback - include if available
            completed_only=True,  # Only completed projects (proven success)
            platform=platform,  # CRITICAL: Platform-specific filtering
            urgent_adhoc=urgency if urgency else None  # Prioritize urgent projects when job is urgent
        )
    
    def _is_ai_ml_job(self, job_data: Dict[str, Any]) -> Tuple[bool, int]:
        """
        Check if this is primarily an AI/ML job.
        
        Returns:
            Tuple of (is_ai_job, ai_score) - True with score if AI keywords found prominently
        """
        job_desc = job_data.get("job_description", "").lower()
        job_title = job_data.get("job_title", "").lower()
        skills = [s.lower() for s in job_data.get("skills_required", [])]
        task_type = job_data.get("task_type", "").lower()
        
        # If task_type is explicitly ai_ml, it's an AI job
        if task_type == "ai_ml":
            return True, 100
        
        # Combine all text for searching
        search_text = f"{job_title} {job_desc} {' '.join(skills)}"
        
        # Count AI/ML keyword matches
        ai_score = 0
        matched_keywords = []
        for keyword in AI_ML_KEYWORDS:
            if keyword in search_text:
                ai_score += 3  # Base score for finding the keyword
                matched_keywords.append(keyword)
                # Extra weight for title matches (very specific)
                if keyword in job_title:
                    ai_score += 5
                # Extra weight for skill matches
                if keyword in skills:
                    ai_score += 3
        
        # Threshold: if we found 3+ AI keywords or score > 10, it's an AI job
        is_ai_job = ai_score >= 10 or len(matched_keywords) >= 3
        
        if is_ai_job:
            logger.info(f"  → AI/ML job detected (score: {ai_score}, keywords: {matched_keywords[:5]})")
        
        return is_ai_job, ai_score
    
    def _detect_platform(self, job_data: Dict[str, Any]) -> Optional[str]:
        """
        Detect the primary platform from job description and skills.
        
        CRITICAL: AI/ML jobs take precedence over platform detection.
        This ensures AI content generation jobs don't get matched to WordPress
        just because they mention "publish to WordPress".
        
        Returns:
            Platform name (lowercase), "ai_ml" for AI jobs, or None if not detected
        """
        # FIRST: Check if this is an AI/ML job - takes precedence
        is_ai_job, ai_score = self._is_ai_ml_job(job_data)
        if is_ai_job:
            logger.info(f"  → Detected as AI/ML job (score: {ai_score}) - prioritizing AI projects")
            return "ai_ml"
        
        job_desc = job_data.get("job_description", "").lower()
        job_title = job_data.get("job_title", "").lower()
        skills = [s.lower() for s in job_data.get("skills_required", [])]
        
        # Combine all text for searching
        search_text = f"{job_title} {job_desc} {' '.join(skills)}"
        
        # Check each platform's keywords
        platform_scores = {}
        for platform, keywords in PLATFORM_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in search_text:
                    score += 1
                    # Extra weight for skill matches (more specific)
                    if keyword in skills:
                        score += 2
            if score > 0:
                platform_scores[platform] = score
        
        if not platform_scores:
            return None
        
        # Return the platform with highest score
        detected = max(platform_scores, key=platform_scores.get)
        logger.debug(f"  Platform scores: {platform_scores} → Selected: {detected}")
        return detected
    def _metadata_filter(
        self,
        jobs: List[Dict[str, Any]],
        criteria: FilterCriteria
    ) -> List[Dict[str, Any]]:
        """
        STAGE 1: Fast metadata filtering with platform exclusion.
        Priority: AI/ML → Urgent → Platform+Industry → Platform → Industry → Skills → Fallback
        
        CRITICAL: AI/ML jobs are now handled specially - they match based on task_type
        and AI-related skills, not CMS platforms.
        """
        # All CMS platforms are mutually exclusive
        CMS_PLATFORMS = {"wordpress", "woocommerce", "shopify", "wix", "magento", "webflow", "squarespace"}
        JS_FRAMEWORKS = {"react": {"angular", "vue"}, "angular": {"react", "vue"}, "vue": {"react", "angular"}}
        
        # SPECIAL HANDLING FOR AI/ML JOBS
        if criteria.platform == "ai_ml":
            logger.info("  → AI/ML filtering mode - prioritizing AI projects")
            return self._filter_ai_ml_jobs(jobs, criteria)
        
        # Get excluded platforms
        if criteria.platform in CMS_PLATFORMS:
            excluded = CMS_PLATFORMS - {criteria.platform, "woocommerce" if criteria.platform == "wordpress" else criteria.platform}
            if criteria.platform in ("wordpress", "woocommerce"):
                excluded -= {"wordpress", "woocommerce"}  # WP and WooCommerce are related
        else:
            excluded = JS_FRAMEWORKS.get(criteria.platform, set())
        
        target_industry = (criteria.industries[0].lower() if criteria.industries else None)
        results = {"urgent": [], "platform": [], "industry": [], "skills": []}
        
        for job in jobs:
            if criteria.completed_only and job.get("project_status", "").lower() != "completed":
                continue
            
            job_platform = self._detect_job_platform(job)
            job_skills_text = " ".join(s.lower() for s in job.get("skills_required", []))
            
            # CRITICAL: Skip if competing platform detected OR in skills
            if criteria.platform:
                if job_platform in excluded:
                    logger.debug(f"  ✗ Excluding {job.get('company_name')} - platform {job_platform} conflicts")
                    continue
                if any(kw in job_skills_text for p in excluded for kw in PLATFORM_KEYWORDS.get(p, [p])):
                    logger.debug(f"  ✗ Excluding {job.get('company_name')} - has competing platform skill")
                    continue
            
            # Check matches
            job_industry = (job.get("industry") or "").lower()
            industry_match = target_industry and (job_industry == target_industry or 
                            target_industry in [t.lower() for t in job.get("industry_tags", [])])
            platform_match = (job_platform == criteria.platform or 
                             self._are_platforms_related(criteria.platform, job_platform))
            
            # Categorize by priority
            if criteria.platform and platform_match:
                if criteria.urgent_adhoc and job.get("urgent_adhoc"):
                    results["urgent"].append(job)
                elif industry_match:
                    results["platform"].insert(0, job)  # Platform+Industry at front
                else:
                    results["platform"].append(job)
            elif industry_match and job_platform not in excluded:
                results["industry"].append(job)
            elif criteria.skills and job_platform not in excluded:
                if set(s.lower() for s in job.get("skills_required", [])) & set(s.lower() for s in criteria.skills):
                    results["skills"].append(job)
        
        # Return by priority
        for key in ["urgent", "platform", "industry", "skills"]:
            if results[key]:
                if key == "urgent":
                    return results["urgent"] + [p for p in results["platform"] if p not in results["urgent"]]
                logger.info(f"  → Found {len(results[key])} {key}-matched projects")
                return results[key]
        
        # Fallback: non-competing completed projects
        logger.warning(f"  ⚠ No matches, returning non-competing completed projects")
        return [j for j in jobs if j.get("project_status", "").lower() == "completed" 
                and self._detect_job_platform(j) not in excluded][:10]
    
    def _filter_ai_ml_jobs(
        self,
        jobs: List[Dict[str, Any]],
        criteria: FilterCriteria
    ) -> List[Dict[str, Any]]:
        """
        Special filtering for AI/ML jobs.
        
        Prioritizes:
        1. Jobs with task_type = 'ai_ml'
        2. Jobs with AI-related skills (openai, langchain, etc.)
        3. Backend/automation jobs as fallback
        
        CRITICAL: Excludes pure CMS jobs (WordPress themes, Shopify stores, etc.)
        """
        CMS_PLATFORMS = {"wordpress", "woocommerce", "shopify", "wix", "magento", "webflow", "squarespace"}
        
        ai_ml_jobs = []
        backend_jobs = []
        
        for job in jobs:
            if criteria.completed_only and job.get("project_status", "").lower() != "completed":
                continue
            
            job_task_type = (job.get("task_type") or "").lower()
            job_platform = self._detect_job_platform(job)
            
            # Skip pure CMS jobs for AI/ML matching
            if job_platform in CMS_PLATFORMS:
                # Unless they also have AI skills
                job_skills = [s.lower() for s in job.get("skills_required", [])]
                has_ai_skills = any(kw in " ".join(job_skills) for kw in ["openai", "gpt", "ai", "llm", "langchain", "automation"])
                if not has_ai_skills:
                    continue
            
            # Check if it's an AI/ML job
            is_ai_job, ai_score = self._is_ai_ml_job(job)
            
            if job_task_type == "ai_ml" or is_ai_job:
                ai_ml_jobs.append((job, ai_score))
            elif job_task_type in ("backend_api", "automation", "full_stack"):
                # Backend/automation as secondary matches
                backend_jobs.append(job)
        
        # Sort AI/ML jobs by their AI score
        ai_ml_jobs.sort(key=lambda x: x[1], reverse=True)
        ai_ml_results = [job for job, score in ai_ml_jobs]
        
        if ai_ml_results:
            logger.info(f"  → Found {len(ai_ml_results)} AI/ML matched projects")
            return ai_ml_results
        
        if backend_jobs:
            logger.info(f"  → Found {len(backend_jobs)} backend/automation projects as fallback")
            return backend_jobs
        
        # Last fallback: any non-CMS completed projects
        logger.warning(f"  ⚠ No AI/ML matches, returning non-CMS completed projects")
        return [j for j in jobs if j.get("project_status", "").lower() == "completed" 
                and self._detect_job_platform(j) not in CMS_PLATFORMS][:10]
    
    def _detect_job_platform(self, job: Dict[str, Any]) -> Optional[str]:
        """
        Detect platform from job data using weighted keyword scoring.
        
        ENHANCED: Now also detects AI/ML jobs as a "platform" type.
        """
        # First check if it's an AI/ML job
        is_ai_job, ai_score = self._is_ai_ml_job(job)
        if is_ai_job:
            return "ai_ml"
        
        text = f"{job.get('job_title', '')} {job.get('job_description', '')}".lower()
        skills = [s.lower() for s in job.get("skills_required", [])]
        skills_text = " ".join(skills)
        
        scores = {}
        for platform, keywords in PLATFORM_KEYWORDS.items():
            score = sum(
                4 if kw in skills else 3 if kw in skills_text else 2 if kw in job.get("job_title", "").lower() else 1 if kw in text else 0
                for kw in keywords
            )
            if score:
                scores[platform] = score
        
        if not scores:
            return None
        detected = max(scores, key=scores.get)
        logger.debug(f"  Platform: {job.get('company_name', '?')} → {detected} (scores: {scores})")
        return detected
    
    def _are_platforms_related(self, platform1: Optional[str], platform2: Optional[str]) -> bool:
        """
        Check if two platforms are related (e.g., WooCommerce is part of WordPress).
        This helps find relevant projects when exact platform match isn't available.
        """
        if not platform1 or not platform2:
            return False
        
        # AI/ML is related to backend technologies
        if platform1 == "ai_ml" or platform2 == "ai_ml":
            backend_related = ["python", "fastapi", "node", "backend", "automation"]
            return platform1 in backend_related or platform2 in backend_related
        
        # Define related platforms (bi-directional relationships)
        related_platforms = {
            "wordpress": ["woocommerce", "elementor", "php"],
            "woocommerce": ["wordpress", "php"],
            "shopify": ["liquid"],  # Shopify uses Liquid templating
            "react": ["next.js", "javascript"],
            "vue": ["nuxt", "javascript"],
            "angular": ["typescript", "javascript"],
            "html_css": ["javascript", "php"],  # Basic web can relate to backend
        }
        
        return platform2 in related_platforms.get(platform1, [])
        
        return platform2 in related_platforms.get(platform1, [])

    def _semantic_rank(
        self,
        new_job: Dict[str, Any],
        candidate_jobs: List[Dict[str, Any]],
        use_semantic_search: bool = True,
        job_requirements: Optional["JobRequirements"] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        STAGE 2: Semantic ranking using Pinecone embeddings + metadata.
        
        Combines:
        1. Semantic similarity from Pinecone embeddings (if available)
        2. Metadata similarity from project attributes
        3. Diversity boost to avoid returning the same portfolios repeatedly
        
        ENHANCED (Issue #24): Uses JobRequirements for better query construction:
        - specific_deliverables boost semantic matching
        - problems_mentioned add context for better relevance
        """
        from app.utils.metadata_extractor import MetadataExtractor

        ranked = []
        pinecone_scores = {}  # item_id -> semantic score from Pinecone (or contract_id for legacy)

        # Try to use Pinecone semantic search for better ranking
        if use_semantic_search and self.pinecone_service:
            try:
                import os
                from app.utils.openai_service import OpenAIService
                from app.config import settings
                
                api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                    
                openai_service = OpenAIService(api_key=api_key)
                
                # Create query text from new job (ENHANCED with JobRequirements - Issue #24)
                query_text = self._build_enhanced_query(new_job, job_requirements)
                logger.debug(f"  → Enhanced query: {query_text[:100]}...")
                
                query_embedding = openai_service.get_embedding(query_text)
                
                if query_embedding:
                    # Query Pinecone for semantic matches
                    pinecone_results = self.pinecone_service.query_vectors(
                        query_embedding=query_embedding,
                        top_k=50,  # Get more results for diversity
                        include_metadata=True
                    )
                    
                    # Build item_id -> score mapping from Pinecone results
                    # New portfolio namespace uses item_id, fallback to contract_id for legacy
                    for result in pinecone_results:
                        metadata = result.get("metadata", {})
                        # Try item_id first (new schema), then contract_id (legacy)
                        entry_id = metadata.get("item_id") or metadata.get("contract_id")
                        if entry_id:
                            # Keep highest score for each entry
                            current_score = pinecone_scores.get(entry_id, 0)
                            pinecone_scores[entry_id] = max(current_score, result.get("score", 0))
                    
                    logger.info(f"  → Pinecone returned {len(pinecone_scores)} unique portfolio items with scores")
            except Exception as e:
                logger.warning(f"  ⚠ Pinecone semantic search failed, using metadata only: {e}")

        # Calculate combined scores for each candidate job
        for job in candidate_jobs:
            contract_id = job.get("contract_id", "")
            
            # Metadata similarity (always computed)
            similarity_scores = MetadataExtractor.compare_projects(new_job, job)
            metadata_sim = MetadataExtractor.calculate_overall_similarity(similarity_scores)
            
            # Semantic similarity from Pinecone (if available)
            semantic_sim = pinecone_scores.get(contract_id, 0.0)
            
            # Combined score: weight semantic search higher when available
            if semantic_sim > 0:
                # 60% semantic + 40% metadata when we have Pinecone scores
                combined_score = (semantic_sim * 0.6) + (metadata_sim * 0.4)
            else:
                # Fall back to metadata only
                combined_score = metadata_sim
            
            # PORTFOLIO BOOST: Prioritize portfolio entries as they have verified work samples
            # Portfolio entries have real portfolio URLs that can be referenced in proposals
            is_portfolio_entry = job.get("is_portfolio_entry", False)
            portfolio_urls = job.get("portfolio_urls", [])
            
            if is_portfolio_entry and portfolio_urls:
                # 15% boost for portfolio entries with actual URLs
                # This ensures portfolio entries compete well against regular entries
                portfolio_boost = 0.15
                combined_score = combined_score + portfolio_boost
                logger.debug(f"  → Portfolio boost applied to {job.get('company_name')}: +{portfolio_boost}")

            ranked.append((job, combined_score))

        # Sort by combined score descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        # Apply diversity boost: avoid returning jobs with same company name or very similar portfolios
        diversified = self._apply_diversity(ranked)

        return diversified

    def _apply_diversity(
        self,
        ranked_projects: List[Tuple[Dict[str, Any], float]],
        max_per_company: int = 1,
        max_shared_urls: int = 1
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Apply diversity to avoid returning the same companies/portfolios repeatedly.
        
        This ensures proposal references are varied and don't always cite the same projects.
        
        ENHANCED: Now tracks INDIVIDUAL portfolio URLs to prevent the same links
        from appearing across different referenced projects.
        
        Args:
            ranked_projects: Ranked list of (project, score) tuples
            max_per_company: Max projects from same company (default 1)
            max_shared_urls: Max times a single portfolio URL can appear (default 1)
        """
        seen_companies = {}  # company_name -> count
        seen_portfolio_urls = {}  # individual URL -> count (track each URL separately)
        seen_feedback_urls = set()  # Track feedback URLs for uniqueness
        diversified = []
        
        for job, score in ranked_projects:
            company = job.get("company_name", "").lower().strip()
            portfolio_urls = job.get("portfolio_urls", [])
            feedback_url = job.get("client_feedback_url", "")
            
            # Skip if we've already selected max_per_company projects from this company
            if company and seen_companies.get(company, 0) >= max_per_company:
                logger.debug(f"  → Skipping duplicate company: {company}")
                continue
            
            # Check if this project has UNIQUE portfolio URLs
            # A project is good if at least ONE of its URLs is fresh
            has_fresh_urls = False
            fresh_url_count = 0
            for url in portfolio_urls:
                if url:
                    url_lower = url.lower().strip()
                    if seen_portfolio_urls.get(url_lower, 0) < max_shared_urls:
                        has_fresh_urls = True
                        fresh_url_count += 1
            
            # Skip if ALL portfolio URLs have already been used in other projects
            if portfolio_urls and not has_fresh_urls:
                logger.debug(f"  → Skipping project with no fresh portfolio URLs: {company}")
                continue
            
            # Prefer projects with MORE unique URLs (diversity bonus)
            # Slight score boost for projects bringing new portfolio examples
            diversity_bonus = min(fresh_url_count * 0.02, 0.1)  # Up to 10% bonus
            adjusted_score = score + diversity_bonus
            
            # Add to diversified results
            diversified.append((job, adjusted_score))
            
            # Track company usage
            if company:
                seen_companies[company] = seen_companies.get(company, 0) + 1
            
            # Track EACH portfolio URL individually
            for url in portfolio_urls:
                if url:
                    url_lower = url.lower().strip()
                    seen_portfolio_urls[url_lower] = seen_portfolio_urls.get(url_lower, 0) + 1
            
            # Track feedback URLs
            if feedback_url:
                seen_feedback_urls.add(feedback_url.lower().strip())
        
        # Re-sort by adjusted score after diversity bonus
        diversified.sort(key=lambda x: x[1], reverse=True)
        
        unique_urls = len(seen_portfolio_urls)
        unique_companies = len(seen_companies)
        logger.info(f"  → Diversity filter: {len(ranked_projects)} → {len(diversified)} unique projects ({unique_companies} companies, {unique_urls} unique portfolio URLs)")
        return diversified

    def _build_enhanced_query(
        self,
        job_data: Dict[str, Any],
        job_requirements: Optional["JobRequirements"] = None
    ) -> str:
        """
        Build enhanced semantic search query using JobRequirements.
        
        Issue #24: Uses extracted specific_deliverables and problems_mentioned
        to create a more targeted query for finding relevant past projects.
        
        Args:
            job_data: Basic job data dict
            job_requirements: Extracted structured requirements (optional)
            
        Returns:
            Enhanced query string for semantic search
        """
        # Base query from job data
        title = job_data.get("job_title", "")
        description = job_data.get("job_description", "")[:400]
        
        # Start with basic query
        query_parts = [title, description]
        
        # ENHANCED: Add extracted requirements for better matching (Issue #24)
        if job_requirements:
            # Add exact task understanding (most important)
            if job_requirements.exact_task:
                query_parts.append(f"Task: {job_requirements.exact_task}")
            
            # Add specific deliverables for precise matching
            if job_requirements.specific_deliverables:
                deliverables_text = " | ".join(job_requirements.specific_deliverables[:4])
                query_parts.append(f"Deliverables: {deliverables_text}")
            
            # Add pain points/problems to find projects that solved similar issues
            if job_requirements.problems_mentioned:
                problems_text = " | ".join(job_requirements.problems_mentioned[:3])
                query_parts.append(f"Problems to solve: {problems_text}")
            
            # Add tech stack for technical matching
            if job_requirements.tech_stack_mentioned:
                tech_text = " ".join(job_requirements.tech_stack_mentioned[:5])
                query_parts.append(f"Tech: {tech_text}")
            
            logger.info(f"  → Query enhanced with {len(job_requirements.specific_deliverables)} deliverables, "
                       f"{len(job_requirements.problems_mentioned)} problems")
        
        # Combine and truncate to reasonable length
        query_text = " ".join(query_parts)
        max_length = 1500  # Reasonable limit for embedding
        if len(query_text) > max_length:
            query_text = query_text[:max_length]
        
        return query_text

    def _extract_insights(
        self,
        similar_projects: List[Tuple[Dict[str, Any], float]],
        new_job: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        STAGE 3: Analyze feedback from top projects for winning patterns.
        
        Extracts:
        - Success patterns (what worked in proposals that won)
        - Common elements (skills, approaches, messaging)
        - Winning sections (how to structure proposals)
        - Client values (what clients praised)
        """
        insights = {
            "success_patterns": [],
            "common_elements": {},
            "winning_sections": [],
            "client_values": [],
            "feedback_sentiment": "positive",
            "success_rate": 0.0
        }

        success_count = 0
        feedback_samples = []
        proposals_samples = []

        for job, similarity in similar_projects:
            if job.get("project_status", "").lower() != "completed":
                continue

            success_count += 1
            
            # Collect feedback
            feedback = job.get("client_feedback_text") or job.get("client_feedback", "")
            if feedback:
                feedback_samples.append(feedback)

            # Collect proposal text for pattern analysis
            proposal = job.get("your_proposal_text", "")
            if proposal:
                proposals_samples.append(proposal)

        # Extract success patterns from proposals and feedback
        if proposals_samples:
            insights["success_patterns"] = self._extract_winning_patterns(proposals_samples, feedback_samples)

        # Extract what clients valued
        if feedback_samples:
            insights["client_values"] = self._extract_client_values(feedback_samples)
            insights["feedback_sentiment"] = self._analyze_feedback_sentiment(feedback_samples)

        # Calculate success rate
        insights["success_rate"] = success_count / len(similar_projects) if similar_projects else 0
        insights["successful_project_count"] = success_count

        # Add winning sections template
        insights["winning_sections"] = self._build_winning_sections_template(new_job)

        logger.debug(f"  Extracted patterns: {len(insights['success_patterns'])} patterns, "
                    f"{len(insights['client_values'])} client values")

        return insights

    def _extract_winning_patterns(self, proposals: List[str], feedbacks: List[str]) -> List[str]:
        """Extract patterns that made proposals win"""
        patterns = []
        
        if not proposals:
            logger.debug("  No proposals to analyze for patterns")
            return patterns
        
        logger.debug(f"  Analyzing {len(proposals)} proposals for winning patterns...")
        
        # Analyze what made proposals successful
        pattern_indicators = {
            "Reference specific past projects": ["completed", "worked on", "delivered", "built", "designed", "developed"],
            "Show understanding of their problem": ["understand", "see you", "dealing with", "challenge", "realize", "know"],
            "Include portfolio proof": ["portfolio", "github", "live site", "project link", "attachment", "check"],
            "Be specific about approach": ["approach", "methodology", "process", "steps", "strategy", "method"],
            "Include realistic timeline": ["timeline", "phases", "weeks", "months", "days", "schedule"],
            "Personal and conversational tone": ["i've", "you're", "let's", "excited", "hey", "thanks", "thank"],
            "Show past results/metrics": ["improved", "increased", "reduced", "%", "faster", "faster", "better"],
            "Address client's specific tech stack": ["wordpress", "shopify", "react", "python", "nodejs", "aws", "html", "css"]
        }
        
        for pattern, indicators in pattern_indicators.items():
            # Check if pattern appears in successful proposals
            matches = 0
            for proposal in proposals[:10]:  # Check top 10 proposals
                if any(indicator in proposal.lower() for indicator in indicators):
                    matches += 1
            
            if matches >= 1:  # Pattern appears in at least 1 winning proposal (lowered threshold)
                patterns.append(pattern)
                logger.debug(f"    ✓ Pattern: {pattern} (found in {matches} proposals)")
        
        logger.debug(f"  Extracted {len(patterns)} patterns total")
        return patterns[:8]  # Return up to 8 patterns

    def _extract_client_values(self, feedbacks: List[str]) -> List[str]:
        """Extract what clients valued from feedback"""
        values = []
        value_keywords = {
            "Responsive communication": ["responsive", "communication", "quick reply", "available"],
            "Professional quality": ["professional", "quality", "polished", "attention to detail"],
            "Timely delivery": ["on time", "deadline", "fast", "quick turnaround"],
            "Problem solving": ["challenges", "obstacles", "creative", "solutions"],
            "Exceeded expectations": ["exceeded", "amazing", "great", "fantastic"],
            "Technical expertise": ["expertise", "knowledge", "skilled", "experienced"],
            "Attention to detail": ["detail", "thorough", "precise", "careful"]
        }
        
        for value, keywords in value_keywords.items():
            for feedback in feedbacks:
                if any(kw in feedback.lower() for kw in keywords):
                    if value not in values:
                        values.append(value)
                    break
        
        return values[:5]

    def _analyze_feedback_sentiment(self, feedbacks: List[str]) -> str:
        """Simple sentiment analysis of feedback"""
        if not feedbacks:
            return "neutral"
        
        positive_words = ["great", "excellent", "amazing", "fantastic", "love", "perfect", "best", "wonderful"]
        negative_words = ["bad", "poor", "terrible", "awful", "hate", "worst", "disappointing"]
        
        positive_count = sum(1 for f in feedbacks if any(word in f.lower() for word in positive_words))
        negative_count = sum(1 for f in feedbacks if any(word in f.lower() for word in negative_words))
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"

    def _build_winning_sections_template(self, new_job: Dict[str, Any]) -> List[str]:
        """Build template of sections that should be in winning proposal"""
        return [
            "HOOK: Acknowledge their specific problem",
            "PROOF: 2-3 past similar projects with outcomes",
            "PROOF: Portfolio links for credibility",
            "APPROACH: Specific solution for their tech stack",
            "TIMELINE: Realistic phases and duration",
            "CTA: Clear, friendly call-to-action"
        ]

    def _extract_phrases(self, texts: List[str], num_phrases: int = 5) -> List[str]:
        """Extract common positive phrases and winning patterns from feedback texts"""
        phrases = []
        
        # Winning patterns that appear in successful proposals
        positive_patterns = {
            "Reference specific past projects": "Mentioning company names and past work",
            "Show understanding of their problem": "Acknowledge their specific challenge",
            "Include portfolio proof": "Add links to live projects and work samples",
            "Be conversational and human": "Use casual, direct language (not AI tone)",
            "Include metrics and results": "Show past performance numbers and outcomes",
            "Specific to their tech stack": "Address their specific technologies",
            "Clear timeline included": "Provide realistic project phases and duration"
        }

        for pattern, description in positive_patterns.items():
            # Check if pattern-related keywords appear in feedback
            pattern_keywords = pattern.lower().split() + description.lower().split()
            
            for text in texts:
                text_lower = text.lower()
                # If multiple keywords match, consider pattern found
                if sum(1 for kw in pattern_keywords if kw in text_lower) >= 2:
                    if pattern not in phrases:
                        phrases.append(pattern)
                    break

        return phrases[:num_phrases]

    def _build_proposal_context(
        self,
        similar_projects: List[Tuple[Dict[str, Any], float]],
        new_job: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build rich context for AI proposal generation.
        
        Combines similar project examples with patterns and insights.
        """
        context = {
            "reference_projects": [],
            "patterns_to_follow": [],
            "sections_to_include": [],
            "tone_style": "professional",
            "estimated_success_rate": 0.0
        }

        # Extract top 3 successful projects as references
        success_count = 0
        for job, similarity in similar_projects:
            if job.get("project_status", "").lower() == "completed" and success_count < 3:
                context["reference_projects"].append({
                    "company": job.get("company_name"),
                    "industry": job.get("industry"),
                    "skills": job.get("skills_required", []),
                    "deliverables": job.get("deliverables", []),  # What was actually built
                    "outcome": job.get("outcome"),  # Structured outcome
                    "feedback": (job.get("client_feedback_text") or 
                               job.get("client_feedback", ""))[:200],  # First 200 chars
                    "feedback_url": job.get("client_feedback_url"),
                    "portfolio_urls": job.get("portfolio_urls", [])
                })
                success_count += 1

        # Calculate estimated success rate (case-insensitive)
        completed = sum(1 for j, _ in similar_projects if j.get("project_status", "").lower() == "completed")
        context["estimated_success_rate"] = completed / len(similar_projects) if similar_projects else 0.5

        return context

    # ============================================================================
    # ENHANCED RETRIEVAL: Deliverable-Level + Problem-to-Solution Matching
    # Issue #25: More granular matching for better proposal relevance
    # ============================================================================

    def retrieve_by_deliverables(
        self,
        job_requirements: "JobRequirements",
        all_jobs: List[Dict[str, Any]],
        top_k_per_deliverable: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve similar projects for EACH specific deliverable.
        
        Instead of one search for the whole job, search for each deliverable
        individually to find the BEST match for each requirement.
        
        Args:
            job_requirements: Extracted requirements with specific_deliverables
            all_jobs: All historical jobs
            top_k_per_deliverable: Max projects per deliverable
            
        Returns:
            Dict mapping each deliverable to its matching projects
            Example: {"Moodle integration": [project1], "Payment system": [project2]}
        """
        if not job_requirements or not job_requirements.specific_deliverables:
            return {}
        
        deliverable_matches = {}
        tech_context = " ".join(job_requirements.tech_stack_mentioned or [])
        
        logger.info(f"[DeliverableRetrieval] Searching for {len(job_requirements.specific_deliverables)} deliverables")
        
        for deliverable in job_requirements.specific_deliverables:
            # Create focused search query for this deliverable
            search_query = f"{deliverable} {tech_context}"
            
            # Score each job against this specific deliverable
            matches = []
            for job in all_jobs:
                score = self._score_job_for_deliverable(job, deliverable, job_requirements.tech_stack_mentioned)
                if score > 0.3:  # Minimum relevance threshold
                    matches.append((job, score))
            
            # Sort by score and take top K
            matches.sort(key=lambda x: x[1], reverse=True)
            top_matches = [job for job, score in matches[:top_k_per_deliverable]]
            
            if top_matches:
                deliverable_matches[deliverable] = top_matches
                logger.debug(f"  → '{deliverable[:40]}...' matched {len(top_matches)} projects")
        
        logger.info(f"[DeliverableRetrieval] Matched {len(deliverable_matches)}/{len(job_requirements.specific_deliverables)} deliverables")
        return deliverable_matches

    def _score_job_for_deliverable(
        self,
        job: Dict[str, Any],
        deliverable: str,
        tech_stack: List[str]
    ) -> float:
        """
        Score how well a job matches a specific deliverable.
        
        Uses keyword matching + tech stack overlap + outcome relevance.
        """
        score = 0.0
        deliverable_lower = deliverable.lower()
        
        # Extract job text for matching
        job_title = (job.get("job_title") or "").lower()
        job_desc = (job.get("job_description") or "").lower()
        skills = [s.lower() for s in job.get("skills_required", []) or []]
        deliverables_list = [d.lower() for d in job.get("deliverables", []) or []]
        outcome_stats = ""
        outcome = job.get("outcome")
        if isinstance(outcome, dict):
            outcome_stats = (outcome.get("stats") or "").lower()
        elif isinstance(outcome, str):
            outcome_stats = outcome.lower()
        
        # Extract keywords from deliverable
        keywords = [w for w in deliverable_lower.split() if len(w) > 3]
        
        # Title match (strongest signal)
        title_match = sum(1 for kw in keywords if kw in job_title)
        score += title_match * 0.25
        
        # Description match
        desc_match = sum(1 for kw in keywords if kw in job_desc)
        score += min(desc_match * 0.1, 0.3)  # Cap at 0.3
        
        # Skills match
        skill_match = sum(1 for kw in keywords if any(kw in s for s in skills))
        score += skill_match * 0.15
        
        # Deliverables list match (very strong signal)
        for d in deliverables_list:
            match_count = sum(1 for kw in keywords if kw in d)
            if match_count >= 2:
                score += 0.35
                break
        
        # Tech stack overlap
        if tech_stack:
            tech_overlap = sum(1 for t in tech_stack if t.lower() in " ".join(skills))
            score += min(tech_overlap * 0.1, 0.2)
        
        # Outcome mention (shows proven delivery)
        if outcome_stats:
            outcome_match = sum(1 for kw in keywords if kw in outcome_stats)
            score += min(outcome_match * 0.15, 0.2)
        
        return min(score, 1.0)  # Cap at 1.0

    def retrieve_by_problems_solved(
        self,
        job_requirements: "JobRequirements",
        all_jobs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find projects where we SOLVED similar problems.
        
        Searches in OUTCOMES and FEEDBACK fields - where we describe
        what we actually fixed/solved.
        
        Args:
            job_requirements: Extracted requirements with problems_mentioned
            all_jobs: All historical jobs
            
        Returns:
            List of {"problem": str, "project": dict, "evidence": str, "confidence": float}
        """
        if not job_requirements or not job_requirements.problems_mentioned:
            return []
        
        problem_solutions = []
        
        logger.info(f"[ProblemMatching] Searching for solutions to {len(job_requirements.problems_mentioned)} problems")
        
        for problem in job_requirements.problems_mentioned:
            problem_lower = problem.lower()
            problem_keywords = [w for w in problem_lower.split() if len(w) > 3]
            
            for job in all_jobs:
                # Search in outcome stats
                outcome = job.get("outcome", {})
                outcome_stats = ""
                if isinstance(outcome, dict):
                    outcome_stats = outcome.get("stats", "") or ""
                elif isinstance(outcome, str):
                    outcome_stats = outcome
                
                # Search in feedback
                feedback = job.get("client_feedback_text", "") or job.get("client_feedback", "") or ""
                
                # Search in job title/description too
                title = job.get("job_title", "") or ""
                
                search_text = f"{outcome_stats} {feedback} {title}".lower()
                
                # Score this job as a solution to the problem
                match_count = sum(1 for kw in problem_keywords if kw in search_text)
                confidence = match_count / len(problem_keywords) if problem_keywords else 0
                
                if confidence >= 0.4:  # At least 40% keyword match
                    evidence = outcome_stats if outcome_stats else feedback[:200]
                    problem_solutions.append({
                        "problem": problem,
                        "project": job,
                        "evidence": evidence,
                        "confidence": confidence,
                        "portfolio_url": (job.get("portfolio_urls", []) or [None])[0]
                    })
        
        # Sort by confidence and remove duplicates
        problem_solutions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Keep only top match per problem
        seen_problems = set()
        unique_solutions = []
        for sol in problem_solutions:
            if sol["problem"] not in seen_problems:
                seen_problems.add(sol["problem"])
                unique_solutions.append(sol)
        
        logger.info(f"[ProblemMatching] Found solutions for {len(unique_solutions)}/{len(job_requirements.problems_mentioned)} problems")
        return unique_solutions

    def analyze_portfolio_coverage(
        self,
        job_requirements: "JobRequirements",
        similar_projects: List[Dict[str, Any]],
        deliverable_matches: Dict[str, List[Dict[str, Any]]] = None,
        problem_solutions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze how well our portfolio covers the job's explicit checklist.
        
        For EACH checklist item, determine if we have proof to back it up.
        
        Args:
            job_requirements: Extracted requirements with explicit_checklist
            similar_projects: General similar projects found
            deliverable_matches: Per-deliverable matches (from retrieve_by_deliverables)
            problem_solutions: Problem-solution matches (from retrieve_by_problems_solved)
            
        Returns:
            Coverage analysis: {
                "items": [{item, has_proof, best_project, portfolio_url, confidence}],
                "overall_coverage": float (0-1),
                "gaps": [items without proof],
                "strategy": "full_proof" | "partial_proof" | "approach_first"
            }
        """
        if not job_requirements:
            return {"items": [], "overall_coverage": 1.0, "gaps": [], "strategy": "approach_first"}
        
        checklist = job_requirements.explicit_checklist or []
        if not checklist:
            # Use deliverables as implicit checklist
            checklist = [{"item_text": d, "item_type": "specific_answer"} 
                        for d in job_requirements.specific_deliverables or []]
        
        coverage_items = []
        
        for item in checklist:
            item_text = item.get("item_text", "")
            item_type = item.get("item_type", "other")
            
            result = {
                "checklist_item": item_text,
                "item_type": item_type,
                "has_proof": False,
                "best_project": None,
                "portfolio_url": None,
                "confidence": 0.0,
                "evidence_source": None
            }
            
            # Check deliverable matches first (most specific)
            if deliverable_matches:
                for deliverable, projects in deliverable_matches.items():
                    if self._text_similarity(item_text, deliverable) > 0.5 and projects:
                        best = projects[0]
                        result["has_proof"] = True
                        result["best_project"] = best.get("job_title") or best.get("project_title")
                        result["portfolio_url"] = (best.get("portfolio_urls", []) or [None])[0]
                        result["confidence"] = 0.9
                        result["evidence_source"] = "deliverable_match"
                        break
            
            # Check problem solutions
            if not result["has_proof"] and problem_solutions:
                for sol in problem_solutions:
                    if self._text_similarity(item_text, sol["problem"]) > 0.4:
                        result["has_proof"] = True
                        result["best_project"] = sol["project"].get("job_title")
                        result["portfolio_url"] = sol.get("portfolio_url")
                        result["confidence"] = sol["confidence"]
                        result["evidence_source"] = "problem_solution"
                        result["evidence_text"] = sol.get("evidence", "")[:100]
                        break
            
            # Fallback to general similar projects
            if not result["has_proof"] and similar_projects:
                for proj in similar_projects[:5]:
                    score = self._score_job_for_deliverable(
                        proj if isinstance(proj, dict) else proj[0],
                        item_text,
                        job_requirements.tech_stack_mentioned or []
                    )
                    if score > 0.4:
                        best = proj if isinstance(proj, dict) else proj[0]
                        result["has_proof"] = True
                        result["best_project"] = best.get("job_title")
                        result["portfolio_url"] = (best.get("portfolio_urls", []) or [None])[0]
                        result["confidence"] = score
                        result["evidence_source"] = "similar_project"
                        break
            
            coverage_items.append(result)
        
        # Calculate overall coverage
        covered = sum(1 for c in coverage_items if c["has_proof"])
        total = len(coverage_items) if coverage_items else 1
        overall_coverage = covered / total
        
        # Identify gaps
        gaps = [c["checklist_item"] for c in coverage_items if not c["has_proof"]]
        
        # Determine strategy based on coverage
        if overall_coverage >= 0.7:
            strategy = "full_proof"
        elif overall_coverage >= 0.4:
            strategy = "partial_proof"
        else:
            strategy = "approach_first"
        
        logger.info(f"[PortfolioCoverage] Coverage: {overall_coverage:.0%} ({covered}/{total}), Strategy: {strategy}")
        
        return {
            "items": coverage_items,
            "overall_coverage": overall_coverage,
            "gaps": gaps,
            "strategy": strategy,
            "covered_count": covered,
            "total_count": total
        }

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word-overlap similarity between two texts."""
        words1 = set(w.lower() for w in text1.split() if len(w) > 3)
        words2 = set(w.lower() for w in text2.split() if len(w) > 3)
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def get_no_portfolio_strategy(
        self,
        job_requirements: "JobRequirements",
        all_jobs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Determine the best strategy when no direct portfolio match exists.
        
        Returns strategy and supporting evidence for generating proposals
        without exact portfolio proof.
        
        Strategies:
        1. SKILL_BASED: 70%+ skill overlap - lead with skill expertise
        2. TRANSFERABLE: 40-70% overlap - use similar domain experience
        3. APPROACH_FIRST: <40% overlap - lead with methodology + ask questions
        
        Args:
            job_requirements: Extracted requirements
            all_jobs: Historical jobs to analyze for transferable skills
            
        Returns:
            {
                "strategy": "skill_based" | "transferable" | "approach_first",
                "skill_coverage": float,
                "matching_skills": [skills we have],
                "missing_skills": [skills we lack],
                "transferable_experience": [related projects],
                "recommended_questions": [diagnostic questions to ask],
                "hook_approach": str (recommended hook style)
            }
        """
        if not job_requirements:
            return {"strategy": "approach_first", "skill_coverage": 0, "matching_skills": [], "missing_skills": []}
        
        required_skills = set(s.lower() for s in (job_requirements.tech_stack_mentioned or []))
        required_skills.update(s.lower() for s in (job_requirements.specific_deliverables or []))
        
        # Collect all skills from portfolio
        all_portfolio_skills = set()
        for job in all_jobs:
            skills = job.get("skills_required", []) or job.get("skills", []) or []
            all_portfolio_skills.update(s.lower() for s in skills)
        
        # Calculate skill coverage
        matching_skills = required_skills & all_portfolio_skills
        missing_skills = required_skills - all_portfolio_skills
        skill_coverage = len(matching_skills) / len(required_skills) if required_skills else 0
        
        # Find transferable experience (similar tech/domain even if not exact)
        transferable = []
        for job in all_jobs[:20]:
            job_skills = set(s.lower() for s in (job.get("skills_required", []) or []))
            overlap = len(job_skills & required_skills)
            if overlap >= 1:  # At least 1 skill overlap
                transferable.append({
                    "project": job.get("job_title"),
                    "matching_skills": list(job_skills & required_skills),
                    "portfolio_url": (job.get("portfolio_urls", []) or [None])[0]
                })
        
        # Determine strategy
        if skill_coverage >= 0.7:
            strategy = "skill_based"
            hook_approach = "I work with [TECH] daily - here's my approach to your specific challenge..."
        elif skill_coverage >= 0.4:
            strategy = "transferable"
            hook_approach = "While I haven't worked with [EXACT_TECH], I've built similar systems using [RELATED_TECH]..."
        else:
            strategy = "approach_first"
            hook_approach = "I'd approach this by first understanding [ASPECT]. Quick question: [DIAGNOSTIC_Q]..."
        
        # Generate recommended questions for approach_first
        recommended_questions = []
        if strategy == "approach_first":
            if job_requirements.problems_mentioned:
                recommended_questions.append(f"Could you share more about when the {job_requirements.problems_mentioned[0]} started? That'll help me diagnose the root cause.")
            if job_requirements.links_mentioned:
                recommended_questions.append("Could you share temporary access so I can take a deeper look at the current setup?")
            else:
                recommended_questions.append("Could you share the site/system URL? I'd love to do a quick audit before proposing a solution.")
        
        logger.info(f"[NoPortfolio] Strategy: {strategy}, Skill coverage: {skill_coverage:.0%}")
        
        return {
            "strategy": strategy,
            "skill_coverage": skill_coverage,
            "matching_skills": list(matching_skills),
            "missing_skills": list(missing_skills),
            "transferable_experience": transferable[:3],  # Top 3
            "recommended_questions": recommended_questions,
            "hook_approach": hook_approach
        }

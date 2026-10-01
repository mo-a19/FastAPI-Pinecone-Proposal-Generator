"""
Proposal Service

Business logic for proposal generation, including:
- Retrieval of similar projects
- Analysis of previous proposals
- Proposal generation with AI  
- Quality scoring and suggestions
- Streaming generation with progress events
"""
import logging
import json
from typing import Optional, List, Dict, Any, Generator, Callable
from dataclasses import dataclass, field
from datetime import datetime

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ProposalRequest:
    """Input parameters for proposal generation."""
    job_title: str
    company_name: str
    job_description: str
    skills_required: List[str]
    industry: Optional[str] = "general"
    task_type: Optional[str] = "other"
    estimated_budget: Optional[float] = None
    project_duration_days: Optional[int] = None
    urgent_adhoc: bool = False
    proposal_style: str = "professional"
    tone: str = "confident"
    max_word_count: int = 300
    include_timeline: bool = False
    timeline_duration: Optional[str] = None
    similar_projects_count: int = 3
    include_previous_proposals: bool = True
    include_portfolio: bool = True
    include_feedback: bool = True
    # Multi-tenant fields (optional - for org-scoped queries)
    org_id: Optional[str] = None
    profile_id: Optional[str] = None
    # Hook question (user-selected opening question)
    selected_hook_question: Optional[str] = None


@dataclass
class ProposalResult:
    """Result of proposal generation."""
    success: bool
    generated_proposal: str = ""
    word_count: int = 0
    similar_projects: List[Dict[str, Any]] = field(default_factory=list)
    previous_proposals_insights: Optional[Dict[str, Any]] = None
    portfolio_links_used: List[str] = field(default_factory=list)
    feedback_urls_used: List[str] = field(default_factory=list)
    insights: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    improvement_suggestions: Optional[List[str]] = None
    error_message: Optional[str] = None
    # Phase 3: Post-generation validation warnings
    validation_warnings: Optional[List[str]] = None


class ProposalService:
    """
    Service for intelligent proposal generation.
    
    Orchestrates:
    - JobRequirementsService for deep job understanding (NEW)
    - RetrievalPipeline for finding similar projects
    - PromptEngine for building prompts
    - OpenAI for text generation
    """
    
    def __init__(
        self,
        db_manager=None,
        openai_service=None,
        retrieval_pipeline=None,
        prompt_engine=None,
        job_requirements_service=None
    ):
        """
        Initialize with dependencies.
        
        Args:
            db_manager: Database manager instance
            openai_service: OpenAI service for text generation
            retrieval_pipeline: Pipeline for finding similar projects
            prompt_engine: Engine for building prompts
            job_requirements_service: Service for extracting structured job requirements (NEW)
        """
        self.db = db_manager
        self.openai_service = openai_service
        self.retrieval_pipeline = retrieval_pipeline
        self.prompt_engine = prompt_engine
        self.job_requirements_service = job_requirements_service
    
    def generate_proposal(self, request: ProposalRequest) -> ProposalResult:
        """
        Generate an intelligent proposal based on job details and history.
        
        ENHANCED Steps (Issue #25):
        1. Get historical jobs
        2. Find similar projects via semantic search
        2a. NEW: Deliverable-level retrieval (find best match per deliverable)
        2b. NEW: Problem-to-solution matching (find where we solved similar problems)
        2c. NEW: Portfolio coverage analysis (what can we prove?)
        2d. NEW: No-portfolio strategy (if no matches, determine approach)
        3. Analyze previous proposals
        4. Collect portfolio and feedback URLs
        5. Build enhanced prompt with specific proof
        6. Generate proposal with AI
        7. Score quality and suggest improvements
        
        Args:
            request: ProposalRequest with job details and options
            
        Returns:
            ProposalResult with generated proposal and metadata
        """
        try:
            logger.info(f"[ProposalService] Generating proposal for {request.company_name} - {request.job_title}")
            
            # Prepare job data
            job_data = self._prepare_job_data(request)
            
            # Step 0: Extract structured job requirements (Issue #23)
            job_requirements = self._extract_job_requirements(request)
            if job_requirements:
                logger.info(f"[ProposalService] Extracted requirements: task='{job_requirements.exact_task[:50]}...' "
                           f"tone={job_requirements.client_tone}, confidence={job_requirements.extraction_confidence:.2f}")
                # Enhance job_data with extracted requirements
                job_data = self._enhance_job_data_with_requirements(job_data, job_requirements)
            
            # Step 1: Get historical jobs (org-scoped if org_id provided)
            all_jobs = self._get_historical_jobs(request.org_id)
            
            # Step 2: Find similar projects (ENHANCED with job_requirements - Issue #24)
            retrieval_result = self._find_similar_projects(
                job_data, all_jobs, request.similar_projects_count,
                job_requirements=job_requirements
            )
            similar_projects = retrieval_result.get("similar_projects", [])
            
            # Filter to projects with portfolio
            similar_projects = self._filter_projects_with_portfolio(similar_projects)
            
            # ============ NEW: ENHANCED RETRIEVAL (Issue #25) ============
            deliverable_matches = {}
            problem_solutions = []
            portfolio_coverage = {"overall_coverage": 0, "strategy": "approach_first", "items": [], "gaps": []}
            no_portfolio_strategy = None
            
            if self.retrieval_pipeline and job_requirements:
                # Step 2a: Deliverable-level retrieval
                deliverable_matches = self.retrieval_pipeline.retrieve_by_deliverables(
                    job_requirements, all_jobs, top_k_per_deliverable=2
                )
                if deliverable_matches:
                    logger.info(f"[ProposalService] Matched {len(deliverable_matches)} deliverables to past projects")
                
                # Step 2b: Problem-to-solution matching
                problem_solutions = self.retrieval_pipeline.retrieve_by_problems_solved(
                    job_requirements, all_jobs
                )
                if problem_solutions:
                    logger.info(f"[ProposalService] Found solutions for {len(problem_solutions)} mentioned problems")
                
                # Step 2c: Portfolio coverage analysis
                portfolio_coverage = self.retrieval_pipeline.analyze_portfolio_coverage(
                    job_requirements, similar_projects, deliverable_matches, problem_solutions
                )
                logger.info(f"[ProposalService] Portfolio coverage: {portfolio_coverage.get('overall_coverage', 0):.0%}, "
                           f"strategy: {portfolio_coverage.get('strategy')}, gaps: {len(portfolio_coverage.get('gaps', []))}")
                
                # Step 2d: No-portfolio strategy (if coverage is low)
                if portfolio_coverage.get("overall_coverage", 0) < 0.4 or not similar_projects:
                    no_portfolio_strategy = self.retrieval_pipeline.get_no_portfolio_strategy(
                        job_requirements, all_jobs
                    )
                    logger.info(f"[ProposalService] No-portfolio strategy: {no_portfolio_strategy.get('strategy')}, "
                               f"skill coverage: {no_portfolio_strategy.get('skill_coverage', 0):.0%}")
            
            # Step 2.5: Phase 2 - Analyze capability and get factual answers
            capability_analysis = None
            factual_answers = {}
            if job_requirements and job_requirements.explicit_checklist:
                capability_analysis = self._analyze_checklist_capability(
                    job_requirements.explicit_checklist,
                    similar_projects,
                    all_jobs
                )
                factual_answers = self._get_factual_answers(
                    job_requirements.explicit_checklist,
                    request.org_id
                )
                logger.info(f"[ProposalService] Capability coverage: {capability_analysis.get('overall_coverage', 0):.0%}, facts: {list(factual_answers.keys())}")
            
            # Step 3: Analyze previous proposals
            previous_insights = None
            if request.include_previous_proposals and similar_projects:
                previous_insights = self._analyze_previous_proposals(
                    similar_projects[:request.similar_projects_count]
                )
            
            # Step 4: Collect portfolio and feedback URLs
            portfolio_links, feedback_urls = self._collect_urls(
                similar_projects[:request.similar_projects_count],
                request.include_portfolio,
                request.include_feedback
            )
            
            # Step 4.5: Get freelancer profile context if available
            profile_context = None
            if request.profile_id and request.org_id:
                profile_context = self._get_freelancer_profile(request.org_id, request.profile_id)
            
            # Step 5: Build prompt (now with job requirements context)
            prompt = self._build_prompt(
                job_data=job_data,
                similar_projects=similar_projects[:request.similar_projects_count],
                previous_insights=previous_insights,
                portfolio_links=portfolio_links,
                feedback_urls=feedback_urls,
                request=request,
                profile_context=profile_context,
                job_requirements=job_requirements if 'job_requirements' in dir() else None,
                capability_analysis=capability_analysis,
                factual_answers=factual_answers,
                # NEW: Enhanced retrieval data (Issue #25)
                deliverable_matches=deliverable_matches,
                problem_solutions=problem_solutions,
                portfolio_coverage=portfolio_coverage,
                no_portfolio_strategy=no_portfolio_strategy
            )
            
            # Step 6: Generate proposal
            proposal_text, word_count = self._generate_with_ai(prompt, request.max_word_count)
            
            # Step 6.5: Validate proposal against requirements (Phase 3)
            validation_warnings = self._validate_proposal(
                proposal_text=proposal_text,
                job_requirements=job_requirements if 'job_requirements' in dir() else None,
                capability_analysis=capability_analysis,
                portfolio_links=portfolio_links
            )
            
            # Step 7: Score quality
            confidence_score, suggestions = self._score_proposal(
                proposal_text=proposal_text,
                portfolio_links=portfolio_links,
                feedback_urls=feedback_urls,
                similar_projects=similar_projects,
                request=request
            )
            
            return ProposalResult(
                success=True,
                generated_proposal=proposal_text,
                word_count=word_count,
                similar_projects=similar_projects[:request.similar_projects_count],
                previous_proposals_insights=previous_insights,
                portfolio_links_used=portfolio_links,
                feedback_urls_used=feedback_urls,
                insights=retrieval_result.get("insights"),
                confidence_score=confidence_score,
                improvement_suggestions=suggestions,
                validation_warnings=validation_warnings
            )
            
        except Exception as e:
            logger.error(f"[ProposalService] Error generating proposal: {e}")
            return ProposalResult(
                success=False,
                error_message=str(e)
            )
    
    def _prepare_job_data(self, request: ProposalRequest) -> Dict[str, Any]:
        """Prepare job data dict from request."""
        job_data = {
            "job_title": request.job_title,
            "company_name": request.company_name,
            "job_description": request.job_description,
            "skills_required": request.skills_required,
            "industry": request.industry or "general",
            "task_type": request.task_type or "other"
        }
        
        # Enhance industry detection if needed
        if not request.industry or request.industry == "general":
            job_data["industry"] = self._detect_industry(job_data)
        
        return job_data
    
    def _extract_job_requirements(self, request: ProposalRequest) -> Optional[Any]:
        """
        Extract structured job requirements using JobRequirementsService.
        
        This is the NEW job understanding layer that deeply analyzes the job
        description BEFORE retrieval and generation.
        
        Args:
            request: ProposalRequest with job details
            
        Returns:
            JobRequirements dataclass or None if service unavailable
        """
        if not self.job_requirements_service:
            logger.debug("[ProposalService] JobRequirementsService not available, skipping extraction")
            return None
        
        try:
            from app.services.job_requirements_service import JobRequirements
            
            requirements = self.job_requirements_service.extract_job_requirements(
                job_description=request.job_description,
                job_title=request.job_title,
                skills_required=request.skills_required
            )
            
            return requirements
            
        except Exception as e:
            logger.warning(f"[ProposalService] Job requirements extraction failed: {e}")
            return None
    
    def _enhance_job_data_with_requirements(
        self,
        job_data: Dict[str, Any],
        requirements: Any
    ) -> Dict[str, Any]:
        """
        Enhance job_data with extracted requirements for better retrieval.
        
        This merges the LLM-extracted understanding into the job_data dict
        so the retrieval pipeline can find truly relevant projects.
        
        Args:
            job_data: Original job data dict
            requirements: JobRequirements dataclass
            
        Returns:
            Enhanced job_data dict
        """
        # Add extracted understanding to job_data
        job_data["exact_task"] = requirements.exact_task
        job_data["client_tone"] = requirements.client_tone
        job_data["client_problems"] = requirements.problems_mentioned
        job_data["client_constraints"] = requirements.constraints
        job_data["tech_stack"] = requirements.tech_stack_mentioned
        job_data["do_not_assume"] = requirements.do_not_assume
        
        # Override industry if extracted with high confidence
        if requirements.inferred_industry and requirements.inferred_industry != "general":
            if requirements.extraction_confidence >= 0.6:
                job_data["industry"] = requirements.inferred_industry
                logger.debug(f"[ProposalService] Industry overridden to: {requirements.inferred_industry}")
        
        # Add complexity for retrieval filtering
        job_data["task_complexity"] = requirements.complexity_level
        
        # Add deliverables for better semantic matching
        if requirements.specific_deliverables:
            job_data["expected_deliverables"] = requirements.specific_deliverables
        
        # NEW FIELDS - Sprint 3 Enhanced Intent Capture
        # Working arrangement (critical for service roles)
        if requirements.working_arrangement:
            job_data["working_arrangement"] = requirements.working_arrangement
            # Log timezone requirements as these are often deal-breakers
            if requirements.working_arrangement.get("timezone"):
                logger.info(f"[ProposalService] Timezone requirement: {requirements.working_arrangement.get('timezone')}")
        
        # Application requirements (critical - missing these = rejection)
        if requirements.application_requirements:
            job_data["application_requirements"] = requirements.application_requirements
            logger.info(f"[ProposalService] Application requirements: {requirements.application_requirements}")
        
        # Soft requirements (what client values beyond tech)
        if requirements.soft_requirements:
            job_data["soft_requirements"] = requirements.soft_requirements
        
        # Client priorities (what matters most)
        if requirements.client_priorities:
            job_data["client_priorities"] = requirements.client_priorities
        
        # Must NOT propose (avoid these suggestions)
        if requirements.must_not_propose:
            job_data["must_not_propose"] = requirements.must_not_propose
        
        # Key phrases to echo (for rapport building)
        if requirements.key_phrases_to_echo:
            job_data["key_phrases_to_echo"] = requirements.key_phrases_to_echo
        
        return job_data
    
    def _detect_industry(self, job_data: Dict[str, Any]) -> str:
        """Detect industry from job data context."""
        try:
            from app.utils.metadata_extractor import MetadataExtractor
            result = MetadataExtractor.detect_industry_with_context(job_data)
            if result.get("industry") != "general" and result.get("confidence", 0) >= 0.5:
                logger.info(f"[ProposalService] Industry detected: {result['industry']}")
                return result["industry"]
        except Exception as e:
            logger.debug(f"Industry detection failed: {e}")
        return job_data.get("industry", "general")
    
    def _get_historical_jobs(self, org_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get historical jobs from portfolio_items collection."""
        if not self.db:
            return []
        try:
            # Fetch from portfolio_items collection
            items = list(self.db.db.portfolio_items.find({}))
            if items:
                # Convert portfolio items to job-like format for retrieval
                return [self._portfolio_to_job_format(item) for item in items]
            return []
        except Exception as e:
            logger.error(f"Error fetching historical jobs: {e}")
            return []
    
    def _portfolio_to_job_format(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert lean portfolio item to job format for retrieval compatibility."""
        return {
            "contract_id": item.get("item_id"),
            "job_title": item.get("project_title", ""),
            "skills_required": item.get("skills", []),
            "industry": item.get("industry", "general"),
            "deliverables": item.get("deliverables", []),
            "outcome": item.get("outcome"),
            "company_name": item.get("project_title", ""),
            "portfolio_urls": [item.get("portfolio_url")] if item.get("portfolio_url") else [],
            "client_feedback_text": item.get("client_feedback", ""),
            "duration_days": item.get("duration_days"),
            "project_status": "completed",
            "is_portfolio_entry": True,  # All portfolio_items are portfolio entries
        }
    
    def _get_freelancer_profile(self, org_id: str, profile_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get freelancer profile for proposal personalization."""
        try:
            from app.infra.mongodb.repositories import get_freelancer_profile_repo
            profile_repo = get_freelancer_profile_repo()
            
            if profile_id:
                return profile_repo.get_by_profile_id(profile_id)
            return profile_repo.get_default_profile(org_id)
        except Exception as e:
            logger.debug(f"Profile fetch failed: {e}")
            return None
    
    def _find_similar_projects(
        self,
        job_data: Dict[str, Any],
        all_jobs: List[Dict[str, Any]],
        top_k: int,
        job_requirements: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Find similar projects using retrieval pipeline.
        
        ENHANCED (Issue #24): Now passes JobRequirements to retrieval for:
        - Better semantic search queries using specific_deliverables
        - Problem-based matching using problems_mentioned
        - More accurate filtering using extracted tech stack and complexity
        """
        if not self.retrieval_pipeline:
            return {"similar_projects": [], "insights": None}
        
        try:
            return self.retrieval_pipeline.retrieve_for_proposal(
                job_data, all_jobs, top_k=top_k,
                job_requirements=job_requirements
            )
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return {"similar_projects": [], "insights": None}
    
    def _filter_projects_with_portfolio(
        self,
        projects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter to projects with actual portfolio URLs."""
        filtered = []
        for project in projects:
            portfolio_urls = project.get("portfolio_urls", [])
            if isinstance(portfolio_urls, str):
                portfolio_urls = [portfolio_urls] if portfolio_urls else []
            portfolio_urls = [url for url in portfolio_urls if url]
            project["portfolio_urls"] = portfolio_urls
            
            if portfolio_urls:
                filtered.append(project)
        
        return filtered if filtered else projects
    
    def _analyze_checklist_capability(
        self,
        checklist: List[Dict[str, Any]],
        similar_projects: List[Dict[str, Any]],
        all_jobs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Phase 2: Analyze if portfolio can satisfy explicit checklist requirements.
        
        Returns capability analysis with status (SATISFIED/PARTIAL/UNSATISFIED) per item.
        """
        if not checklist:
            return {"items": [], "overall_coverage": 1.0}
        
        # Collect all available portfolio URLs and skills
        all_urls = []
        all_skills = set()
        for p in similar_projects + all_jobs[:20]:
            urls = p.get("portfolio_urls", [])
            if isinstance(urls, list):
                all_urls.extend([u for u in urls if u])
            skills = p.get("skills_required", []) or p.get("skills", [])
            all_skills.update(s.lower() for s in skills if s)
        
        analysis = {"items": [], "overall_coverage": 0}
        satisfied_count = 0
        
        for item in checklist:
            item_type = item.get("item_type", "other")
            qty_needed = item.get("quantity_requested", 1) or 1
            item_text = item.get("item_text", "").lower()
            
            result = {"item": item, "status": "UNSATISFIED", "evidence": [], "strategy": ""}
            
            if item_type == "portfolio_links":
                # Check if we have enough portfolio URLs
                if len(all_urls) >= qty_needed:
                    result["status"] = "SATISFIED"
                    result["evidence"] = all_urls[:qty_needed]
                    result["strategy"] = "direct_evidence"
                elif all_urls:
                    result["status"] = "PARTIAL"
                    result["evidence"] = all_urls
                    result["strategy"] = f"have {len(all_urls)} of {qty_needed} requested"
                else:
                    result["strategy"] = "acknowledge_gap_pivot_to_skills"
            
            elif item_type in ["experience_question", "specific_answer"]:
                # Check if we have relevant skills/projects
                keywords = [w for w in item_text.split() if len(w) > 3]
                matches = [s for s in all_skills if any(k in s for k in keywords)]
                if matches:
                    result["status"] = "PARTIAL" if len(matches) < 3 else "SATISFIED"
                    result["evidence"] = list(matches)[:5]
                    result["strategy"] = "cite_relevant_experience"
                else:
                    result["strategy"] = "honest_transferable_skills"
            
            elif item_type in ["time_estimate_total", "time_estimate_phased"]:
                # Always can provide estimate
                result["status"] = "SATISFIED"
                result["strategy"] = "provide_estimate"
            
            elif item_type in ["preference_question", "approach_description"]:
                # Always can provide opinion/approach
                result["status"] = "SATISFIED"
                result["strategy"] = "provide_recommendation"
            
            else:
                result["status"] = "PARTIAL"
                result["strategy"] = "address_directly"
            
            if result["status"] == "SATISFIED":
                satisfied_count += 1
            elif result["status"] == "PARTIAL":
                satisfied_count += 0.5
            
            analysis["items"].append(result)
        
        analysis["overall_coverage"] = satisfied_count / len(checklist) if checklist else 1.0
        return analysis
    
    def _get_factual_answers(
        self,
        checklist: List[Dict[str, Any]],
        org_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Phase 2: Query database for factual answers to client questions.
        
        Answers questions like "How many WordPress sites have you built?"
        """
        if not self.db or not checklist:
            return {}
        
        facts = {}
        
        for item in checklist:
            if item.get("item_type") != "experience_question":
                continue
            
            text = item.get("item_text", "").lower()
            
            # Pattern: "how many [tech] sites/projects"
            if "how many" in text:
                tech_keywords = ["wordpress", "shopify", "react", "woocommerce", "elementor", 
                               "website", "site", "project", "app", "application"]
                for tech in tech_keywords:
                    if tech in text:
                        try:
                            # Query portfolio_items collection
                            collection = self.db.db.portfolio_items
                            query = {
                                "$or": [
                                    {"skills": {"$regex": tech, "$options": "i"}},
                                    {"project_title": {"$regex": tech, "$options": "i"}}
                                ]
                            }
                            count = collection.count_documents(query)
                            if count > 0:
                                facts[f"{tech}_count"] = count
                        except Exception as e:
                            logger.debug(f"Factual query error: {e}")
                        break
        
        return facts
    
    def _analyze_previous_proposals(
        self,
        similar_projects: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze previous proposals for similar jobs."""
        if not self.db:
            return None
        
        previous_proposals = []
        for project in similar_projects:
            contract_id = project.get("contract_id")
            if not contract_id:
                continue
            
            stored = self.db.db.proposals.find_one({
                "contract_id": contract_id,
                "generated_proposal": {"$exists": True}
            })
            
            if stored:
                previous_proposals.append({
                    "contract_id": contract_id,
                    "company": project.get("company"),
                    "proposal_text": stored.get("generated_proposal"),
                    "style": stored.get("proposal_style"),
                    "tone": stored.get("proposal_tone"),
                    "word_count": stored.get("word_count", 0)
                })
        
        if not previous_proposals:
            return None
        
        return {
            "total_previous_proposals": len(previous_proposals),
            "common_patterns": self._extract_proposal_patterns(previous_proposals),
            "effective_phrases": self._extract_effective_phrases(previous_proposals),
            "average_word_count": sum(p["word_count"] for p in previous_proposals) / len(previous_proposals)
        }
    
    def _extract_proposal_patterns(self, proposals: List[Dict[str, Any]]) -> List[str]:
        """Extract common patterns from previous proposals."""
        patterns = []
        for p in proposals:
            text = p.get("proposal_text", "")
            if "specific experience" in text.lower():
                patterns.append("References specific past experience")
            if "similar project" in text.lower():
                patterns.append("Mentions similar past projects")
            if any(word in text.lower() for word in ["timeline", "milestone", "deadline"]):
                patterns.append("Includes timeline/milestones")
            if any(word in text.lower() for word in ["portfolio", "work sample", "example"]):
                patterns.append("References portfolio/samples")
        return list(set(patterns))
    
    def _extract_effective_phrases(self, proposals: List[Dict[str, Any]]) -> List[str]:
        """Extract effective phrases from proposals."""
        phrases = []
        for p in proposals:
            text = p.get("proposal_text", "")
            if "I'd love to" in text:
                phrases.append("I'd love to")
            if "Based on my experience" in text:
                phrases.append("Based on my experience")
            if "I've successfully" in text:
                phrases.append("I've successfully")
        return list(set(phrases))[:5]
    
    def _collect_urls(
        self,
        projects: List[Dict[str, Any]],
        include_portfolio: bool,
        include_feedback: bool
    ) -> tuple:
        """Collect portfolio and feedback URLs from projects."""
        portfolio_links = []
        feedback_urls = []
        
        for project in projects:
            if include_portfolio:
                urls = project.get("portfolio_urls", [])
                if isinstance(urls, list):
                    portfolio_links.extend(urls)
                elif isinstance(urls, str) and urls:
                    portfolio_links.append(urls)
            
            if include_feedback:
                feedback_url = project.get("client_feedback_url")
                if feedback_url:
                    feedback_urls.append(feedback_url)
        
        return portfolio_links, feedback_urls
    
    def _build_prompt(
        self,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        previous_insights: Optional[Dict[str, Any]],
        portfolio_links: List[str],
        feedback_urls: List[str],
        request: ProposalRequest,
        profile_context: Optional[Dict[str, Any]] = None,
        job_requirements: Optional[Any] = None,
        capability_analysis: Optional[Dict[str, Any]] = None,
        factual_answers: Optional[Dict[str, Any]] = None,
        deliverable_matches: Optional[Dict[str, List[Dict]]] = None,
        problem_solutions: Optional[List[Dict]] = None,
        portfolio_coverage: Optional[Dict[str, Any]] = None,
        no_portfolio_strategy: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the proposal generation prompt with enhanced job understanding."""
        if not self.prompt_engine:
            # Fallback to basic prompt
            return self._build_basic_prompt(job_data, request, profile_context, job_requirements)
        
        success_patterns = []
        if previous_insights:
            success_patterns = previous_insights.get("common_patterns", [])
        
        # Build job requirements context for the prompt engine
        requirements_context = None
        if job_requirements:
            requirements_context = {
                "exact_task": job_requirements.exact_task,
                "deliverables": job_requirements.specific_deliverables,
                "client_tone": job_requirements.client_tone,
                "problems": job_requirements.problems_mentioned,
                "constraints": job_requirements.constraints,
                "do_not_assume": job_requirements.do_not_assume,  # CRITICAL: Anti-hallucination
                "resources_provided": job_requirements.resources_provided,
                # NEW FIELDS - Sprint 3 Enhanced Intent Capture
                "working_arrangement": job_requirements.working_arrangement,
                "application_requirements": job_requirements.application_requirements,
                "soft_requirements": job_requirements.soft_requirements,
                "client_priorities": job_requirements.client_priorities,
                "must_not_propose": job_requirements.must_not_propose,  # CRITICAL: What NOT to suggest
                "key_phrases_to_echo": job_requirements.key_phrases_to_echo,
                # Phase 1: Explicit checklist for mandatory requirement tracking
                "explicit_checklist": job_requirements.explicit_checklist,
                # Phase 2: Capability analysis and factual answers
                "capability_analysis": capability_analysis,
                "factual_answers": factual_answers or {}, 
                "smart_question": job_requirements.smart_question,
                # NEW: Enhanced Retrieval Data for Specific Proof
                "deliverable_matches": deliverable_matches or {},
                "problem_solutions": problem_solutions or [],
                "portfolio_coverage": portfolio_coverage or {},
                "no_portfolio_strategy": no_portfolio_strategy,
            }
        
        # Inject user-selected hook question (overrides smart_question)
        if request.selected_hook_question:
            if requirements_context is None:
                requirements_context = {}
            requirements_context["selected_hook_question"] = request.selected_hook_question
        
        return self.prompt_engine.build_proposal_prompt(
            job_data=job_data,
            similar_projects=similar_projects,
            success_patterns=success_patterns,
            style=request.proposal_style,
            tone=request.tone,
            max_words=request.max_word_count,
            include_portfolio=request.include_portfolio and bool(portfolio_links),
            include_feedback=request.include_feedback and bool(feedback_urls),
            include_timeline=request.include_timeline,
            timeline_duration=request.timeline_duration,
            profile_context=profile_context,
            requirements_context=requirements_context  # NEW: Pass extracted requirements
        )
    
    def _build_basic_prompt(
        self, 
        job_data: Dict[str, Any], 
        request: ProposalRequest,
        profile_context: Optional[Dict[str, Any]] = None,
        job_requirements: Optional[Any] = None
    ) -> str:
        """Build a basic prompt when prompt engine is not available."""
        profile_section = ""
        if profile_context:
            profile_section = f"""
Freelancer Profile:
- Name: {profile_context.get('display_name', 'Freelancer')}
- Title: {profile_context.get('headline', '')}
- Skills: {', '.join(profile_context.get('skills', [])[:10])}
- Bio: {profile_context.get('bio', '')[:300]}
"""
        
        # Add requirements section if available (NEW - Issue #23)
        requirements_section = ""
        if job_requirements:
            requirements_section = f"""
EXTRACTED JOB UNDERSTANDING:
- Exact Task: {job_requirements.exact_task}
- Client Tone: {job_requirements.client_tone}
- Problems Mentioned: {', '.join(job_requirements.problems_mentioned) if job_requirements.problems_mentioned else 'None stated'}
- DO NOT ASSUME (avoid mentioning these unless client specified): {', '.join(job_requirements.do_not_assume) if job_requirements.do_not_assume else 'N/A'}
"""
        
        return f"""Generate a {request.proposal_style} proposal for:

Job Title: {job_data['job_title']}
Company: {job_data['company_name']}
Description: {job_data['job_description']}
Skills Required: {', '.join(job_data['skills_required'])}
{profile_section}{requirements_section}
Write a {request.tone} proposal in approximately {request.max_word_count} words."""
    
    def _generate_with_ai(self, prompt: str, max_word_count: int) -> tuple:
        """Generate proposal text using AI."""
        if not self.openai_service:
            raise ValueError("OpenAI service not initialized")
        
        max_tokens = int(max_word_count * 1.5) + 100
        
        proposal_text = self.openai_service.generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        word_count = len(proposal_text.split())
        logger.info(f"[ProposalService] Generated {word_count} words (target ~{max_word_count})")
        
        return proposal_text, word_count
    
    def _validate_proposal(
        self,
        proposal_text: str,
        job_requirements: Optional[Any],
        capability_analysis: Optional[Dict[str, Any]],
        portfolio_links: List[str]
    ) -> Optional[List[str]]:
        """
        Phase 3: Post-generation validation to detect issues.
        
        Validates:
        1. All checklist items are addressed in the proposal
        2. No timezone hallucination (only mention if source was explicit)
        3. UNSATISFIED capabilities aren't claimed as strengths
        4. Requested portfolio links are included
        
        Returns:
            List of warning messages or None if all checks pass
        """
        warnings = []
        proposal_lower = proposal_text.lower()
        
        # === Check 1: Checklist coverage ===
        if job_requirements and hasattr(job_requirements, 'explicit_checklist'):
            checklist = job_requirements.explicit_checklist or []
            for item in checklist:
                item_text = item.get("item_text", "")
                item_type = item.get("item_type", "")
                
                # Extract keywords from the checklist item for matching
                keywords = self._extract_validation_keywords(item_text, item_type)
                
                if keywords and not any(kw in proposal_lower for kw in keywords):
                    warnings.append(f"⚠️ May not address: '{item_text[:60]}...'")
        
        # === Check 2: Timezone hallucination ===
        if job_requirements and hasattr(job_requirements, 'working_arrangement'):
            working = job_requirements.working_arrangement or {}
            tz_source = working.get("timezone_source", "none")
            
            # If timezone wasn't explicitly stated, check for hallucination
            if tz_source != "explicit":
                tz_patterns = ["utc", "gmt", "est", "pst", "cet", "timezone", "time zone"]
                for pattern in tz_patterns:
                    if pattern in proposal_lower:
                        warnings.append(
                            f"🚨 HALLUCINATION: Mentions '{pattern.upper()}' but client didn't specify timezone"
                        )
                        break
        
        # === Check 3: Unsatisfied capability claims ===
        if capability_analysis:
            for item_text, analysis in capability_analysis.items():
                if analysis.get("status") == "UNSATISFIED":
                    # Check if proposal makes strong claims about this area
                    keywords = self._extract_validation_keywords(item_text, "experience_question")
                    strong_claim_patterns = [
                        "extensive experience", "expert in", "deep expertise",
                        "years of experience with", "specialized in"
                    ]
                    for pattern in strong_claim_patterns:
                        if pattern in proposal_lower:
                            # Check if near the unsatisfied topic
                            for kw in keywords[:2]:
                                if kw in proposal_lower:
                                    # Find if they're close together (within 100 chars)
                                    pattern_pos = proposal_lower.find(pattern)
                                    kw_pos = proposal_lower.find(kw)
                                    if abs(pattern_pos - kw_pos) < 100:
                                        warnings.append(
                                            f"🚨 UNSUPPORTED CLAIM: Strong claim near '{kw}' "
                                            f"but portfolio shows UNSATISFIED"
                                        )
                                        break
        
        # === Check 4: Portfolio link inclusion ===
        if portfolio_links:
            links_found = sum(1 for link in portfolio_links if link in proposal_text)
            
            # Check if checklist required specific number of links
            if job_requirements and hasattr(job_requirements, 'explicit_checklist'):
                for item in (job_requirements.explicit_checklist or []):
                    if item.get("item_type") == "portfolio_links":
                        qty_requested = item.get("quantity_requested", 0)
                        if qty_requested and links_found < qty_requested:
                            warnings.append(
                                f"⚠️ Client asked for {qty_requested} portfolio links, "
                                f"but only {links_found} found in proposal"
                            )
                        break
        
        return warnings if warnings else None
    
    def _extract_validation_keywords(self, text: str, item_type: str) -> List[str]:
        """
        Extract keywords from checklist item for validation matching.
        
        Returns lowercase keywords that should appear in proposal if item is addressed.
        """
        # Common stopwords to filter
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "your",
            "you", "i", "we", "they", "it", "this", "that", "these",
            "those", "with", "for", "and", "or", "but", "in", "on",
            "at", "to", "from", "by", "about", "please", "share",
            "provide", "describe", "explain", "tell", "us", "me"
        }
        
        # Extract meaningful keywords
        words = text.lower().split()
        keywords = [w.strip(".,?!:;()[]{}") for w in words if len(w) > 2]
        keywords = [w for w in keywords if w not in stopwords]
        
        # Add type-specific keywords
        if item_type == "portfolio_links":
            keywords.extend(["portfolio", "example", "project", "link", "url", "site"])
        elif item_type == "time_estimate_phased":
            keywords.extend(["day", "week", "phase", "timeline", "estimate"])
        elif item_type == "experience_question":
            keywords.extend(["experience", "built", "developed", "worked"])
        
        return keywords[:10]  # Limit to most relevant

    def _score_proposal(
        self,
        proposal_text: str,
        portfolio_links: List[str],
        feedback_urls: List[str],
        similar_projects: List[Dict[str, Any]],
        request: ProposalRequest
    ) -> tuple:
        """Score proposal quality and generate suggestions."""
        score = 0.5  # Base score
        suggestions = []
        
        # Check for portfolio references
        if portfolio_links and any(link in proposal_text for link in portfolio_links):
            score += 0.15
        elif request.include_portfolio:
            suggestions.append("Consider adding portfolio links to strengthen credibility")
        
        # Check for specific project references
        for project in similar_projects:
            if project.get("company", "").lower() in proposal_text.lower():
                score += 0.1
                break
        
        # Check word count adherence
        word_count = len(proposal_text.split())
        if abs(word_count - request.max_word_count) < request.max_word_count * 0.2:
            score += 0.1
        else:
            suggestions.append(f"Adjust length closer to {request.max_word_count} words")
        
        # Check for personalization
        if request.company_name.lower() in proposal_text.lower():
            score += 0.1
        else:
            suggestions.append("Add more company-specific personalization")
        
        # Check for skill mentions
        skills_mentioned = sum(1 for skill in request.skills_required if skill.lower() in proposal_text.lower())
        if skills_mentioned >= len(request.skills_required) * 0.5:
            score += 0.05
        
        return min(score, 1.0), suggestions if suggestions else None
    
    def save_generated_proposal(
        self,
        proposal_text: str,
        job_data: Dict[str, Any],
        word_count: int,
        style: str,
        tone: str
    ) -> Optional[str]:
        """Save generated proposal to database."""
        if not self.db:
            return None
        
        try:
            from app.infra.mongodb.repositories import get_proposal_repo
            repo = get_proposal_repo()
            
            return repo.save_proposal({
                "job_title": job_data.get("job_title"),
                "company_name": job_data.get("company_name"),
                "generated_proposal": proposal_text,
                "word_count": word_count,
                "proposal_style": style,
                "proposal_tone": tone,
                "job_data": job_data
            })
        except Exception as e:
            logger.error(f"Error saving proposal: {e}")
            return None

    def generate_proposal_stream(self, request: ProposalRequest) -> Generator[str, None, None]:
        """
        Generate proposal with streaming progress events (SSE format).
        
        Yields SSE-formatted strings for real-time progress updates.
        
        Args:
            request: ProposalRequest with job details
            
        Yields:
            SSE event strings: "event: {type}\ndata: {json}\n\n"
        """
        def send_event(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"
        
        try:
            # Step 1: Understanding job
            yield send_event("step", {"step": 1, "message": "Understanding job requirements...", "icon": "brain"})
            
            job_data = self._prepare_job_data(request)
            all_jobs = self._get_historical_jobs(request.org_id)
            
            yield send_event("step", {"step": 1, "message": f"Found {len(all_jobs)} historical projects", "icon": "check", "done": True})
            
            # Step 2: Extract requirements + Find similar projects
            yield send_event("step", {"step": 2, "message": "Analyzing client needs + finding matches...", "icon": "search"})
            
            job_requirements = self._extract_job_requirements(request)
            requirements_context = None
            if job_requirements and job_requirements.extraction_confidence > 0:
                requirements_context = self.job_requirements_service.get_prompt_context(job_requirements)
                job_data = self._enhance_job_data_with_requirements(job_data, job_requirements)
            
            retrieval_result = self._find_similar_projects(
                job_data, all_jobs, request.similar_projects_count, job_requirements=job_requirements
            )
            similar_projects = self._filter_projects_with_portfolio(retrieval_result.get("similar_projects", []))
            
            yield send_event("step", {"step": 2, "message": f"Matched {len(similar_projects)} similar projects", "icon": "check", "done": True})
            
            # Step 3: Collecting context
            yield send_event("step", {"step": 3, "message": "Collecting portfolio & feedback...", "icon": "folder"})
            
            portfolio_links, feedback_urls = self._collect_urls(
                similar_projects[:request.similar_projects_count],
                request.include_portfolio,
                request.include_feedback
            )
            
            profile_context = None
            if request.profile_id and request.org_id:
                profile_context = self._get_freelancer_profile(request.org_id, request.profile_id)
            
            yield send_event("step", {"step": 3, "message": f"Gathered {len(portfolio_links)} portfolio links", "icon": "check", "done": True})
            
            # Step 4: Building prompt
            yield send_event("step", {"step": 4, "message": "Building personalized prompt...", "icon": "edit"})
            
            previous_insights = None
            if request.include_previous_proposals and similar_projects:
                previous_insights = self._analyze_previous_proposals(similar_projects[:request.similar_projects_count])
            
            prompt = self._build_prompt(
                job_data=job_data,
                similar_projects=similar_projects[:request.similar_projects_count],
                previous_insights=previous_insights,
                portfolio_links=portfolio_links,
                feedback_urls=feedback_urls,
                request=request,
                profile_context=profile_context,
                job_requirements=job_requirements
            )
            
            yield send_event("step", {"step": 4, "message": "Prompt optimized", "icon": "check", "done": True})
            
            # Step 5: Generate proposal with STREAMING
            yield send_event("step", {"step": 5, "message": "Writing proposal...", "icon": "sparkles"})
            
            max_tokens = int(request.max_word_count * 1.5) + 100
            proposal_chunks = []
            
            # Use streaming from OpenAI
            stream = self.openai_service.client.chat.completions.create(
                model=self.openai_service.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    proposal_chunks.append(text)
                    yield send_event("token", {"text": text})
            
            proposal_text = "".join(proposal_chunks)
            word_count = len(proposal_text.split())
            
            yield send_event("step", {"step": 5, "message": f"Generated {word_count} words", "icon": "check", "done": True})
            
            # Step 6: Scoring
            yield send_event("step", {"step": 6, "message": "Scoring quality...", "icon": "chart"})
            
            confidence_score, suggestions = self._score_proposal(
                proposal_text, portfolio_links, feedback_urls, similar_projects, request
            )
            
            yield send_event("step", {"step": 6, "message": "Complete!", "icon": "check", "done": True})
            
            # Final result
            yield send_event("complete", {
                "success": True,
                "generated_proposal": proposal_text,
                "word_count": word_count,
                "confidence_score": confidence_score,
                "similar_projects": similar_projects[:request.similar_projects_count],
                "portfolio_links_used": portfolio_links,
                "feedback_urls_used": feedback_urls,
                "improvement_suggestions": suggestions
            })
            
        except Exception as e:
            logger.error(f"[ProposalService] Streaming error: {e}", exc_info=True)
            yield send_event("error", {"message": str(e)})


# ===================== SERVICE FACTORY =====================

_proposal_service_instance: Optional[ProposalService] = None


def get_proposal_service(
    db_manager=None,
    openai_service=None,
    retrieval_pipeline=None,
    prompt_engine=None,
    job_requirements_service=None
) -> ProposalService:
    """
    Get or create singleton ProposalService instance.
    
    Args:
        db_manager: Database manager
        openai_service: OpenAI service
        retrieval_pipeline: Retrieval pipeline
        prompt_engine: Prompt engine
        job_requirements_service: Job requirements service
        
    Returns:
        ProposalService instance
    """
    global _proposal_service_instance
    
    if _proposal_service_instance is None:
        _proposal_service_instance = ProposalService(
            db_manager=db_manager,
            openai_service=openai_service,
            retrieval_pipeline=retrieval_pipeline,
            prompt_engine=prompt_engine,
            job_requirements_service=job_requirements_service
        )
    else:
        # Update dependencies if provided
        if db_manager is not None:
            _proposal_service_instance.db = db_manager
        if openai_service is not None:
            _proposal_service_instance.openai_service = openai_service
        if retrieval_pipeline is not None:
            _proposal_service_instance.retrieval_pipeline = retrieval_pipeline
        if prompt_engine is not None:
            _proposal_service_instance.prompt_engine = prompt_engine
        if job_requirements_service is not None:
            _proposal_service_instance.job_requirements_service = job_requirements_service
    
    return _proposal_service_instance
"""
Hook Strategy System for Winning Proposals

This module creates VARIED, COMPELLING opening hooks that force clients to click.
On Upwork, only ~2.5 lines are visible in preview - making the hook critical.

Key Principles (learned from 20+ winning proposals):
1. DON'T always start with "I see you're dealing with..."
2. Mirror the client's TONE and ENERGY from their job post
3. Lead with VALUE or CURIOSITY, not self-introduction
4. Reference something SPECIFIC from their post (shows you read it)
5. Create urgency or intrigue in the first line

Hook Categories (based on job sentiment analysis):
- URGENCY: Client sounds stressed, needs fast help → Lead with availability/speed
- TECHNICAL: Specific tech challenge → Lead with technical insight/solution hint
- FRUSTRATED: Bad past experience → Lead with empathy + differentiation  
- CURIOUS: Exploring options → Lead with unique value proposition
- BUSINESS_IMPACT: Revenue/conversion focus → Lead with ROI/results
- COLLABORATIVE: Looking for partner → Lead with questions showing interest
"""

import logging
import random
import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

# Import centralized constants
from app.domain.constants import URGENCY_PATTERNS, AI_ML_KEYWORDS

# Import consolidated text analysis utilities
from app.utils.text_analysis import (
    detect_urgency_score,
    extract_pain_points_simple,
    extract_specific_details as _extract_specific_details,
    extract_tone_words as _extract_tone_words,
)

logger = logging.getLogger(__name__)


class JobSentiment(str, Enum):
    """Detected sentiment/mood from job description"""
    URGENT = "urgent"               # Needs it NOW, stressed, time pressure
    FRUSTRATED = "frustrated"       # Bad past experience, something broken
    TECHNICAL = "technical"         # Detailed technical requirements, specs-focused
    CURIOUS = "curious"             # Exploring options, not urgent
    BUSINESS_FOCUS = "business"     # Revenue, conversions, ROI focused
    COLLABORATIVE = "collaborative" # Looking for ongoing partner
    CASUAL = "casual"               # Relaxed, simple request
    PROFESSIONAL = "professional"   # Formal, enterprise-level
    AI_TECHNICAL = "ai_technical"   # AI/ML focused, wants technical depth
    AI_BUILDER = "ai_builder"       # Wants someone who has BUILT AI systems


class JobIntent(str, Enum):
    """What the client actually wants done"""
    FIX_PROBLEM = "fix"            # Something is broken/not working
    BUILD_NEW = "build"            # Create something from scratch
    IMPROVE = "improve"            # Enhance existing (speed, design, features)
    MIGRATE = "migrate"            # Move from one platform/system to another
    MAINTAIN = "maintain"          # Ongoing support/maintenance
    CONSULT = "consult"            # Need advice/strategy, not just execution


@dataclass
class JobAnalysis:
    """Complete analysis of a job posting"""
    sentiment: JobSentiment
    intent: JobIntent
    urgency_level: int  # 1-5 (5 = critical)
    pain_points: List[str]
    specific_details: List[str]  # Numbers, tools, deadlines mentioned
    tone_words: List[str]  # Key emotional words from the post
    platform: str  # WordPress, Shopify, etc.
    task_type: str  # speed optimization, bug fix, etc.
    hook_strategy: str  # Recommended hook approach


class HookStrategyEngine:
    """
    Generates varied, compelling hooks based on job analysis.
    
    The goal: Create hooks so good that clients MUST click to read more.
    Only ~2.5 lines visible on Upwork = every word matters.
    """

    # ============== WINNING PATTERNS FROM REAL PROPOSALS ==============
    # These are actual opening patterns that won jobs
    
    WINNING_HOOK_PATTERNS = {
        "direct_solution": [
            "I know exactly why {problem} is happening and how to fix it.",
            "{problem}? I've fixed this exact issue {count}+ times.",
            "That {problem} you're seeing - there's a 90% chance it's {likely_cause}.",
            "Good news: {problem} is usually a quick fix. Here's why...",
        ],
        "immediate_value": [
            "I built something nearly identical - {portfolio_with_stats}",
            "I have built this for another client: {portfolio_with_stats}",
            "Here's a live example of exactly what you need: {portfolio_with_stats}",
            "Quick preview of what I can do for you: {portfolio_with_stats}",
        ],
        "empathy_first": [
            "I know how frustrating {problem} can be when {consequence}.",
            "Sounds like your last developer left you in a tough spot.",
            "That's a stressful situation - let me help you fix it fast.",
            "I get it - {problem} is the worst, especially when {context}.",
        ],
        "question_hook": [
            "Quick question: Is the {issue} affecting {impact}?",
            "Curious - have you checked if {technical_detail}?",
            "When did the {problem} start? That might tell us a lot.",
            "Is this happening on all {context} or just specific ones?",
        ],
        "result_lead": [
            "Got a similar store from {before} to {after} last month - {portfolio_url}",
            "Took another client's {metric} from {before} to {after}: {portfolio_with_stats}",
            "My last {platform} project: {outcome_stats} - {portfolio_url}",
            "{metric_improvement} on a similar project: {portfolio_url}",
        ],
        "availability_lead": [
            "I can start right now - this shouldn't take more than {timeframe}.",
            "I have built this - {portfolio_with_stats}",
            "At my desk and ready to dive in immediately.",
            "I've got time blocked for urgent projects like this today.",
        ],
        "specific_insight": [
            "Looking at your {context}, the issue is likely {likely_cause}.",
            "Based on what you described, I'd start by checking {technical_detail}.",
            "That {tool} + {issue} combination usually means {insight}.",
            "Interesting challenge - most {platform} sites have this because of {reason}.",
        ],
        "social_proof": [
            "I have built this multiple times with 5-star feedback.",
            "Been doing exactly this for {timeframe} - dozens of happy clients.",
            "This is literally 80% of my work. Here's my latest: {portfolio_with_stats}",
            "I specialize in {task_type} - it's what I do best.",
        ],
        # ============== AI/ML SPECIFIC HOOKS (PROVEN WINNERS) ==============
        "ai_capability_claim": [
            "I Built an AI System That {ai_result}—Now I'm Ready to Build Yours.",
            "I've already built exactly this: {ai_system_name}. It's in production right now.",
            "Your {domain} documents become {output} instantly. No manual work. I've already architected exactly this for production clients.",
            "I built a system that {ai_capability} in {time_saved}. Same approach works perfectly for your {domain}.",
        ],
        "ai_proof_stack": [
            "What I've Actually Built:\n{ai_project_proof}",
            "Here's what I've shipped to production:\n{ai_project_proof}",
            "My AI systems are live right now:\n{ai_project_proof}",
            "This is my specialty. Here's proof:\n{ai_project_proof}",
        ],
        "ai_tech_mirror": [
            "Your stack (LangChain + {llm_provider} + FastAPI) is exactly what I use daily.",
            "{tech_stack} — I've built production systems with this exact stack.",
            "RAG + {document_type} parsing + {output_type} generation? I shipped this last month.",
            "LangChain, Pinecone, {llm_provider}—this is literally my daily toolkit.",
        ],
        "ai_transformation": [
            "{time_before} → {time_after} per {task}. That's what my AI system delivers.",
            "Cuts {task} from {time_before} → {time_after}. I've built this exact pipeline.",
            "Your {manual_process} becomes {automated_process}. I've done this for {similar_domain}.",
            "{painful_process} becomes {simple_process} with the system I'll build you.",
        ],
        "ai_specificity": [
            "PDF parsing + OCR + AI interpretation? I built an Adaptive OCR Engine that handles exactly this.",
            "Document → structured data → AI generation → formatted output. This is my exact production pipeline.",
            "Messy PDFs, scanned documents, handwritten notes—my system handles all of it.",
            "Text extraction + LLM interpretation + cost calculation + PDF generation. I've built each piece.",
        ],
    }

    # Sentiment-based hook strategy selection
    SENTIMENT_STRATEGIES = {
        JobSentiment.URGENT: ["availability_lead", "direct_solution", "immediate_value"],
        JobSentiment.FRUSTRATED: ["empathy_first", "direct_solution", "social_proof"],
        JobSentiment.TECHNICAL: ["specific_insight", "question_hook", "result_lead"],
        JobSentiment.CURIOUS: ["immediate_value", "result_lead", "question_hook"],
        JobSentiment.BUSINESS_FOCUS: ["result_lead", "immediate_value", "social_proof"],
        JobSentiment.COLLABORATIVE: ["question_hook", "specific_insight", "empathy_first"],
        JobSentiment.CASUAL: ["immediate_value", "availability_lead", "social_proof"],
        JobSentiment.PROFESSIONAL: ["result_lead", "specific_insight", "social_proof"],
        # AI/ML specific strategies - these WORK for AI jobs
        JobSentiment.AI_TECHNICAL: ["ai_tech_mirror", "ai_specificity", "ai_proof_stack"],
        JobSentiment.AI_BUILDER: ["ai_capability_claim", "ai_proof_stack", "ai_transformation"],
    }
    
    # NOTE: URGENCY_PATTERNS and AI_ML_KEYWORDS moved to app/domain/constants.py

    # Sentiment detection patterns (uses JobSentiment enum - kept here)
    SENTIMENT_PATTERNS = {
        JobSentiment.URGENT: ["urgent", "asap", "immediately", "emergency", "critical", "today", "now", "fast", "quickly", "deadline"],
        JobSentiment.FRUSTRATED: ["frustrated", "struggling", "issues", "problems", "doesn't work", "broken", "nightmare", "headache", "last developer", "went mia", "ghosted", "tried before"],
        JobSentiment.TECHNICAL: ["architecture", "api", "database", "algorithm", "optimize", "performance", "stack", "server", "deploy", "integration", "custom code"],
        JobSentiment.BUSINESS_FOCUS: ["revenue", "conversions", "sales", "roi", "customers", "business", "growth", "profit", "traffic"],
        JobSentiment.COLLABORATIVE: ["partner", "long-term", "ongoing", "team", "collaborate", "work together", "relationship"],
        JobSentiment.CASUAL: ["simple", "easy", "quick", "small", "minor", "straightforward", "basic"],
        # AI/ML specific sentiment detection - CRITICAL for AI jobs
        JobSentiment.AI_TECHNICAL: [
            "langchain", "llamaindex", "rag", "retrieval", "embedding", "vector", "pinecone", "chromadb",
            "llm", "gpt", "openai", "anthropic", "claude", "prompt engineering", "fine-tuning",
            "ocr", "document processing", "pdf parsing", "text extraction", "nlp",
        ],
        JobSentiment.AI_BUILDER: [
            "ai tool", "ai system", "ai platform", "build ai", "create ai", "develop ai",
            "automation", "automate", "generate", "generator", "ai-powered", "machine learning",
            "content generation", "proposal generator", "document analyzer", "intelligent",
            "similar projects", "proven experience", "portfolio", "built before",
        ],
    }
    
    # NOTE: AI_ML_KEYWORDS moved to app/domain/constants.py - imported at module level

    def __init__(self):
        """Initialize hook strategy engine"""
        logger.info("HookStrategyEngine initialized")

    def analyze_job(self, job_data: Dict[str, Any]) -> JobAnalysis:
        """
        Deeply analyze a job posting to understand client's true needs.
        
        Args:
            job_data: Job data dictionary with job_description, job_title, etc.
            
        Returns:
            JobAnalysis with sentiment, intent, and recommended hook strategy
        """
        job_desc = job_data.get("job_description", "").lower()
        job_title = job_data.get("job_title", "").lower()
        combined_text = f"{job_title} {job_desc}"
        
        # Detect sentiment
        sentiment = self._detect_sentiment(combined_text)
        
        # Detect intent
        intent = self._detect_intent(combined_text)
        
        # Detect urgency level (1-5)
        urgency = self._detect_urgency(combined_text)
        
        # Extract pain points - prefer semantic extraction from job_requirements_service
        # client_problems is added by proposal_service.py from problems_mentioned
        if job_data.get("client_problems"):
            pain_points = job_data["client_problems"]
            logger.debug(f"[JobAnalysis] Using semantic pain points: {pain_points[:2]}")
        else:
            pain_points = self._extract_pain_points(combined_text)
            logger.debug(f"[JobAnalysis] Using rule-based pain points: {pain_points[:2]}")
        
        # Extract specific details (numbers, tools, etc.)
        specific_details = self._extract_specific_details(job_data)
        
        # Get tone words
        tone_words = self._extract_tone_words(combined_text)
        
        # Detect platform
        platform = self._detect_platform(combined_text, job_data.get("skills_required", []))
        
        # Get task type
        task_type = job_data.get("task_type", self._detect_task_type(combined_text))
        
        # Determine best hook strategy
        hook_strategy = self._select_hook_strategy(sentiment, urgency)
        
        analysis = JobAnalysis(
            sentiment=sentiment,
            intent=intent,
            urgency_level=urgency,
            pain_points=pain_points,
            specific_details=specific_details,
            tone_words=tone_words,
            platform=platform,
            task_type=task_type,
            hook_strategy=hook_strategy
        )
        
        logger.info(f"[JobAnalysis] Sentiment: {sentiment.value}, Intent: {intent.value}, "
                   f"Urgency: {urgency}/5, Strategy: {hook_strategy}")
        
        return analysis

    def generate_hook(
        self,
        job_analysis: JobAnalysis,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        portfolio_url: Optional[str] = None
    ) -> str:
        """
        Generate a compelling, varied hook based on job analysis.
        
        Args:
            job_analysis: Analysis results from analyze_job()
            job_data: Original job data
            similar_projects: Similar past projects for references
            portfolio_url: Best portfolio URL to include
            
        Returns:
            Compelling hook string (1-3 sentences)
        """
        strategy = job_analysis.hook_strategy
        templates = self.WINNING_HOOK_PATTERNS.get(strategy, self.WINNING_HOOK_PATTERNS["direct_solution"])
        
        # Select template randomly for variety
        template = random.choice(templates)
        
        # Build context for template filling
        context = self._build_hook_context(job_analysis, job_data, similar_projects, portfolio_url)
        
        # Fill in the template
        try:
            hook = self._fill_template(template, context)
        except Exception as e:
            logger.warning(f"Template fill failed, using fallback: {e}")
            hook = self._generate_fallback_hook(job_analysis, context)
        
        # Add portfolio link if we have one and it's not already in hook
        if portfolio_url and portfolio_url not in hook and "{portfolio_url}" not in template:
            hook = f"{hook}\n\nCheck out a similar project: {portfolio_url}"
        
        return hook

    def get_hook_variations(
        self,
        job_analysis: JobAnalysis,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        portfolio_url: Optional[str] = None,
        count: int = 3
    ) -> List[str]:
        """
        Generate multiple hook variations to choose from.
        
        Returns:
            List of different hook options
        """
        hooks = []
        used_strategies = set()
        
        # Get primary strategy and alternates
        primary_strategy = job_analysis.hook_strategy
        alternate_strategies = self.SENTIMENT_STRATEGIES.get(
            job_analysis.sentiment, 
            ["direct_solution", "immediate_value", "empathy_first"]
        )
        
        all_strategies = [primary_strategy] + [s for s in alternate_strategies if s != primary_strategy]
        
        context = self._build_hook_context(job_analysis, job_data, similar_projects, portfolio_url)
        
        # If no portfolio URL, prioritize strategies that don't require it
        if not portfolio_url:
            no_url_strategies = ["direct_solution", "empathy_first", "question_hook", "availability_lead", "specific_insight"]
            all_strategies = [s for s in all_strategies if s in no_url_strategies] + \
                            [s for s in all_strategies if s not in no_url_strategies]
        
        for strategy in all_strategies[:count + 2]:  # Try a few extra to get enough valid ones
            if len(hooks) >= count:
                break
            if strategy in used_strategies:
                continue
            used_strategies.add(strategy)
            
            templates = self.WINNING_HOOK_PATTERNS.get(strategy, self.WINNING_HOOK_PATTERNS["direct_solution"])
            
            # Filter out templates that need portfolio_url or portfolio_with_stats if we don't have one
            if not portfolio_url:
                templates = [t for t in templates if "{portfolio_url}" not in t and "{portfolio_with_stats}" not in t]
                if not templates:
                    continue
            
            template = random.choice(templates)
            
            try:
                hook = self._fill_template(template, context)
                # Skip hooks that still have unfilled placeholders or end with empty content
                if "{" not in hook and not hook.endswith(": ") and hook.strip():
                    hooks.append(hook)
            except:
                continue
        
        return hooks if hooks else [self._generate_fallback_hook(job_analysis, context)]

    def _detect_sentiment(self, text: str) -> JobSentiment:
        """Detect the primary sentiment/mood from job text"""
        sentiment_scores = {}
        
        # CRITICAL: Check for AI/ML job FIRST - these take precedence
        ai_score = sum(1 for kw in AI_ML_KEYWORDS if kw in text)
        if ai_score >= 3:  # Strong AI signal
            # Determine if it's technical-focused or builder-focused
            builder_words = ["build", "create", "develop", "tool", "system", "platform", "generator", "automate"]
            builder_score = sum(1 for w in builder_words if w in text)
            if builder_score >= 2:
                return JobSentiment.AI_BUILDER
            return JobSentiment.AI_TECHNICAL
        
        for sentiment, patterns in self.SENTIMENT_PATTERNS.items():
            score = sum(1 for p in patterns if p in text)
            if score > 0:
                sentiment_scores[sentiment] = score
        
        if not sentiment_scores:
            return JobSentiment.PROFESSIONAL  # Default
        
        # Return highest scoring sentiment
        return max(sentiment_scores, key=sentiment_scores.get)

    def _detect_intent(self, text: str) -> JobIntent:
        """Detect what the client actually wants done"""
        intent_patterns = {
            JobIntent.FIX_PROBLEM: ["fix", "broken", "not working", "error", "issue", "problem", "bug", "repair"],
            JobIntent.BUILD_NEW: ["build", "create", "new", "develop", "make", "from scratch", "design"],
            JobIntent.IMPROVE: ["improve", "optimize", "enhance", "upgrade", "better", "faster", "increase"],
            JobIntent.MIGRATE: ["migrate", "transfer", "move", "import", "export", "switch", "convert"],
            JobIntent.MAINTAIN: ["maintain", "support", "ongoing", "regular", "manage", "update"],
            JobIntent.CONSULT: ["advise", "consult", "strategy", "review", "audit", "recommend", "assess"],
        }
        
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for p in patterns if p in text)
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return JobIntent.BUILD_NEW  # Default
        
        return max(intent_scores, key=intent_scores.get)

    def _detect_urgency(self, text: str) -> int:
        """
        Detect urgency level (1-5).
        
        Delegates to app/utils/text_analysis.detect_urgency_score()
        """
        return detect_urgency_score(text)

    def _extract_pain_points(self, text: str) -> List[str]:
        """
        Extract specific pain points mentioned.
        
        Delegates to app/utils/text_analysis.extract_pain_points_simple()
        """
        return extract_pain_points_simple(text)

    def _extract_specific_details(self, job_data: Dict[str, Any]) -> List[str]:
        """
        Extract specific numbers, tools, metrics from job.
        
        Delegates to app/utils/text_analysis.extract_specific_details()
        """
        return _extract_specific_details(job_data)

    def _extract_tone_words(self, text: str) -> List[str]:
        """
        Extract emotional/tone words.
        
        Delegates to app/utils/text_analysis.extract_tone_words()
        """
        result = _extract_tone_words(text)
        # Flatten the dict to a list for backward compatibility
        tone_words = []
        for category_words in result.values():
            tone_words.extend(category_words)
        return tone_words[:5]

    def _detect_platform(self, text: str, skills: List[str]) -> str:
        """Detect the primary platform/technology - AI/ML jobs take priority"""
        combined = f"{text} {' '.join(skills)}".lower()
        
        # AI/ML detection takes priority over platforms
        ai_keywords = [
            "langchain", "llamaindex", "llama-index", "rag", "retrieval augmented",
            "vector database", "pinecone", "weaviate", "chroma", "embeddings",
            "llm", "large language model", "gpt-4", "gpt-3", "claude", "gemini",
            "openai", "anthropic", "ai agent", "ai assistant", "chatbot",
            "natural language", "nlp", "machine learning", "ml model",
            "neural network", "deep learning", "transformer", "fine-tuning",
            "prompt engineering", "ai integration", "ai-powered", "ai tool"
        ]
        
        if any(kw in combined for kw in ai_keywords):
            return "ai_ml"
        
        platforms = {
            "wordpress": ["wordpress", "wp", "elementor", "divi", "wpbakery", "geo directory", "geodirectory"],
            "shopify": ["shopify", "liquid", "gempages", "shopify store"],
            "woocommerce": ["woocommerce", "woo commerce"],
            "wix": ["wix", "wix site", "wix website"],
            "webflow": ["webflow"],
            "squarespace": ["squarespace"],
            "custom": ["react", "node", "python", "django", "flask", "next.js"],
        }
        
        for platform, indicators in platforms.items():
            if any(ind in combined for ind in indicators):
                return platform
        
        return "general"

    def _detect_task_type(self, text: str) -> str:
        """Detect the type of task"""
        task_patterns = {
            "speed optimization": ["speed", "pagespeed", "core web vitals", "performance", "optimize", "slow", "loading"],
            "bug fixes": ["fix", "bug", "broken", "error", "issue", "not working", "repair"],
            "complete website": ["build", "create", "new website", "from scratch", "design"],
            "migration": ["migrate", "transfer", "move", "import", "convert"],
            "enhance functionality": ["add feature", "enhance", "integrate", "custom", "functionality"],
            "redesign": ["redesign", "refresh", "makeover", "new look", "modernize"],
        }
        
        for task_type, patterns in task_patterns.items():
            if any(p in text for p in patterns):
                return task_type
        
        return "general"

    def _select_hook_strategy(self, sentiment: JobSentiment, urgency: int) -> str:
        """Select the best hook strategy based on analysis"""
        # High urgency always leads with availability
        if urgency >= 4:
            return "availability_lead"
        
        # Use sentiment-based strategy selection
        strategies = self.SENTIMENT_STRATEGIES.get(sentiment, ["direct_solution"])
        return strategies[0]  # Return primary strategy for sentiment

    def _build_hook_context(
        self,
        job_analysis: JobAnalysis,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        portfolio_url: Optional[str]
    ) -> Dict[str, str]:
        """Build context dictionary for template filling"""
        job_desc = job_data.get("job_description", "")
        
        # Extract the main problem
        problem = self._extract_main_problem(job_desc, job_analysis)
        
        # Get consequence of problem
        consequence = self._infer_consequence(job_analysis)
        
        # Get likely cause for technical issues
        likely_cause = self._suggest_likely_cause(job_analysis, job_data)
        
        # Get metrics from similar projects (now includes outcome_stats)
        metrics = self._get_project_metrics(similar_projects)
        
        # Extract outcome stats for hooks
        outcome_stats = metrics.get("outcome_stats", "")
        
        # Build portfolio_with_stats: combines URL + stats for compelling proof
        portfolio_with_stats = portfolio_url or ""
        if portfolio_url and outcome_stats:
            portfolio_with_stats = f"{outcome_stats} - see here: {portfolio_url}"
        elif outcome_stats:
            portfolio_with_stats = outcome_stats
        
        context = {
            "problem": problem,
            "consequence": consequence,
            "likely_cause": likely_cause,
            "portfolio_url": portfolio_url or "",
            "outcome_stats": outcome_stats,  # NEW: Raw outcome stats
            "portfolio_with_stats": portfolio_with_stats,  # NEW: Stats + URL combined
            "platform": job_analysis.platform.title(),
            "task_type": job_analysis.task_type,
            "timeframe": self._estimate_timeframe(job_analysis),
            "count": str(random.randint(10, 50)),
            "before": metrics.get("before", "30s"),
            "after": metrics.get("after", "2s"),
            "metric": metrics.get("metric", "load time"),
            "metric_improvement": metrics.get("improvement", "90+ mobile score"),
            "issue": self._extract_specific_issue(job_analysis),
            "impact": self._extract_business_impact(job_data),
            "technical_detail": self._get_technical_detail(job_analysis),
            "context": job_analysis.specific_details[0] if job_analysis.specific_details else "your site",
            "insight": likely_cause,
            "tool": job_analysis.platform.title(),
            "reason": likely_cause,
        }
        
        # Add AI-specific context if this is an AI/ML job
        if job_analysis.sentiment in (JobSentiment.AI_TECHNICAL, JobSentiment.AI_BUILDER):
            ai_context = self._build_ai_context(job_analysis, job_data, similar_projects)
            context.update(ai_context)
        
        return context
    
    def _build_ai_context(
        self,
        job_analysis: JobAnalysis,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Build AI-specific context for AI/ML job hooks"""
        job_desc = job_data.get("job_description", "").lower()
        
        # Detect domain (plumbing, legal, medical, etc.)
        domain = self._detect_ai_domain(job_desc)
        
        # Detect document type being processed
        document_type = self._detect_document_type(job_desc)
        
        # Detect output type
        output_type = self._detect_output_type(job_desc)
        
        # Detect LLM provider mentioned
        llm_provider = self._detect_llm_provider(job_desc)
        
        # Detect tech stack
        tech_stack = self._extract_ai_tech_stack(job_desc)
        
        # Build AI project proof from similar projects
        ai_project_proof = self._build_ai_project_proof(similar_projects)
        
        # Detect what's being transformed
        manual_process = self._detect_manual_process(job_desc)
        automated_process = self._detect_automated_process(job_desc)
        
        return {
            "domain": domain,
            "document_type": document_type,
            "output_type": output_type,
            "output": f"professional {output_type}s",
            "llm_provider": llm_provider,
            "tech_stack": tech_stack,
            "ai_project_proof": ai_project_proof,
            "ai_system_name": "AI Proposal Generator",
            "ai_result": "Turns Documents Into Proposals in 10 Seconds",
            "ai_capability": "generates professional proposals from raw documents",
            "time_saved": "10 seconds",
            "time_before": "20+ minutes",
            "time_after": "10 seconds",
            "task": "proposal",
            "manual_process": manual_process,
            "automated_process": automated_process,
            "painful_process": f"Manual {manual_process}",
            "simple_process": f"instant {automated_process}",
            "similar_domain": domain,
        }
    
    def _detect_ai_domain(self, text: str) -> str:
        """Detect the business domain for AI application"""
        domains = {
            "plumbing": ["plumbing", "plumber", "pipe", "fixture", "drainage"],
            "construction": ["construction", "building", "contractor", "blueprint"],
            "legal": ["legal", "law", "contract", "attorney", "lawyer"],
            "medical": ["medical", "healthcare", "patient", "diagnosis"],
            "real estate": ["real estate", "property", "listing", "mortgage"],
            "finance": ["finance", "banking", "loan", "investment"],
            "insurance": ["insurance", "claim", "policy", "coverage"],
            "e-commerce": ["e-commerce", "product", "inventory", "orders"],
        }
        
        for domain, keywords in domains.items():
            if any(kw in text for kw in keywords):
                return domain
        return "documents"
    
    def _detect_document_type(self, text: str) -> str:
        """Detect the type of document being processed"""
        doc_types = {
            "PDF": ["pdf", "document"],
            "blueprints": ["blueprint", "plan", "drawing", "schematic"],
            "contracts": ["contract", "agreement", "legal document"],
            "invoices": ["invoice", "receipt", "billing"],
            "forms": ["form", "application", "questionnaire"],
        }
        
        for doc_type, keywords in doc_types.items():
            if any(kw in text for kw in keywords):
                return doc_type
        return "documents"
    
    def _detect_output_type(self, text: str) -> str:
        """Detect what output the AI should generate"""
        outputs = {
            "proposal": ["proposal", "estimate", "quote", "bid"],
            "report": ["report", "analysis", "summary"],
            "content": ["content", "article", "blog", "copy"],
            "response": ["response", "reply", "answer"],
        }
        
        for output, keywords in outputs.items():
            if any(kw in text for kw in keywords):
                return output
        return "output"
    
    def _detect_llm_provider(self, text: str) -> str:
        """Detect which LLM provider is mentioned"""
        providers = {
            "OpenAI": ["openai", "gpt-4", "gpt-3", "chatgpt"],
            "Anthropic": ["anthropic", "claude"],
            "Google": ["gemini", "palm", "bard"],
            "Grok": ["grok", "xai"],
        }
        
        for provider, keywords in providers.items():
            if any(kw in text for kw in keywords):
                return provider
        return "OpenAI"
    
    def _extract_ai_tech_stack(self, text: str) -> str:
        """Extract the AI tech stack mentioned"""
        techs = []
        tech_keywords = {
            "LangChain": ["langchain"],
            "LlamaIndex": ["llamaindex", "llama-index"],
            "FastAPI": ["fastapi"],
            "React": ["react"],
            "Pinecone": ["pinecone"],
            "OpenAI": ["openai", "gpt"],
        }
        
        for tech, keywords in tech_keywords.items():
            if any(kw in text for kw in keywords):
                techs.append(tech)
        
        return " + ".join(techs[:4]) if techs else "LangChain + OpenAI + FastAPI"
    
    def _build_ai_project_proof(self, similar_projects: List[Dict[str, Any]]) -> str:
        """Build proof text from similar AI projects"""
        if not similar_projects:
            return """• AI Proposal Generator — Generates proposals in 10 seconds using RAG + GPT-4o
• Adaptive OCR Engine — Reads messy PDFs, scanned docs, handwritten text
Tech: LangChain + Pinecone + FastAPI + MongoDB"""
        
        proof_lines = []
        for proj in similar_projects[:2]:
            title = proj.get("job_title", "AI Project")
            # Truncate title if too long
            if len(title) > 50:
                title = title[:47] + "..."
            
            skills = proj.get("skills_required", [])[:4]
            skills_str = ", ".join(skills) if skills else "AI/ML"
            
            proof_lines.append(f"• {title}\n  Tech: {skills_str}")
        
        return "\n".join(proof_lines)
    
    def _detect_manual_process(self, text: str) -> str:
        """Detect what manual process is being automated"""
        if "proposal" in text or "estimate" in text:
            return "proposal writing"
        if "extract" in text or "parse" in text:
            return "data extraction"
        if "content" in text or "article" in text:
            return "content creation"
        return "document processing"
    
    def _detect_automated_process(self, text: str) -> str:
        """Detect what automated process will replace manual work"""
        if "proposal" in text or "estimate" in text:
            return "AI-generated proposals"
        if "extract" in text or "parse" in text:
            return "automated extraction"
        if "content" in text or "article" in text:
            return "AI content generation"
        return "automated processing"

    def _extract_main_problem(self, job_desc: str, analysis: JobAnalysis) -> str:
        """Extract the main problem being described"""
        # First check specific details
        if analysis.specific_details:
            for detail in analysis.specific_details:
                if any(word in detail.lower() for word in ["slow", "seconds", "not working", "broken"]):
                    return detail
        
        # Check for common problem phrases
        problem_patterns = [
            r"(\w+\s+(?:is|are)\s+(?:not working|broken|slow|failing))",
            r"((?:slow|poor|bad)\s+\w+)",
            r"(need\s+to\s+(?:fix|repair|improve)\s+\w+)",
        ]
        
        for pattern in problem_patterns:
            match = re.search(pattern, job_desc, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback to task type
        return f"{analysis.task_type}" if analysis.task_type != "general" else "this issue"

    def _infer_consequence(self, analysis: JobAnalysis) -> str:
        """Infer the consequence of the problem"""
        consequences = {
            JobIntent.FIX_PROBLEM: "you're losing potential customers",
            JobIntent.IMPROVE: "it's affecting your conversions",
            JobIntent.BUILD_NEW: "you need this to launch",
            JobIntent.MIGRATE: "your content is stuck on the old platform",
            JobSentiment.URGENT: "every hour counts",
            JobSentiment.BUSINESS_FOCUS: "it's impacting your bottom line",
        }
        
        if analysis.sentiment == JobSentiment.URGENT:
            return consequences[JobSentiment.URGENT]
        
        return consequences.get(analysis.intent, "it needs to get fixed")

    def _suggest_likely_cause(self, analysis: JobAnalysis, job_data: Dict[str, Any]) -> str:
        """Suggest a likely cause for the issue"""
        causes = {
            "speed optimization": "unoptimized images or render-blocking JS",
            "bug fixes": "a plugin conflict or theme issue",
            "migration": "different data structures between platforms",
            "enhance functionality": "the current setup's limitations",
        }
        
        return causes.get(analysis.task_type, "something in the current setup")

    def _get_project_metrics(self, similar_projects: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract metrics from similar projects for social proof"""
        defaults = {
            "before": "30+ seconds",
            "after": "under 2s",
            "metric": "load time",
            "improvement": "90+ mobile score",
            "outcome_stats": "",  # Stats from outcome field
        }
        
        if not similar_projects:
            return defaults
        
        # FIRST PRIORITY: Extract outcome stats from outcome field (structured dict)
        for proj in similar_projects[:3]:
            outcome = proj.get("outcome")
            if isinstance(outcome, dict):
                stats = outcome.get("stats", "")
                if stats and stats.strip():
                    defaults["outcome_stats"] = stats
                    defaults["improvement"] = stats  # Also use as improvement
                    logger.debug(f"[HookStrategy] Found outcome stats: {stats[:50]}...")
                    break
            elif isinstance(outcome, str) and outcome.strip():
                defaults["outcome_stats"] = outcome
                defaults["improvement"] = outcome
                logger.debug(f"[HookStrategy] Found outcome string: {outcome[:50]}...")
                break
            # Fall back to legacy 'outcomes' field
            outcomes = proj.get("outcomes", "")
            if isinstance(outcomes, str) and outcomes.strip():
                defaults["outcome_stats"] = outcomes
                defaults["improvement"] = outcomes
                logger.debug(f"[HookStrategy] Found legacy outcomes: {outcomes[:50]}...")
                break
        
        # SECOND PRIORITY: Try to find real metrics from project feedback
        if not defaults["outcome_stats"]:
            for proj in similar_projects[:3]:
                feedback = proj.get("client_feedback_text", "")
                if feedback:
                    # Look for numbers in feedback
                    numbers = re.findall(r'\d+[\+]?(?:\s*%|\s*seconds?|\s*/\s*5)?', feedback)
                    if numbers:
                        defaults["improvement"] = numbers[0]
                        break
        
        return defaults

    def _estimate_timeframe(self, analysis: JobAnalysis) -> str:
        """Estimate timeframe based on task type"""
        timeframes = {
            "bug fixes": "a few hours",
            "speed optimization": "1-2 days",
            "enhance functionality": "2-3 days",
            "migration": "3-5 days",
            "complete website": "1-2 weeks",
            "redesign": "1-2 weeks",
        }
        return timeframes.get(analysis.task_type, "a few days")

    def _extract_specific_issue(self, analysis: JobAnalysis) -> str:
        """Extract a specific issue to reference"""
        if analysis.specific_details:
            return analysis.specific_details[0]
        return analysis.task_type

    def _extract_business_impact(self, job_data: Dict[str, Any]) -> str:
        """Extract business impact mentioned"""
        desc = job_data.get("job_description", "").lower()
        
        impacts = {
            "sales": "your sales numbers",
            "conversion": "your conversion rate",
            "customers": "customer experience",
            "revenue": "your revenue",
            "traffic": "your traffic",
        }
        
        for keyword, impact in impacts.items():
            if keyword in desc:
                return impact
        
        return "your business"

    def _get_technical_detail(self, analysis: JobAnalysis) -> str:
        """Get a technical detail to reference"""
        details = {
            "wordpress": "your theme's functions.php",
            "shopify": "your Liquid templates",
            "speed optimization": "your Core Web Vitals",
            "bug fixes": "your browser console",
        }
        
        return details.get(analysis.platform, details.get(analysis.task_type, "the code"))

    def _fill_template(self, template: str, context: Dict[str, str]) -> str:
        """Fill a template with context values"""
        result = template
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                # Handle empty values for URL/stats placeholders
                if not value and key in ['portfolio_url', 'feedback_url', 'portfolio_with_stats', 'outcome_stats']:
                    # Replace with a fallback or remove
                    if key == 'portfolio_with_stats':
                        result = result.replace(placeholder, context.get('portfolio_url', '[portfolio link]'))
                    elif key == 'outcome_stats':
                        result = result.replace(placeholder, "")
                    else:
                        result = result.replace(placeholder, "[portfolio link]")
                else:
                    result = result.replace(placeholder, str(value))
        return result

    def _generate_fallback_hook(self, analysis: JobAnalysis, context: Dict[str, str]) -> str:
        """Generate a safe fallback hook"""
        problem = context.get("problem", "this")
        platform = context.get("platform", "your site")
        
        fallbacks = [
            f"This is exactly the kind of {analysis.task_type} work I specialize in.",
            f"I've solved this exact {platform} issue multiple times.",
            f"Let me show you how I'd approach {problem}.",
        ]
        return random.choice(fallbacks)


# Create singleton instance
_hook_engine: Optional[HookStrategyEngine] = None


def get_hook_engine() -> HookStrategyEngine:
    """Get singleton hook engine instance"""
    global _hook_engine
    if _hook_engine is None:
        _hook_engine = HookStrategyEngine()
    return _hook_engine

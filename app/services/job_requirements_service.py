"""
Job Requirements Extractor Service

Extracts structured requirements from job descriptions BEFORE proposal generation.
Uses OpenAI function calling for reliable structured output.

Key Benefits:
1. Deep semantic understanding of client needs (not just keyword matching)
2. Explicit "do_not_assume" field prevents hallucination
3. Structured output enables better retrieval and generation
4. MongoDB caching avoids re-processing same jobs

Created: Issue #23 - Sprint 2
"""

import logging
import hashlib
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class JobRequirements:
    """Structured representation of extracted job requirements"""
    
    # Core task understanding
    exact_task: str = ""                          # What client specifically wants done
    specific_deliverables: List[str] = field(default_factory=list)  # Expected outputs
    
    # Context & constraints
    resources_provided: List[str] = field(default_factory=list)     # Figma, content, API docs, etc.
    constraints: List[str] = field(default_factory=list)            # Budget, timeline, tech restrictions
    tech_stack_mentioned: List[str] = field(default_factory=list)   # Explicit tech requirements
    
    # Links & references
    links_mentioned: List[str] = field(default_factory=list)        # Any URLs in posting
    
    # Sentiment & tone
    client_tone: str = "professional"             # urgent|casual|technical|frustrated|professional|exploratory
    
    # Pain points & problems
    problems_mentioned: List[str] = field(default_factory=list)     # Pain points client states
    
    # CRITICAL: Anti-hallucination guard
    do_not_assume: List[str] = field(default_factory=list)          # Things NOT mentioned - don't invent
    
    # Inferred context (for retrieval)
    inferred_industry: str = ""                   # Best-guess industry
    complexity_level: str = "medium"              # low|medium|high
    
    # ====== NEW FIELDS FOR BETTER INTENT CAPTURE (Sprint 3) ======
    
    # Working arrangement details
    working_arrangement: Dict[str, Any] = field(default_factory=dict)  # timezone, hours, ongoing/one-time
    
    # Application-specific requirements (what to include IN the proposal/application)
    application_requirements: List[str] = field(default_factory=list)  # Loom video, portfolio samples, etc.
    
    # Soft requirements (non-technical expectations)
    soft_requirements: List[str] = field(default_factory=list)  # responsiveness, communication style, reliability
    
    # Client priorities (ordered by importance)
    client_priorities: List[str] = field(default_factory=list)  # What matters MOST to client
    
    # Must NOT propose/mention (irrelevant things to avoid)
    must_not_propose: List[str] = field(default_factory=list)  # Things outside scope / unwanted suggestions
    
    # Key phrases to echo (client's exact words for resonance)
    key_phrases_to_echo: List[str] = field(default_factory=list)  # Words to use in proposal for rapport
    
    # ====== PHASE 1: Explicit Checklist & Source Tracking ======
    
    # Parsed checklist from "Please include" / "How to Apply" sections
    explicit_checklist: List[Dict[str, Any]] = field(default_factory=list)
    
    # Smart question injection (for vague/diagnostic jobs)
    smart_question: Optional[Dict[str, Any]] = None  # {ask: bool, question: str, reason: str}
    
    # Metadata
    extraction_confidence: float = 0.0            # How confident in extraction (0-1)
    extracted_at: str = ""                        # ISO timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRequirements":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class HookQuestion:
    """A suggested hook question for proposal opening."""
    question_text: str = ""        # The actual question
    category: str = "engagement"    # diagnostic|scope_clarification|empathy|engagement
    relevance_reason: str = ""      # Why this question is relevant to the job

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HookQuestion":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# LLM function definition for hook question generation
HOOK_QUESTIONS_FUNCTION = {
    "name": "generate_hook_questions",
    "description": "Generate 1-4 specific questions for a freelancer proposal. PRIORITY ORDER: 1) Access request (if they mention having resources but didn't share), 2) Technical diagnostic (show expertise), 3) Constraint clarification (if they hint at past problems), 4) Empathy (acknowledge frustration).",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_text": {
                            "type": "string",
                            "description": "A conversational question (1-2 sentences). Sound like a helpful colleague, not an interrogator. Combine value with the question."
                        },
                        "category": {
                            "type": "string",
                            "enum": ["access_request", "technical_diagnostic", "integration_specific", "constraint_clarification", "empathy", "scope_clarification"],
                            "description": "'access_request' = asking for URL/files/access (HIGHEST PRIORITY), 'technical_diagnostic' = showing expertise with a smart technical question, 'integration_specific' = asking about specific integrations/connectors/plugins, 'constraint_clarification' = understanding constraints like 'do not rebuild', 'empathy' = acknowledging past frustration, 'scope_clarification' = clarifying vague deliverables"
                        },
                        "relevance_reason": {
                            "type": "string",
                            "description": "Why this question matters (e.g., 'Client mentioned Moodle integration but specific connector is unclear')"
                        },
                        "trigger_detected": {
                            "type": "string",
                            "description": "What specific phrase/context triggered this question (e.g., 'do not rebuild LMS', 'moodle connection issues', 'site URL mentioned but no access')"
                        }
                    },
                    "required": ["question_text", "category", "relevance_reason", "trigger_detected"]
                },
                "minItems": 1,
                "maxItems": 4,
                "description": "1-4 questions in priority order. Generate MORE questions for complex jobs."
            }
        },
        "required": ["questions"]
    }
}

HOOK_QUESTIONS_SYSTEM_PROMPT = """You are an expert Upwork proposal strategist. Generate 1-4 specific questions that help the freelancer START A CONVERSATION with the client.

## QUESTION GENERATION TRIGGERS - ALWAYS check for these:

### TRIGGER 1: RESOURCES MENTIONED BUT NOT SHARED (HIGHEST PRIORITY)
If client says "I have X" but doesn't provide X → Ask for it!
- "I have example websites" + no links → Ask for the links
- "migrating from X to Y" + no access → Ask for access
- "site is slow/broken" + URL mentioned but no credentials → Ask for admin access
- "I have designs/Figma" + no link → Ask for Figma link
- Website URL mentioned like "nexacentre.com" but need access → Ask for staging/admin access

### TRIGGER 2: INTEGRATION/CONNECTOR TYPE IS UNCLEAR
If they mention a technology integration but don't specify which connector:
- "WordPress-Moodle connection" → "Which Moodle connector are you using—the official Moodle plugin, LearnDash integration, or a custom SSO setup?"
- "Zapier integration" → "Which Zapier triggers/actions are you using currently?"
- "API connection" → "Is this a REST API, GraphQL, or websocket connection?"
- "LMS integration" → "Which LMS are you connecting—Moodle, LearnDash, Tutor LMS, or another system?"
- "Payment gateway" → "Which payment provider—Stripe, PayPal, or another gateway?"

### TRIGGER 3: CONSTRAINT HINTS AT PAST PROBLEMS
If client says "do NOT do X" or expresses a strong constraint:
- "Do not rebuild the LMS" → "I noticed you mentioned not rebuilding the LMS—totally respect that. Was there a previous attempt, or is this more about budget/timeline? Understanding helps me scope accurately."
- "Last developer left/ghosted" → "I understand you had issues with a previous developer. What went wrong? I want to make sure we avoid those same problems."
- "Tried X before" → "You mentioned trying X before—what didn't work about that approach?"

### TRIGGER 4: DIAGNOSTIC QUESTIONS (show expertise)
For technical issues, ask SMART diagnostic questions:
- "Site is slow" → "Is the slowness mainly on mobile or initial page load? That narrows down whether it's images, JS, or server-side."
- "Connection issues" → "Are users getting specific error messages, or is it silently failing?"
- "Registration problems" → "Is the registration issue on the WordPress side, the LMS side, or during the sync between them?"
- "LMS usability issues" → "What specific usability pain points are users reporting? Navigation, course progress, or something else?"

### TRIGGER 5: EMPATHY ACKNOWLEDGMENT
If client expresses frustration or had bad experience:
- "Frustrated with..." → Acknowledge it before asking anything
- "Nightmare", "Horrible", "Disaster" → Show understanding first

## QUESTION TYPES (categories):
1. **access_request** - Ask for URL/files/access they mentioned but didn't share
2. **technical_diagnostic** - Smart technical question showing expertise  
3. **integration_specific** - Clarify which connector/plugin/integration type
4. **constraint_clarification** - Understand "do not" constraints (often hint at past problems)
5. **empathy** - Acknowledge frustration
6. **scope_clarification** - Clarify ambiguous deliverables

## RULES:
- Generate 2-4 questions for complex jobs (multiple integrations, technical issues)
- Generate 1-2 questions for simple jobs
- Questions must reference SPECIFIC things from the job post
- Sound like a helpful colleague, not an interrogator
- NEVER ask about budget or timeline
- Combine value with questions: "I've worked with Moodle integrations before—which connector are you using?"

## GOOD EXAMPLES BY TRIGGER:

ACCESS_REQUEST:
- "Could you share temporary admin access to nexacentre.com? I'd love to diagnose the Moodle sync issue before giving you a timeline."

INTEGRATION_SPECIFIC:  
- "Quick question: Which Moodle connector are you using—the official plugin or a custom SSO solution? This helps me pinpoint where the registration sync is breaking."

CONSTRAINT_CLARIFICATION:
- "I see you want to improve the LMS rather than rebuild it—smart approach. Was there a past rebuild attempt that didn't work out, or is this more about maintaining stability while fixing issues?"

TECHNICAL_DIAGNOSTIC:
- "When users try to register, are they getting a specific error message or is the process just silently failing? This tells me whether it's a frontend validation issue or a backend sync problem."

## BAD EXAMPLES (NEVER do these):
- "What's your timeline?" (they didn't ask for this)
- "Have you considered using X instead?" (unsolicited advice)
- "Can you share more details?" (vague, lazy)
- "What's your budget?" (NEVER ask this)
- Questions about things they already answered in the post"""


# OpenAI Function Definition for structured extraction
JOB_REQUIREMENTS_FUNCTION = {
    "name": "extract_job_requirements",
    "description": "Extract structured requirements from a job description. Be thorough and precise. Focus on capturing the client's TRUE INTENT including working arrangements, application requirements, and what matters most to them.",
    "parameters": {
        "type": "object",
        "properties": {
            "exact_task": {
                "type": "string",
                "description": "A clear, specific summary of what the client wants done INCLUDING the context (ongoing, one-time, part of team). Example: 'Join team as ongoing Shopify developer designing and building stores, available EST hours' NOT just 'Build Shopify websites'"
            },
            "specific_deliverables": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of concrete outputs the client expects. Examples: 'Migrated content with images', 'Working membership system', 'Admin documentation'"
            },
            "resources_provided": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Resources the client will provide: Figma files, content, API access, brand guidelines, etc. Extract ONLY what's explicitly mentioned."
            },
            "constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit constraints: budget hints ('tight budget'), timeline ('need this week'), tech restrictions ('must use WordPress'), platform requirements"
            },
            "tech_stack_mentioned": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific technologies, platforms, tools mentioned: WordPress, React, Shopify, specific plugins, APIs, etc."
            },
            "links_mentioned": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any URLs mentioned in the job posting - current site, reference sites, Figma links, docs"
            },
            "client_tone": {
                "type": "string",
                "enum": ["urgent", "casual", "technical", "frustrated", "professional", "exploratory"],
                "description": "The client's communication tone. 'urgent' = stressed/time pressure, 'frustrated' = had bad experiences, 'technical' = detailed specs, 'casual' = relaxed, 'professional' = formal/enterprise, 'exploratory' = just researching options"
            },
            "problems_mentioned": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Pain points and problems the client explicitly states: 'site is slow', 'previous developer disappeared', 'getting hacked', etc."
            },
            "do_not_assume": {
                "type": "array",
                "items": {"type": "string"},
                "description": "CRITICAL: Things NOT mentioned that a proposal should NOT assume or invent. If client doesn't mention budget, add 'specific budget'. If no timeline, add 'deadline'. If no tech preference, add 'technology choice'. This prevents hallucination."
            },
            "inferred_industry": {
                "type": "string",
                "description": "Best-guess industry based on context: e-commerce, saas, healthcare, media, education, finance, etc. Use 'general' if unclear."
            },
            "complexity_level": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "'low' = simple task (<1 day), 'medium' = typical project (days-weeks), 'high' = complex system (weeks-months)"
            },
            "working_arrangement": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Required timezone if specified (e.g., 'EST', 'PST', 'UTC'). Leave empty if not mentioned."},
                    "timezone_source": {"type": "string", "enum": ["explicit", "inferred", "none"], "description": "CRITICAL: 'explicit' ONLY if client wrote exact timezone (EST, PST, UTC, etc). 'inferred' if you guessed from context. 'none' if no timezone mentioned at all."},
                    "hours": {"type": "string", "description": "Required working hours if specified (e.g., '9 AM - 5 PM')"},
                    "arrangement_type": {"type": "string", "enum": ["ongoing", "one_time", "contract", "full_time", "part_time", "flexible"], "description": "Type of engagement"},
                    "overlap_required": {"type": "boolean", "description": "Whether timezone overlap is mandatory"},
                    "responsiveness_expectation": {"type": "string", "description": "Expected response time during work hours"}
                },
                "description": "Working arrangement details. CRITICAL: timezone_source must be 'explicit' ONLY if client literally wrote a timezone. Otherwise use 'none'."
            },
            "application_requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "CRITICAL: What client requires IN THE APPLICATION/PROPOSAL. Examples: 'Loom video', 'portfolio samples', 'answer specific questions', 'budget proposal'. Failing to include these = automatic rejection."
            },
            "soft_requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Non-technical expectations: 'responsive during work hours', 'clear communication', 'follow systems and deadlines', 'reliability', 'team player'. These signal what client values beyond technical skills."
            },
            "client_priorities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What matters MOST to the client, in order of importance. Extract from emphasis, repetition, and explicit priorities in the job post. Example: ['reliability', 'speed', 'quality', 'communication']"
            },
            "must_not_propose": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Things the proposal should NOT include or suggest. If client wants Shopify, don't suggest WordPress. If they have a process, don't suggest changing it. Avoid unsolicited advice outside scope."
            },
            "key_phrases_to_echo": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact phrases from the job post that the proposal should echo to show understanding. Examples: 'fast-moving team', 'design and development', 'ready for launch'"
            },
            "extraction_confidence": {
                "type": "number",
                "description": "Your confidence in this extraction from 0.0 to 1.0. Lower if job description is vague or ambiguous."
            },
            "explicit_checklist": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_text": {"type": "string", "description": "The exact requirement text from client"},
                        "item_type": {"type": "string", "enum": ["portfolio_links", "time_estimate_total", "time_estimate_phased", "experience_question", "preference_question", "approach_description", "specific_answer", "other"], "description": "Category of requirement"},
                        "quantity_requested": {"type": "integer", "description": "Number requested if specified (e.g., '2-3 examples' = 2)"},
                        "specificity_required": {"type": "boolean", "description": "True if client says 'be specific' or similar"},
                        "answer_hint": {"type": "string", "description": "What kind of answer is expected"}
                    }
                },
                "description": "CRITICAL: Parse numbered/bulleted requirements from 'Please include', 'How to Apply', 'In your proposal' sections. Each item becomes one checklist entry. This ensures we address ALL requirements."
            },
            "smart_question": {
                "type": "object",
                "properties": {
                    "ask": {"type": "boolean", "description": "True if asking a question would help. CRITICAL: For clear jobs, ONLY ask if client mentions specific assets (Figma, designs, docs) but doesn't share them."},
                    "question": {"type": "string", "description": "Natural conversational question. For mentioned-but-not-shared assets: 'Could you share the Figma file? I'd love to review the designs to give you an accurate timeline.' For diagnostics: 'Could you share the site URL so I can take a look?'"},
                    "reason": {"type": "string", "enum": ["needs_url", "needs_design_files", "vague_scope", "missing_details", "clarify_approach"], "description": "Why asking helps. Use 'needs_design_files' when client mentions having Figma/designs/mockups but doesn't share the link."}
                },
                "description": "SMART QUESTION: Ask when it genuinely helps. PRIORITY: If client mentions having assets (Figma, designs, docs, API specs) but doesn't share them → ask for the link. For clear jobs with all info provided → set ask: false. For vague jobs → ask clarifying question."
            }
        },
        "required": [
            "exact_task",
            "specific_deliverables", 
            "client_tone",
            "problems_mentioned",
            "do_not_assume",
            "inferred_industry",
            "complexity_level",
            "working_arrangement",
            "application_requirements",
            "soft_requirements",
            "client_priorities",
            "must_not_propose",
            "key_phrases_to_echo",
            "extraction_confidence",
            "explicit_checklist",
            "smart_question"
        ]
    }
}

# System prompt for extraction
EXTRACTION_SYSTEM_PROMPT = """You are an expert job requirements analyst for freelance proposals on Upwork. Your task is to deeply understand what a client REALLY wants from their job posting - not just the technical task, but the full context of who they want to work with and how.

CRITICAL RULES:
1. Extract ONLY what is explicitly stated or clearly implied
2. The 'do_not_assume' field is ESSENTIAL - list anything the client did NOT specify that a proposal writer might wrongly assume
3. Be SPECIFIC in 'exact_task' - include the working context (ongoing, team role, etc.), not just the technical task
4. 'problems_mentioned' should capture pain points in the client's own words
5. 'client_tone' should reflect their emotional state and communication style
6. If something is ambiguous, note it in 'do_not_assume'

CRITICAL NEW FIELDS - PAY ATTENTION:
7. 'working_arrangement' - ALWAYS extract timezone, hours, and engagement type if mentioned. Many jobs are rejected for timezone mismatch alone!
8. 'application_requirements' - CRITICAL: Extract what client REQUIRES in the application (Loom video, portfolio link, specific questions). Missing these = instant rejection!
9. 'soft_requirements' - Communication style, reliability, responsiveness preferences. Often MORE important than technical skills.
10. 'client_priorities' - ORDER MATTERS. What do they emphasize? What do they repeat? What do they put in CAPS or bold?
11. 'must_not_propose' - Don't suggest alternatives to what they asked for. Don't offer unsolicited advice. Stay in scope.
12. 'key_phrases_to_echo' - Use their exact words to show you read the post carefully.

COMMON EXTRACTION MISTAKES TO AVOID:
- Missing timezone/availability requirements (leads to wasted proposals)
- Missing application requirements like Loom videos (leads to instant rejection)
- Being too generic in exact_task (loses the context of WHO they want)
- Not capturing what matters MOST to the client (misaligned proposals)
- Suggesting technologies/approaches they didn't ask for (annoying to clients)

PHASE 1 CRITICAL ADDITIONS:

13. 'timezone_source' in working_arrangement:
   - Set to 'explicit' ONLY if client literally wrote: EST, PST, UTC, GMT, or specific timezone
   - Set to 'none' if no timezone is mentioned at all
   - Set to 'inferred' ONLY if you're guessing from context (rare - avoid this)
   - DEFAULT TO 'none' if unsure - this prevents hallucination

14. 'explicit_checklist' - PARSE EVERY NUMBERED/BULLETED REQUIREMENT:
   - Look for: "Please include", "How to Apply", "In your proposal", numbered lists
   - Parse quantity: "2-3 examples" → quantity_requested: 2
   - Parse specificity: "be specific", "please elaborate" → specificity_required: true
   - Determine type: portfolio links, time estimates, experience questions, etc.
   - This ensures the proposal addresses EVERY client requirement

EXAMPLE explicit_checklist parsing:
Input: "Please include: 1. 2-3 WordPress examples 2. Time estimate for each phase 3. Do you have RTL experience? Be specific"
Output: [
  {"item_text": "2-3 WordPress examples", "item_type": "portfolio_links", "quantity_requested": 2, "specificity_required": false, "answer_hint": "WordPress sites"},
  {"item_text": "Time estimate for each phase", "item_type": "time_estimate_phased", "quantity_requested": 0, "specificity_required": false, "answer_hint": "breakdown per phase"},
  {"item_text": "Do you have RTL experience? Be specific", "item_type": "experience_question", "quantity_requested": 0, "specificity_required": true, "answer_hint": "RTL/Hebrew projects"}
]

15. 'smart_question' - ASK A QUESTION ONLY WHEN IT GENUINELY HELPS:
   
   PRIORITY 1 - MENTIONED BUT NOT SHARED ASSETS:
   If client says "We have Figma designs" / "I have mockups" / "designs are ready" BUT doesn't share link:
   → ask: true, question: "Could you share the Figma file? I'd love to review the designs to give you an accurate timeline.", reason: "needs_design_files"
   This is ALWAYS worth asking - it's practical, not interrogating.
   
   PRIORITY 2 - DIAGNOSTIC/PERFORMANCE JOBS:
   - Speed optimization/performance jobs → need URL to diagnose
   - Debugging/fixing jobs → need access or more details
   → ask: true, question: "Could you share the URL? I'll take a look...", reason: "needs_url"
   
   PRIORITY 3 - VAGUE SCOPE:
   - Vague 1-2 sentence descriptions → scope is unclear
   - Missing critical info that affects approach
   → ask: true, reason: "vague_scope" or "missing_details"
   
   DON'T ASK when:
   - Job is clear AND all assets/resources are already shared
   - Simple installation/setup tasks with full details
   - Client provided everything needed
   
   Question style: Conversational, combine with value. Never interrogate.

Your extraction directly influences proposal quality - precision matters. A great extraction enables a great proposal that addresses EXACTLY what the client cares about."""


class JobRequirementsService:
    """
    Service for extracting structured requirements from job descriptions.
    
    Uses OpenAI function calling for reliable structured output.
    Caches results in MongoDB to avoid re-processing.
    """
    
    def __init__(self, openai_service=None, db_manager=None):
        """
        Initialize with dependencies.
        
        Args:
            openai_service: OpenAI service for extraction
            db_manager: Database manager for caching
        """
        self.openai_service = openai_service
        self.db = db_manager
        self._cache_collection = "job_requirements_cache"
        logger.info("JobRequirementsService initialized")
    
    def extract_job_requirements(
        self,
        job_description: str,
        job_title: str = "",
        skills_required: List[str] = None,
        force_refresh: bool = False
    ) -> JobRequirements:
        """
        Extract structured requirements from job description.
        
        Uses OpenAI function calling for semantic extraction.
        Results are cached in MongoDB by content hash.
        
        Args:
            job_description: Full job description text
            job_title: Job title (optional, improves extraction)
            skills_required: Required skills list (optional)
            force_refresh: If True, bypass cache and re-extract
            
        Returns:
            JobRequirements dataclass with structured data
        """
        if not job_description:
            logger.warning("Empty job description provided")
            return self._create_fallback_requirements(job_description, job_title)
        
        # Generate cache key from content
        cache_key = self._generate_cache_key(job_description, job_title)
        
        # Check cache first (unless force refresh)
        if not force_refresh and self.db:
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.info(f"[JobRequirements] Cache hit for hash {cache_key[:12]}...")
                return cached
        
        # Extract using OpenAI
        try:
            requirements = self._extract_with_openai(job_description, job_title, skills_required)
            
            # Cache the result
            if self.db and requirements.extraction_confidence > 0:
                self._save_to_cache(cache_key, requirements)
            
            return requirements
            
        except Exception as e:
            logger.error(f"[JobRequirements] Extraction failed: {e}")
            return self._create_fallback_requirements(job_description, job_title)
    
    def _extract_with_openai(
        self,
        job_description: str,
        job_title: str,
        skills_required: List[str] = None
    ) -> JobRequirements:
        """
        Extract requirements using OpenAI function calling.
        
        Args:
            job_description: Job description text
            job_title: Job title
            skills_required: Required skills
            
        Returns:
            JobRequirements with extracted data
        """
        if not self.openai_service:
            logger.warning("OpenAI service not available, using fallback")
            return self._create_fallback_requirements(job_description, job_title)
        
        # Build the extraction prompt
        skills_text = ", ".join(skills_required) if skills_required else "Not specified"
        user_prompt = f"""Analyze this job posting and extract structured requirements:

JOB TITLE: {job_title or 'Not provided'}

REQUIRED SKILLS: {skills_text}

JOB DESCRIPTION:
{job_description}

Extract all requirements using the provided function. Be thorough with 'do_not_assume' - this prevents the proposal from making up requirements the client never mentioned."""

        try:
            # Call OpenAI with function calling
            response = self.openai_service.client.chat.completions.create(
                model=self.openai_service.llm_model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[{"type": "function", "function": JOB_REQUIREMENTS_FUNCTION}],
                tool_choice={"type": "function", "function": {"name": "extract_job_requirements"}}
            )
            
            # Parse the function call response
            tool_call = response.choices[0].message.tool_calls[0]
            extracted_data = json.loads(tool_call.function.arguments)
            
            # Add timestamp
            extracted_data["extracted_at"] = datetime.utcnow().isoformat()
            
            # Create JobRequirements from extracted data
            requirements = JobRequirements.from_dict(extracted_data)
            
            logger.info(f"[JobRequirements] Extracted: task='{requirements.exact_task[:50]}...', "
                       f"tone={requirements.client_tone}, confidence={requirements.extraction_confidence:.2f}")
            
            # Log smart_question for debugging
            if requirements.smart_question:
                logger.info(f"[JobRequirements] Smart question: ask={requirements.smart_question.get('ask')}, "
                           f"reason={requirements.smart_question.get('reason')}")
            
            return requirements
            
        except Exception as e:
            logger.error(f"[JobRequirements] OpenAI extraction error: {e}")
            raise
    
    def _generate_cache_key(self, job_description: str, job_title: str) -> str:
        """Generate cache key from content hash."""
        content = f"{job_title}::{job_description}".lower().strip()
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[JobRequirements]:
        """Get cached requirements by key."""
        try:
            cached = self.db.db[self._cache_collection].find_one({"cache_key": cache_key})
            if cached:
                return JobRequirements.from_dict(cached.get("requirements", {}))
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, requirements: JobRequirements) -> None:
        """Save requirements to cache."""
        try:
            self.db.db[self._cache_collection].update_one(
                {"cache_key": cache_key},
                {
                    "$set": {
                        "cache_key": cache_key,
                        "requirements": requirements.to_dict(),
                        "updated_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            logger.debug(f"[JobRequirements] Cached with key {cache_key[:12]}...")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _create_fallback_requirements(
        self,
        job_description: str,
        job_title: str
    ) -> JobRequirements:
        """
        Create fallback requirements using simple heuristics.
        Used when OpenAI is unavailable or fails.
        """
        from app.utils.metadata_extractor import MetadataExtractor
        from app.utils.text_analysis import detect_urgency_level, extract_pain_points
        
        # Simple extraction
        intents = MetadataExtractor.extract_client_intents({
            "job_description": job_description,
            "job_title": job_title,
            "skills_required": []
        })
        
        # Basic pain point extraction
        pain_points = extract_pain_points(job_description)
        problems = []
        for category, points in pain_points.items():
            problems.extend(points[:2])
        
        # Urgency detection
        urgency = detect_urgency_level(job_description, job_title)
        tone_map = {
            "critical": "urgent",
            "high": "urgent", 
            "medium": "professional",
            "low": "casual"
        }
        
        return JobRequirements(
            exact_task=job_title or "Project requirements unclear",
            specific_deliverables=[],
            resources_provided=[],
            constraints=[],
            tech_stack_mentioned=[],
            links_mentioned=self._extract_urls(job_description),
            client_tone=tone_map.get(urgency, "professional"),
            problems_mentioned=problems[:3],
            do_not_assume=["specific budget", "exact timeline", "detailed specifications"],
            inferred_industry="general",
            complexity_level="medium",
            extraction_confidence=0.3,  # Low confidence for fallback
            extracted_at=datetime.utcnow().isoformat()
        )
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.findall(url_pattern, text)
    
    def generate_hook_questions(
        self,
        requirements: JobRequirements,
        job_description: str = "",
        job_title: str = ""
    ) -> List[HookQuestion]:
        """
        Generate 1-4 suggested hook questions based on extracted job requirements.

        Uses a focused LLM call with trigger detection to produce specific,
        compelling opening questions for the user to SELECT from.

        Enhanced triggers:
        - Access requests (URL/credentials mentioned but not shared)
        - Integration specifics (which connector/plugin)
        - Constraint clarification (why "do not X")
        - Technical diagnostics (smart questions showing expertise)
        - Empathy acknowledgment (past frustration)

        Args:
            requirements: Previously extracted JobRequirements
            job_description: Original job description (for additional context)
            job_title: Job title

        Returns:
            List of HookQuestion dataclasses (1-4 items)
        """
        if not self.openai_service:
            logger.warning("[HookQuestions] OpenAI not available, returning empty")
            return []

        # Build enhanced context with EXPLICIT trigger detection
        context_parts = []
        triggers_detected = []
        job_lower = job_description.lower()
        
        # Core task
        if requirements.exact_task:
            context_parts.append(f"Task: {requirements.exact_task}")
        
        # ============ TRIGGER 1: ACCESS/RESOURCES MENTIONED BUT NOT SHARED ============
        links_shared = requirements.links_mentioned or []
        mentioned_but_not_shared = []
        
        # Website URL mentioned but need access
        if links_shared:
            context_parts.append(f"URLs mentioned: {', '.join(links_shared[:3])}")
            # They have URL but likely need admin access to diagnose
            if any(prob in str(requirements.problems_mentioned).lower() for prob in ['issue', 'broken', 'slow', 'not working', 'connection', 'error']):
                triggers_detected.append("TRIGGER: Website URL provided but admin access likely needed to diagnose issues")
                mentioned_but_not_shared.append("admin/staging access (URL mentioned, but need access to diagnose)")
        else:
            context_parts.append("No URLs provided in job post")
        
        # Figma/design files mentioned but not shared
        if any(phrase in job_lower for phrase in ['figma', 'design file', 'mockup', 'psd', 'sketch', 'has designs', 'have designs']):
            if not any('figma' in link.lower() for link in links_shared):
                triggers_detected.append("TRIGGER: Design files mentioned but not shared")
                mentioned_but_not_shared.append("Figma/design files (mentioned but link not provided)")
        
        # Examples mentioned but not shared
        if any(phrase in job_lower for phrase in ['example website', 'example site', 'reference site', 'similar to', 'like this site']):
            if len(links_shared) < 2:
                triggers_detected.append("TRIGGER: Example/reference sites mentioned but not all shared")
                mentioned_but_not_shared.append("example/reference websites")
        
        if mentioned_but_not_shared:
            context_parts.append(f"⚠️ ACCESS_REQUEST TRIGGER: {'; '.join(mentioned_but_not_shared)}")
        
        # ============ TRIGGER 2: INTEGRATION TYPE UNCLEAR ============
        integration_triggers = []
        
        # Build exclusion context from must_not_propose (LLM already parsed negations)
        excluded_topics = ' '.join(requirements.must_not_propose).lower() if requirements.must_not_propose else ''
        
        # Moodle integration
        if 'moodle' in job_lower and 'moodle' not in excluded_topics:
            if not any(connector in job_lower for connector in ['learndash', 'tutor lms', 'official moodle plugin', 'sso']):
                triggers_detected.append("TRIGGER: Moodle mentioned but connector type unclear")
                integration_triggers.append("Moodle connector (which plugin/integration?)")
        
        # LMS mentioned generically
        if 'lms' in job_lower and 'moodle' not in job_lower and 'lms' not in excluded_topics:
            triggers_detected.append("TRIGGER: LMS mentioned but specific system unclear")
            integration_triggers.append("LMS system (which one?)")
        
        # Payment gateway - skip if payment/store/commerce explicitly excluded
        payment_excluded = any(ex in excluded_topics for ex in ['payment', 'store', 'paywall', 'checkout', 'commerce', 'membership'])
        if not payment_excluded and any(term in job_lower for term in ['payment gateway', 'payment integration', 'accept payment', 'checkout system']):
            if not any(gateway in job_lower for gateway in ['stripe', 'paypal', 'square', 'woocommerce payments']):
                triggers_detected.append("TRIGGER: Payment mentioned but gateway unclear")
                integration_triggers.append("payment gateway (Stripe, PayPal, etc.?)")
        
        # API integration
        if 'api' in job_lower and 'which api' not in job_lower and 'api' not in excluded_topics:
            integration_triggers.append("API type (REST, GraphQL, specific service?)")
        
        if integration_triggers:
            context_parts.append(f"⚠️ INTEGRATION_SPECIFIC TRIGGER: {'; '.join(integration_triggers)}")
        
        # ============ TRIGGER 3: CONSTRAINT HINTS AT PAST PROBLEMS ============
        constraint_triggers = []
        
        if requirements.must_not_propose:
            for constraint in requirements.must_not_propose:
                constraint_lower = constraint.lower()
                if any(word in constraint_lower for word in ['rebuild', 'redesign', 'start over', 'from scratch']):
                    triggers_detected.append(f"TRIGGER: 'Do not {constraint}' - possible past bad experience")
                    constraint_triggers.append(f"'{constraint}' constraint (why? past attempt?)")
        
        # Direct constraint phrases in job
        if any(phrase in job_lower for phrase in ['do not rebuild', "don't rebuild", 'do not redesign', "don't redesign", 'no redesign', 'no rebuild']):
            triggers_detected.append("TRIGGER: Explicit 'do not rebuild/redesign' constraint")
            constraint_triggers.append("rebuild/redesign constraint (understanding the concern)")
        
        # Past developer issues
        if any(phrase in job_lower for phrase in ['last developer', 'previous developer', 'developer left', 'ghosted', 'went mia', 'disappeared']):
            triggers_detected.append("TRIGGER: Past developer problems mentioned")
            constraint_triggers.append("past developer issues (what went wrong?)")
        
        if constraint_triggers:
            context_parts.append(f"⚠️ CONSTRAINT_CLARIFICATION TRIGGER: {'; '.join(constraint_triggers)}")
        
        # ============ TRIGGER 4: TECHNICAL DIAGNOSTIC OPPORTUNITIES ============
        diagnostic_triggers = []
        
        if requirements.problems_mentioned:
            context_parts.append(f"Problems mentioned: {', '.join(requirements.problems_mentioned[:4])}")
            for problem in requirements.problems_mentioned:
                problem_lower = problem.lower()
                if 'connection' in problem_lower or 'sync' in problem_lower:
                    diagnostic_triggers.append(f"'{problem}' - need to understand error messages/failure mode")
                elif 'slow' in problem_lower or 'performance' in problem_lower:
                    diagnostic_triggers.append(f"'{problem}' - need to know which pages/mobile vs desktop")
                elif 'usability' in problem_lower:
                    diagnostic_triggers.append(f"'{problem}' - need specific user complaints")
        
        if diagnostic_triggers:
            triggers_detected.append("TRIGGER: Technical issues that need diagnostic questions")
            context_parts.append(f"⚠️ TECHNICAL_DIAGNOSTIC TRIGGER: {'; '.join(diagnostic_triggers[:2])}")
        
        # ============ TRIGGER 5: EMPATHY NEEDED ============
        empathy_needed = False
        if requirements.client_tone in ['frustrated', 'urgent']:
            empathy_needed = True
            triggers_detected.append(f"TRIGGER: Client tone is {requirements.client_tone} - acknowledge it")
        
        if any(word in job_lower for word in ['nightmare', 'disaster', 'horrible', 'terrible', 'frustrated', 'struggling']):
            empathy_needed = True
            triggers_detected.append("TRIGGER: Frustration words detected - acknowledge empathetically")
        
        if empathy_needed:
            context_parts.append("⚠️ EMPATHY TRIGGER: Client seems frustrated - acknowledge before asking")
        
        # ============ OTHER CONTEXT ============
        if requirements.tech_stack_mentioned:
            context_parts.append(f"Tech stack: {', '.join(requirements.tech_stack_mentioned[:6])}")
        
        if requirements.client_priorities:
            context_parts.append(f"Client priorities: {', '.join(requirements.client_priorities[:3])}")
        
        if requirements.do_not_assume:
            context_parts.append(f"DO NOT ask about: {', '.join(requirements.do_not_assume[:4])}")
        
        # Explicit checklist from client
        if requirements.explicit_checklist:
            items = [c.get('item_text', '') for c in requirements.explicit_checklist[:4] if c.get('item_text')]
            if items:
                context_parts.append(f"Client's explicit checklist: {'; '.join(items)}")

        # Build final prompt with all detected triggers
        triggers_summary = "\n".join(f"  • {t}" for t in triggers_detected) if triggers_detected else "  • No strong triggers detected - generate general engagement questions"
        
        user_prompt = f"""Generate 1-4 hook questions for a proposal responding to this job.

JOB TITLE: {job_title or 'Not provided'}

DETECTED TRIGGERS (MUST address these):
{triggers_summary}

EXTRACTED CONTEXT:
{chr(10).join(context_parts)}

ORIGINAL JOB DESCRIPTION (reference):
{job_description[:1500]}

Generate questions that would make this specific client want to click and read the full proposal."""

        try:
            response = self.openai_service.client.chat.completions.create(
                model=self.openai_service.llm_model,
                messages=[
                    {"role": "system", "content": HOOK_QUESTIONS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[{"type": "function", "function": HOOK_QUESTIONS_FUNCTION}],
                tool_choice={"type": "function", "function": {"name": "generate_hook_questions"}},
                temperature=0.8  # Slightly creative for varied questions
            )

            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)

            questions = [
                HookQuestion.from_dict(q)
                for q in result.get("questions", [])
            ]

            logger.info(f"[HookQuestions] Generated {len(questions)} questions")
            for i, q in enumerate(questions, 1):
                logger.debug(f"  {i}. [{q.category}] {q.question_text[:60]}...")

            return questions

        except Exception as e:
            logger.error(f"[HookQuestions] Generation failed: {e}")
            return []

    def get_retrieval_context(self, requirements: JobRequirements) -> Dict[str, Any]:
        """
        Convert requirements to retrieval-optimized context.
        
        This is used by the retrieval pipeline to find better matches.
        
        Args:
            requirements: Extracted job requirements
            
        Returns:
            Dictionary optimized for retrieval queries
        """
        return {
            "search_query": f"{requirements.exact_task} {' '.join(requirements.specific_deliverables[:3])}",
            "required_skills": requirements.tech_stack_mentioned,
            "industry": requirements.inferred_industry,
            "complexity": requirements.complexity_level,
            "client_problems": requirements.problems_mentioned,
            "client_tone": requirements.client_tone,
            "constraints": requirements.constraints,
        }
    
    def get_prompt_context(self, requirements: JobRequirements) -> Dict[str, Any]:
        """
        Convert requirements to prompt-optimized context.
        
        This is used by the prompt engine to generate better proposals.
        
        Args:
            requirements: Extracted job requirements
            
        Returns:
            Dictionary optimized for prompt building
        """
        return {
            "exact_task": requirements.exact_task,
            "deliverables": requirements.specific_deliverables,
            "client_tone": requirements.client_tone,
            "problems": requirements.problems_mentioned,
            "resources_available": requirements.resources_provided,
            "constraints": requirements.constraints,
            "do_not_mention": requirements.do_not_assume,  # CRITICAL: What to avoid assuming
            "links": requirements.links_mentioned,
            # NEW FIELDS - Sprint 3
            "working_arrangement": requirements.working_arrangement,
            "application_requirements": requirements.application_requirements,
            "soft_requirements": requirements.soft_requirements,
            "client_priorities": requirements.client_priorities,
            "must_not_propose": requirements.must_not_propose,  # CRITICAL: What NOT to suggest
            "key_phrases_to_echo": requirements.key_phrases_to_echo,
            # Phase 1: Explicit checklist for mandatory requirements
            "explicit_checklist": requirements.explicit_checklist,
            # Smart question injection
            "smart_question": requirements.smart_question,
        }


# Singleton instance getter
_service_instance: Optional[JobRequirementsService] = None


def get_job_requirements_service(
    openai_service=None,
    db_manager=None
) -> JobRequirementsService:
    """
    Get or create singleton JobRequirementsService instance.
    
    Args:
        openai_service: OpenAI service for extraction
        db_manager: Database manager for caching
        
    Returns:
        JobRequirementsService instance
    """
    global _service_instance
    
    if _service_instance is None:
        _service_instance = JobRequirementsService(
            openai_service=openai_service,
            db_manager=db_manager
        )
    elif openai_service is not None:
        # Update dependencies if provided
        _service_instance.openai_service = openai_service
    
    if db_manager is not None:
        _service_instance.db = db_manager
    
    return _service_instance

"""
Prompt Engine v2 - Streamlined Proposal Generation

Optimized from 1,249 → ~400 lines by:
- Moving templates to constants.py
- Removing duplicate rules
- Consolidating methods
"""
import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

from app.domain.constants import (
    PAIN_POINT_INDICATORS, URGENCY_TIMELINE_PROMISES, EMPATHY_RESPONSES,
    STYLE_INSTRUCTIONS, TONE_INSTRUCTIONS, HOOK_STRATEGIES, STALE_OPENINGS,
    ANTI_HALLUCINATION_RULES, PROPOSAL_FORMAT_RULES, PROPOSAL_STRUCTURE,
    CONVERSATIONAL_EXAMPLES, FORBIDDEN_PHRASES, SYSTEM_ROLES,
)

# Import BAD_HOOK_PATTERNS if available
try:
    from app.domain.constants import BAD_HOOK_PATTERNS
except ImportError:
    BAD_HOOK_PATTERNS = []
from app.utils.text_analysis import (
    detect_urgency_level as _detect_urgency_level,
    extract_pain_points as _extract_pain_points,
)

logger = logging.getLogger(__name__)

# Import hook strategy engine
try:
    from app.utils.hook_strategy import get_hook_engine, JobAnalysis
except ImportError:
    get_hook_engine = None
    JobAnalysis = None


class ProposalStyle(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    DATA_DRIVEN = "data_driven"


class ProposalTone(str, Enum):
    CONFIDENT = "confident"
    HUMBLE = "humble"
    ENTHUSIASTIC = "enthusiastic"
    ANALYTICAL = "analytical"
    FRIENDLY = "friendly"


@dataclass
class PromptConfig:
    style: ProposalStyle = ProposalStyle.PROFESSIONAL
    tone: ProposalTone = ProposalTone.CONFIDENT
    include_metrics: bool = True
    include_timeline: bool = True
    include_portfolio: bool = True
    max_word_count: int = 300


class PromptEngine:
    """Streamlined prompt engine for proposal generation."""

    # Delegate to text_analysis
    @staticmethod
    def detect_urgency_level(job_description: str, job_title: str = "") -> str:
        return _detect_urgency_level(job_description, job_title)

    @staticmethod
    def get_urgency_timeline_promise(urgency_level: str) -> Optional[str]:
        return URGENCY_TIMELINE_PROMISES.get(urgency_level)

    @staticmethod
    def extract_pain_points(job_description: str) -> Dict[str, List[str]]:
        return _extract_pain_points(job_description)

    @staticmethod
    def build_empathy_statement(pain_points: Dict[str, List[str]]) -> str:
        if not pain_points:
            return ""
        import random
        for category in ["business_impact", "urgency", "frustration", "previous_failure"]:
            if category in pain_points and category in EMPATHY_RESPONSES:
                return random.choice(EMPATHY_RESPONSES[category])
        return ""

    def build_proposal_prompt(
        self,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        success_patterns: List[str],
        style: str = "professional",
        tone: str = "confident",
        max_words: int = 300,
        include_portfolio: bool = True,
        include_feedback: bool = True,
        include_timeline: bool = False,
        timeline_duration: str = None,
        profile_context: Optional[Dict[str, Any]] = None,
        requirements_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build optimized prompt using HOOK→PROOF→APPROACH→CTA structure."""
        
        style_key = style.value if isinstance(style, ProposalStyle) else style
        tone_key = tone.value if isinstance(tone, ProposalTone) else tone

        # Build sections
        system_role = SYSTEM_ROLES.get(style_key, SYSTEM_ROLES["professional"])
        style_guide = STYLE_INSTRUCTIONS.get(style_key, "")
        tone_guide = TONE_INSTRUCTIONS.get(tone_key, "")
        
        job_section = self._build_job_section(job_data)
        projects_section = self._build_projects_section(similar_projects, include_portfolio, include_feedback)
        profile_section = self._build_profile_section(profile_context) if profile_context else ""
        
        # NEW: Build requirements section from extracted job understanding
        requirements_section = self._build_requirements_section(requirements_context) if requirements_context else ""
        
        # NEW: Build specific proof section from enhanced retrieval
        specific_proof_section = self._build_specific_proof_section(requirements_context) if requirements_context else ""
        
        # Dynamic hook generation - NOW REQUIREMENTS-AWARE
        hook_section = self._build_hook_section(job_data, similar_projects, requirements_context)
        
        # Timeline instruction
        timeline_inst = self._get_timeline_instruction(include_timeline, timeline_duration)
        
        # Build anti-hallucination rules (enhanced with do_not_assume AND must_not_propose)
        anti_hallucination = self._build_anti_hallucination_rules(requirements_context)

        # Build strategic instructions based on requirements
        strategic_instructions = self._build_strategic_instructions(requirements_context, job_data)
        
        # Phase 1: Build mandatory checklist section
        checklist_section = self._build_mandatory_checklist_section(requirements_context, similar_projects)
        
        return f"""
{system_role}

{self._get_core_rules()}

{job_section}
{requirements_section}
{profile_section}
{projects_section}
{specific_proof_section}
{checklist_section}
{hook_section}

{PROPOSAL_STRUCTURE}

{style_guide}
{tone_guide}

{strategic_instructions}

Generate the proposal NOW. Target: {max_words} words (ideal: 200-350).

{anti_hallucination}

🚨 ABSOLUTE RULES - VIOLATION = REJECTED PROPOSAL:
1. NO "As an AI" - sound like a REAL human freelancer
2. NO MARKDOWN: No **bold**, no *italic*, no "- **Label:**" formatting
3. NO SECTION HEADERS: Write in natural paragraphs, not structured sections
4. Reference ONLY verified projects above - NO inventing
5. Use PLAIN URLs only (https://example.com)
{timeline_inst}
6. End with casual CTA: "Happy to chat" or "Let me know"
7. NEVER say "While I haven't done this exact..." or similar weak language
8. NEVER mention timezone/availability UNLESS client explicitly required it
9. NEVER mention "one-time project" or "ongoing" UNLESS client specified engagement type
10. NEVER provide rates, hours, or timeline estimates UNLESS client asked for them
11. If client provided a document to review, speak as if you've ALREADY reviewed it

{PROPOSAL_FORMAT_RULES}
"""

    def _get_core_rules(self) -> str:
        """Core system rules - consolidated."""
        hooks = "\n".join(f"   {i}. {h}" for i, h in enumerate(HOOK_STRATEGIES, 1))
        stale = "\n".join(f'   - "{s}"' for s in STALE_OPENINGS)
        examples = "\n".join(f'   ✓ "{e}"' for e in CONVERSATIONAL_EXAMPLES)
        forbidden = "\n".join(f'   - "{p}"' for p in FORBIDDEN_PHRASES[:10])
        
        return f"""
SYSTEM: SHORT, HUMAN, WINNING proposal writer. Write in PLAIN TEXT paragraphs - NO markdown, NO section headers.

🧠 WINNING FORMULA:
1. HOOK with RELEVANT portfolio link (NOT generic questions)
2. PROVE with similar work (ONLY if truly relevant)
3. APPROACH - your specific solution for THEIR situation
4. CTA - casual close "Happy to chat" or "Let me know"

✅ WINNING HOOK STRATEGIES:
{hooks}

❌ STALE/BAD OPENINGS - NEVER USE:
{stale}

❌ FORBIDDEN PHRASES - NEVER USE:
{forbidden}

🚨 CRITICAL RULES - READ CAREFULLY:
1. ONLY mention timezone/availability IF client explicitly asks for it in the job post
2. ONLY mention "one-time" or "ongoing" IF client specifies engagement type
3. ONLY provide rates/estimates IF client explicitly requests them
4. ONLY include portfolio links that are DIRECTLY relevant to the job
5. NEVER say "While I haven't done this exact scope" or any self-deprecating language
6. NEVER say "I'll review the document" - speak as if you've ALREADY reviewed it
7. Write in PLAIN TEXT - absolutely NO markdown formatting (**bold**, - bullet with **label**:, etc.)
8. Write in NATURAL PARAGRAPHS - do NOT use section headers or structured formatting

CONVERSATIONAL TONE:
{examples}
"""

    def _build_hook_section(
        self, 
        job_data: Dict[str, Any], 
        similar_projects: List[Dict[str, Any]],
        requirements_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build dynamic hook suggestions based on job analysis AND requirements context.
        
        ENHANCED Sprint 3: Now considers working_arrangement, application_requirements,
        soft_requirements, and client_priorities when generating hooks.
        """
        section = ""
        
        # Build requirements-aware hook guidance FIRST (now with outcome stats from similar projects)
        hook_guidance = self._build_requirements_based_hook_guidance(requirements_context, job_data, similar_projects)
        
        # Try hook engine
        if job_data and get_hook_engine is not None:
            try:
                hook_engine = get_hook_engine()
                analysis = hook_engine.analyze_job(job_data)
                
                # Get portfolio URL for hook
                portfolio_url = self._get_best_portfolio_url(similar_projects)
                hooks = hook_engine.get_hook_variations(analysis, job_data, similar_projects, portfolio_url, count=3)
                
                section = f"""
🎯 JOB ANALYSIS:
• Sentiment: {analysis.sentiment.value.upper()} | Intent: {analysis.intent.value.upper()}
• Urgency: {analysis.urgency_level}/5 | Platform: {analysis.platform.upper()}
• Strategy: {analysis.hook_strategy}
"""
                if hooks:
                    section += "\n🔥 GENERATED HOOKS:\n"
                    for i, hook in enumerate(hooks, 1):
                        section += f'   {i}. "{hook}"\n'
                        
            except Exception as e:
                logger.warning(f"Hook engine error: {e}")
        
        # Add requirements-based hook guidance
        if hook_guidance:
            section += hook_guidance
        
        # Fallback pain points
        if not section and job_data:
            pain_points = self.extract_pain_points(job_data.get('job_description', ''))
            if pain_points:
                empathy = self.build_empathy_statement(pain_points)
                section = f"\n💡 DETECTED PAIN POINTS: {list(pain_points.keys())[:3]}\n"
                if empathy:
                    section += f'   Empathy opener: "{empathy}"\n'
        
        return section
    
    def _build_requirements_based_hook_guidance(
        self,
        requirements: Optional[Dict[str, Any]],
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]] = None
    ) -> str:
        """
        Build hook guidance based on extracted requirements AND outcome stats.
        
        This ensures the hook addresses what CLIENT CARES ABOUT, not generic tech questions.
        Now includes REAL outcome stats from similar projects for compelling hooks.
        """
        guidance_parts = []
        
        # Extract best outcome stats from similar projects (TOP PRIORITY for compelling hooks)
        outcome_stats = self._extract_best_outcome_stats(similar_projects)
        if outcome_stats:
            guidance_parts.append(f'''   📊 USE THESE REAL STATS IN YOUR HOOK:
      "{outcome_stats['stats']}"
      Company: {outcome_stats['company']} | URL: {outcome_stats.get('url', 'N/A')}
      ✅ Pattern: "I achieved [stat] on a similar project - [URL]"''')
        
        if not requirements:
            return self._format_hook_guidance(guidance_parts) if guidance_parts else ""
        
        # Only use question if user explicitly selected one
        selected_q = requirements.get("selected_hook_question")
        if selected_q:
            logger.info(f"[PromptEngine] Using user-selected hook question: {selected_q[:50]}...")
            guidance_parts.append(f'''   🚨 OPEN WITH THIS QUESTION (FIRST 1-2 SENTENCES):
      "{selected_q}"
      ❌ DO NOT put this at the end - START with it
      ✅ Pattern: "[Question] + I've done similar work at [example]"
      Example: "{selected_q} I built a similar project: [URL]"''')
        
        # Working arrangement hook (CRITICAL for service roles)
        working = requirements.get("working_arrangement", {})
        if working.get("timezone") or working.get("arrangement_type"):
            tz = working.get("timezone", "")
            arr_type = working.get("arrangement_type", "")
            
            if tz:
                guidance_parts.append(f'   ✅ MENTION AVAILABILITY: Address {tz} timezone requirement in your hook or opening')
            if arr_type == "ongoing":
                guidance_parts.append('   ✅ SHOW PARTNERSHIP MINDSET: Client wants ongoing work - show you value long-term relationships')
            elif arr_type == "one_time":
                guidance_parts.append('   ✅ SHOW EFFICIENCY: Client wants one-time work - emphasize quick, quality delivery')
        
        # Client priorities hook
        if requirements.get("client_priorities"):
            top_priority = requirements["client_priorities"][0] if requirements["client_priorities"] else None
            if top_priority:
                guidance_parts.append(f'   ✅ ADDRESS TOP PRIORITY: "{top_priority}" matters MOST - lead with this')
        
        # Soft requirements hook
        if requirements.get("soft_requirements"):
            soft = requirements["soft_requirements"][0] if requirements["soft_requirements"] else None
            if soft:
                guidance_parts.append(f'   ✅ ADDRESS SOFT REQUIREMENT: Client values "{soft}" - demonstrate this quality')
        
        # Application requirements awareness
        if requirements.get("application_requirements"):
            app_req = requirements["application_requirements"][0] if requirements["application_requirements"] else None
            if app_req and "loom" in app_req.lower():
                guidance_parts.append('   ✅ LOOM VIDEO REQUIRED: Mention you will/can provide a Loom video')
            elif app_req:
                guidance_parts.append(f'   ✅ APPLICATION REQUIREMENT: Address "{app_req}" in your proposal')
        
        # Key phrases to use
        if requirements.get("key_phrases_to_echo"):
            phrases = requirements["key_phrases_to_echo"][:2]
            phrase_text = '", "'.join(phrases)
            guidance_parts.append(f'   ✅ USE THEIR WORDS: Echo "{phrase_text}" to show you read the post')
        
        # Must NOT propose (prevent irrelevant hooks)
        if requirements.get("must_not_propose"):
            dont_items = requirements["must_not_propose"][:2]
            guidance_parts.append(f'   ❌ AVOID IN HOOK: Do not lead with questions about {", ".join(dont_items)} - not what client asked for')
        
        return self._format_hook_guidance(guidance_parts)
    
    def _extract_best_outcome_stats(self, similar_projects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract the most compelling outcome stats from similar projects."""
        if not similar_projects:
            return None
        for proj in similar_projects:
            outcome = proj.get('outcome')
            # Primary: structured dict {stats, loom_url}
            if isinstance(outcome, dict):
                stats = outcome.get('stats')
            elif isinstance(outcome, str):
                stats = outcome
            else:
                # Fallback: unmigrated legacy 'outcomes' string from old training_data
                stats = proj.get('outcomes') if isinstance(proj.get('outcomes'), str) else None
            if stats and stats.strip() and 'test' not in stats.lower():
                urls = proj.get('portfolio_urls', []) or []
                url = proj.get('portfolio_url') or (urls[0] if urls else None)
                return {
                    'stats': stats,
                    'company': proj.get('company_name') or proj.get('company') or proj.get('client_name') or 'Previous client',
                    'url': url
                }
        return None
    
    def _format_hook_guidance(self, guidance_parts: List[str]) -> str:
        """Format hook guidance parts into final string."""
        if not guidance_parts:
            return ""
        return """\n🎯 REQUIREMENTS-BASED HOOK GUIDANCE (CRITICAL):\n""" + "\n".join(guidance_parts) + """\n\n⚠️ DO NOT open with generic technical questions about things the client didn't ask about!\n   ✅ GOOD hook: "[Question/Statement] + I achieved [STAT] on [Company] - [URL]"
   ❌ BAD hook: Generic questions without proof or stats
"""
    
    def _build_strategic_instructions(
        self,
        requirements: Optional[Dict[str, Any]],
        job_data: Dict[str, Any]
    ) -> str:
        """
        Build strategic instructions that act as an ORCHESTRATOR.
        
        This method analyzes the job and provides explicit guidance on:
        - What TO include (based on what client asked)
        - What NOT to include (avoid assumptions)
        
        This prevents common issues like:
        - Mentioning timezone when not asked
        - Saying "one-time project" when not specified
        - Giving rates/timelines when not requested
        - Self-deprecating statements
        """
        include_items = []
        exclude_items = []
        
        # Analyze job for what's actually required
        job_desc = job_data.get("job_description", "").lower()
        
        # === TIMEZONE/AVAILABILITY (Phase 1: Use timezone_source) ===
        working = requirements.get("working_arrangement", {}) if requirements else {}
        tz_source = working.get("timezone_source", "none")
        tz_value = working.get("timezone", "")
        
        if tz_source == "explicit" and tz_value:
            # Client EXPLICITLY stated timezone - MUST mention
            include_items.append(f"✅ MENTION TIMEZONE: Client EXPLICITLY requires {tz_value} - confirm availability")
        else:
            # Timezone not explicitly stated OR was inferred - DO NOT mention
            exclude_items.append("❌ DO NOT mention timezone or 'available in your timezone' - client didn't explicitly ask")
        
        # === ENGAGEMENT TYPE ===
        ongoing_mentioned = any(word in job_desc for word in ["ongoing", "long-term", "monthly", "retainer", "continuous"])
        onetime_mentioned = any(word in job_desc for word in ["one-time", "single project", "one time", "fixed price project"])
        
        if ongoing_mentioned:
            include_items.append("✅ SHOW LONG-TERM MINDSET: Client wants ongoing relationship")
        elif onetime_mentioned:
            include_items.append("✅ SHOW EFFICIENCY: Client wants one-time delivery")
        else:
            exclude_items.append("❌ DO NOT mention 'one-time project' or 'ongoing' - client didn't specify engagement type")
        
        # === RATES/ESTIMATES ===
        rates_requested = any(word in job_desc for word in ["hourly rate", "your rate", "send your rate", "estimate", "how long", "how many hours", "budget proposal", "fixed price proposal"])
        if requirements and requirements.get("application_requirements"):
            for req in requirements["application_requirements"]:
                if any(word in req.lower() for word in ["rate", "estimate", "hours", "budget", "price"]):
                    rates_requested = True
        
        if rates_requested:
            include_items.append("✅ PROVIDE RATE/ESTIMATE: Client explicitly asked for pricing/timeline")
        else:
            exclude_items.append("❌ DO NOT provide hourly rates, estimated hours, or timelines - client didn't ask")
        
        # === DOCUMENT REVIEW ===
        has_document = any(word in job_desc for word in ["document", "attached", "provided", "review the", "based on our", "concept document", "plan document"])
        if has_document:
            include_items.append("✅ SPEAK AS IF YOU'VE REVIEWED: Write as if you already understand the document")
            exclude_items.append("❌ DO NOT say 'I'll review the document' - assume you HAVE reviewed it already")
        
        # === PORTFOLIO RELEVANCE ===
        exclude_items.append("❌ DO NOT include portfolio links that aren't directly relevant to the job")
        exclude_items.append("❌ DO NOT say 'although not identical' or 'haven't done this exact scope' - NEVER be self-deprecating")
        
        # === APPLICATION REQUIREMENTS ===
        if requirements and requirements.get("application_requirements"):
            app_reqs = requirements["application_requirements"]
            for req in app_reqs:
                if "loom" in req.lower():
                    include_items.append("✅ LOOM VIDEO: Mention you'll provide a Loom video as requested")
                elif "portfolio" in req.lower() or "example" in req.lower():
                    include_items.append(f"✅ PORTFOLIO: Include relevant examples as client requested")
                elif any(word in req.lower() for word in ["happy", "specific word", "code word"]):
                    include_items.append(f"✅ INCLUDE: {req}")
        
        # === FORMAT RULES ===
        exclude_items.append("❌ NO MARKDOWN: Do not use **bold**, *italic*, or '- **Label:**' formatting")
        exclude_items.append("❌ NO SECTION HEADERS: Write natural paragraphs, not structured sections")
        
        # Build the output
        output = "\n📋 STRATEGIC INSTRUCTIONS (FOLLOW EXACTLY):\n"
        
        if include_items:
            output += "\nWHAT TO INCLUDE:\n" + "\n".join(include_items)
        
        if exclude_items:
            output += "\n\nWHAT TO EXCLUDE (CRITICAL):\n" + "\n".join(exclude_items)
        
        # Add smart question / selected hook question opener instruction
        selected_q = requirements.get("selected_hook_question") if requirements else None
        if selected_q:
            output = f"""
⚡ MANDATORY OPENER - YOUR FIRST SENTENCE:
"{selected_q}"
Then immediately follow with relevant experience/portfolio link.
Example: "{selected_q} I built something similar - here's the result: [URL]"
DO NOT put this question at the end. START your proposal with it.
""" + output
        else:
            smart_q = requirements.get("smart_question") or {} if requirements else {}
            if smart_q.get("ask") and smart_q.get("question"):
                question = smart_q["question"]
                output = f"""
⚡ MANDATORY OPENER - YOUR FIRST SENTENCE:
"{question}"
Then immediately follow with relevant experience/portfolio link.
Example: "{question} I built something similar - here's the result: [URL]"
DO NOT put this question at the end. START your proposal with it.
""" + output
        
        return output

    def _build_mandatory_checklist_section(
        self,
        requirements: Optional[Dict[str, Any]],
        similar_projects: List[Dict[str, Any]]
    ) -> str:
        """
        Phase 1+2: Build MANDATORY checklist section with capability analysis.
        
        Enforces addressing every numbered/bulleted requirement from the job post.
        Now includes capability status (SATISFIED/PARTIAL/UNSATISFIED) and factual data.
        """
        if not requirements:
            return ""
        
        checklist = requirements.get("explicit_checklist", [])
        if not checklist:
            return ""
        
        # Phase 2: Get capability analysis results (with safe fallback to empty dict)
        capability_analysis = requirements.get("capability_analysis") or {}
        factual_answers = requirements.get("factual_answers") or {}
        
        # Collect available portfolio URLs
        available_urls = []
        for p in similar_projects[:5]:
            urls = p.get("portfolio_urls", [])
            if isinstance(urls, list):
                available_urls.extend([u for u in urls if u])
        
        lines = [
            "",
            "=" * 60,
            "🚨 MANDATORY CHECKLIST - ADDRESS EVERY ITEM 🚨",
            "=" * 60,
            ""
        ]
        
        for i, item in enumerate(checklist, 1):
            item_type = item.get("item_type", "other")
            qty = item.get("quantity_requested", 0)
            specific = item.get("specificity_required", False)
            hint = item.get("answer_hint", "")
            text = item.get("item_text", "Requirement")
            
            # Phase 2: Get capability status for this item
            item_capability = capability_analysis.get(text, {})
            status = item_capability.get("status", "unknown")
            supporting = item_capability.get("supporting_evidence", [])
            suggestion = item_capability.get("suggested_approach", "")
            
            lines.append(f"{i}. {text.upper()}")
            
            # Add capability status indicator
            if status == "SATISFIED":
                lines.append(f"   ✅ CAPABILITY: Strong evidence in portfolio")
                if supporting:
                    lines.append(f"   📋 Evidence: {'; '.join(supporting[:2])}")
            elif status == "PARTIAL":
                lines.append(f"   ⚠️ CAPABILITY: Related but not exact match")
                if suggestion:
                    lines.append(f"   💡 Approach: {suggestion}")
            elif status == "UNSATISFIED":
                lines.append(f"   ❌ CAPABILITY: No direct evidence - BE HONEST")
                if suggestion:
                    lines.append(f"   💡 Approach: {suggestion}")
            
            if item_type == "portfolio_links":
                urls_to_show = available_urls[:qty] if qty else available_urls[:2]
                if urls_to_show:
                    lines.append(f"   📎 USE THESE: {', '.join(urls_to_show)}")
                else:
                    lines.append("   ⚠️ No direct matches - acknowledge and pivot to related experience")
                if qty:
                    lines.append(f"   → Client wants {qty}+ examples")
            
            elif item_type == "time_estimate_phased":
                lines.append("   📅 Provide PER-PHASE breakdown, not just total")
                lines.append("   → Example: 'Homepage: 3-4 days, Catalog: 2-3 days...'")
            
            elif item_type == "time_estimate_total":
                lines.append("   📅 Provide overall timeline estimate")
            
            elif item_type == "experience_question":
                # Check for factual answers (e.g., "How many WordPress sites?")
                factual_key = self._find_factual_key(text, factual_answers)
                if factual_key and factual_answers.get(factual_key):
                    fact = factual_answers[factual_key]
                    lines.append(f"   📊 FACTUAL ANSWER: {fact}")
                
                if specific:
                    lines.append("   ⚠️ BE SPECIFIC - name actual projects, not vague claims")
                    lines.append("   ❌ BAD: 'I have experience with RTL'")
                    lines.append("   ✅ GOOD: 'I built [project] with full Hebrew RTL support'")
                if hint:
                    lines.append(f"   🎯 Focus on: {hint}")
            
            elif item_type == "preference_question":
                lines.append("   💡 Give specific recommendation with reasoning")
            
            elif item_type == "approach_description":
                lines.append("   📝 Describe YOUR process for THIS project")
            
            lines.append("")
        
        lines.extend([
            "=" * 60,
            "⛔ MISSING ANY ITEM = LIKELY REJECTION",
            "=" * 60,
            ""
        ])
        
        return "\n".join(lines)

    def _find_factual_key(self, item_text: str, factual_answers: Dict[str, Any]) -> Optional[str]:
        """
        Phase 2: Match checklist item to factual answer key.
        
        Uses keyword matching to find relevant factual data for the question.
        """
        if not factual_answers:
            return None
        
        item_lower = item_text.lower()
        
        # Direct key match
        for key in factual_answers.keys():
            if key.lower() in item_lower or item_lower in key.lower():
                return key
        
        # Keyword mapping for common patterns
        keyword_map = {
            "wordpress": ["wordpress_count", "wordpress_sites", "wp_count"],
            "how many": ["count", "total", "number"],
            "years": ["years_experience", "experience_years"],
            "sites": ["site_count", "websites_count"],
            "projects": ["project_count", "completed_projects"],
        }
        
        for keyword, possible_keys in keyword_map.items():
            if keyword in item_lower:
                for pk in possible_keys:
                    if pk in factual_answers:
                        return pk
        
        return None

    def _get_best_portfolio_url(self, projects: List[Dict[str, Any]]) -> Optional[str]:
        """Get best portfolio URL (prefer non-Upwork)."""
        for proj in projects[:3]:
            for url in proj.get("portfolio_urls", []):
                if url and 'upwork.com' not in url.lower():
                    return url
        return None

    def _get_timeline_instruction(self, include: bool, duration: str = None) -> str:
        if include:
            dur = duration or "2-3 weeks"
            return f'5. Include casual timeline: "Looking at about {dur} to wrap this up"'
        return "5. NO timeline - skip timeline section"

    def _build_requirements_section(self, requirements: Dict[str, Any]) -> str:
        """
        Build section from extracted job requirements.
        
        This provides the LLM with structured understanding of what the client
        ACTUALLY wants, improving proposal relevance and reducing hallucination.
        
        ENHANCED Sprint 3: Now includes working arrangement, application requirements,
        soft requirements, client priorities, and must_not_propose.
        """
        if not requirements:
            return ""
        
        sections = []
        
        # Exact task (most important)
        if requirements.get("exact_task"):
            sections.append(f"📌 EXACT CLIENT NEED: {requirements['exact_task']}")
        
        # Deliverables
        if requirements.get("deliverables"):
            delivs = ", ".join(requirements["deliverables"][:5])
            sections.append(f"📦 EXPECTED DELIVERABLES: {delivs}")
        
        # Client tone (for matching communication style)
        if requirements.get("client_tone"):
            tone_guide = {
                "urgent": "⚡ Client is URGENT - emphasize availability and speed",
                "frustrated": "😤 Client is FRUSTRATED - show empathy, highlight reliability",
                "technical": "🔧 Client is TECHNICAL - use precise language, show expertise",
                "casual": "😊 Client is CASUAL - keep it friendly and relaxed",
                "professional": "💼 Client is PROFESSIONAL - formal but warm",
                "exploratory": "🤔 Client is EXPLORING - educate and guide gently"
            }
            sections.append(tone_guide.get(requirements["client_tone"], ""))
        
        # Problems mentioned (for empathy)
        if requirements.get("problems"):
            probs = ", ".join(requirements["problems"][:3])
            sections.append(f"🎯 CLIENT PAIN POINTS: {probs}")
        
        # Constraints
        if requirements.get("constraints"):
            cons = ", ".join(requirements["constraints"][:3])
            sections.append(f"⚠️ CONSTRAINTS: {cons}")
        
        # Resources provided
        if requirements.get("resources_provided"):
            res = ", ".join(requirements["resources_provided"][:3])
            sections.append(f"📁 RESOURCES CLIENT WILL PROVIDE: {res}")
        
        # ====== NEW FIELDS - Sprint 3 Enhanced Intent Capture ======
        
        # Working arrangement (CRITICAL for service roles)
        working = requirements.get("working_arrangement", {})
        if working:
            work_parts = []
            if working.get("timezone"):
                work_parts.append(f"Timezone: {working['timezone']}")
            if working.get("hours"):
                work_parts.append(f"Hours: {working['hours']}")
            if working.get("arrangement_type"):
                work_parts.append(f"Type: {working['arrangement_type']}")
            if working.get("responsiveness_expectation"):
                work_parts.append(f"Responsiveness: {working['responsiveness_expectation']}")
            if work_parts:
                sections.append(f"🕐 WORKING ARRANGEMENT (MUST ADDRESS): {' | '.join(work_parts)}")
        
        # Application requirements (CRITICAL - missing = instant rejection)
        if requirements.get("application_requirements"):
            app_reqs = ", ".join(requirements["application_requirements"])
            sections.append(f"🚨 APPLICATION REQUIREMENTS (MUST INCLUDE OR MENTION): {app_reqs}")
        
        # Soft requirements (what client values beyond tech)
        if requirements.get("soft_requirements"):
            soft = ", ".join(requirements["soft_requirements"][:4])
            sections.append(f"💪 SOFT REQUIREMENTS (address these EXPLICITLY): {soft}")
        
        # Client priorities (what matters most - ORDER MATTERS)
        if requirements.get("client_priorities"):
            prios = " > ".join(requirements["client_priorities"][:5])
            sections.append(f"🎯 CLIENT PRIORITIES (in order): {prios}")
        
        # Key phrases to echo (for rapport)
        if requirements.get("key_phrases_to_echo"):
            phrases = '", "'.join(requirements["key_phrases_to_echo"][:4])
            sections.append(f'💬 USE THESE PHRASES: "{phrases}"')

        if sections:
            return "\n🧠 EXTRACTED JOB UNDERSTANDING (FOLLOW THESE CLOSELY):\n" + "\n".join(sections) + "\n"
        return ""

    def _build_specific_proof_section(self, requirements: Optional[Dict[str, Any]]) -> str:
        """
        Build SPECIFIC PROOF section from enhanced retrieval data.
        
        Uses deliverable_matches, problem_solutions, and portfolio_coverage
        to provide the LLM with EXACT proof points for each job requirement.
        """
        if not requirements:
            return ""
        
        sections = []
        
        # === DELIVERABLE MATCHES: Proof for each specific deliverable ===
        deliverable_matches = requirements.get("deliverable_matches", {})
        if deliverable_matches:
            sections.append("\n🎯 SPECIFIC PROOF FOR EACH DELIVERABLE:")
            for deliverable, matches in deliverable_matches.items():
                if matches:
                    best_match = matches[0]  # Best match first
                    company = best_match.get('company') or best_match.get('company_name') or 'Previous client'
                    urls = best_match.get('portfolio_urls', [])
                    url = urls[0] if urls else best_match.get('portfolio_url', 'N/A')
                    score = best_match.get('deliverable_score', 0)
                    sections.append(f'   ✅ "{deliverable}" → Proven at {company}: {url} (match: {score:.0%})')
                else:
                    sections.append(f'   ⚠️ "{deliverable}" → No direct proof - use skill-based approach')
        
        # === PROBLEM SOLUTIONS: Past problems solved similar to client's pain points ===
        problem_solutions = requirements.get("problem_solutions", [])
        if problem_solutions:
            sections.append("\n🔧 PAST PROBLEMS SOLVED (use these for empathy hooks):")
            for solution in problem_solutions[:3]:
                problem = solution.get('problem_matched', '')
                company = solution.get('company') or solution.get('company_name') or 'Previous client'
                outcome = solution.get('outcome_text', '')
                urls = solution.get('portfolio_urls', [])
                url = urls[0] if urls else solution.get('portfolio_url', 'N/A')
                sections.append(f'   ✅ Problem: "{problem[:50]}..." → Solved at {company}')
                if outcome:
                    sections.append(f'      Result: "{outcome[:80]}..."')
                sections.append(f'      URL: {url}')
        
        # === PORTFOLIO COVERAGE ANALYSIS ===
        portfolio_coverage = requirements.get("portfolio_coverage", {})
        if portfolio_coverage:
            total = portfolio_coverage.get("total_items", 0)
            covered = portfolio_coverage.get("covered", 0)
            partial = portfolio_coverage.get("partial", 0)
            uncovered = portfolio_coverage.get("uncovered", 0)
            coverage_pct = portfolio_coverage.get("coverage_percentage", 0)
            
            sections.append(f"\n📊 PORTFOLIO COVERAGE: {coverage_pct:.0%} of requirements")
            sections.append(f"   Full proof: {covered}/{total} | Partial: {partial}/{total} | No proof: {uncovered}/{total}")
            
            # Specific guidance based on coverage
            if coverage_pct >= 0.7:
                sections.append("   ✅ STRONG MATCH: Lead with specific project proofs")
            elif coverage_pct >= 0.4:
                sections.append("   ⚠️ PARTIAL MATCH: Combine proven work + transferable skills")
            else:
                sections.append("   🔶 LIMITED MATCH: Lead with approach and problem-solving skills")
        
        # === NO PORTFOLIO STRATEGY: What to do when portfolio doesn't match ===
        no_portfolio_strategy = requirements.get("no_portfolio_strategy")
        if no_portfolio_strategy and no_portfolio_strategy.get("strategy_type") != "direct_proof":
            strategy_type = no_portfolio_strategy.get("strategy_type", "approach_first")
            recommendation = no_portfolio_strategy.get("recommendation", "")
            
            strategy_guidance = {
                "skill_based": "🎯 Use SKILL-BASED hooks: 'I specialize in X and Y which directly apply here'",
                "transferable": "🔄 Use TRANSFERABLE hooks: 'I did similar work for [industry] that directly applies'",
                "approach_first": "💡 Use APPROACH hooks: Lead with HOW you'd solve their specific problem"
            }
            
            sections.append(f"\n🚨 NO DIRECT PORTFOLIO MATCH - STRATEGY: {strategy_type.upper()}")
            sections.append(f"   {strategy_guidance.get(strategy_type, recommendation)}")
            if recommendation:
                sections.append(f"   Recommendation: {recommendation}")
        
        if sections:
            return "\n" + "=" * 60 + "\n🎯 SPECIFIC PROOF POINTS (USE THESE EXACTLY)\n" + "=" * 60 + "\n" + "\n".join(sections) + "\n"
        return ""

    def _build_anti_hallucination_rules(self, requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        Build anti-hallucination rules, enhanced with do_not_assume AND must_not_propose.
        
        The do_not_assume field lists things the client did NOT specify.
        The must_not_propose field lists things the proposal should NOT suggest.
        Together, these prevent irrelevant/unwanted content in proposals.
        """
        base_rules = ANTI_HALLUCINATION_RULES
        
        enhanced_rules = base_rules
        
        if requirements:
            additional_rules = []
            
            # Things NOT to assume
            if requirements.get("do_not_assume"):
                avoid_list = "\n".join(f"   ❌ Do not assume: {item}" for item in requirements["do_not_assume"][:5])
                additional_rules.append(f"""
🚫 DO NOT ASSUME (client did not specify):
{avoid_list}
If you need to discuss any of the above, ask questions instead of assuming!""")
            
            # Things NOT to propose (CRITICAL NEW - prevents irrelevant suggestions)
            if requirements.get("must_not_propose"):
                dont_suggest = "\n".join(f"   🛑 Do not suggest: {item}" for item in requirements["must_not_propose"][:5])
                additional_rules.append(f"""
🛑 DO NOT PROPOSE/SUGGEST (stay in scope):
{dont_suggest}
Only address what the client ASKED for. No unsolicited advice!""")
            
            # Application requirements warning
            if requirements.get("application_requirements"):
                app_reqs = ", ".join(requirements["application_requirements"])
                additional_rules.append(f"""
📋 APPLICATION REQUIREMENT CHECK:
   Client requires: {app_reqs}
   → If you cannot include this in the proposal, ACKNOWLEDGE it clearly!
   → Example: "I'll prepare a Loom video walkthrough for you" """)
            
            # Working arrangement warning
            working = requirements.get("working_arrangement", {})
            if working.get("timezone") or working.get("hours"):
                tz = working.get("timezone", "not specified")
                hrs = working.get("hours", "not specified")
                additional_rules.append(f"""
🕐 AVAILABILITY REQUIREMENT:
   Client requires: {tz} timezone, {hrs}
   → You MUST confirm you can meet this availability
   → If unclear, mention you're "happy to discuss availability" """)
            
            if additional_rules:
                enhanced_rules = f"""
{base_rules}
{"".join(additional_rules)}
"""
        
        return enhanced_rules

    def _build_job_section(self, job_data: Dict[str, Any]) -> str:
        """Build job details section."""
        return f"""
JOB DETAILS:
Company: {job_data.get('company_name')}
Position: {job_data.get('job_title')}
Industry: {job_data.get('industry', 'Not specified')}
Skills: {', '.join(job_data.get('skills_required', []))}
Urgency: {'URGENT' if job_data.get('urgent_adhoc') else 'Planned'}

DESCRIPTION:
{job_data.get('job_description', 'No description')}
"""

    def _build_profile_section(self, profile: Optional[Dict[str, Any]]) -> str:
        """Build freelancer profile section."""
        if not profile:
            return ""
        parts = ["FREELANCER PROFILE:"]
        if profile.get('display_name'):
            parts.append(f"Name: {profile['display_name']}")
        if profile.get('headline'):
            parts.append(f"Title: {profile['headline']}")
        if profile.get('skills'):
            parts.append(f"Skills: {', '.join(profile['skills'][:10])}")
        return "\n".join(parts) + "\n"

    def _build_projects_section(
        self,
        projects: List[Dict[str, Any]],
        include_portfolio: bool,
        include_feedback: bool
    ) -> str:
        """Build verified projects section."""
        if not projects:
            return """VERIFIED PROJECTS: None available.
🚫 NO PAST PROJECTS - Focus on APPROACH and SKILLS only. DO NOT fabricate experience."""

        # Check relevance
        has_relevant = any(p.get('similarity_score', 0) >= 0.45 for p in projects[:3])
        if not has_relevant:
            return """VERIFIED PROJECTS: Available but not closely relevant.
🚫 Focus on APPROACH. Don't force irrelevant project references."""

        section = f"VERIFIED PAST PROJECTS (Reference ONLY these):\n\n{ANTI_HALLUCINATION_RULES}\n\n"
        used_urls = set()
        added = 0

        for proj in projects[:5]:
            if added >= 3:
                break
            
            company = proj.get('company') or proj.get('title', 'Past project')
            urls = proj.get('portfolio_urls', [])
            
            # Get best URL (non-Upwork, not used)
            best_url = None
            for url in urls:
                if url and url.lower() not in used_urls:
                    if 'upwork.com' not in url.lower():
                        best_url = url
                        break
            if not best_url:
                for url in urls:
                    if url and url.lower() not in used_urls:
                        best_url = url
                        break
            
            if not best_url:
                continue
            
            used_urls.add(best_url.lower())
            added += 1
            
            section += f"PROJECT {added}: {company}\n"
            if proj.get('task_type'):
                section += f"   Task: {proj['task_type']}\n"
            if proj.get('deliverables'):
                delivs = proj['deliverables']
                section += f"   Deliverables: {', '.join(delivs[:4]) if isinstance(delivs, list) else delivs}\n"
            
            # Show outcome stats if available (critical for compelling proposals)
            outcome = proj.get('outcome')
            if outcome:
                if isinstance(outcome, dict):
                    if outcome.get('stats'):
                        section += f"   Results: {outcome['stats']}\n"
                    if outcome.get('loom_url'):
                        section += f"   Video: {outcome['loom_url']}\n"
                elif isinstance(outcome, str) and outcome.strip():
                    section += f"   Results: {outcome}\n"
            # Fallback: unmigrated legacy 'outcomes' string
            elif proj.get('outcomes') and isinstance(proj.get('outcomes'), str) and proj['outcomes'].strip():
                section += f"   Results: {proj['outcomes']}\n"
            
            if include_portfolio:
                section += f"   Live: {best_url}\n"
            
            feedback_url = proj.get('client_feedback_url')
            if include_feedback and feedback_url and feedback_url.strip():
                section += f"   Feedback: {feedback_url}\n"
            section += "\n"

        return section if added > 0 else "PROJECTS: No projects with portfolio links found."

    def build_improvement_prompt(self, current: str, feedback: str, aspect: str) -> str:
        """Build prompt to improve existing proposal."""
        improvements = {
            "tone": "Adjust tone to be more engaging",
            "length": "Adjust length while keeping key info",
            "details": "Add relevant details and examples",
            "structure": "Reorganize for better flow",
            "impact": "Make more persuasive and compelling"
        }
        return f"""
Improve this proposal based on feedback.

CURRENT: {current}

FEEDBACK: {feedback}

FOCUS: {improvements.get(aspect, improvements['impact'])}

Provide improved version maintaining professional tone.
"""

    def score_proposal_quality(
        self,
        proposal_text: str,
        job_data: Dict[str, Any],
        references: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Score proposal quality."""
        score = 0.0
        feedback = []
        text_lower = proposal_text.lower()
        word_count = len(proposal_text.split())
        
        # Word count (target 200-350)
        if 200 <= word_count <= 350:
            score += 0.25
        elif 150 < word_count < 400:
            score += 0.15
            feedback.append(f"Word count {word_count} - ideal is 200-350")
        else:
            feedback.append(f"Word count {word_count} - should be 200-350")

        # No AI language
        if not any(p in text_lower for p in ["as an ai", "i'm an ai", "as a language model"]):
            score += 0.2
        else:
            feedback.append("Remove AI language")

        # Conversational
        if ("i've" in text_lower or "i have" in text_lower) and "your" in text_lower:
            score += 0.15
        else:
            feedback.append("Add conversational tone - contractions, personal touch")

        # Problem acknowledgment
        if any(w in text_lower for w in ["deal with", "challenge", "problem", "looking for"]):
            score += 0.15

        # Project references
        refs = len(references.get("projects_referenced", []))
        score += min(refs / 3, 1.0) * 0.15

        # Portfolio/feedback links
        links = len(references.get("portfolio_links_used", [])) + len(references.get("feedback_urls_cited", []))
        score += min(links / 3, 1.0) * 0.1

        return {
            "overall_score": round(min(score, 1.0), 2),
            "word_count": word_count,
            "feedback": feedback,
            "is_short_human_winning": score >= 0.8 and 200 <= word_count <= 350
        }

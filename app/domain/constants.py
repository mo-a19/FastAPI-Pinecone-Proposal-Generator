"""
Centralized Constants for Upwork Proposal Generator

SINGLE SOURCE OF TRUTH for all keyword lists and mappings.
No duplication - all modules should import from here.

Created: Sprint 1 - Consolidation Phase
"""

from typing import Dict, List, Set
from enum import Enum


# =============================================================================
# SHARED ENUMS
# =============================================================================

class ImportSource(str, Enum):
    """Where profile data was imported from."""
    MANUAL = "manual"
    UPWORK = "upwork"
    LINKEDIN = "linkedin"  # Future support


# =============================================================================
# PLATFORM DETECTION KEYWORDS
# =============================================================================
# Platform keywords for smart detection in job descriptions
# CRITICAL: More specific keywords should come first in each list

PLATFORM_KEYWORDS: Dict[str, List[str]] = {
    "wordpress": [
        "wordpress", "wp-admin", "elementor", "divi", "theme", "plugin", 
        "gutenberg", "acf", "wp theme", "wp plugin", "geo directory", 
        "geodirectory"
    ],
    "shopify": [
        "shopify", "shopify theme", "shopify app", "shopify store", 
        "liquid", "shopify plus"
    ],
    "woocommerce": [
        "woocommerce", "woo commerce", "woo membership", "woomembership", 
        "woo subscription", "woo-"
    ],
    "wix": ["wix", "wix site", "wix website", "wix editor"],
    "webflow": ["webflow"],
    "squarespace": ["squarespace"],
    "magento": ["magento", "adobe commerce"],
    "drupal": ["drupal"],
    "joomla": ["joomla"],
    "react": [
        "react", "reactjs", "react.js", "next.js", "nextjs", "react native"
    ],
    "vue": ["vue", "vuejs", "vue.js", "nuxt"],
    "angular": ["angular", "angularjs"],
    # NOTE: Removed html/css as they're too generic
    "html_css": ["static site", "landing page"],
}


# =============================================================================
# AI/ML DETECTION KEYWORDS
# =============================================================================
# Comprehensive list of AI/ML keywords for detecting AI-focused jobs
# These take precedence over platform detection when found
# MERGED from retrieval_pipeline.py AND hook_strategy.py

AI_ML_KEYWORDS: List[str] = [
    # LLM providers & models
    "openai", "gpt", "gpt-4", "gpt-3", "chatgpt", "claude", "anthropic", 
    "llm", "large language model", "deepseek",
    
    # AI concepts
    "ai model", "ai api", "ai integration", "machine learning", "deep learning", 
    "neural network", "ai-powered", "ai automation", "ai agent", "ai assistant",
    "fine-tuning", "prompt engineering",
    
    # RAG & Vector DBs
    "langchain", "llamaindex", "rag", "retrieval augmented", "embedding", 
    "vector database", "pinecone", "chromadb", "weaviate",
    
    # ML frameworks
    "huggingface", "transformers", "pytorch", "tensorflow",
    
    # Computer vision & NLP
    "computer vision", "ocr", "nlp", "natural language processing",
    
    # Content generation
    "text generation", "content generation", "automated content",
    
    # Document processing
    "document processing", "pdf parsing", "text extraction", "document analyzer",
    "unstructured", "llamaparse", "docling",
    
    # Image generation
    "stability ai", "dall-e", "midjourney", "image generation",
    
    # Automation platforms
    "n8n", "make.com", "zapier automation",
    
    # Chatbots
    "chatbot", "conversational ai",
]


# =============================================================================
# INDUSTRY DETECTION KEYWORDS
# =============================================================================
# Industry mapping for standardization
# NOTE: For complex cases (like "similar to TMZ"), use LLM-based detection
# IMPORTANT: Avoid short/generic words that match unintended contexts

INDUSTRY_KEYWORDS: Dict[str, List[str]] = {
    "saas": [
        "saas", "software as a service", "cloud platform", 
        "subscription platform", "web platform", "dashboard app"
    ],
    "e-commerce": [
        "e-commerce", "ecommerce", "shopify", "woo commerce", "online store", 
        "products", "cart", "checkout", "shop"
    ],
    "healthcare": [
        "healthcare", "health care", "medical", "hospital", "clinic", 
        "telemedicine", "patient portal", "doctor"
    ],
    "finance": [
        "finance", "financial", "banking", "fintech", "crypto", 
        "investment", "trading platform"
    ],
    "education": [
        "education", "edtech", "learning platform", "course platform", 
        "university", "school", "student", "lms"
    ],
    "real_estate": [
        "real estate", "realestate", "property listing", "rental property", 
        "real estate agent", "broker"
    ],
    "manufacturing": [
        "manufacturing", "factory", "industrial", "logistics", 
        "warehouse", "supply chain"
    ],
    "travel": [
        "travel", "tourism", "booking", "hotel", "flight", "vacation", "resort"
    ],
    "social": [
        "social network", "social platform", "community", "forum", 
        "members area", "social media marketing"
    ],
    # ENHANCED: Media industry with brand references and contextual keywords
    # NOTE: Use specific terms, avoid generic words like "press" (matches WordPress)
    "media": [
        "media company", "entertainment", "streaming", "video platform", 
        "podcast", "news site", "magazine", "celebrity", "gossip", 
        "journalism", "editorial content", "content site", "blog network",
        "tmz", "justjared", "wwd", "variety", "hollywood reporter", 
        "entertainment news", "media outlet", "publishing company", 
        "news articles", "newsroom", "breaking news"
    ],
    "technology": [
        "technology", "tech", "software", "it services", "consulting", 
        "development"
    ],
    "professional_services": [
        "consulting", "agency", "b2b", "services", "professional"
    ],
    "non_profit": [
        "non-profit", "nonprofit", "charity", "ngo", "foundation", "donation"
    ],
}


# =============================================================================
# BRAND TO INDUSTRY MAPPING
# =============================================================================
# When someone says "like X", we can infer the industry

BRAND_INDUSTRY_MAP: Dict[str, str] = {
    # Media/Entertainment brands
    "tmz": "media", "justjared": "media", "wwd": "media", "variety": "media",
    "buzzfeed": "media", "huffpost": "media", "cnn": "media", "bbc": "media",
    "techcrunch": "media", "mashable": "media", "verge": "media", "engadget": "media",
    "eonline": "media", "people": "media", "us weekly": "media", 
    "entertainment weekly": "media",
    # E-commerce brands
    "amazon": "e-commerce", "shopify": "e-commerce", "etsy": "e-commerce", 
    "ebay": "e-commerce",
    # SaaS brands  
    "salesforce": "saas", "hubspot": "saas", "slack": "saas", "notion": "saas",
    # Social brands
    "facebook": "social", "twitter": "social", "linkedin": "social", 
    "reddit": "social",
}


# =============================================================================
# PAIN POINT INDICATORS
# =============================================================================
# Pain point keywords that indicate client frustration/urgency

PAIN_POINT_INDICATORS: Dict[str, List[str]] = {
    "frustration": [
        "frustrated", "struggling", "can't figure out", "doesn't work", 
        "broken", "not working", "issues", "problems", "nightmare", 
        "headache", "stuck"
    ],
    "urgency": [
        "urgent", "asap", "immediately", "deadline", "time-sensitive", 
        "need quickly", "fast turnaround", "rush", "critical", "today", 
        "right now", "this week", "costing us", "every hour", "emergency"
    ],
    "previous_failure": [
        "previous developer", "last freelancer", "didn't work out", 
        "need someone new", "past experience", "tried before", 
        "went mia", "disappeared", "ghosted"
    ],
    "business_impact": [
        "losing sales", "losing customers", "revenue", "customers complaining", 
        "bad reviews", "conversion", "cart abandonment", "costing us money", 
        "affecting business"
    ],
    "complexity": [
        "complex", "complicated", "difficult", "challenging", "advanced", 
        "sophisticated"
    ],
    "growth": [
        "scaling", "growing", "expansion", "more traffic", "increased demand", 
        "outgrown"
    ]
}


# =============================================================================
# URGENCY PATTERNS
# =============================================================================
# Urgency level detection patterns (5 = most urgent, 1 = least urgent)

URGENCY_PATTERNS: Dict[int, List[str]] = {
    5: [
        "emergency", "site down", "not working", "broken", "losing sales", 
        "every hour", "critical", "asap today"
    ],
    4: ["urgent", "asap", "immediately", "rush"],
    3: [
        "soon", "this week", "deadline", "time-sensitive", "quickly", "fast"
    ],
    2: ["when you can", "flexible timeline"],
    1: [
        "no rush", "exploring options", "considering", "thinking about", 
        "eventually", "nothing urgent", "not urgent"
    ],
}


# =============================================================================
# COMPLEXITY INDICATORS
# =============================================================================
# Complexity factors for job classification

COMPLEXITY_INDICATORS: Dict[str, List[str]] = {
    "high": [
        "machine learning", "ai", "blockchain", "real-time", "high volume", 
        "complex integration", "multi-tenant", "microservices", "distributed"
    ],
    "medium": [
        "api", "database", "authentication", "payment", "integration", 
        "mobile responsive"
    ],
    "low": [
        "landing page", "blog", "portfolio", "static", "basic crud"
    ],
}

COMPLEXITY_LEVELS: Dict[str, int] = {"low": 1, "medium": 2, "high": 3}


# =============================================================================
# CLIENT INTENT KEYWORDS
# =============================================================================
# What the client ACTUALLY wants done (not just platform detection)
# CRITICAL for matching jobs by actual requirement

CLIENT_INTENT_KEYWORDS: Dict[str, List[str]] = {
    # Migration/Transfer intents
    "content_migration": [
        "migrate", "migration", "transfer content", "move content", 
        "import content", "export content", "convert", "transition", 
        "switch from", "move from", "substack", "mailchimp import", 
        "medium import"
    ],
    "platform_switch": [
        "switch to wordpress", "move to shopify", "migrate to", 
        "convert to wordpress", "rebuild on", "recreate on"
    ],
    
    # Membership/Subscription intents  
    "membership_setup": [
        "membership", "woomembership", "subscription site", "paid content", 
        "paywall", "member area", "paid subscriber", "recurring payment", 
        "member pages", "restrict content", "premium content"
    ],
    "newsletter_email": [
        "newsletter", "email list", "mailing list", "email marketing", 
        "mailchimp", "convertkit", "email subscribers", "email campaign", 
        "email automation"
    ],
    
    # E-commerce intents
    "store_setup": [
        "online store", "e-commerce store", "ecommerce store", "sell products", 
        "product listing", "shopping cart", "checkout page"
    ],
    "payment_integration": [
        "payment gateway", "stripe integration", "paypal integration", 
        "woocommerce payments", "checkout integration", "buy button", 
        "accept payments"
    ],
    
    # Performance/Optimization intents
    "speed_optimization": [
        "speed up", "pagespeed", "core web vitals", "performance optimization", 
        "load time", "site optimization", "gtmetrix", "lighthouse score", 
        "website slow", "slow loading"
    ],
    "seo_optimization": [
        "seo optimization", "search engine optimization", "google ranking", 
        "keyword optimization", "organic traffic", "meta tags", "on-page seo"
    ],
    
    # Design/Development intents
    "website_redesign": [
        "redesign website", "restyle website", "makeover", "new look", 
        "refresh design", "modernize website", "update design", 
        "update my existing", "more professional", "unique layout", 
        "similar to", "like tmz", "like variety", "like justjared",
        "redesign", "professional look", "brand aligned", "brand consistency"
    ],
    "new_website": [
        "build website", "create website", "new website", "from scratch", 
        "brand new site", "develop website"
    ],
    "bug_fixes": [
        "fix bug", "fix issue", "broken", "not working", "website issue", 
        "website problem", "fix error", "repair"
    ],
    "feature_addition": [
        "add feature", "new feature", "add functionality", "integrate", 
        "custom feature", "enhancement", "extend"
    ],
    
    # Content/Blog intents
    "content_management": [
        "blog posts", "articles", "content management", "cms setup", 
        "publish content", "editorial", "written articles", "add articles", 
        "easy to add"
    ],
    "form_setup": [
        "contact form", "web form", "form submission", "lead capture form", 
        "cf7", "gravity forms", "formidable forms", "form plugin"
    ],
    
    # RSS/Content Integration intents
    "rss_integration": [
        "rss integration", "rss feed", "youtube integration", "pull content", 
        "content syndication", "auto import", "feed integration", "youtube feed",
        "youtube content", "video feed", "content aggregation", "auto-publish"
    ],
}


# =============================================================================
# EMPATHY RESPONSES
# =============================================================================
# Pre-written empathy responses for different pain points

EMPATHY_RESPONSES: Dict[str, List[str]] = {
    "frustration": [
        "I know how frustrating that can be",
        "That sounds really annoying to deal with", 
        "I get it - that's a common pain point",
        "Been there, totally understand the headache"
    ],
    "urgency": [
        "I understand you need this handled quickly - I'm available now",
        "Time pressure is real - I can start immediately",
        "No worries, I can prioritize this and jump on it today",
        "I know every hour counts - let's fix this ASAP",
        "I can see this is urgent - I'm free to start right now"
    ],
    "previous_failure": [
        "Sorry to hear the last experience didn't work out",
        "I understand the hesitation after a bad experience",
        "Let me show you a different approach"
    ],
    "business_impact": [
        "Losing sales is never fun - let's fix that",
        "I know how much that impacts your bottom line",
        "Revenue matters, so let's get this working ASAP"
    ],
    "complexity": [
        "I love a good technical challenge",
        "This is exactly the kind of project I specialize in",
        "Complex projects are where I do my best work"
    ],
    "growth": [
        "Congrats on the growth - exciting problem to have!",
        "Scaling challenges are a good sign",
        "Love helping businesses level up"
    ]
}


# =============================================================================
# URGENCY TIMELINE PROMISES
# =============================================================================
# Faster promises for urgent jobs

URGENCY_TIMELINE_PROMISES: Dict[str, str] = {
    "critical": "I can start right now and have this fixed within hours",
    "today": "I'm available today - can jump on this immediately",
    "asap": "Can get started today and have it sorted by tomorrow",
    "this_week": "I can prioritize this and wrap it up in 2-3 days",
    # NOTE: "standard" returns None - use normal timeline
}


# =============================================================================
# PROPOSAL STYLE INSTRUCTIONS
# =============================================================================
# Style-specific instructions for proposal generation
# Used by PromptEngine to guide AI response formatting

STYLE_INSTRUCTIONS: Dict[str, str] = {
    "professional": """
Your response should be:
- Formal yet warm
- Structured with clear sections
- Emphasis on expertise and credentials
- Professional formatting with proper grammar
- Bullet points for key benefits
- Clear Call-to-Action at end
""",
    "casual": """
Your response should be:
- Conversational and approachable
- Personal and relatable
- Friendly tone without losing professionalism
- Shorter paragraphs
- Genuine enthusiasm visible
- Direct communication style
""",
    "technical": """
Your response should be:
- Deep technical details
- Architecture discussions
- Technology stack explanation
- Performance metrics and benchmarks
- Technical trade-offs explained
- Implementation approach detailed
""",
    "creative": """
Your response should be:
- Engaging and memorable
- Unique positioning
- Creative problem-solving emphasis
- Innovation highlighted
- Visual/metaphorical language
- Storytelling approach
""",
    "data_driven": """
Your response should be:
- Backed by metrics and data
- Success rate references
- ROI focused
- Performance guarantees
- Historical data cited
- Quantifiable results promised
""",
}


# =============================================================================
# PROPOSAL TONE INSTRUCTIONS
# =============================================================================
# Tone-specific guidance for proposal generation

TONE_INSTRUCTIONS: Dict[str, str] = {
    "confident": "Convey strong confidence in abilities. Use words like 'will deliver', 'proven', 'guaranteed'.",
    "humble": "Show humility while demonstrating competence. Use words like 'eager to', 'honored to', 'privileged'.",
    "enthusiastic": "Show genuine excitement. Use energetic language, exclamation marks, positive energy.",
    "analytical": "Focus on facts and analysis. Logical structure, data references, systematic approach.",
    "friendly": "Warm and approachable. Show personality, use 'we', collaborative language, genuine interest.",
}


# =============================================================================
# PROMPT TEMPLATES - SINGLE SOURCE OF TRUTH
# =============================================================================
# Consolidated prompt rules - defined ONCE, used everywhere

HOOK_STRATEGIES: List[str] = [
    'PROOF-LED HOOK: "I built [similar project] - [portfolio_url] - same [tech/scope] you need."',
    'UNDERSTANDING HOOK: "[Specific detail from job] - I\'ve done exactly this for [portfolio_url]."',
    'EMPATHY HOOK: "Sounds like [their situation] - I fixed this for [client]: [portfolio_url]"',
    'DIRECT MATCH: "[Tech they need] + [deliverable they want] = my specialty. Recent example: [portfolio_url]"',
    'RESULT HOOK: "Took [similar project] from [problem] to [solution] - here\'s the result: [portfolio_url]"',
]

# HOOKS TO AVOID - These often misfire
BAD_HOOK_PATTERNS: List[str] = [
    'AVAILABILITY without urgency - Don\'t lead with "I can start right now" unless client is URGENT',
    'TIMEZONE without requirement - Don\'t mention timezone unless job post requires it',
    'ONE-TIME/ONGOING - Don\'t categorize engagement unless client specifies',
    'RATES/ESTIMATES - Don\'t offer pricing unless explicitly asked',
]

STALE_OPENINGS: List[str] = [
    "I see you're dealing with...",
    "I came across your job post...",
    "I'm excited to help...",
    "I have X years of experience...",
    "Curious - have you...",  # Generic technical questions NOT relevant to job
    "Quick question about...",  # Questions about things client didn't ask
    "Have you considered...",  # Unsolicited advice
    "Have you optimized...",  # Assuming problems that weren't mentioned
]

ANTI_HALLUCINATION_RULES: str = """
🚨 ANTI-HALLUCINATION RULES (CRITICAL):
1. ONLY reference projects listed in "VERIFIED PAST PROJECTS" section
2. If NO verified projects: Focus on APPROACH, not past work
3. NEVER invent company names, deliverables, or outcomes
4. Match client's NEED to project DELIVERABLES - only include RELEVANT portfolio
5. Use ONLY portfolio URLs provided - if none, include ZERO links
6. NEVER suggest technologies/approaches the client didn't ask for
7. NEVER open with questions about things not mentioned in the job post
8. ONLY mention timezone/availability IF client explicitly requires it
9. ONLY mention "one-time" or "ongoing" IF client specifies engagement type
10. NEVER give hourly rates, timelines, or estimates UNLESS client asks for them
11. If client provides a document or any materials to review, DON'T say "I'll review it" - assume you HAVE reviewed it
12. NEVER use self-deprecating language like "haven't done this exact scope" or "although not identical"
"""

PROPOSAL_FORMAT_RULES: str = """
CRITICAL FORMAT RULES:
1. Use PLAIN URLs (https://example.com) - NOT markdown [text](url)
2. ABSOLUTELY NO MARKDOWN - no **bold**, no *italic*, no # headers, no - bullet points with bold
3. Target: 200-350 words MAXIMUM
4. Sound human - casual, genuine, conversational
5. PLATFORM MATCH: WordPress job = WordPress examples ONLY
6. Write in PLAIN TEXT paragraphs - no structured sections like "**Approach:**" or "- **Fit:**"
7. NO section headers or labels - just natural flowing paragraphs
"""

PROPOSAL_STRUCTURE: str = """
SUCCESS PATTERN (HOOK→PROOF→APPROACH→CTA):
1. HOOK (1-2 sentences): Show you understand THEIR specific need + include RELEVANT portfolio link
2. PROOF (1-2 sentences): Reference similar work WITH portfolio URL - ONLY if truly relevant
3. APPROACH (2-3 sentences): Your specific solution for THEIR situation
4. CTA (1 sentence): Casual "Happy to chat" or "Let me know"

CRITICAL - ONLY MENTION IF CLIENT ASKED:
- Timezone/availability: ONLY if job post mentions timezone requirements
- Engagement type (ongoing/one-time): ONLY if job post specifies
- Rates/timeline: ONLY if client explicitly requests estimates
- Your availability: ONLY if client mentions urgency or timeline
"""

CONVERSATIONAL_EXAMPLES: List[str] = [
    "I have built something similar",
    "I've tackled this exact issue before",
    "Here's what worked for a similar client",
    "Happy to hop on a quick call if you want to discuss",
]

FORBIDDEN_PHRASES: List[str] = [
    # AI/formal language
    "As an AI", "I'm an AI", "I would be delighted", "I am eager to",
    "Best regards", "Sincerely", "Looking forward to hearing from you",
    # Self-deprecating/weak language - NEVER use these
    "While I haven't", "although not identical", "haven't tackled this exact",
    "less experience with", "new to this", "first time", "learning",
    # Unsolicited assumptions - don't mention unless client asked
    "one-time project", "I'm available in the PST", "I'm available in your timezone",
    "Hourly Rate:", "Estimated Hours:", "Total Timeframe:",
    # Markdown formatting - FORBIDDEN
    "**", "*", "- **", "##", "###",
    # Promising to do what should be done already
    "I'll begin by thoroughly reviewing", "I will review", "after reviewing",
]

SYSTEM_ROLES: Dict[str, str] = {
    "professional": "You are an expert proposal writer with 10+ years winning high-value contracts.",
    "casual": "You are a friendly freelancer who writes approachable, personable proposals.",
    "technical": "You are a senior technical architect demonstrating deep expertise.",
    "creative": "You are a creative innovator who writes memorable, unique proposals.",
    "data_driven": "You are a data analyst who writes proposals backed by metrics and ROI.",
}


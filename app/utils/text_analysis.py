"""
Consolidated Text Analysis Utilities

Single source of truth for:
- Urgency detection
- Pain point extraction
- Tone analysis

Previously duplicated in prompt_engine.py and hook_strategy.py.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from app.domain.constants import PAIN_POINT_INDICATORS, URGENCY_PATTERNS

logger = logging.getLogger(__name__)


@dataclass
class UrgencyResult:
    """Urgency detection result with both numeric level and string label."""
    level: int          # 1-5 scale
    label: str          # 'critical', 'today', 'asap', 'this_week', 'standard'
    matched_keyword: Optional[str] = None


# Mapping between urgency levels and labels
URGENCY_LEVEL_MAP = {
    5: "critical",
    4: "today", 
    3: "asap",
    2: "this_week",
    1: "standard"
}

URGENCY_LABEL_MAP = {v: k for k, v in URGENCY_LEVEL_MAP.items()}

# No rush patterns - override other signals
NO_RUSH_PATTERNS = [
    "no rush", "not urgent", "nothing urgent", "no hurry", 
    "take your time", "flexible timeline", "whenever you can"
]

# Critical urgency keywords (level 5)
CRITICAL_KEYWORDS = [
    "emergency", "critical", "site down", "broken", "not working", 
    "right now", "within hours", "losing sales", "costing us", "every hour"
]

# Today urgency keywords (level 4)
TODAY_KEYWORDS = ["today", "immediately", "asap", "urgent help"]

# ASAP urgency keywords (level 3)
ASAP_KEYWORDS = ["asap", "urgent", "rush", "quick turnaround", "fast", "need quickly"]

# This week urgency keywords (level 2)
WEEK_KEYWORDS = ["this week", "deadline", "time-sensitive"]


def detect_urgency(text: str) -> UrgencyResult:
    """
    Detect urgency level from text.
    
    Combines logic from:
    - prompt_engine.py:detect_urgency_level()
    - hook_strategy.py:_detect_urgency()
    
    Args:
        text: Job description, title, or combined text
        
    Returns:
        UrgencyResult with level (1-5), label, and matched keyword
    """
    text_lower = text.lower()
    
    # Check for explicit "no rush" first - these override other signals
    for pattern in NO_RUSH_PATTERNS:
        if pattern in text_lower:
            return UrgencyResult(level=1, label="standard", matched_keyword=pattern)
    
    # Check critical (level 5)
    for kw in CRITICAL_KEYWORDS:
        if kw in text_lower:
            return UrgencyResult(level=5, label="critical", matched_keyword=kw)
    
    # Check today (level 4)
    for kw in TODAY_KEYWORDS:
        if kw in text_lower:
            return UrgencyResult(level=4, label="today", matched_keyword=kw)
    
    # Check ASAP (level 3)
    for kw in ASAP_KEYWORDS:
        if kw in text_lower:
            return UrgencyResult(level=3, label="asap", matched_keyword=kw)
    
    # Check this week (level 2)
    for kw in WEEK_KEYWORDS:
        if kw in text_lower:
            return UrgencyResult(level=2, label="this_week", matched_keyword=kw)
    
    # Also check URGENCY_PATTERNS from constants (for any we missed)
    for level, patterns in sorted(URGENCY_PATTERNS.items(), reverse=True):
        for pattern in patterns:
            if pattern in text_lower:
                label = URGENCY_LEVEL_MAP.get(level, "standard")
                return UrgencyResult(level=level, label=label, matched_keyword=pattern)
    
    # Default: standard priority
    return UrgencyResult(level=1, label="standard")


def detect_urgency_level(job_description: str, job_title: str = "") -> str:
    """
    Legacy compatibility wrapper - returns string label only.
    
    Use detect_urgency() for full result with both level and label.
    """
    text = f"{job_title} {job_description}"
    result = detect_urgency(text)
    return result.label


def detect_urgency_score(text: str) -> int:
    """
    Legacy compatibility wrapper - returns numeric level only.
    
    Use detect_urgency() for full result with both level and label.
    """
    result = detect_urgency(text)
    return result.level


def extract_pain_points(job_description: str) -> Dict[str, List[str]]:
    """
    Extract specific pain points and frustrations from job description.
    
    Combines logic from:
    - prompt_engine.py:extract_pain_points() - category-based with sentences
    - hook_strategy.py:_extract_pain_points() - regex pattern matching
    
    Args:
        job_description: Full job description text
        
    Returns:
        Dict with pain point categories and matching phrases/sentences
    """
    job_desc_lower = job_description.lower()
    found_pain_points: Dict[str, List[str]] = {}
    
    # Method 1: Use PAIN_POINT_INDICATORS from constants (categorical)
    for category, keywords in PAIN_POINT_INDICATORS.items():
        matches = []
        for keyword in keywords:
            if keyword in job_desc_lower:
                # Extract the sentence containing this keyword for context
                sentences = job_description.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower() and sentence.strip():
                        matches.append(sentence.strip())
                        break
        if matches:
            found_pain_points[category] = matches[:2]  # Max 2 per category
    
    # Method 2: Regex pattern matching for specific structures
    regex_patterns = [
        (r"(\w+)\s+(?:is|are)\s+(?:broken|not working|slow)", "technical_issue"),
        (r"(?:losing|lost)\s+([\w\s]+?)(?:\.|,|$)", "business_loss"),
        (r"(?:can't|cannot|unable to)\s+([\w\s]+?)(?:\.|,|$)", "blocker"),
        (r"(?:frustrated|struggling|stuck)\s+(?:with\s+)?([\w\s]+?)(?:\.|,|$)", "frustration"),
        (r"(?:previous\s+)?(?:developer|freelancer)\s+(?:didn't|failed|couldn't)", "previous_failure"),
    ]
    
    for pattern, category in regex_patterns:
        if category not in found_pain_points:
            found_pain_points[category] = []
        
        matches = re.findall(pattern, job_description, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            match_str = match.strip()
            if match_str and match_str not in found_pain_points[category]:
                found_pain_points[category].append(match_str)
    
    # Clean up empty categories
    return {k: v for k, v in found_pain_points.items() if v}


def extract_pain_points_simple(text: str) -> List[str]:
    """
    Legacy compatibility wrapper - returns flat list.
    
    Use extract_pain_points() for categorized results.
    """
    result = extract_pain_points(text)
    pain_points = []
    for category, matches in result.items():
        for match in matches[:1]:  # Take first match per category
            pain_points.append(f"{category}: {match[:50]}")  # Truncate long matches
    return pain_points[:3]  # Limit to top 3


def extract_specific_details(job_data: Dict[str, Any]) -> List[str]:
    """
    Extract specific numbers, tools, metrics from job.
    
    Moved from hook_strategy.py:_extract_specific_details()
    
    Args:
        job_data: Job data dictionary
        
    Returns:
        List of specific details (numbers, tools, metrics)
    """
    details = []
    text = f"{job_data.get('job_title', '')} {job_data.get('job_description', '')}"
    
    # Numbers with context
    number_patterns = re.findall(
        r'(\d+[\+]?\s*(?:subscribers?|products?|pages?|seconds?|ms|visitors?|items?|%|hours?|days?|users?))',
        text, re.IGNORECASE
    )
    details.extend(number_patterns[:3])
    
    # Tool/platform mentions
    tools = [
        "wordpress", "shopify", "woocommerce", "elementor", "substack", "mailchimp", 
        "stripe", "paypal", "gempages", "liquid", "php", "javascript", "react",
        "python", "fastapi", "django", "flask", "node", "nextjs", "vue"
    ]
    for tool in tools:
        if tool in text.lower():
            details.append(tool.title())
    
    # Speed metrics
    speed_patterns = re.findall(
        r'(\d+[\.\d]*\s*(?:seconds?|ms|s)\s*(?:load|time)?)', 
        text, re.IGNORECASE
    )
    details.extend(speed_patterns[:2])
    
    return details[:5]  # Limit to 5 most important


def extract_tone_words(text: str) -> Dict[str, List[str]]:
    """
    Extract emotional/tone words from text.
    
    Moved from hook_strategy.py:_extract_tone_words()
    
    Returns:
        Dict with 'positive', 'negative', 'urgent' categories
    """
    text_lower = text.lower()
    result = {"positive": [], "negative": [], "urgent": []}
    
    positive = ["excited", "love", "great", "awesome", "amazing", "perfect", "excellent", "happy", "thrilled"]
    negative = ["frustrated", "angry", "disappointed", "struggling", "nightmare", "horrible", "terrible", "annoyed"]
    urgent = ["urgent", "asap", "immediately", "critical", "emergency", "rush", "quickly", "fast"]
    
    for word in positive:
        if word in text_lower:
            result["positive"].append(word)
    
    for word in negative:
        if word in text_lower:
            result["negative"].append(word)
    
    for word in urgent:
        if word in text_lower:
            result["urgent"].append(word)
    
    return result


def analyze_job_text(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive text analysis for a job posting.
    
    Combines all analysis functions into a single call.
    
    Args:
        job_data: Job data dictionary
        
    Returns:
        Dict with urgency, pain_points, details, tone
    """
    job_desc = job_data.get("job_description", "")
    job_title = job_data.get("job_title", "")
    combined_text = f"{job_title} {job_desc}"
    
    urgency = detect_urgency(combined_text)
    
    return {
        "urgency": {
            "level": urgency.level,
            "label": urgency.label,
            "matched_keyword": urgency.matched_keyword
        },
        "pain_points": extract_pain_points(job_desc),
        "specific_details": extract_specific_details(job_data),
        "tone": extract_tone_words(combined_text)
    }

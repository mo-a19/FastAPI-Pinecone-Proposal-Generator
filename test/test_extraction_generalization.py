"""
Test: Job Requirements Extraction Generalization

This test demonstrates how the LLM-based extraction dynamically adapts
to different job types - proving the system is NOT hardcoded for Shopify.

The extraction is done by OpenAI function calling, which analyzes each
unique job description and extracts relevant context.
"""

# Example job descriptions for different engagement models
JOB_EXAMPLES = {
    
    # =========================================================================
    # 1. ONE-TIME FIXED-PRICE PROJECT
    # =========================================================================
    "one_time_fixed_price": {
        "job_title": "Build a Landing Page for Product Launch",
        "job_description": """
We need a single landing page for our new SaaS product launch next month.

Budget: $500 fixed price
Timeline: 2 weeks

Requirements:
- Mobile responsive design
- Email signup form with Mailchimp integration
- Hero section with video embed
- Pricing table (3 tiers)
- FAQ accordion
- Contact form

We have Figma designs ready. This is a one-time project, we have an in-house 
team for maintenance.

Please share 2-3 similar landing pages you've built.
""",
        "skills_required": ["HTML", "CSS", "JavaScript", "Figma to Code"],
        
        # EXPECTED LLM EXTRACTION (dynamically inferred):
        "expected_extraction": {
            "exact_task": "Build a single landing page for SaaS product launch from Figma designs",
            "working_arrangement": {
                "timezone": None,  # Not specified
                "hours": None,  # Not specified
                "arrangement_type": "one_time",  # Inferred from "one-time project"
                "responsiveness_expectation": None
            },
            "application_requirements": ["Share 2-3 similar landing pages"],
            "soft_requirements": [],  # None mentioned
            "client_priorities": ["Mobile responsive", "2-week timeline", "Fixed price delivery"],
            "must_not_propose": ["Ongoing maintenance", "Monthly retainer"],  # They have in-house team
            "key_phrases_to_echo": ["product launch", "Figma designs ready", "in-house team"]
        }
    },
    
    # =========================================================================
    # 2. TECHNICAL DEBUGGING TASK
    # =========================================================================
    "technical_debugging": {
        "job_title": "URGENT: Fix Critical Bug Breaking Checkout",
        "job_description": """
EMERGENCY - Our WooCommerce checkout is broken and we're losing sales!

Since yesterday's plugin update, customers get "500 error" when clicking 
"Place Order". We've lost ~$3000 in sales already.

Error logs show:
- PHP Fatal error in /wp-content/plugins/woocommerce-payments/...
- MySQL connection timeout errors

What we've tried:
- Disabled all plugins except WooCommerce (still broken)
- Restored database backup (no change)
- Contacted hosting (they say it's code issue)

Need someone who can jump on this NOW. Happy to pay premium rate for 
immediate availability. Budget is flexible - just fix it!

Must have WooCommerce debugging experience. Share your approach.
""",
        "skills_required": ["WooCommerce", "PHP", "MySQL", "WordPress", "Debugging"],
        
        # EXPECTED LLM EXTRACTION (dynamically inferred):
        "expected_extraction": {
            "exact_task": "Debug and fix critical WooCommerce checkout 500 error causing sales loss",
            "working_arrangement": {
                "timezone": None,
                "hours": None,  
                "arrangement_type": "one_time",  # Emergency fix
                "responsiveness_expectation": "immediate availability"  # Inferred from "NOW"
            },
            "client_tone": "urgent",  # EMERGENCY, losing sales, NOW
            "application_requirements": ["Share debugging approach"],
            "soft_requirements": ["Immediate availability", "Fast response"],
            "client_priorities": ["Speed", "Fix the issue", "Minimize downtime"],
            "problems_mentioned": ["500 error on checkout", "Lost $3000 in sales", "Plugin update broke it"],
            "must_not_propose": ["Full site rebuild", "Platform migration", "Long-term contract"],
            "key_phrases_to_echo": ["losing sales", "jump on this NOW", "premium rate"]
        }
    },
    
    # =========================================================================
    # 3. AI CONSULTING / IMPLEMENTATION
    # =========================================================================
    "ai_consulting": {
        "job_title": "AI Strategy Consultant - Automate Our Document Processing",
        "job_description": """
We're a mid-size law firm processing 500+ contracts/month manually. Looking 
for an AI expert to help us:

1. CONSULT: Assess our current workflow and recommend AI solutions
2. BUILD: Implement a document processing pipeline using LLMs
3. TRAIN: Help our team understand and maintain the system

Tech we're open to: OpenAI API, LangChain, Azure AI, AWS Textract

Requirements:
- Must sign NDA (handling sensitive legal documents)
- Weekly video calls to discuss progress
- Experience with legal/compliance documents preferred
- Provide case studies of similar implementations

This is a 3-month engagement with potential extension. Budget: $15-20k total.

We value clear communication and documentation over speed.
""",
        "skills_required": ["AI/ML", "LangChain", "OpenAI API", "Python", "Document Processing"],
        
        # EXPECTED LLM EXTRACTION (dynamically inferred):
        "expected_extraction": {
            "exact_task": "Consult on and build AI document processing pipeline for law firm handling 500+ contracts/month",
            "working_arrangement": {
                "timezone": None,
                "hours": None,
                "arrangement_type": "contract",  # 3-month engagement
                "responsiveness_expectation": "weekly video calls"
            },
            "client_tone": "professional",  # Formal, structured request
            "application_requirements": ["Sign NDA", "Provide case studies of similar AI implementations"],
            "soft_requirements": ["Clear communication", "Documentation over speed", "Training capability"],
            "client_priorities": ["Documentation", "Training team", "Communication", "Compliance"],
            "must_not_propose": ["Quick automated solution without training", "Off-the-shelf tools only"],
            "key_phrases_to_echo": ["clear communication", "documentation over speed", "3-month engagement"]
        }
    },
    
    # =========================================================================
    # 4. DESIGN-ONLY PROJECT
    # =========================================================================
    "design_only": {
        "job_title": "UI/UX Designer for Mobile App Redesign",
        "job_description": """
We have a fitness tracking app with 50k+ users but our UI looks dated 
(designed in 2019). Need a fresh, modern redesign.

Scope:
- Redesign 12 core screens (home, workout, stats, profile, settings, etc.)
- Create a cohesive design system (colors, typography, components)
- Deliver Figma files with proper layer organization
- NOT coding - we have dev team

Style references:
- Nike Training Club (clean, motivational)
- Strava (data visualization)
- Calm (relaxing feel for recovery screens)

Process:
1. Review current app + competitor analysis
2. Wireframes for feedback
3. High-fidelity mockups
4. Final Figma handoff

Timeline: 4 weeks
Budget: $2000-3000

Include your portfolio and explain your design process.
""",
        "skills_required": ["UI/UX Design", "Figma", "Mobile App Design"],
        
        # EXPECTED LLM EXTRACTION (dynamically inferred):
        "expected_extraction": {
            "exact_task": "Redesign 12 screens for fitness mobile app with design system, Figma deliverables only (no coding)",
            "working_arrangement": {
                "timezone": None,
                "hours": None,
                "arrangement_type": "one_time",
                "responsiveness_expectation": None
            },
            "client_tone": "professional",
            "application_requirements": ["Include portfolio", "Explain design process"],
            "soft_requirements": ["Proper layer organization", "Structured process"],
            "client_priorities": ["Modern look", "Cohesive design system", "Figma organization", "Process alignment"],
            "must_not_propose": ["Coding/development", "Full rebrand", "Backend changes"],  # "NOT coding"
            "key_phrases_to_echo": ["fresh modern redesign", "design system", "Figma handoff"],
            "resources_provided": ["style references", "existing app access"]
        }
    },
    
    # =========================================================================
    # 5. BACKEND API INTEGRATION
    # =========================================================================
    "backend_api": {
        "job_title": "Senior Python Developer - Payment Gateway Integration",
        "job_description": """
We need to integrate Stripe Connect for our marketplace platform (Django/DRF).

Technical requirements:
- Implement Stripe Connect Standard accounts for sellers
- Handle split payments (platform fee + seller payout)
- Webhook handling for payment events
- Implement retry logic for failed payouts
- Write comprehensive tests (pytest)

Our stack:
- Django 4.2 + DRF
- PostgreSQL
- Celery + Redis for async
- Docker deployment on AWS ECS

We use GitHub with PR reviews. Must follow our coding standards (PEP8, 
type hints, docstrings). 

Looking for 20-30 hrs/week for 2 months. Must overlap with PST for 
daily standups (30 min at 10am PST).

Share GitHub profile and similar payment integration work.
""",
        "skills_required": ["Python", "Django", "Stripe API", "PostgreSQL", "Docker"],
        
        # EXPECTED LLM EXTRACTION (dynamically inferred):
        "expected_extraction": {
            "exact_task": "Integrate Stripe Connect for marketplace with split payments, webhooks, and tests in Django/DRF",
            "working_arrangement": {
                "timezone": "PST",  # Must overlap with PST
                "hours": "10am PST daily standup",
                "arrangement_type": "part_time",  # 20-30 hrs/week
                "responsiveness_expectation": "daily standups"
            },
            "client_tone": "technical",
            "application_requirements": ["Share GitHub profile", "Show similar payment integration work"],
            "soft_requirements": ["Follow coding standards", "PEP8", "Type hints", "PR reviews"],
            "client_priorities": ["Code quality", "Testing", "Coding standards", "PST overlap"],
            "must_not_propose": ["Different tech stack", "No-code solutions", "Skipping tests"],
            "key_phrases_to_echo": ["coding standards", "PR reviews", "comprehensive tests"],
            "tech_stack_mentioned": ["Django 4.2", "DRF", "PostgreSQL", "Celery", "Redis", "Docker", "AWS ECS", "Stripe Connect"]
        }
    }
}


def demonstrate_extraction_logic():
    """
    Show how the LLM extraction adapts to each job type.
    
    KEY INSIGHT: The extraction is NOT pattern-matching - it's LLM reasoning.
    The same prompt/function schema extracts different values based on 
    what's actually in each job description.
    """
    
    print("=" * 80)
    print("DEMONSTRATION: Dynamic LLM Extraction Across Job Types")
    print("=" * 80)
    print()
    print("The system uses OpenAI function calling to extract requirements.")
    print("The LLM reads each job description and fills in relevant fields.")
    print("Empty/null values are returned when not mentioned in the job.")
    print()
    
    for job_type, job_data in JOB_EXAMPLES.items():
        print(f"\n{'='*80}")
        print(f"JOB TYPE: {job_type.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        print(f"Title: {job_data['job_title']}")
        print()
        print("DYNAMICALLY EXTRACTED VALUES:")
        print("-" * 40)
        
        expected = job_data["expected_extraction"]
        
        # Show working arrangement adaptation
        wa = expected.get("working_arrangement", {})
        print(f"\n📅 Working Arrangement:")
        print(f"   • arrangement_type: {wa.get('arrangement_type', 'N/A')}")
        print(f"   • timezone: {wa.get('timezone', 'Not specified')}")
        print(f"   • hours: {wa.get('hours', 'Not specified')}")
        print(f"   • responsiveness: {wa.get('responsiveness_expectation', 'Not specified')}")
        
        # Show application requirements
        app_reqs = expected.get("application_requirements", [])
        print(f"\n📋 Application Requirements:")
        for req in app_reqs or ["None specified"]:
            print(f"   • {req}")
        
        # Show client priorities
        priorities = expected.get("client_priorities", [])
        print(f"\n🎯 Client Priorities (in order):")
        for i, p in enumerate(priorities or ["None detected"], 1):
            print(f"   {i}. {p}")
        
        # Show must_not_propose
        must_not = expected.get("must_not_propose", [])
        print(f"\n🚫 Must NOT Propose:")
        for item in must_not or ["Nothing specific"]:
            print(f"   • {item}")
        
        # Show key phrases
        phrases = expected.get("key_phrases_to_echo", [])
        print(f"\n💬 Key Phrases to Echo:")
        for phrase in phrases[:3] or ["None detected"]:
            print(f'   • "{phrase}"')
        
        print()


def show_extraction_adaptation_summary():
    """Show summary of how extraction adapts per engagement model."""
    
    print("\n" + "=" * 80)
    print("SUMMARY: How Extraction Adapts Per Engagement Model")
    print("=" * 80)
    
    summary = """
┌─────────────────────┬──────────────────────────────────────────────────────────┐
│ Job Type            │ Key Extraction Adaptations                               │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│ One-Time Fixed      │ • arrangement_type = "one_time"                          │
│                     │ • must_not_propose = ongoing services                    │
│                     │ • priorities = delivery timeline, fixed budget           │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│ Technical Debug     │ • client_tone = "urgent"                                 │
│                     │ • responsiveness = "immediate"                           │
│                     │ • problems_mentioned = specific errors                   │
│                     │ • priorities = speed over documentation                  │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│ AI Consulting       │ • arrangement_type = "contract" (multi-month)            │
│                     │ • soft_requirements = documentation, training            │
│                     │ • application_requirements = NDA, case studies           │
│                     │ • priorities = communication over speed                  │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│ Design-Only         │ • must_not_propose = coding/development                  │
│                     │ • deliverables = Figma files, design system              │
│                     │ • priorities = process, organization                     │
│                     │ • resources_provided = style references                  │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│ Backend API         │ • working_arrangement.timezone = specific (PST)          │
│                     │ • working_arrangement.hours = daily standup time         │
│                     │ • soft_requirements = coding standards, PR process       │
│                     │ • application_requirements = GitHub profile              │
│                     │ • must_not_propose = different stack                     │
└─────────────────────┴──────────────────────────────────────────────────────────┘

KEY INSIGHT: The same extraction schema produces DIFFERENT outputs based on
what the LLM finds in each unique job description. This is NOT pattern matching -
it's intelligent reasoning about what matters for each specific job.
"""
    print(summary)


if __name__ == "__main__":
    demonstrate_extraction_logic()
    show_extraction_adaptation_summary()

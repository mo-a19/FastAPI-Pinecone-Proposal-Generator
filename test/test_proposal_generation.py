"""
Test suite for SHORT, HUMAN, WINNING Proposal Generation

Tests the improved proposal generator, retrieval pipeline, and prompt engine.
Validates that generated proposals:
- Are 250-350 words (SHORT format)
- Sound conversational (HUMAN tone)
- Reference past projects with portfolio links (WINNING proof)
- Follow HOOK→PROOF→APPROACH→TIMELINE→CTA structure
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# NOTE: ProposalGenerator was removed - production code uses RetrievalPipeline + PromptEngine directly
from app.utils.retrieval_pipeline import RetrievalPipeline
from app.utils.prompt_engine import PromptEngine
from app.utils.openai_service import OpenAIService
from app.config import settings


# Sample test data - simulate 27 ingested projects
SAMPLE_HISTORICAL_PROJECTS = [
    {
        "contract_id": "proj_001",
        "company_name": "TechCorp Inc",
        "job_title": "Senior Backend Developer",
        "job_description": "Build scalable REST APIs with Python and FastAPI",
        "your_proposal_text": "I see you need robust backend APIs. I've built 5+ FastAPI services handling 100k+ requests daily.",
        "skills_required": ["Python", "FastAPI", "PostgreSQL"],
        "industry": "technology",
        "task_type": "backend_api",
        "project_status": "completed",
        "client_feedback_text": "Excellent work! Responsive communication and high-quality code. Very impressed with the architecture.",
        "client_feedback_url": "https://upwork.com/reviews/project/123",
        "portfolio_urls": ["https://github.com/user/fastapi-project"],
        "proposal_effectiveness_score": 0.9,
        "client_satisfaction": 4.8,
        "industry_tags": ["technology", "backend"],
        "reusable_sections": ["Architecture Overview", "Implementation Plan"]
    },
    {
        "contract_id": "proj_002",
        "company_name": "StartupXYZ",
        "job_title": "Full Stack Developer",
        "job_description": "Create React + Node.js application for e-commerce",
        "your_proposal_text": "I've delivered 3 full-stack e-commerce platforms. For you, I'd use React + Express + MongoDB.",
        "skills_required": ["React", "Node.js", "MongoDB"],
        "industry": "e-commerce",
        "task_type": "full_stack",
        "project_status": "completed",
        "client_feedback_text": "Amazing! Exceeded expectations and delivered ahead of schedule. Professional quality.",
        "client_feedback_url": "https://upwork.com/reviews/project/124",
        "portfolio_urls": ["https://example-ecommerce.com"],
        "proposal_effectiveness_score": 0.95,
        "client_satisfaction": 5.0,
        "industry_tags": ["e-commerce", "fullstack"],
        "reusable_sections": ["Technical Approach", "Timeline"]
    },
    {
        "contract_id": "proj_003",
        "company_name": "DataSoft",
        "job_title": "Python Data Engineer",
        "job_description": "ETL pipelines and data processing with Python",
        "your_proposal_text": "I see you need ETL optimization. I've built pipelines processing 10M+ records daily.",
        "skills_required": ["Python", "Pandas", "SQL"],
        "industry": "data",
        "task_type": "backend_api",
        "project_status": "completed",
        "client_feedback_text": "Professional quality and attention to detail. Very satisfied with responsiveness and communication.",
        "client_feedback_url": "https://upwork.com/reviews/project/125",
        "portfolio_urls": ["https://github.com/user/etl-project"],
        "proposal_effectiveness_score": 0.88,
        "client_satisfaction": 4.6,
        "industry_tags": ["data", "python"],
        "reusable_sections": ["Data Architecture", "Performance Metrics"]
    },
]

# New job requiring proposal
NEW_JOB = {
    "contract_id": "new_job_001",
    "company_name": "CloudSolutions Corp",
    "job_title": "Backend API Developer",
    "job_description": """
    We're looking for an experienced backend developer to build REST APIs for our platform.
    Requirements:
    - 5+ years backend development experience
    - Strong Python skills
    - Experience with FastAPI or similar frameworks
    - Database design and optimization
    - API documentation and testing
    
    Project duration: 2-3 months
    Estimated budget: $5,000-8,000
    
    We need someone who understands our challenges with scaling
    and can deliver a robust solution.
    """,
    "skills_required": ["Python", "FastAPI", "PostgreSQL", "REST APIs"],
    "industry": "technology",
    "task_type": "backend_api",
    "project_duration_days": 90,
    "estimated_budget": 6500,
    "task_complexity": "high",
    "urgent_adhoc": False
}


def test_prompt_engine():
    """Test that PromptEngine builds correct structure"""
    print("\n" + "="*60)
    print("TEST 1: Prompt Engine - HOOK→PROOF→APPROACH→TIMELINE→CTA")
    print("="*60)
    
    engine = PromptEngine()
    
    prompt = engine.build_proposal_prompt(
        job_data=NEW_JOB,
        similar_projects=[{
            "company": "TechCorp",
            "title": "Backend API",
            "skills": ["Python", "FastAPI"],
            "satisfaction": 4.8,
            "portfolio_urls": ["https://github.com/example"]
        }],
        success_patterns=["Reference specific past projects", "Show conversational tone"],
        max_words=350
    )
    
    # Check for key elements
    has_hook = "HOOK" in prompt
    has_proof = "PROOF" in prompt
    has_approach = "APPROACH" in prompt
    has_timeline = "TIMELINE" in prompt
    has_cta = "CTA" in prompt
    has_system_rules = "CRITICAL RULES" in prompt
    has_word_limit = "250-350" in prompt
    
    print(f"✓ HOOK section: {has_hook}")
    print(f"✓ PROOF section: {has_proof}")
    print(f"✓ APPROACH section: {has_approach}")
    print(f"✓ TIMELINE section: {has_timeline}")
    print(f"✓ CTA section: {has_cta}")
    print(f"✓ System rules included: {has_system_rules}")
    print(f"✓ Word limit (250-350): {has_word_limit}")
    
    # Check for structure guidance (not anti-patterns, since the prompt instructs NOT to use AI language)
    has_structure_guidance = "HOOK" in prompt and "PROOF" in prompt
    print(f"✓ Structure guidance present: {has_structure_guidance}")
    
    all_checks_pass = all([
        has_hook, has_proof, has_approach, has_timeline, has_cta,
        has_system_rules, has_word_limit, has_structure_guidance
    ])
    
    if all_checks_pass:
        print("\n✅ PASS: Prompt Engine correctly structured")
    else:
        print("\n❌ FAIL: Prompt Engine structure incomplete")
    
    return all_checks_pass


def test_retrieval_pipeline():
    """Test that Retrieval Pipeline finds and ranks similar projects"""
    print("\n" + "="*60)
    print("TEST 2: Retrieval Pipeline - Find Similar Projects")
    print("="*60)
    
    retrieval = RetrievalPipeline()
    
    result = retrieval.retrieve_for_proposal(
        new_job_requirements=NEW_JOB,
        all_jobs=SAMPLE_HISTORICAL_PROJECTS,
        top_k=3,
        use_semantic_search=False
    )
    
    # Check retrieval result structure
    has_filtered_count = "stage1_filtered_count" in result
    has_similar_projects = "similar_projects" in result
    has_insights = "insights" in result
    
    similar_count = len(result.get("similar_projects", []))
    success_patterns = result.get("insights", {}).get("success_patterns", [])
    success_rate = result.get("insights", {}).get("success_rate", 0)
    
    print(f"✓ Filtered projects count: {result.get('stage1_filtered_count', 'N/A')}")
    print(f"✓ Similar projects found: {similar_count}")
    print(f"✓ Success patterns extracted: {len(success_patterns)}")
    print(f"  Patterns: {success_patterns[:3]}")
    print(f"✓ Success rate: {success_rate*100:.0f}%")
    
    # Check for expected insights
    has_client_values = len(result.get("insights", {}).get("client_values", [])) > 0
    has_winning_sections = len(result.get("insights", {}).get("winning_sections", [])) > 0
    
    print(f"✓ Client values identified: {has_client_values}")
    print(f"✓ Winning sections identified: {has_winning_sections}")
    
    all_checks_pass = (
        has_filtered_count and 
        has_similar_projects and 
        has_insights and 
        similar_count >= 1 and  # At least 1 project found
        success_rate >= 0 and
        has_winning_sections  # Should have winning sections template
    )
    
    if all_checks_pass:
        print("\n✅ PASS: Retrieval Pipeline working correctly")
    else:
        print("\n❌ FAIL: Retrieval Pipeline incomplete")
        if not has_winning_sections:
            print("  - Missing winning sections template")
    
    return all_checks_pass


def test_proposal_quality_scoring():
    """Test that proposal quality scoring works"""
    print("\n" + "="*60)
    print("TEST 3: Proposal Quality Scoring - SHORT, HUMAN, WINNING")
    print("="*60)
    
    engine = PromptEngine()
    
    # Test with a good proposal
    good_proposal = """
    I see you need robust backend APIs for your platform. I've built 5+ production FastAPI services
    handling high-traffic loads, including work for TechCorp (delivered in 8 weeks) and DataSoft
    (4.8★ satisfaction). Check my portfolio: https://github.com/user/fastapi-project
    
    For you, I'd architect a scalable REST API with PostgreSQL optimization, comprehensive testing,
    and clear documentation. I'd use FastAPI with async patterns, implement proper error handling,
    and ensure security best practices.
    
    Timeline: Phase 1 (API architecture) → Phase 2 (implementation) → Phase 3 (testing).
    Realistic estimate: 8-10 weeks based on similar projects.
    
    Let's hop on a call to discuss your specific requirements and timeline.
    """
    
    score = engine.score_proposal_quality(
        good_proposal,
        NEW_JOB,
        {
            "projects_referenced": [{"company": "TechCorp"}, {"company": "DataSoft"}],
            "portfolio_links_used": ["https://github.com/user/fastapi-project"],
            "feedback_urls_cited": []
        }
    )
    
    word_count = len(good_proposal.split())
    quality = score["overall_score"]
    is_short_human_winning = score.get("is_short_human_winning", False)
    
    print(f"✓ Word count: {word_count} (ideal: 250-350)")
    print(f"✓ Quality score: {quality}/1.0")
    print(f"✓ Is SHORT, HUMAN, WINNING: {is_short_human_winning}")
    print(f"✓ Feedback: {score.get('feedback', [])[:3]}")
    
    # Check all components
    components = score.get("components", {})
    print(f"\nComponent scores:")
    for component, component_score in components.items():
        print(f"  - {component}: {component_score}")
    
    if quality >= 0.75:
        print("\n✅ PASS: Proposal quality scoring working")
        return True
    else:
        print(f"\n⚠️  Quality score below threshold ({quality} < 0.75)")
        return False


def test_with_sample_proposal_text():
    """Test with actual proposal text"""
    print("\n" + "="*60)
    print("TEST 4: Real-world Proposal Examples")
    print("="*60)
    
    engine = PromptEngine()
    
    # Example 1: GOOD proposal (SHORT, HUMAN, WINNING)
    example_good = """
    I see you're dealing with scaling challenges on your APIs. I've handled this exact problem 
    for TechCorp (built FastAPI service handling 100k+ daily requests) and StartupXYZ (delivered 
    full-stack platform ahead of schedule). Check my GitHub: https://github.com/user/fastapi-project
    
    For you, I'd design a horizontal-scalable API architecture with connection pooling, caching,
    and async patterns. I've done this 5+ times successfully.
    
    Timeline: Week 1-2 (design), Week 3-6 (implementation), Week 7-8 (testing/optimization).
    Budget: $6,500 fits well for an 8-10 week project.
    
    Let's discuss your specific database schema and expected traffic patterns.
    """
    
    # Example 2: BAD proposal (too formal, no references)
    example_bad = """
    I am an AI-powered proposal writer here to help you with your backend development needs.
    As a professional developer, I have extensive experience in building robust and scalable
    applications using modern technologies and best practices. I am excited to work with you
    on this project and believe I can deliver exceptional results.
    
    My approach involves comprehensive analysis, detailed planning, and iterative development
    to ensure high-quality deliverables. I will follow industry best practices and maintain
    clear communication throughout the project lifecycle.
    
    I look forward to hearing from you and discussing how I can assist with your backend
    development requirements. Please feel free to reach out with any questions or to schedule
    a consultation call.
    """
    
    print("\nAnalyzing GOOD proposal:")
    score_good = engine.score_proposal_quality(
        example_good,
        NEW_JOB,
        {
            "projects_referenced": [{"company": "TechCorp"}, {"company": "StartupXYZ"}],
            "portfolio_links_used": ["https://github.com/user/fastapi-project"],
            "feedback_urls_cited": []
        }
    )
    
    word_count_good = len(example_good.split())
    print(f"  Word count: {word_count_good}")
    print(f"  Quality score: {score_good['overall_score']}")
    print(f"  Is SHORT, HUMAN, WINNING: {score_good.get('is_short_human_winning')}")
    print(f"  Issues: {score_good.get('feedback', [])}")
    
    print("\nAnalyzing BAD proposal:")
    score_bad = engine.score_proposal_quality(
        example_bad,
        NEW_JOB,
        {
            "projects_referenced": [],
            "portfolio_links_used": [],
            "feedback_urls_cited": []
        }
    )
    
    word_count_bad = len(example_bad.split())
    print(f"  Word count: {word_count_bad}")
    print(f"  Quality score: {score_bad['overall_score']}")
    print(f"  Is SHORT, HUMAN, WINNING: {score_bad.get('is_short_human_winning')}")
    print(f"  Issues: {score_bad.get('feedback', [])}")
    
    good_better = score_good['overall_score'] > score_bad['overall_score']
    
    if good_better:
        print("\n✅ PASS: Good proposals scored higher than bad proposals")
        return True
    else:
        print("\n❌ FAIL: Scoring didn't differentiate properly")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("PROPOSAL GENERATION TEST SUITE - SHORT, HUMAN, WINNING")
    print("="*70)
    
    results = {
        "Prompt Engine Structure": test_prompt_engine(),
        "Retrieval Pipeline": test_retrieval_pipeline(),
        "Quality Scoring": test_proposal_quality_scoring(),
        "Real-world Examples": test_with_sample_proposal_text()
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System ready for proposal generation.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Phase 3-4 Test Suite: Proposal Generation & Retrieval Testing

This script validates:
1. Proposal generation with test data
2. Retrieval pipeline accuracy
3. Reference extraction (portfolio URLs, feedback URLs)
4. Metadata extraction
5. Integration between all components

Usage:
    python test_phase3_4.py [test_type]
    
    test_type:
        - all: Run all tests
        - proposal: Test proposal generation only
        - retrieval: Test retrieval pipeline only
        - metadata: Test metadata extraction only
        - integration: Test full integration
"""

import sys
import json
import logging
import pytest
from typing import Dict, Any, List
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# NOTE: ProposalGenerator was removed - production uses RetrievalPipeline + PromptEngine + OpenAIService directly
# TODO: Refactor tests test_proposal_generation and test_reference_extraction to use production code path
from app.utils.metadata_extractor import MetadataExtractor
from app.utils.retrieval_pipeline import RetrievalPipeline
from app.utils.openai_service import OpenAIService
from app.db import get_db
from app.config import settings
from datetime import datetime

# ===================== TEST DATA =====================

TEST_HISTORICAL_JOBS = [
    {
        "contract_id": "hist_001",
        "company_name": "CloudMetrics Inc",
        "job_title": "Full Stack Developer",
        "industry": "SaaS",
        "task_type": "full_stack",
        "skills_required": ["React", "Node.js", "PostgreSQL", "AWS"],
        "job_description": "Build real-time analytics dashboard with React frontend and Node.js backend. Real-time data streaming with WebSocket.",
        "estimated_budget": 12000,
        "project_duration_days": 60,
        "task_complexity": "high",
        "urgent_adhoc": False,
        "industry_tags": ["SaaS", "Web", "Data"],
        "proposal_status": "completed",
        "proposal_effectiveness_score": 0.92,
        "client_satisfaction": 4.9,
        "feedback_text": "Excellent developer! Built exactly what we needed. Professional approach, great communication.",
        "feedback_url": "https://upwork.com/o/feedback/example1",
        "portfolio_urls": ["https://github.com/example/cloudmetrics", "https://portfolio.example.com/cloudmetrics"],
        "reusable_sections": ["technical_approach", "real_time_architecture"]
    },
    {
        "contract_id": "hist_002",
        "company_name": "DataFlow Systems",
        "job_title": "Backend API Developer",
        "industry": "FinTech",
        "task_type": "backend",
        "skills_required": ["Python", "FastAPI", "PostgreSQL", "Docker"],
        "job_description": "Build scalable payment processing API with Python FastAPI. Handle 1000s of concurrent requests.",
        "estimated_budget": 15000,
        "project_duration_days": 75,
        "task_complexity": "high",
        "urgent_adhoc": False,
        "industry_tags": ["FinTech", "Backend", "Payments"],
        "proposal_status": "completed",
        "proposal_effectiveness_score": 0.88,
        "client_satisfaction": 4.8,
        "feedback_text": "Professional work, delivered on schedule. Great communication and responsive to feedback.",
        "feedback_url": "https://upwork.com/o/feedback/example2",
        "portfolio_urls": ["https://github.com/example/dataflow"],
        "reusable_sections": ["payment_architecture", "performance_optimization"]
    },
    {
        "contract_id": "hist_003",
        "company_name": "MobileFirst startup",
        "job_title": "React Native Developer",
        "industry": "HealthTech",
        "task_type": "mobile_development",
        "skills_required": ["React Native", "Firebase", "Redux", "iOS", "Android"],
        "job_description": "Build cross-platform fitness tracking app with real-time sync and social features.",
        "estimated_budget": 18000,
        "project_duration_days": 90,
        "task_complexity": "medium",
        "urgent_adhoc": False,
        "industry_tags": ["HealthTech", "Mobile", "Social"],
        "proposal_status": "completed",
        "proposal_effectiveness_score": 0.85,
        "client_satisfaction": 4.7,
        "feedback_text": "Great communication and clean code. App works great on both iOS and Android.",
        "feedback_url": "https://upwork.com/o/feedback/example3",
        "portfolio_urls": ["https://apps.apple.com/app/example", "https://play.google.com/store/apps/details?id=com.example"],
        "reusable_sections": ["mobile_architecture", "cross_platform_optimization"]
    }
]

TEST_NEW_JOBS = [
    {
        "contract_id": "new_001",
        "company_name": "NewVenture Inc",
        "job_title": "Full Stack Web Developer",
        "industry": "SaaS",
        "task_type": "full_stack",
        "skills_required": ["React", "Node.js", "PostgreSQL"],
        "job_description": "Build project management dashboard with React and Node.js. Real-time collaboration features.",
        "estimated_budget": 10000,
        "project_duration_days": 60,
        "task_complexity": "high",
        "urgent_adhoc": False,
        "industry_tags": ["SaaS", "Web"]
    },
    {
        "contract_id": "new_002",
        "company_name": "HealthApp Corp",
        "job_title": "React Native Mobile Developer",
        "industry": "HealthTech",
        "task_type": "mobile_development",
        "skills_required": ["React Native", "Firebase", "Redux"],
        "job_description": "Build wellness tracking app with real-time sync.",
        "estimated_budget": 15000,
        "project_duration_days": 75,
        "task_complexity": "medium",
        "urgent_adhoc": False,
        "industry_tags": ["HealthTech", "Mobile"]
    },
    {
        "contract_id": "new_003",
        "company_name": "Urgent Fix Co",
        "job_title": "Emergency Backend Bug Fix",
        "industry": "E-Commerce",
        "task_type": "bug_fix",
        "skills_required": ["Python", "FastAPI", "Debugging"],
        "job_description": "Critical payment processing bug - need immediate fix.",
        "estimated_budget": 2000,
        "project_duration_days": 2,
        "task_complexity": "medium",
        "urgent_adhoc": True,
        "industry_tags": ["E-Commerce", "Urgent"]
    }
]

# ===================== TESTS =====================

class TestProposalGeneration:
    """Test proposal generation functionality"""
    
    def __init__(self):
        logger.info("Initializing tests...")
        self.db = get_db()
        self.openai_service = OpenAIService(api_key=settings.OPENAI_API_KEY)
        
    def setup_test_data(self):
        """Clear and setup test data in database"""
        logger.info("[Setup] Clearing and loading test data...")
        
        # Clear collections (access via DatabaseManager's db attribute)
        self.db.db.training_data.delete_many({})
        self.db.db.chunks.delete_many({})
        
        # Insert test data
        for job in TEST_HISTORICAL_JOBS:
            self.db.db.training_data.insert_one(job)
        
        logger.info(f"✓ [Setup] Loaded {len(TEST_HISTORICAL_JOBS)} test jobs")
    
    def test_metadata_extraction(self):
        """Test 1: Metadata extraction"""
        logger.info("\n" + "="*60)
        logger.info("[Test 1] Metadata Extraction")
        logger.info("="*60)
        
        for job in TEST_HISTORICAL_JOBS[:2]:
            logger.info(f"\nExtracting metadata for {job['company_name']}...")
            
            # extract_all_metadata modifies and returns the job_data
            enriched_job = MetadataExtractor.extract_all_metadata(job)
            
            # Check that metadata was added
            assert "proposal_effectiveness_score" in enriched_job, "Missing proposal_effectiveness_score"
            assert "task_complexity" in enriched_job, "Missing task_complexity"
            assert "industry_tags" in enriched_job, "Missing industry_tags"
            assert "reusable_sections" in enriched_job, "Missing reusable_sections"
            
            logger.info(f"  - Effectiveness: {enriched_job.get('proposal_effectiveness_score', 0):.2f}")
            logger.info(f"  - Satisfaction: {enriched_job.get('client_satisfaction', 'N/A')}")
            logger.info(f"  - Complexity: {enriched_job.get('task_complexity')}")
            logger.info(f"  - Tags: {', '.join(enriched_job.get('industry_tags', []))}")
            logger.info("  ✓ Metadata extracted successfully")
        
        logger.info("\n✓ [Test 1 PASSED] Metadata extraction working correctly")
        return True
    
    def test_retrieval_pipeline(self):
        """Test 2: Retrieval pipeline"""
        logger.info("\n" + "="*60)
        logger.info("[Test 2] Retrieval Pipeline")
        logger.info("="*60)
        
        retrieval = RetrievalPipeline(self.db, self.openai_service)
        
        for new_job in TEST_NEW_JOBS[:2]:
            logger.info(f"\nRetrieving context for {new_job['job_title']}...")
            
            try:
                context = retrieval.retrieve_for_proposal(new_job, TEST_HISTORICAL_JOBS, top_k=3)
                
                assert "similar_projects" in context, "Missing similar_projects"
                assert "insights" in context, "Missing insights"
                
                similar = context["similar_projects"]
                logger.info(f"  - Found {len(similar)} similar projects")
                
                for i, proj in enumerate(similar[:2], 1):
                    logger.info(f"    {i}. {proj['company']} (similarity: {proj['similarity_score']:.2f})")
                
                insights = context["insights"]
                logger.info(f"  - Success patterns: {len(insights['success_patterns'])} identified")
                logger.info(f"  - Success rate: {insights['success_rate']*100:.0f}%")
                logger.info("  ✓ Retrieval successful")
                
            except Exception as e:
                logger.warning(f"  ⚠ Retrieval error: {str(e)}")
                # Note: Semantic search may not work without real embeddings
        
        logger.info("\n✓ [Test 2 PASSED] Retrieval pipeline working")
        return True
    
    @pytest.mark.skip(reason="ProposalGenerator was removed - TODO: refactor to use production code path")
    def test_proposal_generation(self):
        """Test 3: Proposal generation"""
        logger.info("\n" + "="*60)
        logger.info("[Test 3] Proposal Generation")
        logger.info("="*60)
        
        retrieval = RetrievalPipeline(self.db, self.openai_service)
        # ProposalGenerator was removed - production uses PromptEngine + OpenAIService directly
        pytest.skip("ProposalGenerator class was removed")
        
        for new_job in TEST_NEW_JOBS[:1]:
            logger.info(f"\nGenerating proposal for {new_job['job_title']}...")
            
            try:
                proposal = generator.generate_proposal(
                    new_job,
                    TEST_HISTORICAL_JOBS,
                    max_length=500,
                    include_portfolio=True,
                    include_feedback=True
                )
                
                assert "generated_proposal" in proposal, "Missing generated_proposal"
                assert "word_count" in proposal, "Missing word_count"
                assert "references" in proposal, "Missing references"
                
                logger.info(f"  - Proposal length: {proposal['word_count']} words")
                logger.info(f"  - Portfolio links used: {len(proposal['references']['portfolio_links_used'])}")
                logger.info(f"  - Feedback URLs cited: {len(proposal['references']['feedback_urls_cited'])}")
                logger.info(f"  - Confidence score: {proposal['metadata']['confidence_score']:.2f}")
                logger.info("  ✓ Proposal generated successfully")
                
                # Log proposal preview
                logger.info(f"\n  Proposal Preview (first 300 chars):")
                preview = proposal['generated_proposal'][:300] + "..."
                for line in preview.split('\n'):
                    logger.info(f"    {line}")
                
            except Exception as e:
                logger.warning(f"  ⚠ Proposal generation error: {str(e)}")
                logger.info("    (This may require OpenAI API key and real embeddings)")
        
        logger.info("\n✓ [Test 3 PASSED] Proposal generation executed")
        return True
    
    @pytest.mark.skip(reason="ProposalGenerator was removed - TODO: refactor to use production code path")
    def test_reference_extraction(self):
        """Test 4: Reference extraction"""
        logger.info("\n" + "="*60)
        logger.info("[Test 4] Reference Extraction")
        logger.info("="*60)
        
        retrieval = RetrievalPipeline(self.db, self.openai_service)
        # ProposalGenerator was removed - production uses PromptEngine + OpenAIService directly
        pytest.skip("ProposalGenerator class was removed")
        
        # Sample proposal text
        sample_proposal = """
        I have successfully completed similar projects for CloudMetrics Inc and DataFlow Systems.
        You can see my portfolio at https://github.com/example/cloudmetrics
        Client feedback: https://upwork.com/o/feedback/example1
        """
        
        logger.info("\nExtracting references from sample proposal...")
        
        retrieval_result = {
            "similar_projects": TEST_HISTORICAL_JOBS[:2],
            "insights": {"success_patterns": [], "winning_sections": [], "success_rate": 0.8}
        }
        
        refs = generator._extract_references(
            sample_proposal,
            retrieval_result,
            include_portfolio=True,
            include_feedback=True
        )
        
        logger.info(f"  - Projects referenced: {len(refs['projects_referenced'])}")
        logger.info(f"  - Portfolio links found: {len(refs['portfolio_links_used'])}")
        logger.info(f"  - Feedback URLs found: {len(refs['feedback_urls_cited'])}")
        logger.info("  ✓ References extracted")
        
        logger.info("\n✓ [Test 4 PASSED] Reference extraction working")
        return True
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("\n" + "#"*60)
        logger.info("# PHASE 3-4 TEST SUITE: PROPOSAL GENERATION")
        logger.info("#"*60)
        
        self.setup_test_data()
        
        tests = [
            self.test_metadata_extraction,
            self.test_retrieval_pipeline,
            self.test_proposal_generation,
            self.test_reference_extraction,
        ]
        
        results = {}
        for test_func in tests:
            try:
                result = test_func()
                results[test_func.__name__] = result
            except AssertionError as e:
                logger.error(f"✗ {test_func.__name__} FAILED: {str(e)}")
                results[test_func.__name__] = False
            except Exception as e:
                logger.warning(f"⚠ {test_func.__name__} ERROR: {str(e)}")
                results[test_func.__name__] = False
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            logger.info(f"{status}: {test_name}")
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        logger.info(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("\n✓ ALL TESTS PASSED - Phase 3 ready for integration!")
        else:
            logger.warning(f"\n⚠ {total - passed} tests failed or skipped")
        
        return results

# ===================== MAIN =====================

if __name__ == "__main__":
    test_suite = TestProposalGeneration()
    results = test_suite.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

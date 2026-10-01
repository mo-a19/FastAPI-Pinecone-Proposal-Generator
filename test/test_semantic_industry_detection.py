"""
Test script for LLM-based semantic industry and intent detection.

This tests the fix for the issue where:
- Job description mentions "TMZ, JustJared, WWD, Variety" (media websites)
- But the system was not matching it to media-industry portfolio projects

The solution uses:
1. Enhanced keyword + brand detection in MetadataExtractor
2. LLM-based semantic detection in OpenAIService for complex cases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from app.utils.metadata_extractor import MetadataExtractor

def test_keyword_brand_detection():
    """Test the enhanced keyword + brand detection"""
    print("\n" + "="*60)
    print("TEST 1: Keyword + Brand Detection (Fast)")
    print("="*60)
    
    # User's exact job description
    job_data = {
        "job_title": "WordPress Website Redesign",
        "company_name": "Client",
        "job_description": """I need someone to update my existing WordPress website to be more professional, 
        similar to TMZ, JustJared, WWD, or Variety. All sections are already defined, but I want a more unique 
        layout aligned with my brand. I also need:

        RSS integration to pull my YouTube content into relevant areas of the website
        An easy way to add written articles""",
        "skills_required": ["WordPress", "Web Design", "Web Development", "SEO", "SEO Writing", "Social Media Marketing"],
        "task_type": "redesign"
    }
    
    # Test industry detection with context
    result = MetadataExtractor.detect_industry_with_context(job_data)
    
    print(f"\nJob: {job_data['job_title']}")
    print(f"Description mentions: TMZ, JustJared, WWD, Variety")
    print(f"\nDetection Result:")
    print(f"  Industry: {result['industry']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Detected Brands: {result['detected_brands']}")
    print(f"  Secondary Industries: {result['secondary_industries']}")
    print(f"  Method: {result['method']}")
    
    # Verify it detected media
    assert result['industry'] == 'media', f"Expected 'media', got '{result['industry']}'"
    assert 'tmz' in result['detected_brands'] or 'justjared' in result['detected_brands'] or 'wwd' in result['detected_brands'] or 'variety' in result['detected_brands'], \
        f"Expected to detect media brands, got {result['detected_brands']}"
    
    print("\n✅ TEST 1 PASSED: Correctly detected MEDIA industry from brand references!")
    return True

def test_client_intent_extraction():
    """Test client intent extraction for redesign + RSS integration"""
    print("\n" + "="*60)
    print("TEST 2: Client Intent Extraction")
    print("="*60)
    
    job_data = {
        "job_title": "WordPress Website Redesign",
        "job_description": """I need someone to update my existing WordPress website to be more professional, 
        similar to TMZ, JustJared, WWD, or Variety. All sections are already defined, but I want a more unique 
        layout aligned with my brand. I also need:

        RSS integration to pull my YouTube content into relevant areas of the website
        An easy way to add written articles""",
        "skills_required": ["WordPress", "Web Design", "Web Development", "SEO"]
    }
    
    intents = MetadataExtractor.extract_client_intents(job_data)
    
    print(f"\nDetected Intents: {intents}")
    
    # Should detect: website_redesign, rss_integration, content_management
    assert 'website_redesign' in intents, f"Expected 'website_redesign' in intents, got {intents}"
    
    print("\n✅ TEST 2 PASSED: Correctly detected client intents!")
    return True

def test_industry_tags_extraction():
    """Test industry tags extraction with brand detection"""
    print("\n" + "="*60)
    print("TEST 3: Industry Tags Extraction")
    print("="*60)
    
    job_data = {
        "industry": "",  # Not provided
        "job_description": "Build a media website similar to TMZ and Variety for celebrity news",
        "company_name": "NewsMedia Inc"
    }
    
    tags = MetadataExtractor.extract_industry_tags(job_data)
    
    print(f"\nJob Description: {job_data['job_description']}")
    print(f"Extracted Industry Tags: {tags}")
    
    assert 'media' in tags, f"Expected 'media' in tags, got {tags}"
    
    print("\n✅ TEST 3 PASSED: Correctly extracted media industry tag from brand reference!")
    return True

def test_llm_detection():
    """Test LLM-based semantic detection (requires API key)"""
    print("\n" + "="*60)
    print("TEST 4: LLM-Based Semantic Detection")
    print("="*60)
    
    try:
        from app.config import settings
        from app.utils.openai_service import OpenAIService
        
        if not settings.OPENAI_API_KEY:
            print("\n⚠️  SKIPPED: OPENAI_API_KEY not set")
            return True
        
        openai_service = OpenAIService(api_key=settings.OPENAI_API_KEY)
        
        job_data = {
            "job_title": "WordPress Website Redesign",
            "company_name": "Client",
            "job_description": """I need someone to update my existing WordPress website to be more professional, 
            similar to TMZ, JustJared, WWD, or Variety. All sections are already defined, but I want a more unique 
            layout aligned with my brand. I also need:

            RSS integration to pull my YouTube content into relevant areas of the website
            An easy way to add written articles""",
            "skills_required": ["WordPress", "Web Design", "Web Development", "SEO"],
            "task_type": "redesign"
        }
        
        result = openai_service.detect_industry_and_intent(job_data)
        
        print(f"\nLLM Detection Result:")
        print(f"  Industry: {result['industry']}")
        print(f"  Confidence: {result['industry_confidence']}")
        print(f"  User Intents: {result['user_intents']}")
        print(f"  Context Brands: {result['context_brands']}")
        print(f"  Reasoning: {result['reasoning']}")
        
        assert result['industry'] == 'media', f"Expected 'media', got '{result['industry']}'"
        
        print("\n✅ TEST 4 PASSED: LLM correctly detected MEDIA industry!")
        return True
        
    except Exception as e:
        print(f"\n⚠️  TEST 4 ERROR: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SEMANTIC INDUSTRY & INTENT DETECTION TESTS")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Run tests
    try:
        if test_keyword_brand_detection():
            tests_passed += 1
        else:
            tests_failed += 1
    except AssertionError as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        tests_failed += 1
    
    try:
        if test_client_intent_extraction():
            tests_passed += 1
        else:
            tests_failed += 1
    except AssertionError as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        tests_failed += 1
    
    try:
        if test_industry_tags_extraction():
            tests_passed += 1
        else:
            tests_failed += 1
    except AssertionError as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        tests_failed += 1
    
    try:
        if test_llm_detection():
            tests_passed += 1
        else:
            tests_failed += 1
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Semantic industry detection is working correctly.")
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed. Please review the output above.")
    
    sys.exit(0 if tests_failed == 0 else 1)

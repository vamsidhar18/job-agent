#!/usr/bin/env python3
"""
Test script for the JobScoringAgent
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from scorer.job_agent import JobScoringAgent, score_job
import json

def test_new_agent():
    """Test the new JobScoringAgent class"""
    print("🧪 Testing JobScoringAgent class...")
    
    test_job = {
        "title": "Software Engineer I", 
        "company": "Tesla",
        "description": "We are looking for a backend developer with experience in Python, AWS, and microservices. H1B candidates welcome."
    }
    
    try:
        # Initialize agent
        agent = JobScoringAgent()
        print(f"✅ Agent initialized with model: {agent.current_model}")
        
        # Test health check
        health = agent.health_check()
        print(f"🏥 Health status: {health['status']}")
        
        # Score the job
        result = agent.score(test_job)
        print("📊 Scoring result:")
        print(json.dumps(result, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def test_backward_compatibility():
    """Test backward compatibility with original scorer"""
    print("\n🔄 Testing backward compatibility...")
    
    test_job = {
        "title": "Frontend Developer",
        "company": "Startup Inc", 
        "description": "React, TypeScript, modern frontend development. Remote friendly."
    }
    
    try:
        result = score_job(test_job)
        print("✅ Backward compatibility test passed")
        print("Result preview:", result[:200] + "..." if len(result) > 200 else result)
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def main():
    print("🚀 Starting JobScoringAgent tests...\n")
    
    # Test environment setup
    print("🔧 Environment check:")
    print(f"OpenAI API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    print(f"Anthropic API Key: {'✅ Set' if os.getenv('ANTHROPIC_API_KEY') else '❌ Missing'}")
    print()
    
    # Run tests
    agent_test = test_new_agent()
    compat_test = test_backward_compatibility()
    
    # Summary
    print(f"\n📋 Test Results:")
    print(f"Agent Class: {'✅ PASS' if agent_test else '❌ FAIL'}")
    print(f"Backward Compatibility: {'✅ PASS' if compat_test else '❌ FAIL'}")
    
    if agent_test and compat_test:
        print("\n🎉 All tests passed! Ready for FastAPI integration.")
    else:
        print("\n⚠️  Some tests failed. Check your API keys and network connection.")

if __name__ == "__main__":
    main()
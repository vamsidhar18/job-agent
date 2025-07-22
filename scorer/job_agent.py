import os
import json
import logging
import time
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Handle pandas import fallback for timestamps
try:
    import pandas as pd
    now = lambda: pd.Timestamp.now()
except ImportError:
    from datetime import datetime
    now = lambda: datetime.utcnow()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class JobScoringAgent:
    """
    LangChain-based job scoring agent with multi-model fallback support.
    Supports OpenAI GPT models and Anthropic Claude as fallbacks.
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Model priority order
        self.models_config = [
            {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.2},
            {"provider": "openai", "model": "gpt-3.5-turbo", "temperature": 0.2},
            {"provider": "anthropic", "model": "claude-3-sonnet-20240229", "temperature": 0.2},
            {"provider": "openai", "model": "gpt-4", "temperature": 0.2},
        ]
        
        self.llm = None
        self.current_model = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the first available LLM from the priority list."""
        for config in self.models_config:
            try:
                if config["provider"] == "openai" and self.openai_api_key:
                    llm = ChatOpenAI(
                        model=config["model"],
                        temperature=config["temperature"],
                        openai_api_key=self.openai_api_key,
                        max_tokens=1000
                    )
                elif config["provider"] == "anthropic" and self.anthropic_api_key:
                    llm = ChatAnthropic(
                        model=config["model"],
                        temperature=config["temperature"],
                        anthropic_api_key=self.anthropic_api_key,
                        max_tokens=1000
                    )
                else:
                    continue
                
                # Test the model with a simple call
                test_response = llm.invoke([HumanMessage(content="test")])
                
                self.llm = llm
                self.current_model = f"{config['provider']}:{config['model']}"
                logger.info(f"✅ Successfully initialized model: {self.current_model}")
                return
                
            except Exception as e:
                logger.warning(f"❌ Model {config['provider']}:{config['model']} failed: {e}")
                continue
        
        raise Exception("No available models found. Check your API keys and billing status.")
    
    def _try_fallback_model(self):
        """Try the next available model in the priority list."""
        current_index = None
        for i, config in enumerate(self.models_config):
            model_name = f"{config['provider']}:{config['model']}"
            if model_name == self.current_model:
                current_index = i
                break
        
        if current_index is None or current_index >= len(self.models_config) - 1:
            raise Exception("No more fallback models available")
        
        # Try remaining models
        for config in self.models_config[current_index + 1:]:
            try:
                if config["provider"] == "openai" and self.openai_api_key:
                    llm = ChatOpenAI(
                        model=config["model"],
                        temperature=config["temperature"],
                        openai_api_key=self.openai_api_key,
                        max_tokens=1000
                    )
                elif config["provider"] == "anthropic" and self.anthropic_api_key:
                    llm = ChatAnthropic(
                        model=config["model"],
                        temperature=config["temperature"],
                        anthropic_api_key=self.anthropic_api_key,
                        max_tokens=1000
                    )
                else:
                    continue
                
                # Test the model
                test_response = llm.invoke([HumanMessage(content="test")])
                
                self.llm = llm
                self.current_model = f"{config['provider']}:{config['model']}"
                logger.info(f"✅ Switched to fallback model: {self.current_model}")
                return True
                
            except Exception as e:
                logger.warning(f"❌ Fallback model {config['provider']}:{config['model']} failed: {e}")
                continue
        
        return False
    
    def score(self, job: Dict[str, Any], max_retries: int = 2, use_exponential_backoff: bool = True) -> Dict[str, Any]:
        """
        Score a job posting and return structured results.
        
        Args:
            job: Dictionary with 'title', 'company', and 'description' keys
            max_retries: Maximum number of retry attempts with fallback models
            use_exponential_backoff: Whether to use exponential backoff between retries
            
        Returns:
            Dictionary with scoring results and metadata
        """
        system_prompt = (
            "You are an expert career advisor helping a software engineer "
            "evaluate job opportunities. Analyze the job posting and return "
            "a JSON response with detailed scoring and reasoning. "
            "Focus on technical fit, company reputation, visa sponsorship likelihood, "
            "and overall career growth potential."
        )
        
        user_prompt = f"""
Analyze this job posting:

Job Title: {job.get('title', 'Not specified')}
Company: {job.get('company', 'Not specified')}
Description: {job.get('description', 'Not specified')}

Return a JSON response with this exact structure:
{{
  "totalScore": <0-100 integer>,
  "sponsorshipScore": <0-100 integer>,
  "fitScore": <0-100 integer>,
  "techScore": <0-100 integer>,
  "companyScore": <0-100 integer>,
  "shouldApply": <true or false>,
  "reasoning": "<brief explanation of the scores>",
  "keyTechnologies": ["<list of relevant technologies mentioned>"],
  "redFlags": ["<list of potential concerns>"],
  "positives": ["<list of positive aspects>"]
}}

Scoring criteria:
- totalScore: Overall recommendation (weighted average)
- sponsorshipScore: Likelihood of H1B/visa sponsorship (keywords, company size, etc.)
- fitScore: Match with software engineering career goals
- techScore: Technical stack appeal and growth potential
- companyScore: Company reputation, stability, and culture
"""

        for attempt in range(max_retries + 1):
            try:
                # Exponential backoff for retries (but not first attempt)
                if attempt > 0 and use_exponential_backoff:
                    delay = min(2 ** (attempt - 1), 30)  # Cap at 30 seconds
                    logger.info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)
                
                logger.info(f"Scoring job with model: {self.current_model} (attempt {attempt + 1})")
                
                response = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                
                # Try to parse the JSON response
                try:
                    # Extract JSON from response if it's wrapped in markdown or other text
                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    if content.startswith("```"):
                        content = content[3:]
                    
                    result = json.loads(content.strip())
                    
                    # Add metadata
                    result["metadata"] = {
                        "model_used": self.current_model,
                        "attempt": attempt + 1,
                        "timestamp": str(now())
                    }
                    
                    logger.info(f"✅ Successfully scored job: {result.get('totalScore', 'unknown')}/100")
                    return result
                    
                except json.JSONDecodeError as json_error:
                    logger.warning(f"JSON parsing failed: {json_error}")
                    # Return raw response if JSON parsing fails
                    return {
                        "error": "Failed to parse JSON response",
                        "raw_response": response.content,
                        "model_used": self.current_model,
                        "attempt": attempt + 1,
                        "timestamp": str(now())
                    }
                
            except Exception as e:
                logger.error(f"❌ Scoring failed with {self.current_model}: {e}")
                
                # Try fallback model if available and not on last attempt
                if attempt < max_retries:
                    if self._try_fallback_model():
                        continue
                    else:
                        break
                else:
                    # Last attempt failed
                    raise Exception(f"All scoring attempts failed. Last error: {e}")
        
        raise Exception("Maximum retries exceeded")
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the agent and available models."""
        try:
            test_job = {
                "title": "Test Engineer",
                "company": "Test Company",
                "description": "This is a test job posting."
            }
            
            start_time = now()
            test_response = self.llm.invoke([HumanMessage(content="Health check - respond with 'OK'")])
            end_time = now()
            
            response_time = (end_time - start_time).total_seconds() if hasattr((end_time - start_time), 'total_seconds') else None
            
            return {
                "status": "healthy",
                "current_model": self.current_model,
                "response_time_seconds": response_time,
                "test_response": test_response.content[:100] + "..." if len(test_response.content) > 100 else test_response.content,
                "available_keys": {
                    "openai": bool(self.openai_api_key),
                    "anthropic": bool(self.anthropic_api_key)
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "current_model": self.current_model,
                "available_keys": {
                    "openai": bool(self.openai_api_key),
                    "anthropic": bool(self.anthropic_api_key)
                }
            }


# Convenience function to maintain backward compatibility
def score_job(job: Dict[str, Any]) -> str:
    """
    Backward-compatible function that mimics the original scorer.py behavior.
    Returns the raw string response for compatibility with existing code.
    """
    agent = JobScoringAgent()
    result = agent.score(job)
    
    # If it's a structured result, convert back to JSON string
    if isinstance(result, dict) and "raw_response" not in result:
        return json.dumps(result, indent=2)
    elif isinstance(result, dict) and "raw_response" in result:
        return result["raw_response"]
    else:
        return str(result)


# Example usage and testing
if __name__ == "__main__":
    # Test the agent
    test_job = {
        "title": "Software Engineer I",
        "company": "Tesla",
        "description": "We are looking for a backend developer with experience in Python, AWS, and microservices. H1B candidates welcome."
    }
    
    try:
        agent = JobScoringAgent()
        print(f"🤖 Initialized agent with model: {agent.current_model}")
        
        print("🔍 Scoring test job...")
        result = agent.score(test_job)
        print("✅ Scoring result:")
        print(json.dumps(result, indent=2))
        
        print("\n🏥 Health check...")
        health = agent.health_check()
        print(json.dumps(health, indent=2))
        
    except Exception as e:
        print(f"❌ Error: {e}")
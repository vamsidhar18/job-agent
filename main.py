#!/usr/bin/env python3
"""
FastAPI Job Scoring Service
Provides REST endpoints for scoring job postings using LangChain agents
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our job scoring agent
from scorer.job_agent import JobScoringAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import security functions (optional - only if you want API key protection)
try:
    from scorer.security import verify_api_key, rate_limit_check
    SECURITY_ENABLED = True
    logger.info("🔐 Security features loaded")
except ImportError:
    SECURITY_ENABLED = False
    logger.info("🔓 Security features disabled (security.py not found)")

# Initialize FastAPI app
app = FastAPI(
    title="Job Scoring API",
    description="AI-powered job posting scorer with visa sponsorship analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration - environment-based security
import os

# Determine allowed origins based on environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    # Production: Restrict to your actual domains
    allowed_origins = [
        # Add your Railway URL here after deployment
        # "https://job-agent-production-xxxx.up.railway.app",
        "https://yourdomain.com",           # Replace with your actual domain
        "https://app.yourdomain.com",       # Replace with your app subdomain
        "https://www.yourdomain.com",       # www version if needed
    ]
elif ENVIRONMENT == "staging":
    # Staging: Allow staging domains
    allowed_origins = [
        "https://staging.yourdomain.com",
        "https://yourapp-staging.up.railway.app"  # Railway staging
    ]
else:
    # Development: Allow localhost for testing
    allowed_origins = [
        "http://localhost:3000",    # React default
        "http://localhost:5173",    # Vite default
        "http://localhost:8080",    # Vue default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict to needed methods
    allow_headers=["*"],
)

# Initialize the job scoring agent
try:
    job_agent = JobScoringAgent()
    logger.info(f"✅ Job agent initialized with model: {job_agent.current_model}")
except Exception as e:
    logger.error(f"❌ Failed to initialize job agent: {e}")
    job_agent = None

# Pydantic models for request/response validation
class JobPosting(BaseModel):
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    description: str = Field(..., description="Job description")
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Software Engineer I",
                "company": "Tesla", 
                "description": "We are looking for a backend developer with experience in Python, AWS, and microservices. H1B candidates welcome."
            }
        }

class BatchJobRequest(BaseModel):
    jobs: List[JobPosting] = Field(..., description="List of job postings to score")
    
    class Config:
        schema_extra = {
            "example": {
                "jobs": [
                    {
                        "title": "Software Engineer",
                        "company": "Google",
                        "description": "Build scalable systems..."
                    },
                    {
                        "title": "Frontend Developer", 
                        "company": "Startup Inc",
                        "description": "React and TypeScript..."
                    }
                ]
            }
        }

class ScoringResult(BaseModel):
    totalScore: int = Field(..., description="Overall score 0-100")
    sponsorshipScore: int = Field(..., description="Visa sponsorship likelihood 0-100")
    fitScore: int = Field(..., description="Job fit score 0-100")
    techScore: int = Field(..., description="Technology stack score 0-100")
    companyScore: int = Field(..., description="Company attractiveness score 0-100")
    shouldApply: bool = Field(..., description="Recommendation to apply")
    reasoning: str = Field(..., description="Brief explanation")
    keyTechnologies: List[str] = Field(default=[], description="Technologies mentioned")
    redFlags: List[str] = Field(default=[], description="Potential concerns")
    positives: List[str] = Field(default=[], description="Positive aspects")
    metadata: Dict[str, Any] = Field(default={}, description="Scoring metadata")

class BatchScoringResponse(BaseModel):
    total_jobs: int = Field(..., description="Total number of jobs submitted")
    successful: int = Field(..., description="Number of successfully scored jobs")
    failed: int = Field(..., description="Number of failed scoring attempts")
    results: List[ScoringResult] = Field(..., description="List of scoring results")
    errors: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of errors if any occurred")
    
    class Config:
        schema_extra = {
            "example": {
                "total_jobs": 2,
                "successful": 2,
                "failed": 0,
                "results": [
                    {
                        "totalScore": 85,
                        "sponsorshipScore": 90,
                        "shouldApply": True,
                        "reasoning": "Strong tech match with explicit visa support"
                    }
                ],
                "errors": None
            }
        }

# Helper function to load job samples
def load_job_samples() -> Dict[str, Any]:
    """Load job samples from the data directory."""
    try:
        samples_path = os.path.join("data", "job_samples.json")
        if not os.path.exists(samples_path):
            # Fallback to original location
            samples_path = "job_samples.json"
        
        with open(samples_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load job samples: {e}")
        # Return minimal fallback data
        return {
            "test_jobs": [{
                "id": "fallback_job",
                "title": "Software Engineer",
                "company": "Test Company",
                "description": "Test job description for fallback."
            }]
        }

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - API status check."""
    return {
        "message": "Job Scoring API is running",
        "version": "1.0.0",
        "status": "active",
        "agent_model": job_agent.current_model if job_agent else "not initialized",
        "endpoints": {
            "POST /score-job": "Score a single job posting",
            "POST /score-batch": "Score multiple job postings", 
            "GET /score-samples": "Score all jobs from job_samples.json",
            "GET /health": "Agent health check",
            "GET /docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check for the job scoring agent."""
    if not job_agent:
        raise HTTPException(
            status_code=503, 
            detail="Job agent not initialized"
        )
    
    try:
        health_result = job_agent.health_check()
        
        if health_result.get("status") == "healthy":
            return JSONResponse(
                status_code=200,
                content=health_result
            )
        else:
            return JSONResponse(
                status_code=503,
                content=health_result
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "agent_initialized": job_agent is not None
            }
        )

@app.post("/score-job", response_model=ScoringResult, dependencies=[Depends(verify_api_key)] if SECURITY_ENABLED else [])
async def score_job(job: JobPosting, request: Request = None):
    """Score a single job posting."""
    # Apply rate limiting if security is enabled
    if SECURITY_ENABLED and request:
        rate_limit_check(request.client.host)
    
    if not job_agent:
        raise HTTPException(
            status_code=503,
            detail="Job agent not available"
        )
    
    try:
        # Convert Pydantic model to dict
        job_dict = job.dict()
        
        # Score the job
        result = job_agent.score(job_dict)
        
        # Handle error responses
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Scoring failed: {result.get('error', 'Unknown error')}"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scoring job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while scoring job: {str(e)}"
        )

@app.post("/score-batch", response_model=BatchScoringResponse, dependencies=[Depends(verify_api_key)] if SECURITY_ENABLED else [])
async def score_batch(request_data: BatchJobRequest, request: Request = None):
    """Score multiple job postings in a batch."""
    # Apply rate limiting if security is enabled
    if SECURITY_ENABLED and request:
        rate_limit_check(request.client.host)
        
    if not job_agent:
        raise HTTPException(
            status_code=503,
            detail="Job agent not available"
        )
    
    if len(request_data.jobs) > 20:  # Reasonable limit for batch processing
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 20 jobs per request."
        )
    
    results = []
    errors = []
    
    for i, job in enumerate(request_data.jobs):
        try:
            job_dict = job.dict()
            result = job_agent.score(job_dict)
            
            # Add index for tracking
            result["batch_index"] = i
            results.append(result)
            
        except Exception as e:
            error_info = {
                "batch_index": i,
                "job_title": job.title,
                "error": str(e)
            }
            errors.append(error_info)
            logger.error(f"Error scoring job {i}: {e}")
    
    return {
        "total_jobs": len(request_data.jobs),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None
    }

@app.get("/score-samples")
async def score_samples():
    """Score all jobs from the job_samples.json file."""
    if not job_agent:
        raise HTTPException(
            status_code=503,
            detail="Job agent not available"
        )
    
    try:
        # Load job samples
        samples_data = load_job_samples()
        
        # Extract jobs (handle both old and new format)
        if "test_jobs" in samples_data:
            jobs = samples_data["test_jobs"]
        elif isinstance(samples_data, list):
            jobs = samples_data
        else:
            # Old format - single job object
            jobs = [samples_data]
        
        results = []
        errors = []
        
        for job in jobs:
            try:
                # Ensure job has required fields
                job_dict = {
                    "title": job.get("title", "Unknown"),
                    "company": job.get("company", "Unknown"),
                    "description": job.get("description", "No description provided")
                }
                
                result = job_agent.score(job_dict)
                
                # Add sample metadata
                result["sample_id"] = job.get("id", "unknown")
                result["expected_sponsorship"] = job.get("expected_sponsorship", "unknown")
                results.append(result)
                
            except Exception as e:
                error_info = {
                    "sample_id": job.get("id", "unknown"),
                    "job_title": job.get("title", "Unknown"),
                    "error": str(e)
                }
                errors.append(error_info)
                logger.error(f"Error scoring sample {job.get('id', 'unknown')}: {e}")
        
        return {
            "total_samples": len(jobs),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors if errors else None,
            "source": "job_samples.json"
        }
        
    except Exception as e:
        logger.error(f"Error loading/scoring samples: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load or score job samples: {str(e)}"
        )

@app.post("/apply-job")
async def apply_job(job: JobPosting, background_tasks: BackgroundTasks):
    """
    Stub endpoint for job application automation.
    Future: integrate with Make.com or direct application systems.
    """
    # For now, just return a confirmation
    # In the future, this could trigger:
    # - Resume customization
    # - Cover letter generation  
    # - Actual application submission via Make.com
    
    background_tasks.add_task(log_application_intent, job.dict())
    
    return {
        "status": "application_queued",
        "message": f"Application intent logged for {job.title} at {job.company}",
        "job": job.dict(),
        "next_steps": [
            "Resume customization (coming soon)",
            "Cover letter generation (coming soon)", 
            "Application submission via Make.com (coming soon)"
        ],
        "note": "This is currently a stub endpoint for future automation"
    }

# Background task function
async def log_application_intent(job_data: Dict[str, Any]):
    """Log application intent for future processing."""
    logger.info(f"Application intent logged: {job_data['title']} at {job_data['company']}")
    # Future: Could write to database, trigger workflows, etc.

# Development server runner
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )
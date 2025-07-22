"""
Optional security enhancements for production deployment
Place this file as: scorer/security.py
"""

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import time
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Simple API Key Authentication (optional)
API_KEY = os.getenv("API_KEY", None)
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Optional API key verification for production"""
    if not API_KEY:
        return True  # No API key required if not set
    
    if not credentials or credentials.credentials != API_KEY:
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# Simple rate limiting (in-memory, resets on restart)
request_counts = defaultdict(list)
RATE_LIMIT = int(os.getenv("RATE_LIMIT", 100))  # requests per hour
RATE_WINDOW = int(os.getenv("RATE_WINDOW", 3600))  # 1 hour in seconds

def rate_limit_check(client_ip: str):
    """Simple rate limiting by IP address"""
    if not client_ip:
        return  # Skip if no IP available
        
    now = time.time()
    # Clean old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip] 
        if now - req_time < RATE_WINDOW
    ]
    
    # Check if over limit
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per hour. Try again later."
        )
    
    # Add current request
    request_counts[client_ip].append(now)
    logger.debug(f"Request logged for {client_ip}: {len(request_counts[client_ip])}/{RATE_LIMIT}")
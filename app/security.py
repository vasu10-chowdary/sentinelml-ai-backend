from fastapi import Request, HTTPException
from collections import defaultdict
import time
from app.config import settings

# Simple in-memory rate limiting (per IP)
request_counts = defaultdict(list)

def check_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    # Clean up old timestamps
    request_counts[client_ip] = [t for t in request_counts[client_ip] if now - t < 60]
    
    if len(request_counts[client_ip]) >= settings.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    
    request_counts[client_ip].append(now)

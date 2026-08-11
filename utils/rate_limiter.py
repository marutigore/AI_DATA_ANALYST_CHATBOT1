"""
Rate Limiting Middleware Utility.
Implements in-memory token bucket rate limiting per IP / Session to prevent API abuse, DoS attacks,
and unthrottled LLM token consumption.
"""
import time
import logging
from typing import Dict, List, Tuple
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RateLimiter:
    """Fixed-window in-memory Rate Limiter."""
    
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests_store: Dict[str, List[float]] = {}
        
    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Checks if a key (IP or session_id) is allowed to make a request.
        
        Args:
            key (str): Identifier key (e.g. client IP).
            
        Returns:
            Tuple[bool, int]: (allowed, seconds_until_reset)
        """
        now = time.time()
        window_start = now - 60.0
        
        if key not in self.requests_store:
            self.requests_store[key] = []
            
        # Filter out timestamps outside current 60s window
        self.requests_store[key] = [t for t in self.requests_store[key] if t > window_start]
        
        if len(self.requests_store[key]) >= self.requests_per_minute:
            oldest_request = self.requests_store[key][0]
            retry_after = int(60.0 - (now - oldest_request)) + 1
            return False, max(retry_after, 1)
            
        self.requests_store[key].append(now)
        return True, 0

# Global Limiter Instance (30 requests per minute)
global_rate_limiter = RateLimiter(requests_per_minute=30)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware to enforce rate limits on incoming HTTP requests."""
    
    async def dispatch(self, request: Request, call_next):
        # Exclude documentation and health check endpoints from rate limiting
        if request.url.path in ["/docs", "/openapi.json", "/api/health", "/redoc"]:
            return await call_next(request)
            
        client_ip = request.client.host if request.client else "127.0.0.1"
        allowed, retry_after = global_rate_limiter.is_allowed(client_ip)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for client {client_ip}. Retry after {retry_after}s.")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum 30 requests per minute allowed. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )
            
        response = await call_next(request)
        return response

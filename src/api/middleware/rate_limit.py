# =============================================================================
# RATE LIMITING MODULE
# =============================================================================
#
# Implements sliding window rate limiting with Redis backend.
# Falls back to in-memory for development.
#
# Features:
#   - Per-API-key rate limits
#   - Tier-based limits (free, pro, enterprise)
#   - Sliding window algorithm
#   - Rate limit headers in response
#
# =============================================================================

from fastapi import Request, HTTPException, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional, Tuple
from datetime import datetime
import time
import asyncio
from collections import defaultdict

from src.core.config import settings
from src.core.logging import logger


# =============================================================================
# RATE LIMIT CONFIGURATION
# =============================================================================

RATE_LIMITS = {
    "free": {
        "requests_per_minute": 10,
        "requests_per_hour": 100,
        "requests_per_day": 500,
    },
    "pro": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "requests_per_day": 10000,
    },
    "enterprise": {
        "requests_per_minute": 300,
        "requests_per_hour": 10000,
        "requests_per_day": 100000,
    },
    "anonymous": {
        "requests_per_minute": 5,
        "requests_per_hour": 20,
        "requests_per_day": 50,
    },
}


# =============================================================================
# IN-MEMORY RATE LIMITER (Development)
# =============================================================================

class InMemoryRateLimiter:
    """Simple in-memory rate limiter using sliding window."""
    
    def __init__(self):
        # {key: [(timestamp, count), ...]}
        self._windows: dict = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> Tuple[bool, int, int]:
        """
        Check if request is allowed.
        
        Returns:
            (allowed, remaining, reset_time)
        """
        async with self._lock:
            now = time.time()
            window_start = now - window_seconds
            
            # Clean old entries
            self._windows[key] = [
                (ts, count) for ts, count in self._windows[key]
                if ts > window_start
            ]
            
            # Count requests in window
            total = sum(count for _, count in self._windows[key])
            
            if total >= limit:
                # Calculate reset time
                if self._windows[key]:
                    oldest = min(ts for ts, _ in self._windows[key])
                    reset_time = int(oldest + window_seconds - now)
                else:
                    reset_time = window_seconds
                return False, 0, reset_time
            
            # Add this request
            self._windows[key].append((now, 1))
            remaining = limit - total - 1
            reset_time = window_seconds
            
            return True, remaining, reset_time
    
    async def get_usage(self, key: str, window_seconds: int) -> int:
        """Get current usage count."""
        now = time.time()
        window_start = now - window_seconds
        
        return sum(
            count for ts, count in self._windows.get(key, [])
            if ts > window_start
        )


# =============================================================================
# REDIS RATE LIMITER (Production)
# =============================================================================

class RedisRateLimiter:
    """Redis-backed rate limiter for distributed systems."""
    
    def __init__(self, redis_url: str):
        import redis.asyncio as redis
        self._redis = redis.from_url(redis_url)
    
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> Tuple[bool, int, int]:
        """Check if request is allowed using Redis sliding window."""
        now = time.time()
        window_key = f"ratelimit:{key}:{int(now // window_seconds)}"
        
        pipe = self._redis.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, window_seconds)
        results = await pipe.execute()
        
        current = results[0]
        
        if current > limit:
            ttl = await self._redis.ttl(window_key)
            return False, 0, ttl
        
        remaining = limit - current
        return True, remaining, window_seconds
    
    async def get_usage(self, key: str, window_seconds: int) -> int:
        """Get current usage count."""
        now = time.time()
        window_key = f"ratelimit:{key}:{int(now // window_seconds)}"
        count = await self._redis.get(window_key)
        return int(count) if count else 0


# =============================================================================
# RATE LIMITER FACTORY
# =============================================================================

def get_rate_limiter():
    """Get appropriate rate limiter based on environment."""
    if settings.REDIS_URL:
        return RedisRateLimiter(settings.REDIS_URL)
    return InMemoryRateLimiter()


rate_limiter = get_rate_limiter()


# =============================================================================
# RATE LIMIT MIDDLEWARE
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce rate limits.
    
    Extracts API key from request and applies tier-based limits.
    """
    
    def __init__(self, app, limiter=None):
        super().__init__(app)
        self.limiter = limiter or rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/healthz", "/ready"]:
            return await call_next(request)
        
        # Get API key and tier
        api_key = request.headers.get("X-API-Key", "")
        tier = self._get_tier(api_key)
        limits = RATE_LIMITS.get(tier, RATE_LIMITS["anonymous"])
        
        # Use API key or IP as rate limit key
        rate_key = api_key if api_key else self._get_client_ip(request)
        
        # Check rate limit (per minute)
        allowed, remaining, reset = await self.limiter.is_allowed(
            key=f"{rate_key}:minute",
            limit=limits["requests_per_minute"],
            window_seconds=60,
        )
        
        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_ip": self._get_client_ip(request),
                    "api_key": api_key[:8] + "..." if api_key else None,
                    "tier": tier,
                    "path": request.url.path,
                }
            )
            
            return Response(
                content='{"detail": "Rate limit exceeded. Please try again later."}',
                status_code=429,
                headers={
                    "Content-Type": "application/json",
                    "X-RateLimit-Limit": str(limits["requests_per_minute"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset),
                    "Retry-After": str(reset),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limits["requests_per_minute"])
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset)
        
        return response
    
    def _get_tier(self, api_key: str) -> str:
        """Get tier from API key."""
        from src.api.middleware.auth import api_key_manager
        
        if not api_key:
            return "anonymous"
        
        key_data = api_key_manager.validate(api_key)
        if not key_data:
            return "anonymous"
        
        return key_data.get("tier", "free")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, considering proxies."""
        # Check X-Forwarded-For (set by load balancers)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check X-Real-IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return request.client.host if request.client else "unknown"


# =============================================================================
# DECORATOR FOR CUSTOM RATE LIMITS
# =============================================================================

def rate_limit(requests: int, window: int = 60):
    """
    Decorator to apply custom rate limit to specific endpoints.
    
    Usage:
        @router.post("/expensive-operation")
        @rate_limit(requests=5, window=60)  # 5 per minute
        async def expensive_operation():
            ...
    """
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            api_key = request.headers.get("X-API-Key", "")
            rate_key = api_key or request.client.host
            
            allowed, remaining, reset = await rate_limiter.is_allowed(
                key=f"{rate_key}:{func.__name__}",
                limit=requests,
                window_seconds=window,
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for this endpoint. Retry after {reset} seconds.",
                    headers={"Retry-After": str(reset)},
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator

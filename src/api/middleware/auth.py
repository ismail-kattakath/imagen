# =============================================================================
# AUTHENTICATION MODULE
# =============================================================================
#
# Supports:
#   - API Key authentication (simple, good for M2M)
#   - JWT authentication (for user-based auth)
#   - Optional: Google Cloud Identity Platform integration
#
# =============================================================================

from datetime import datetime, timedelta

import jwt
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.core.config import settings
from src.core.logging import logger

# =============================================================================
# API KEY AUTHENTICATION
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyManager:
    """
    Manages API keys with tier-based rate limits.

    In production, this should be backed by a database (Firestore/PostgreSQL).
    """

    def __init__(self):
        # In production, load from database
        # For now, use environment-based config
        self._keys: dict[str, dict] = {}
        self._load_keys()

    def _load_keys(self):
        """Load API keys from config or database."""
        # Default development key
        if settings.is_development():
            self._keys["dev-key-12345"] = {
                "name": "Development",
                "tier": "free",
                "rate_limit": 100,  # requests per minute
                "daily_limit": 1000,
                "max_file_size_mb": 10,
                "created_at": datetime.utcnow(),
                "active": True,
            }

        # Load from environment if configured
        if settings.API_KEYS:
            for key_config in settings.API_KEYS:
                self._keys[key_config["key"]] = key_config

    def validate(self, api_key: str) -> dict | None:
        """Validate API key and return metadata."""
        if not api_key:
            return None

        key_data = self._keys.get(api_key)
        if not key_data:
            return None

        if not key_data.get("active", True):
            return None

        return key_data

    def get_rate_limit(self, api_key: str) -> int:
        """Get rate limit for API key."""
        key_data = self._keys.get(api_key, {})
        return key_data.get("rate_limit", 10)  # Default: 10/min

    def get_file_size_limit(self, api_key: str) -> int:
        """Get max file size in bytes."""
        key_data = self._keys.get(api_key, {})
        max_mb = key_data.get("max_file_size_mb", 5)
        return max_mb * 1024 * 1024


# Singleton instance
api_key_manager = APIKeyManager()


async def get_api_key(
    api_key: str = Security(api_key_header),
) -> dict:
    """
    Dependency to validate API key.

    Usage:
        @router.post("/endpoint")
        async def endpoint(auth: dict = Depends(get_api_key)):
            print(auth["tier"])  # "free", "pro", "enterprise"
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_data = api_key_manager.validate(api_key)
    if not key_data:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return {"api_key": api_key, **key_data}


async def get_optional_api_key(
    api_key: str = Security(api_key_header),
) -> dict | None:
    """Optional API key - returns None if not provided."""
    if not api_key:
        return None
    return api_key_manager.validate(api_key)


# =============================================================================
# JWT AUTHENTICATION
# =============================================================================

jwt_bearer = HTTPBearer(auto_error=False)


class JWTManager:
    """JWT token management."""

    def __init__(self):
        self.secret = settings.JWT_SECRET or "dev-secret-change-in-production"
        self.algorithm = "HS256"
        self.expiry_hours = 24

    def create_token(self, user_id: str, claims: dict = None) -> str:
        """Create a JWT token."""
        payload = {
            "sub": user_id,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.expiry_hours),
            **(claims or {}),
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict | None:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None


jwt_manager = JWTManager()


async def get_jwt_user(
    credentials: HTTPAuthorizationCredentials = Security(jwt_bearer),
) -> dict:
    """
    Dependency to validate JWT token.

    Usage:
        @router.post("/endpoint")
        async def endpoint(user: dict = Depends(get_jwt_user)):
            print(user["sub"])  # user ID
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = jwt_manager.verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


# =============================================================================
# FLEXIBLE AUTH (API Key OR JWT)
# =============================================================================


async def get_auth(
    request: Request,
    api_key: str = Security(api_key_header),
    jwt_credentials: HTTPAuthorizationCredentials = Security(jwt_bearer),
) -> dict:
    """
    Accept either API key or JWT token.

    Checks API key first, then JWT.
    """
    # Try API key first
    if api_key:
        key_data = api_key_manager.validate(api_key)
        if key_data:
            return {"type": "api_key", "api_key": api_key, **key_data}

    # Try JWT
    if jwt_credentials:
        payload = jwt_manager.verify_token(jwt_credentials.credentials)
        if payload:
            return {"type": "jwt", **payload}

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide X-API-Key header or Bearer token.",
    )

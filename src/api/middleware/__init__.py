# =============================================================================
# MIDDLEWARE PACKAGE
# =============================================================================
#
# Production-ready middleware for:
#   - Authentication (API keys, JWT)
#   - Rate limiting (per-user, tiered)
#   - Request validation & size limiting
#   - Structured logging & request tracing
#   - Prometheus metrics
#   - Error handling
#
# =============================================================================

from src.api.middleware.auth import (
    get_api_key,
    get_optional_api_key,
    get_jwt_user,
    get_auth,
    api_key_manager,
    jwt_manager,
)

from src.api.middleware.rate_limit import (
    RateLimitMiddleware,
    rate_limit,
    RATE_LIMITS,
)

from src.api.middleware.validation import (
    SizeLimitMiddleware,
    FileValidator,
    file_validator,
    get_validated_image,
    FILE_SIZE_LIMITS,
)

from src.api.middleware.logging import (
    RequestLoggingMiddleware,
    ctx_logger,
    get_request_id,
    setup_logging,
)

from src.api.middleware.metrics import (
    MetricsMiddleware,
    metrics_endpoint,
    metrics,
)

from src.api.middleware.errors import (
    register_exception_handlers,
    ImagenException,
    JobNotFoundException,
    JobProcessingException,
    QuotaExceededException,
    InvalidImageException,
)


__all__ = [
    # Auth
    "get_api_key",
    "get_optional_api_key",
    "get_jwt_user",
    "get_auth",
    "api_key_manager",
    "jwt_manager",
    
    # Rate limiting
    "RateLimitMiddleware",
    "rate_limit",
    "RATE_LIMITS",
    
    # Validation
    "SizeLimitMiddleware",
    "FileValidator",
    "file_validator",
    "get_validated_image",
    "FILE_SIZE_LIMITS",
    
    # Logging
    "RequestLoggingMiddleware",
    "ctx_logger",
    "get_request_id",
    "setup_logging",
    
    # Metrics
    "MetricsMiddleware",
    "metrics_endpoint",
    "metrics",
    
    # Errors
    "register_exception_handlers",
    "ImagenException",
    "JobNotFoundException",
    "JobProcessingException",
    "QuotaExceededException",
    "InvalidImageException",
]

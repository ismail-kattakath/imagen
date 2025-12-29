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
    api_key_manager,
    get_api_key,
    get_auth,
    get_jwt_user,
    get_optional_api_key,
    jwt_manager,
)
from src.api.middleware.errors import (
    ImagenException,
    InvalidImageException,
    JobNotFoundException,
    JobProcessingException,
    QuotaExceededException,
    register_exception_handlers,
)
from src.api.middleware.logging import (
    RequestLoggingMiddleware,
    ctx_logger,
    get_request_id,
    setup_logging,
)
from src.api.middleware.metrics import (
    MetricsMiddleware,
    metrics,
    metrics_endpoint,
)
from src.api.middleware.rate_limit import (
    RATE_LIMITS,
    RateLimitMiddleware,
    rate_limit,
)
from src.api.middleware.validation import (
    FILE_SIZE_LIMITS,
    FileValidator,
    SizeLimitMiddleware,
    file_validator,
    get_validated_image,
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

# =============================================================================
# LOGGING & REQUEST TRACING MODULE
# =============================================================================
#
# Provides:
#   - Structured JSON logging (for Cloud Logging)
#   - Request ID / Correlation ID tracking
#   - Request/Response logging
#   - Performance timing
#
# =============================================================================

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime

from fastapi import Request, Response
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import settings

# =============================================================================
# CONTEXT VARIABLES
# =============================================================================

# Request ID available throughout request lifecycle
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
start_time_var: ContextVar[float] = ContextVar("start_time", default=0)


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get()


# =============================================================================
# STRUCTURED JSON FORMATTER
# =============================================================================


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Compatible with Google Cloud Logging format.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_entry["request_id"] = request_id

        # Add OpenTelemetry trace context
        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, "032x")
            span_id = format(span_context.span_id, "016x")

            # Cloud Logging trace format
            log_entry["logging.googleapis.com/trace"] = (
                f"projects/{settings.GOOGLE_CLOUD_PROJECT}/traces/{trace_id}"
            )
            log_entry["logging.googleapis.com/spanId"] = span_id
            log_entry["logging.googleapis.com/trace_sampled"] = span_context.trace_flags.sampled

            # Also add as standard fields
            log_entry["trace_id"] = trace_id
            log_entry["span_id"] = span_id

        # Add extra fields
        if hasattr(record, "extra"):
            log_entry.update(record.extra)

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class DevelopmentFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        request_id = request_id_var.get()
        rid_str = f"[{request_id[:8]}] " if request_id else ""

        return (
            f"{color}{record.levelname:8}{self.RESET} "
            f"{rid_str}"
            f"{record.name}:{record.lineno} - {record.getMessage()}"
        )


# =============================================================================
# LOGGER SETUP
# =============================================================================


def setup_logging():
    """Configure application logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    root_logger.handlers = []

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    # Use appropriate formatter
    if settings.is_production():
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(DevelopmentFormatter())

    root_logger.addHandler(handler)

    # Set levels for noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return root_logger


# Application logger
logger = setup_logging()


# =============================================================================
# CONTEXT-AWARE LOGGER
# =============================================================================


class ContextLogger:
    """Logger that automatically includes request context."""

    def __init__(self, name: str = "imagen"):
        self._logger = logging.getLogger(name)

    def _log(self, level: int, msg: str, extra: dict = None):
        record_extra = extra or {}
        record_extra["request_id"] = request_id_var.get()

        self._logger.log(level, msg, extra={"extra": record_extra})

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, kwargs)

    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, kwargs)

    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, kwargs)


# Default context-aware logger
ctx_logger = ContextLogger()


# =============================================================================
# REQUEST LOGGING MIDDLEWARE
# =============================================================================


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging.

    Features:
    - Generates unique request ID
    - Logs request details
    - Logs response status and timing
    - Adds request ID to response headers
    """

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = self._get_or_create_request_id(request)
        request_id_var.set(request_id)

        # Record start time
        start_time = time.time()
        start_time_var.set(start_time)

        # Log request
        self._log_request(request, request_id)

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception
            duration = time.time() - start_time
            ctx_logger.error(
                "Request failed with exception",
                path=request.url.path,
                method=request.method,
                duration_ms=round(duration * 1000, 2),
                error=str(e),
            )
            raise

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        self._log_response(request, response, duration, request_id)

        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"

        return response

    def _get_or_create_request_id(self, request: Request) -> str:
        """Get request ID from header or generate new one."""
        # Check for existing ID (from load balancer or client)
        existing = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Correlation-ID")
            or request.headers.get("X-Cloud-Trace-Context", "").split("/")[0]
        )

        if existing:
            return existing

        return str(uuid.uuid4())

    def _log_request(self, request: Request, request_id: str):
        """Log incoming request."""
        # Skip health checks in production
        if request.url.path in ["/health", "/healthz", "/ready"]:
            return

        ctx_logger.info(
            "Request received",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else None,
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("User-Agent", "")[:100],
            content_type=request.headers.get("Content-Type"),
            content_length=request.headers.get("Content-Length"),
        )

    def _log_response(
        self,
        request: Request,
        response: Response,
        duration: float,
        request_id: str,
    ):
        """Log outgoing response."""
        # Skip health checks in production
        if request.url.path in ["/health", "/healthz", "/ready"]:
            return

        log_func = ctx_logger.info if response.status_code < 400 else ctx_logger.warning

        log_func(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

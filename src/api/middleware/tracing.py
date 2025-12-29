# =============================================================================
# OPENTELEMETRY TRACING MIDDLEWARE
# =============================================================================
#
# Provides:
#   - Automatic span creation for all HTTP requests
#   - Integration with existing request logging
#   - Custom span attributes (request ID, user info, etc.)
#   - Error and exception tracking in spans
#
# =============================================================================

from fastapi import Request
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware.logging import get_request_id


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Enhanced tracing middleware that adds custom attributes to spans.

    Works alongside FastAPIInstrumentor to add business-specific context:
    - Request ID (for correlation with logs)
    - User/API key information
    - Custom business attributes
    """

    async def dispatch(self, request: Request, call_next):
        # Get current span (created by FastAPIInstrumentor)
        span = trace.get_current_span()

        if span and span.is_recording():
            # Add request ID for log correlation
            request_id = get_request_id()
            if request_id:
                span.set_attribute("request.id", request_id)

            # Add client information
            client_ip = self._get_client_ip(request)
            if client_ip:
                span.set_attribute("client.ip", client_ip)

            # Add user agent
            user_agent = request.headers.get("User-Agent", "")
            if user_agent:
                span.set_attribute("http.user_agent", user_agent[:200])

            # Add custom headers if present
            if "X-API-Key" in request.headers:
                # Don't log the actual key, just that it was present
                span.set_attribute("auth.api_key_present", True)

        # Process request
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Record exception in span
            if span and span.is_recording():
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# =============================================================================
# SETUP FUNCTION
# =============================================================================


def setup_fastapi_instrumentation(app):
    """
    Configure OpenTelemetry instrumentation for FastAPI.

    Args:
        app: FastAPI application instance
    """
    # Use FastAPIInstrumentor for automatic instrumentation
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="/health,/healthz,/ready,/metrics",  # Don't trace health checks
    )

# =============================================================================
# METRICS MODULE (Prometheus)
# =============================================================================
#
# Exposes Prometheus metrics for the API layer:
#   - Request counts by endpoint, method, status
#   - Request latency histograms
#   - Active requests gauge
#   - Business metrics (jobs created, file uploads)
#
# NOTE: Job completion metrics are tracked by workers, not the API.
#       See src/workers/triton_worker.py for worker metrics.
#
# Endpoint: GET /metrics
#
# =============================================================================

import time

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

from src.core.config import settings

# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

# Application info
APP_INFO = Info(
    "imagen_app",
    "Application information",
)
APP_INFO.info(
    {
        "version": "1.0.0",
        "environment": "production" if settings.is_production() else "development",
    }
)

# Request metrics
REQUEST_COUNT = Counter(
    "imagen_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "imagen_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REQUESTS_IN_PROGRESS = Gauge(
    "imagen_http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    ["method", "endpoint"],
)

# Error metrics
ERROR_COUNT = Counter(
    "imagen_errors_total",
    "Total errors by type and endpoint",
    ["type", "endpoint"],
)

# Business metrics - API layer only tracks job creation
JOBS_CREATED = Counter(
    "imagen_jobs_created_total",
    "Total jobs created (submitted to queue)",
    ["job_type"],
)

# File metrics
FILE_SIZE_BYTES = Histogram(
    "imagen_upload_file_size_bytes",
    "Uploaded file sizes in bytes",
    ["job_type"],
    buckets=(
        100 * 1024,  # 100 KB
        500 * 1024,  # 500 KB
        1 * 1024 * 1024,  # 1 MB
        5 * 1024 * 1024,  # 5 MB
        10 * 1024 * 1024,  # 10 MB
        25 * 1024 * 1024,  # 25 MB
        50 * 1024 * 1024,  # 50 MB
    ),
)


# =============================================================================
# METRICS MIDDLEWARE
# =============================================================================


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""

    async def dispatch(self, request: Request, call_next):
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        # Get endpoint name (route path, not actual URL)
        endpoint = self._get_endpoint(request)
        method = request.method

        # Track in-progress requests
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()

        # Time the request
        start_time = time.time()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            ERROR_COUNT.labels(type=type(e).__name__, endpoint=endpoint).inc()
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()

            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            REQUESTS_IN_PROGRESS.labels(
                method=method,
                endpoint=endpoint,
            ).dec()

        return response

    def _get_endpoint(self, request: Request) -> str:
        """Get the route pattern (e.g., /api/v1/jobs/{job_id})."""
        # Try to match against routes to get pattern
        for route in request.app.routes:
            match, scope = route.matches(request.scope)
            if match == Match.FULL:
                return route.path

        # Fall back to actual path (normalize IDs)
        return self._normalize_path(request.url.path)

    def _normalize_path(self, path: str) -> str:
        """Normalize path to prevent high cardinality."""
        import re

        # Replace UUIDs with placeholder
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            path,
        )
        # Replace numeric IDs
        path = re.sub(r"/\d+", "/{id}", path)
        return path


# =============================================================================
# METRICS ENDPOINT
# =============================================================================


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


# =============================================================================
# BUSINESS METRICS HELPERS
# =============================================================================


class MetricsRecorder:
    """
    Helper class to record business metrics in the API layer.

    NOTE: Job completion and SLO metrics are tracked by workers.
    This class only handles API-layer metrics like job creation and uploads.
    """

    @staticmethod
    def job_created(job_type: str):
        """Record job creation (job submitted to queue)."""
        JOBS_CREATED.labels(job_type=job_type).inc()

    @staticmethod
    def file_uploaded(job_type: str, size_bytes: int):
        """Record file upload size."""
        FILE_SIZE_BYTES.labels(job_type=job_type).observe(size_bytes)

    @staticmethod
    def record_error(error_type: str, endpoint: str):
        """Record an error."""
        ERROR_COUNT.labels(type=error_type, endpoint=endpoint).inc()


metrics = MetricsRecorder()

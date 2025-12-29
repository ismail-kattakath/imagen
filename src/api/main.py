# =============================================================================
# IMAGEN API - Main Application
# =============================================================================
#
# Production-ready FastAPI application with:
#   - Authentication (API keys, JWT)
#   - Rate limiting (tiered)
#   - Request validation & size limits
#   - Structured logging & request tracing
#   - Prometheus metrics
#   - Standardized error handling
#
# =============================================================================

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import (
    MetricsMiddleware,
    # Middleware classes
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    SizeLimitMiddleware,
    metrics_endpoint,
    # Functions
    register_exception_handlers,
    setup_logging,
)
from src.api.routes import health, images, jobs
from src.core.config import settings

# =============================================================================
# LIFESPAN (Startup/Shutdown)
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger = setup_logging()
    logger.info("Starting Imagen API...")

    # Initialize OpenTelemetry
    if settings.otel_enabled:
        from src.core.telemetry import setup_auto_instrumentation, setup_telemetry, shutdown_telemetry

        environment = "production" if settings.is_production() else "development"
        setup_telemetry(
            service_name=settings.otel_service_name or "imagen-api",
            environment=environment,
            endpoint=settings.otel_exporter_otlp_endpoint,
            enable_console_export=settings.otel_exporter_console,
            enable_gcp_trace=settings.otel_exporter_gcp_trace,
        )
        setup_auto_instrumentation()
        logger.info("OpenTelemetry initialized")

    # Validate configuration
    if settings.is_production():
        logger.info("Running in PRODUCTION mode")
        try:
            settings.validate_gcp_config()
            logger.info("GCP configuration validated")

            # Validate CORS in production
            if not settings.cors_origins:
                raise ValueError("CORS_ORIGINS must be set in production (not default to wildcard)")
            logger.info(f"CORS origins: {settings.cors_origins}")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise
    else:
        logger.info("Running in DEVELOPMENT mode")

    yield

    # Shutdown
    if settings.otel_enabled:
        shutdown_telemetry()
    logger.info("Shutting down Imagen API...")


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title="Imagen API",
    description="AI-powered image processing microservices",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production() else None,  # Disable in prod
    redoc_url="/redoc" if not settings.is_production() else None,
    lifespan=lifespan,
)


# =============================================================================
# MIDDLEWARE (Order matters! First added = outermost = runs first)
# =============================================================================

# 1. Request logging (outermost - logs everything)
app.add_middleware(RequestLoggingMiddleware)

# 2. OpenTelemetry tracing (adds custom span attributes)
if settings.otel_enabled:
    from src.api.middleware.tracing import TracingMiddleware, setup_fastapi_instrumentation

    setup_fastapi_instrumentation(app)
    app.add_middleware(TracingMiddleware)

# 3. Metrics collection
app.add_middleware(MetricsMiddleware)

# 4. Rate limiting
app.add_middleware(RateLimitMiddleware)

# 5. Request size limiting
app.add_middleware(SizeLimitMiddleware)

# 6. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

register_exception_handlers(app)


# =============================================================================
# ROUTES
# =============================================================================

# Health checks (no auth required)
app.include_router(health.router, tags=["health"])

# API routes
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(images.router, prefix="/api/v1/images", tags=["images"])

# Metrics endpoint (for Prometheus scraping)
app.add_api_route("/metrics", metrics_endpoint, methods=["GET"], include_in_schema=False)


# =============================================================================
# ROOT ENDPOINT
# =============================================================================


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - basic info."""
    return {
        "service": "Imagen API",
        "version": "1.0.0",
        "docs": "/docs" if not settings.is_production() else "disabled",
        "health": "/health",
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, jobs, images
from src.core.config import settings

app = FastAPI(
    title="Imagen API",
    description="AI-powered image processing microservices",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(images.router, prefix="/api/v1/images", tags=["images"])


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    from src.core.logging import logger

    # Validate GCP config in production mode
    if settings.is_production():
        logger.info("Validating GCP configuration for production...")
        try:
            settings.validate_gcp_config()
            logger.info("GCP configuration validated successfully")
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    else:
        logger.info("Running in development mode - skipping GCP validation")

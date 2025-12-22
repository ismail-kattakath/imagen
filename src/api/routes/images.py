# =============================================================================
# IMAGE PROCESSING ROUTES
# =============================================================================
#
# Endpoints for image processing with:
#   - Authentication (API key required)
#   - File validation
#   - Usage tracking
#
# =============================================================================

from fastapi import APIRouter, UploadFile, File, Depends, Request
from PIL import Image
import io
import uuid

from src.api.schemas import JobResponse, UpscaleParams, EnhanceParams, StyleParams
from src.api.middleware import (
    get_api_key,
    get_validated_image,
    metrics,
    ctx_logger,
    FILE_SIZE_LIMITS,
)
from src.services import get_job_service
from src.services.storage import get_storage_service
from src.services.queue import get_queue_service

router = APIRouter()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _process_upload(
    file: UploadFile,
    job_type: str,
    params: dict,
    auth: dict,
) -> JobResponse:
    """Common logic for processing image uploads."""
    job_id = str(uuid.uuid4())
    
    # Get services
    storage = get_storage_service()
    queue = get_queue_service()
    job_service = get_job_service()
    
    # Read and validate image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Record metrics
    metrics.file_uploaded(job_type, len(contents))
    metrics.job_created(job_type)
    
    # Log
    ctx_logger.info(
        "Processing upload",
        job_id=job_id,
        job_type=job_type,
        file_size=len(contents),
        image_size=f"{image.size[0]}x{image.size[1]}",
        api_key=auth.get("name", "unknown"),
    )
    
    # Upload input image
    input_path = f"inputs/{job_id}/{file.filename}"
    storage.upload_image(image, input_path)
    
    # Create job record with user info
    job_service.create(
        job_id=job_id,
        job_type=job_type,
        input_path=input_path,
        params=params,
        metadata={
            "api_key_name": auth.get("name"),
            "tier": auth.get("tier"),
        },
    )
    
    # Enqueue for processing
    queue.enqueue(
        topic_name=f"{job_type}-jobs",
        payload={"input_path": input_path, "params": params},
        job_id=job_id,
    )
    
    return JobResponse(job_id=job_id, status="queued")


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/upscale", response_model=JobResponse)
async def upscale_image(
    request: Request,
    file: UploadFile = File(..., description="Image file to upscale"),
    params: UpscaleParams = Depends(),
    auth: dict = Depends(get_api_key),
):
    """
    Upscale an image by 4x using Real-ESRGAN.
    
    Requires API key authentication.
    
    **Rate limits:**
    - Free: 10/min, 5MB max file
    - Pro: 60/min, 25MB max file
    - Enterprise: 300/min, 100MB max file
    """
    return await _process_upload(file, "upscale", params.model_dump(), auth)


@router.post("/enhance", response_model=JobResponse)
async def enhance_image(
    request: Request,
    file: UploadFile = File(..., description="Image file to enhance"),
    params: EnhanceParams = Depends(),
    auth: dict = Depends(get_api_key),
):
    """
    Enhance image quality using SDXL Refiner.
    
    Requires API key authentication.
    """
    return await _process_upload(file, "enhance", params.model_dump(), auth)


@router.post("/style/comic", response_model=JobResponse)
async def comic_style(
    request: Request,
    file: UploadFile = File(..., description="Image file to convert"),
    params: StyleParams = Depends(),
    auth: dict = Depends(get_api_key),
):
    """
    Convert image to comic/cartoon style.
    
    Requires API key authentication.
    """
    return await _process_upload(file, "style-comic", params.model_dump(), auth)


@router.post("/style/aged", response_model=JobResponse)
async def aged_style(
    request: Request,
    file: UploadFile = File(..., description="Image file to age"),
    params: StyleParams = Depends(),
    auth: dict = Depends(get_api_key),
):
    """
    Apply vintage/aged effect to image.
    
    Requires API key authentication.
    """
    return await _process_upload(file, "style-aged", params.model_dump(), auth)


@router.post("/background/remove", response_model=JobResponse)
async def remove_background(
    request: Request,
    file: UploadFile = File(..., description="Image file for background removal"),
    auth: dict = Depends(get_api_key),
):
    """
    Remove background from image using RMBG-1.4.
    
    Requires API key authentication.
    """
    return await _process_upload(file, "background-remove", {}, auth)


# =============================================================================
# INFO ENDPOINT
# =============================================================================

@router.get("/limits")
async def get_limits(auth: dict = Depends(get_api_key)):
    """Get rate limits and file size limits for authenticated user."""
    from src.api.middleware import RATE_LIMITS
    
    tier = auth.get("tier", "free")
    
    return {
        "tier": tier,
        "rate_limits": RATE_LIMITS.get(tier, RATE_LIMITS["free"]),
        "max_file_size_mb": FILE_SIZE_LIMITS.get(tier, 5 * 1024 * 1024) // (1024 * 1024),
        "max_image_dimension": 8192,
    }

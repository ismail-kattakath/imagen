from fastapi import APIRouter, UploadFile, File, Depends
from PIL import Image
import io
import uuid

from src.api.schemas import JobResponse, UpscaleParams, EnhanceParams, StyleParams
from src.services import GCSStorageService, PubSubQueueService, JobService

router = APIRouter()
storage = GCSStorageService()
queue = PubSubQueueService()
job_service = JobService()


async def _process_upload(file: UploadFile, job_type: str, params: dict) -> JobResponse:
    """Common logic for processing image uploads."""
    job_id = str(uuid.uuid4())
    
    # Read and validate image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Upload input image
    input_path = f"inputs/{job_id}/{file.filename}"
    storage.upload_image(image, input_path)
    
    # Create job record
    job_service.create(
        job_id=job_id,
        job_type=job_type,
        input_path=input_path,
        params=params,
    )
    
    # Enqueue for processing
    queue.enqueue(
        topic_name=f"{job_type}-jobs",
        payload={"input_path": input_path, "params": params},
        job_id=job_id,
    )
    
    return JobResponse(job_id=job_id, status="queued")


@router.post("/upscale", response_model=JobResponse)
async def upscale_image(
    file: UploadFile = File(...),
    params: UpscaleParams = Depends(),
):
    """Upscale an image by 4x."""
    return await _process_upload(file, "upscale", params.model_dump())


@router.post("/enhance", response_model=JobResponse)
async def enhance_image(
    file: UploadFile = File(...),
    params: EnhanceParams = Depends(),
):
    """Enhance image quality."""
    return await _process_upload(file, "enhance", params.model_dump())


@router.post("/style/comic", response_model=JobResponse)
async def comic_style(
    file: UploadFile = File(...),
    params: StyleParams = Depends(),
):
    """Convert image to comic/cartoon style."""
    return await _process_upload(file, "style-comic", params.model_dump())


@router.post("/background/remove", response_model=JobResponse)
async def remove_background(file: UploadFile = File(...)):
    """Remove background from image."""
    return await _process_upload(file, "background-remove", {})

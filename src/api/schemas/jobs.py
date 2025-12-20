from pydantic import BaseModel
from datetime import datetime
from src.services.jobs import JobStatus


class JobCreate(BaseModel):
    """Request to create a job."""
    job_type: str
    params: dict = {}


class JobResponse(BaseModel):
    """Response after creating a job."""
    job_id: str
    status: JobStatus
    message: str = "Job queued successfully"


class JobDetail(BaseModel):
    """Full job details."""
    job_id: str
    type: str
    status: JobStatus
    input_path: str | None = None
    output_path: str | None = None
    output_url: str | None = None
    params: dict = {}
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str | None = None

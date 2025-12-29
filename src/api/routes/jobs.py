from fastapi import APIRouter, HTTPException

from src.api.schemas import ErrorResponse, JobDetail
from src.core.exceptions import JobNotFoundError
from src.services import GCSStorageService, JobService, JobStatus

router = APIRouter()
job_service = JobService()
storage = GCSStorageService()


@router.get(
    "/{job_id}",
    response_model=JobDetail,
    responses={404: {"model": ErrorResponse}},
)
async def get_job(job_id: str):
    """Get job status and details."""
    try:
        job = job_service.get(job_id)

        # Generate signed URL if completed
        output_url = None
        if job["status"] == JobStatus.COMPLETED and job.get("output_path"):
            output_url = storage.get_signed_url(job["output_path"])

        return JobDetail(
            job_id=job["job_id"],
            type=job["type"],
            status=job["status"],
            input_path=job.get("input_path"),
            output_path=job.get("output_path"),
            output_url=output_url,
            params=job.get("params", {}),
            error=job.get("error"),
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )

    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

from google.cloud import firestore
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.config import settings
from src.core.logging import logger
from src.core.exceptions import JobNotFoundError


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobService:
    """Firestore-based job tracking service."""
    
    COLLECTION = "jobs"
    
    def __init__(self):
        self._db: firestore.Client | None = None
    
    @property
    def db(self) -> firestore.Client:
        if self._db is None:
            self._db = firestore.Client(project=settings.google_cloud_project)
        return self._db
    
    @property
    def collection(self):
        return self.db.collection(self.COLLECTION)
    
    def create(
        self,
        job_id: str,
        job_type: str,
        input_path: str,
        params: dict[str, Any] | None = None,
    ) -> dict:
        """Create a new job record."""
        now = datetime.now(timezone.utc)
        
        job_data = {
            "job_id": job_id,
            "type": job_type,
            "status": JobStatus.QUEUED,
            "input_path": input_path,
            "output_path": None,
            "params": params or {},
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        
        self.collection.document(job_id).set(job_data)
        logger.info(f"Created job {job_id}")
        return job_data
    
    def get(self, job_id: str) -> dict:
        """Get a job by ID."""
        doc = self.collection.document(job_id).get()
        if not doc.exists:
            raise JobNotFoundError(f"Job {job_id} not found")
        return doc.to_dict()
    
    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        output_path: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update job status."""
        update_data = {
            "status": status,
            "updated_at": datetime.now(timezone.utc),
        }
        
        if output_path:
            update_data["output_path"] = output_path
        if error:
            update_data["error"] = error
        
        self.collection.document(job_id).update(update_data)
        logger.info(f"Updated job {job_id} to {status}")

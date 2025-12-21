from abc import ABC, abstractmethod
from typing import Type

from src.pipelines.base import BasePipeline
from src.services import GCSStorageService, PubSubQueueService, JobService, JobStatus
from src.core.logging import logger


class BaseWorker(ABC):
    """Base class for GPU workers."""
    
    def __init__(
        self,
        pipeline_cls: Type[BasePipeline],
        subscription_name: str,
    ):
        self.pipeline = pipeline_cls()
        self.subscription_name = subscription_name
        self.storage = GCSStorageService()
        self.queue = PubSubQueueService()
        self.job_service = JobService()
    
    def process_message(self, data: dict) -> None:
        """Process a single job message."""
        job_id = data["job_id"]
        input_path = data["input_path"]
        params = data.get("params", {})
        
        try:
            # Update status to processing
            self.job_service.update_status(job_id, JobStatus.PROCESSING)
            
            # Download input image
            logger.info(f"Processing job {job_id}")
            image = self.storage.download_image(input_path)
            
            # Process image
            result = self.pipeline.process(image, **params)
            
            # Upload result
            output_path = f"outputs/{job_id}/result.png"
            self.storage.upload_image(result, output_path)
            
            # Update job as completed
            self.job_service.update_status(
                job_id,
                JobStatus.COMPLETED,
                output_path=output_path,
            )
            logger.info(f"Completed job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed job {job_id}: {e}", exc_info=True)
            self.job_service.update_status(
                job_id,
                JobStatus.FAILED,
                error=str(e),
            )
            # Don't re-raise to prevent worker crash and message re-delivery
            # The job is already marked as failed in the database
    
    def run(self) -> None:
        """Start the worker."""
        from src.core.config import settings

        # Validate GCP configuration before starting (production only)
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

        logger.info(f"Starting worker for {self.subscription_name}")

        # Load model into GPU memory
        self.pipeline.load()

        # Start listening for messages
        self.queue.subscribe(
            subscription_name=self.subscription_name,
            callback=self.process_message,
        )

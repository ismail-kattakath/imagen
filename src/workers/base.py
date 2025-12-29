from abc import ABC, abstractmethod
from typing import Type
import time
import threading

from prometheus_client import start_http_server, Counter, Histogram, Gauge, Info

from src.pipelines.base import BasePipeline
from src.services import JobStatus, get_job_service
from src.services.queue import get_queue_service
from src.services.storage import get_storage_service
from src.core.logging import logger


# =============================================================================
# WORKER METRICS (Prometheus)
# =============================================================================
# These metrics are scraped by Google Managed Prometheus via PodMonitoring

WORKER_INFO = Info(
    "imagen_worker",
    "Worker information",
)

JOBS_IN_PROGRESS = Gauge(
    "imagen_jobs_in_progress",
    "Number of jobs currently being processed",
    ["job_type"],
)

JOBS_COMPLETED = Counter(
    "imagen_jobs_completed_total",
    "Total jobs completed by this worker",
    ["job_type", "status"],
)

JOB_PROCESSING_TIME = Histogram(
    "imagen_job_processing_seconds",
    "Job processing time in seconds",
    ["job_type"],
    buckets=(1, 5, 10, 15, 30, 45, 60, 90, 120, 180, 300, 600),
)

MODEL_LOAD_TIME = Histogram(
    "imagen_model_load_seconds",
    "Time to load ML model into GPU memory",
    ["model_name"],
    buckets=(1, 5, 10, 30, 60, 120, 300),
)

WORKER_ERRORS = Counter(
    "imagen_worker_errors_total",
    "Total errors encountered by worker",
    ["job_type", "error_type"],
)

# SLO tracking
JOB_SLO_MET = Counter(
    "imagen_job_slo_met_total",
    "Jobs completed within SLO target latency",
    ["job_type"],
)

JOB_SLO_VIOLATED = Counter(
    "imagen_job_slo_violated_total",
    "Jobs that exceeded SLO target latency",
    ["job_type"],
)

# SLO thresholds by job type (in seconds)
SLO_THRESHOLDS = {
    "upscale": 30,
    "enhance": 45,
    "style-comic": 60,
    "style-aged": 60,
    "background-remove": 15,
}


class BaseWorker(ABC):
    """Base class for GPU workers with Prometheus metrics."""

    # Metrics server port (exposed for GMP scraping)
    METRICS_PORT = 8080

    def __init__(
        self,
        pipeline_cls: Type[BasePipeline],
        subscription_name: str,
    ):
        self.pipeline = pipeline_cls()
        self.subscription_name = subscription_name
        self.storage = get_storage_service()
        self.queue = get_queue_service()
        self.job_service = get_job_service()
        
        # Derive job type from class name (e.g., UpscaleWorker -> upscale)
        self.job_type = self._get_job_type()
        
        # Set worker info metric
        WORKER_INFO.info({
            "job_type": self.job_type,
            "subscription": subscription_name,
        })

    def _get_job_type(self) -> str:
        """Derive job type from worker class name."""
        class_name = self.__class__.__name__
        # UpscaleWorker -> upscale, StyleAgedWorker -> style-aged
        job_type = class_name.replace("Worker", "")
        # Convert CamelCase to kebab-case
        import re
        job_type = re.sub(r'(?<!^)(?=[A-Z])', '-', job_type).lower()
        return job_type

    def _start_metrics_server(self) -> None:
        """Start Prometheus metrics HTTP server in background thread."""
        def serve():
            logger.info(f"Starting metrics server on port {self.METRICS_PORT}")
            start_http_server(self.METRICS_PORT)
        
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        logger.info(f"Metrics available at http://localhost:{self.METRICS_PORT}/metrics")

    def process_message(self, data: dict) -> None:
        """Process a single job message with metrics tracking."""
        job_id = data["job_id"]
        input_path = data["input_path"]
        params = data.get("params", {})
        
        # Track job in progress
        JOBS_IN_PROGRESS.labels(job_type=self.job_type).inc()
        start_time = time.time()
        
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
            
            # Record success metrics
            duration = time.time() - start_time
            JOBS_COMPLETED.labels(job_type=self.job_type, status="success").inc()
            JOB_PROCESSING_TIME.labels(job_type=self.job_type).observe(duration)
            
            # Track SLO
            slo_threshold = SLO_THRESHOLDS.get(self.job_type, 60)
            if duration <= slo_threshold:
                JOB_SLO_MET.labels(job_type=self.job_type).inc()
            else:
                JOB_SLO_VIOLATED.labels(job_type=self.job_type).inc()
                logger.warning(
                    f"Job {job_id} exceeded SLO: {duration:.2f}s > {slo_threshold}s"
                )
            
            logger.info(f"Completed job {job_id} in {duration:.2f}s")
            
        except Exception as e:
            # Record failure metrics
            duration = time.time() - start_time
            error_type = type(e).__name__
            
            JOBS_COMPLETED.labels(job_type=self.job_type, status="failed").inc()
            JOB_PROCESSING_TIME.labels(job_type=self.job_type).observe(duration)
            WORKER_ERRORS.labels(job_type=self.job_type, error_type=error_type).inc()
            
            logger.error(f"Failed job {job_id}: {e}", exc_info=True)
            self.job_service.update_status(
                job_id,
                JobStatus.FAILED,
                error=str(e),
            )
            # Don't re-raise to prevent worker crash and message re-delivery
            # The job is already marked as failed in the database
            
        finally:
            # Always decrement in-progress counter
            JOBS_IN_PROGRESS.labels(job_type=self.job_type).dec()

    def run(self) -> None:
        """Start the worker with metrics server."""
        from src.core.config import settings

        # Start Prometheus metrics server
        self._start_metrics_server()

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

        # Load model into GPU memory with timing
        logger.info(f"Loading model for {self.job_type}...")
        load_start = time.time()
        self.pipeline.load()
        load_duration = time.time() - load_start
        
        MODEL_LOAD_TIME.labels(model_name=self.job_type).observe(load_duration)
        logger.info(f"Model loaded in {load_duration:.2f}s")

        # Start listening for messages
        self.queue.subscribe(
            subscription_name=self.subscription_name,
            callback=self.process_message,
        )

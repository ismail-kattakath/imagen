"""
Triton-based Worker Base Class

This is an alternative to the standard BaseWorker that uses Triton
Inference Server instead of loading models directly.

Benefits:
- Automatic batching handled by Triton
- Multi-model serving on single GPU
- No model loading in worker (faster startup)
- Built-in metrics from Triton
"""

from abc import ABC, abstractmethod
from typing import Callable
import time
import threading

from prometheus_client import start_http_server, Counter, Histogram, Gauge, Info

from src.services import JobStatus, get_job_service
from src.services.queue import get_queue_service
from src.services.storage import get_storage_service
from src.services.triton import get_triton_client
from src.core.logging import logger


# =============================================================================
# WORKER METRICS
# =============================================================================

WORKER_INFO = Info("imagen_triton_worker", "Triton worker information")

JOBS_IN_PROGRESS = Gauge(
    "imagen_triton_jobs_in_progress",
    "Jobs currently being processed via Triton",
    ["job_type"],
)

JOBS_COMPLETED = Counter(
    "imagen_triton_jobs_completed_total",
    "Jobs completed via Triton",
    ["job_type", "status"],
)

JOB_PROCESSING_TIME = Histogram(
    "imagen_triton_job_processing_seconds",
    "Job processing time (including Triton inference)",
    ["job_type"],
    buckets=(0.5, 1, 2, 5, 10, 15, 30, 45, 60, 90, 120),
)

TRITON_LATENCY = Histogram(
    "imagen_triton_inference_seconds",
    "Pure Triton inference time",
    ["model_name"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 15, 30),
)

WORKER_ERRORS = Counter(
    "imagen_triton_worker_errors_total",
    "Errors in Triton workers",
    ["job_type", "error_type"],
)


class TritonWorker(ABC):
    """
    Base class for workers that use Triton for inference.
    
    Subclasses only need to implement:
    - model_name: The Triton model to call
    - process_with_triton(): The inference logic
    """
    
    METRICS_PORT = 8080
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the Triton model to use."""
        pass
    
    @property
    @abstractmethod
    def subscription_name(self) -> str:
        """Pub/Sub subscription to listen on."""
        pass
    
    def __init__(self):
        self.storage = get_storage_service()
        self.queue = get_queue_service()
        self.job_service = get_job_service()
        self.triton = get_triton_client()
        
        # Derive job type from model name
        self.job_type = self.model_name.replace("_", "-")
        
        WORKER_INFO.info({
            "job_type": self.job_type,
            "model_name": self.model_name,
            "subscription": self.subscription_name,
            "mode": "triton",
        })
    
    @abstractmethod
    def process_with_triton(self, image, params: dict):
        """
        Process image using Triton client.
        
        Args:
            image: PIL Image
            params: Job parameters
        
        Returns:
            Processed PIL Image
        """
        pass
    
    def _start_metrics_server(self) -> None:
        """Start Prometheus metrics HTTP server."""
        def serve():
            logger.info(f"Starting metrics server on port {self.METRICS_PORT}")
            start_http_server(self.METRICS_PORT)
        
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
    
    def process_message(self, data: dict) -> None:
        """Process a single job message."""
        job_id = data["job_id"]
        input_path = data["input_path"]
        params = data.get("params", {})
        
        JOBS_IN_PROGRESS.labels(job_type=self.job_type).inc()
        start_time = time.time()
        
        try:
            # Update status
            self.job_service.update_status(job_id, JobStatus.PROCESSING)
            
            # Download image
            logger.info(f"Processing job {job_id} via Triton/{self.model_name}")
            image = self.storage.download_image(input_path)
            
            # Run Triton inference
            triton_start = time.time()
            result = self.process_with_triton(image, params)
            triton_duration = time.time() - triton_start
            TRITON_LATENCY.labels(model_name=self.model_name).observe(triton_duration)
            
            # Upload result
            output_path = f"outputs/{job_id}/result.png"
            self.storage.upload_image(result, output_path)
            
            # Mark completed
            self.job_service.update_status(
                job_id,
                JobStatus.COMPLETED,
                output_path=output_path,
            )
            
            # Record metrics
            duration = time.time() - start_time
            JOBS_COMPLETED.labels(job_type=self.job_type, status="success").inc()
            JOB_PROCESSING_TIME.labels(job_type=self.job_type).observe(duration)
            
            logger.info(
                f"Completed job {job_id} in {duration:.2f}s "
                f"(Triton: {triton_duration:.2f}s)"
            )
            
        except Exception as e:
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
        finally:
            JOBS_IN_PROGRESS.labels(job_type=self.job_type).dec()
    
    def run(self) -> None:
        """Start the worker."""
        from src.core.config import settings
        
        # Start metrics server
        self._start_metrics_server()
        
        # Validate config
        if settings.is_production():
            logger.info("Validating GCP configuration...")
            settings.validate_gcp_config()
        
        # Wait for Triton to be ready
        logger.info(f"Waiting for Triton model '{self.model_name}' to be ready...")
        retries = 0
        while not self.triton.is_model_ready(self.model_name):
            retries += 1
            if retries > 30:  # 5 minutes max
                raise RuntimeError(f"Triton model '{self.model_name}' not ready")
            time.sleep(10)
        
        logger.info(f"Triton model '{self.model_name}' is ready!")
        logger.info(f"Starting worker for {self.subscription_name}")
        
        # Start listening
        self.queue.subscribe(
            subscription_name=self.subscription_name,
            callback=self.process_message,
        )


# =============================================================================
# CONCRETE TRITON WORKERS
# =============================================================================

class TritonUpscaleWorker(TritonWorker):
    """Upscale worker using Triton."""
    
    model_name = "upscale"
    subscription_name = "upscale-jobs-sub"
    
    def process_with_triton(self, image, params: dict):
        scale = params.get("scale", 4.0)
        return self.triton.upscale(image, scale=scale)


class TritonEnhanceWorker(TritonWorker):
    """Enhance worker using Triton."""
    
    model_name = "enhance"
    subscription_name = "enhance-jobs-sub"
    
    def process_with_triton(self, image, params: dict):
        prompt = params.get("prompt")
        strength = params.get("strength", 0.3)
        return self.triton.enhance(image, prompt=prompt, strength=strength)


class TritonBackgroundRemoveWorker(TritonWorker):
    """Background removal worker using Triton."""
    
    model_name = "background_remove"
    subscription_name = "background-remove-jobs-sub"
    
    def process_with_triton(self, image, params: dict):
        return self.triton.remove_background(image)


class TritonStyleComicWorker(TritonWorker):
    """Comic style worker using Triton."""
    
    model_name = "style_comic"
    subscription_name = "style-comic-jobs-sub"
    
    def process_with_triton(self, image, params: dict):
        return self.triton.style_comic(image)


class TritonStyleAgedWorker(TritonWorker):
    """Aged style worker using Triton."""
    
    model_name = "style_aged"
    subscription_name = "style-aged-jobs-sub"
    
    def process_with_triton(self, image, params: dict):
        return self.triton.style_aged(image)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
# Usage: python -m src.workers.triton_worker TritonUpscaleWorker

WORKER_CLASSES = {
    "TritonUpscaleWorker": TritonUpscaleWorker,
    "TritonEnhanceWorker": TritonEnhanceWorker,
    "TritonBackgroundRemoveWorker": TritonBackgroundRemoveWorker,
    "TritonStyleComicWorker": TritonStyleComicWorker,
    "TritonStyleAgedWorker": TritonStyleAgedWorker,
}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.workers.triton_worker <WorkerClassName>")
        print(f"Available workers: {', '.join(WORKER_CLASSES.keys())}")
        sys.exit(1)
    
    worker_name = sys.argv[1]
    
    if worker_name not in WORKER_CLASSES:
        print(f"Unknown worker: {worker_name}")
        print(f"Available workers: {', '.join(WORKER_CLASSES.keys())}")
        sys.exit(1)
    
    worker = WORKER_CLASSES[worker_name]()
    worker.run()

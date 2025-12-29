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

import threading
import time
from abc import ABC, abstractmethod

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

from src.core.logging import logger
from src.services import JobStatus, get_job_service
from src.services.queue import get_queue_service
from src.services.storage import get_storage_service
from src.services.triton import get_triton_client

# =============================================================================
# SLO CONFIGURATION
# =============================================================================
# Latency thresholds in seconds - jobs exceeding these are SLO violations

SLO_THRESHOLDS = {
    "upscale": 30,
    "enhance": 45,
    "style-comic": 60,
    "style-aged": 60,
    "background-remove": 15,
}

DEFAULT_SLO_THRESHOLD = 60  # Default for unknown job types

# =============================================================================
# WORKER METRICS
# =============================================================================
# NOTE: Metric names use 'imagen_' prefix (not 'imagen_triton_') to match
# the alerting rules in k8s/monitoring/rules.yaml

WORKER_INFO = Info("imagen_worker", "Worker information")

JOBS_IN_PROGRESS = Gauge(
    "imagen_jobs_in_progress",
    "Jobs currently being processed",
    ["job_type"],
)

JOBS_COMPLETED = Counter(
    "imagen_jobs_completed_total",
    "Total jobs completed",
    ["job_type", "status"],
)

JOB_PROCESSING_TIME = Histogram(
    "imagen_job_processing_seconds",
    "Job processing time in seconds",
    ["job_type"],
    buckets=(0.5, 1, 2, 5, 10, 15, 30, 45, 60, 90, 120),
)

TRITON_LATENCY = Histogram(
    "imagen_triton_inference_seconds",
    "Pure Triton inference time (excludes I/O)",
    ["model_name"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 15, 30),
)

WORKER_ERRORS = Counter(
    "imagen_worker_errors_total",
    "Errors in workers",
    ["job_type", "error_type"],
)

# SLO Metrics - tracks whether jobs meet latency SLOs
JOB_SLO_MET = Counter(
    "imagen_job_slo_met_total",
    "Jobs completed within SLO threshold",
    ["job_type"],
)

JOB_SLO_VIOLATED = Counter(
    "imagen_job_slo_violated_total",
    "Jobs that exceeded SLO threshold",
    ["job_type"],
)

# Queue depth gauge - updated periodically by workers
QUEUE_DEPTH = Gauge(
    "imagen_queue_depth",
    "Current queue depth (unprocessed messages)",
    ["queue_name"],
)


class TritonWorker(ABC):
    """
    Base class for workers that use Triton for inference.

    Subclasses only need to implement:
    - model_name: The Triton model to call
    - process_with_triton(): The inference logic
    """

    METRICS_PORT = 8080
    QUEUE_DEPTH_UPDATE_INTERVAL = 30  # seconds

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
        self._queue_depth_thread: threading.Thread | None = None
        self._running = False

        # Convert underscores to hyphens for job type naming
        # Model names use underscores (style_comic) but job types use hyphens (style-comic)
        self.job_type = self.model_name.replace("_", "-")

        # Get SLO threshold for this job type
        self.slo_threshold = SLO_THRESHOLDS.get(self.job_type, DEFAULT_SLO_THRESHOLD)

        WORKER_INFO.info(
            {
                "job_type": self.job_type,
                "model_name": self.model_name,
                "subscription": self.subscription_name,
                "mode": "triton",
                "slo_threshold_seconds": str(self.slo_threshold),
            }
        )

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

    def _start_queue_depth_monitor(self) -> None:
        """Start background thread to periodically update queue depth metric."""

        def monitor():
            while self._running:
                try:
                    depth = self._get_queue_depth()
                    if depth is not None:
                        QUEUE_DEPTH.labels(queue_name=self.subscription_name).set(depth)
                except Exception as e:
                    logger.warning(f"Failed to get queue depth: {e}")
                time.sleep(self.QUEUE_DEPTH_UPDATE_INTERVAL)

        self._queue_depth_thread = threading.Thread(target=monitor, daemon=True)
        self._queue_depth_thread.start()
        logger.info(f"Started queue depth monitor (interval: {self.QUEUE_DEPTH_UPDATE_INTERVAL}s)")

    def _get_queue_depth(self) -> int | None:
        """
        Get the current queue depth (undelivered messages).

        For Pub/Sub, uses the Monitoring API. For Redis, uses LLEN.
        Returns None if unable to retrieve.
        """
        try:
            # Check if using Redis (local dev)
            if hasattr(self.queue, 'redis'):
                queue_name = self.subscription_name.removesuffix("-sub")
                return self.queue.redis.llen(queue_name)

            # For Pub/Sub, use the Admin API to get subscription info
            # Note: This requires additional permissions (roles/pubsub.viewer)
            from google.cloud import monitoring_v3
            from src.core.config import settings

            client = monitoring_v3.MetricServiceClient()
            project_name = f"projects/{settings.google_cloud_project}"

            # Query the Pub/Sub metric for undelivered messages
            now = time.time()
            interval = monitoring_v3.TimeInterval(
                {
                    "end_time": {"seconds": int(now)},
                    "start_time": {"seconds": int(now - 60)},
                }
            )

            results = client.list_time_series(
                request={
                    "name": project_name,
                    "filter": (
                        f'metric.type="pubsub.googleapis.com/subscription/num_undelivered_messages" '
                        f'AND resource.labels.subscription_id="{self.subscription_name}"'
                    ),
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                }
            )

            for result in results:
                if result.points:
                    return result.points[0].value.int64_value

            return 0  # No messages

        except ImportError:
            logger.debug("Monitoring API not available for queue depth")
            return None
        except Exception as e:
            logger.debug(f"Could not get queue depth: {e}")
            return None

    def _record_slo_metric(self, duration: float) -> None:
        """Record whether job met or violated SLO threshold."""
        if duration <= self.slo_threshold:
            JOB_SLO_MET.labels(job_type=self.job_type).inc()
        else:
            JOB_SLO_VIOLATED.labels(job_type=self.job_type).inc()
            logger.warning(
                f"SLO violated for {self.job_type}: {duration:.2f}s > {self.slo_threshold}s threshold"
            )

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
            self._record_slo_metric(duration)

            logger.info(
                f"Completed job {job_id} in {duration:.2f}s "
                f"(Triton: {triton_duration:.2f}s, SLO: {self.slo_threshold}s)"
            )

        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__

            JOBS_COMPLETED.labels(job_type=self.job_type, status="failed").inc()
            JOB_PROCESSING_TIME.labels(job_type=self.job_type).observe(duration)
            WORKER_ERRORS.labels(job_type=self.job_type, error_type=error_type).inc()
            # Failed jobs also count as SLO violations
            JOB_SLO_VIOLATED.labels(job_type=self.job_type).inc()

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

        self._running = True

        # Start metrics server
        self._start_metrics_server()

        # Start queue depth monitor
        self._start_queue_depth_monitor()

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
        logger.info(
            f"Starting worker for {self.subscription_name} "
            f"(SLO threshold: {self.slo_threshold}s)"
        )

        # Start listening
        try:
            self.queue.subscribe(
                subscription_name=self.subscription_name,
                callback=self.process_message,
            )
        finally:
            self._running = False


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

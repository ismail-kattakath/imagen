import json
import time
import uuid
from collections.abc import Callable
from concurrent.futures import TimeoutError
from typing import Protocol

from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.subscriber.message import Message
from opentelemetry import trace
from opentelemetry.propagate import inject
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from src.core.config import settings
from src.core.exceptions import QueueError
from src.core.logging import logger


class QueueService(Protocol):
    """Protocol for queue services."""

    def enqueue(self, topic_name: str, payload: dict, job_id: str | None = None) -> str:
        """Publish a message to a queue."""
        ...

    def subscribe(
        self,
        subscription_name: str,
        callback: Callable[[dict], None],
        ack_deadline: int = 600,
    ) -> None:
        """Subscribe to a queue with callback."""
        ...


class PubSubQueueService:
    """Google Cloud Pub/Sub queue service."""

    def __init__(self, project_id: str | None = None):
        self.project_id = project_id or settings.google_cloud_project
        self._publisher: pubsub_v1.PublisherClient | None = None
        self._subscriber: pubsub_v1.SubscriberClient | None = None

    @property
    def publisher(self) -> pubsub_v1.PublisherClient:
        if self._publisher is None:
            self._publisher = pubsub_v1.PublisherClient()
        return self._publisher

    @property
    def subscriber(self) -> pubsub_v1.SubscriberClient:
        if self._subscriber is None:
            self._subscriber = pubsub_v1.SubscriberClient()
        return self._subscriber

    def enqueue(
        self,
        topic_name: str,
        payload: dict,
        job_id: str | None = None,
    ) -> str:
        """Publish a message to a Pub/Sub topic with trace context."""
        try:
            job_id = job_id or str(uuid.uuid4())
            topic_path = self.publisher.topic_path(self.project_id, topic_name)

            message_data = {
                "job_id": job_id,
                **payload,
            }

            # Inject trace context into message attributes
            carrier: dict[str, str] = {}
            inject(carrier)  # Injects current trace context into carrier

            message_bytes = json.dumps(message_data).encode("utf-8")
            future = self.publisher.publish(
                topic_path,
                message_bytes,
                **carrier,  # Pass trace context as message attributes
            )
            message_id = future.result()

            logger.info(f"Published job {job_id} to {topic_name}")
            return job_id

        except Exception as e:
            raise QueueError(f"Failed to enqueue message: {e}") from e

    def subscribe(
        self,
        subscription_name: str,
        callback: Callable[[dict], None],
        ack_deadline: int = 600,
    ) -> None:
        """Subscribe to a Pub/Sub subscription with streaming pull and trace context extraction."""
        subscription_path = self.subscriber.subscription_path(self.project_id, subscription_name)
        propagator = TraceContextTextMapPropagator()

        def wrapped_callback(message: Message) -> None:
            try:
                data = json.loads(message.data.decode("utf-8"))

                # Extract trace context from message attributes
                carrier = dict(message.attributes) if message.attributes else {}
                ctx = propagator.extract(carrier)

                # Check delivery attempt count to prevent infinite retries
                delivery_attempt = (
                    message.delivery_attempt if hasattr(message, "delivery_attempt") else 0
                )
                if delivery_attempt > 5:
                    logger.error(
                        f"Job {data.get('job_id')} exceeded max retries ({delivery_attempt} attempts), "
                        f"acknowledging to prevent redelivery"
                    )
                    message.ack()
                    return

                # Process callback with extracted trace context
                with trace.use_span(trace.NonRecordingSpan(ctx), end_on_exit=False):
                    callback(data)

                message.ack()
                logger.info(f"Processed job {data.get('job_id')}")
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                # Nack the message to retry, but only if we haven't exceeded retry limit
                # The delivery_attempt check above will prevent infinite retries
                message.nack()

        streaming_pull = self.subscriber.subscribe(
            subscription_path,
            callback=wrapped_callback,
            flow_control=pubsub_v1.types.FlowControl(max_messages=1),
        )

        logger.info(f"Listening on {subscription_path}")

        with self.subscriber:
            try:
                streaming_pull.result()
            except TimeoutError:
                streaming_pull.cancel()
                streaming_pull.result()


class RedisQueueService:
    """Redis-based queue service for local development."""

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or settings.redis_url or "redis://localhost:6379"
        self._redis: redis.Redis[bytes] | None = None
        self._running = False

    @property
    def redis(self) -> "redis.Redis[bytes]":
        if self._redis is None:
            import redis

            self._redis = redis.from_url(self.redis_url, decode_responses=False)
        return self._redis

    def enqueue(
        self,
        topic_name: str,
        payload: dict,
        job_id: str | None = None,
    ) -> str:
        """Publish a message to a Redis list (queue)."""
        try:
            job_id = job_id or str(uuid.uuid4())

            message_data = {
                "job_id": job_id,
                **payload,
            }

            message_bytes = json.dumps(message_data).encode("utf-8")
            # Use RPUSH to add to the end of the list (queue)
            self.redis.rpush(topic_name, message_bytes)

            logger.info(f"Published job {job_id} to {topic_name}")
            return job_id

        except Exception as e:
            raise QueueError(f"Failed to enqueue message: {e}") from e

    def subscribe(
        self,
        subscription_name: str,
        callback: Callable[[dict], None],
        ack_deadline: int = 600,
    ) -> None:
        """Subscribe to a Redis list (queue) with blocking pop."""
        # Strip '-sub' suffix if present (Pub/Sub convention)
        queue_name = subscription_name.removesuffix("-sub")
        logger.info(f"Listening on Redis queue: {queue_name}")
        self._running = True

        retry_count = {}  # Track retry counts per job_id

        while self._running:
            try:
                # BLPOP blocks until a message is available (timeout: 1 second)
                result = self.redis.blpop(queue_name, timeout=1)

                if result is None:
                    # No message available, continue waiting
                    continue

                _, message_bytes = result
                data = json.loads(message_bytes.decode("utf-8"))
                job_id = data.get("job_id")

                # Check retry count
                current_retries = retry_count.get(job_id, 0)
                if current_retries > 5:
                    logger.error(
                        f"Job {job_id} exceeded max retries ({current_retries} attempts), "
                        f"discarding message"
                    )
                    retry_count.pop(job_id, None)
                    continue

                try:
                    callback(data)
                    logger.info(f"Processed job {job_id}")
                    # Remove from retry count on success
                    retry_count.pop(job_id, None)
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    # Increment retry count and re-queue
                    retry_count[job_id] = current_retries + 1
                    # Re-queue the message for retry (with a small delay)
                    time.sleep(1)
                    self.redis.rpush(queue_name, message_bytes)
                    logger.info(f"Re-queued job {job_id} (attempt {retry_count[job_id]})")

            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self._running = False
                break
            except Exception as e:
                logger.error(f"Error in Redis subscribe loop: {e}", exc_info=True)
                time.sleep(1)  # Brief pause before retrying


def get_queue_service() -> QueueService:
    """Get the appropriate queue service based on configuration.

    Returns RedisQueueService if REDIS_URL is configured (local development),
    otherwise returns PubSubQueueService for production.
    """
    if settings.redis_url and not settings.is_production():
        logger.info("Using Redis queue service (local development)")
        return RedisQueueService()  # type: ignore
    else:
        logger.info("Using Pub/Sub queue service (production)")
        return PubSubQueueService()  # type: ignore

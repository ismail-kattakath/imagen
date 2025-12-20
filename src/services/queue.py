from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.subscriber.message import Message
from typing import Callable
from concurrent.futures import TimeoutError
import json
import uuid

from src.core.config import settings
from src.core.logging import logger
from src.core.exceptions import QueueError


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
        """Publish a message to a Pub/Sub topic."""
        try:
            job_id = job_id or str(uuid.uuid4())
            topic_path = self.publisher.topic_path(self.project_id, topic_name)
            
            message_data = {
                "job_id": job_id,
                **payload,
            }
            
            message_bytes = json.dumps(message_data).encode("utf-8")
            future = self.publisher.publish(topic_path, message_bytes)
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
        """Subscribe to a Pub/Sub subscription with streaming pull."""
        subscription_path = self.subscriber.subscription_path(
            self.project_id, subscription_name
        )
        
        def wrapped_callback(message: Message) -> None:
            try:
                data = json.loads(message.data.decode("utf-8"))
                callback(data)
                message.ack()
                logger.info(f"Processed job {data.get('job_id')}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
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

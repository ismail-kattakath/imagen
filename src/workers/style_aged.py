from src.workers.base import BaseWorker
from src.pipelines import AgedStylePipeline
from src.core.config import settings


class AgedStyleWorker(BaseWorker):
    """Worker for vintage/aged style conversion."""
    
    def __init__(self):
        super().__init__(
            pipeline_cls=AgedStylePipeline,
            subscription_name=settings.pubsub_subscription_aged,
        )


if __name__ == "__main__":
    worker = AgedStyleWorker()
    worker.run()

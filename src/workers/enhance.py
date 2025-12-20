from src.workers.base import BaseWorker
from src.pipelines import EnhancePipeline
from src.core.config import settings


class EnhanceWorker(BaseWorker):
    """Worker for enhancing images."""
    
    def __init__(self):
        super().__init__(
            pipeline_cls=EnhancePipeline,
            subscription_name=settings.pubsub_subscription_enhance,
        )


if __name__ == "__main__":
    worker = EnhanceWorker()
    worker.run()

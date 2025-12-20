from src.workers.base import BaseWorker
from src.pipelines import BackgroundRemovePipeline
from src.core.config import settings


class BackgroundRemoveWorker(BaseWorker):
    """Worker for background removal."""

    def __init__(self):
        super().__init__(
            pipeline_cls=BackgroundRemovePipeline,
            subscription_name=settings.pubsub_subscription_background_remove,
        )


if __name__ == "__main__":
    worker = BackgroundRemoveWorker()
    worker.run()

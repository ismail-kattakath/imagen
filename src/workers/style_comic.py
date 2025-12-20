from src.workers.base import BaseWorker
from src.pipelines import ComicStylePipeline
from src.core.config import settings


class ComicStyleWorker(BaseWorker):
    """Worker for comic style conversion."""
    
    def __init__(self):
        super().__init__(
            pipeline_cls=ComicStylePipeline,
            subscription_name=settings.pubsub_subscription_comic,
        )


if __name__ == "__main__":
    worker = ComicStyleWorker()
    worker.run()

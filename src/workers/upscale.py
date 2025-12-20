from src.workers.base import BaseWorker
from src.pipelines import UpscalePipeline
from src.core.config import settings


class UpscaleWorker(BaseWorker):
    """Worker for upscaling images."""
    
    def __init__(self):
        super().__init__(
            pipeline_cls=UpscalePipeline,
            subscription_name=settings.pubsub_subscription_upscale,
        )


if __name__ == "__main__":
    worker = UpscaleWorker()
    worker.run()

from src.pipelines.base import BasePipeline
from src.pipelines.upscale import UpscalePipeline
from src.pipelines.enhance import EnhancePipeline
from src.pipelines.style_comic import ComicStylePipeline
from src.pipelines.background_remove import BackgroundRemovePipeline

__all__ = [
    "BasePipeline",
    "UpscalePipeline",
    "EnhancePipeline",
    "ComicStylePipeline",
    "BackgroundRemovePipeline",
]

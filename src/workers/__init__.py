"""
Imagen Workers

Thin workers that call Triton Inference Server for ML inference.
"""

from src.workers.triton_worker import (
    TritonBackgroundRemoveWorker,
    TritonEnhanceWorker,
    TritonStyleAgedWorker,
    TritonStyleComicWorker,
    TritonUpscaleWorker,
    TritonWorker,
)

__all__ = [
    "TritonWorker",
    "TritonUpscaleWorker",
    "TritonEnhanceWorker",
    "TritonBackgroundRemoveWorker",
    "TritonStyleComicWorker",
    "TritonStyleAgedWorker",
]

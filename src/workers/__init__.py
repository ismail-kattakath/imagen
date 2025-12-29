"""
Imagen Workers

Thin workers that call Triton Inference Server for ML inference.
"""

from src.workers.triton_worker import (
    TritonWorker,
    TritonUpscaleWorker,
    TritonEnhanceWorker,
    TritonBackgroundRemoveWorker,
    TritonStyleComicWorker,
    TritonStyleAgedWorker,
)

__all__ = [
    "TritonWorker",
    "TritonUpscaleWorker",
    "TritonEnhanceWorker",
    "TritonBackgroundRemoveWorker",
    "TritonStyleComicWorker",
    "TritonStyleAgedWorker",
]

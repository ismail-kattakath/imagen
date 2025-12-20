from abc import ABC, abstractmethod
from typing import Any
from PIL import Image
import torch
from src.core.logging import logger
from src.core.exceptions import ModelLoadError


class BasePipeline(ABC):
    """Abstract base class for all image processing pipelines."""

    def __init__(
        self,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {device}")

        # Set dtype based on device if not specified
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32

        self.device = device
        self.dtype = dtype
        self._pipeline = None
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB if needed."""
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def _load_with_error_handling(
        self, model_name: str, load_func: Any
    ) -> Any:
        """Common error handling wrapper for model loading.

        Args:
            model_name: Human-readable model name for logging
            load_func: Function that loads and returns the pipeline

        Returns:
            The loaded pipeline

        Raises:
            ModelLoadError: If loading fails
        """
        try:
            logger.info(f"Loading {model_name}")
            pipeline = load_func()
            logger.info(f"{model_name} loaded successfully")
            return pipeline
        except Exception as e:
            raise ModelLoadError(f"Failed to load {model_name}: {e}") from e

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        """Process an image. Must be implemented by subclasses."""
        pass
    
    def unload(self) -> None:
        """Free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"{self.__class__.__name__} unloaded from memory")
    
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self.unload()
        return False

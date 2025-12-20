from abc import ABC, abstractmethod
from PIL import Image
import torch
from src.core.logging import logger


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
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

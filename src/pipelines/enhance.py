from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torch

from src.pipelines.base import BasePipeline
from src.core.logging import logger
from src.core.exceptions import ModelLoadError, ImageProcessingError


class EnhancePipeline(BasePipeline):
    """Image enhancement using Stable Diffusion img2img."""
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
    
    def load(self) -> None:
        if self._loaded:
            return

        try:
            from src.core.config import settings

            logger.info(f"Loading enhance model: {self.model_id}")
            logger.info(f"Cache directory: {settings.model_cache_dir}")

            self._pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                cache_dir=settings.model_cache_dir,
            ).to(self.device)
            
            self._pipeline.enable_attention_slicing()
            self._loaded = True
            logger.info("Enhance model loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load enhance model: {e}") from e
    
    def process(
        self,
        image: Image.Image,
        prompt: str = "high quality, detailed, sharp",
        strength: float = 0.3,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> Image.Image:
        """Enhance an image."""
        if not self._loaded:
            raise ImageProcessingError("Pipeline not loaded")
        
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            result = self._pipeline(
                prompt=prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
            ).images[0]
            
            return result
            
        except Exception as e:
            raise ImageProcessingError(f"Enhance failed: {e}") from e

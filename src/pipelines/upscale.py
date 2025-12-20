from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch

from src.pipelines.base import BasePipeline
from src.core.logging import logger
from src.core.exceptions import ModelLoadError, ImageProcessingError


class UpscalePipeline(BasePipeline):
    """4x image upscaling using Stable Diffusion Upscaler."""
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
    
    def load(self) -> None:
        """Load the upscale model into GPU memory."""
        if self._loaded:
            return

        try:
            from src.core.config import settings

            logger.info(f"Loading upscale model: {self.model_id}")
            logger.info(f"Cache directory: {settings.model_cache_dir}")

            self._pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                cache_dir=settings.model_cache_dir,
            ).to(self.device)
            
            # Enable memory optimizations
            self._pipeline.enable_attention_slicing()
            
            self._loaded = True
            logger.info("Upscale model loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load upscale model: {e}") from e
    
    def process(
        self,
        image: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> Image.Image:
        """Upscale an image by 4x."""
        if not self._loaded:
            raise ImageProcessingError("Pipeline not loaded. Call load() first.")
        
        try:
            # Ensure image is RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            result = self._pipeline(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            
            return result
            
        except Exception as e:
            raise ImageProcessingError(f"Upscale processing failed: {e}") from e

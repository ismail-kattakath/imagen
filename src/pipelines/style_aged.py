from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torch

from src.pipelines.base import BasePipeline
from src.core.logging import logger
from src.core.exceptions import ModelLoadError, ImageProcessingError


class AgedStylePipeline(BasePipeline):
    """Apply aged/vintage look to images."""
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
    
    def load(self) -> None:
        if self._loaded:
            return
        
        try:
            logger.info(f"Loading aged style model: {self.model_id}")
            self._pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            ).to(self.device)
            
            self._pipeline.enable_attention_slicing()
            self._loaded = True
            logger.info("Aged style model loaded")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load: {e}") from e
    
    def process(
        self,
        image: Image.Image,
        prompt: str = "aged photograph, vintage, sepia tones, old photo, faded colors, film grain, nostalgic",
        negative_prompt: str = "modern, digital, sharp, vibrant colors",
        strength: float = 0.5,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> Image.Image:
        """Apply aged/vintage effect to image."""
        if not self._loaded:
            raise ImageProcessingError("Pipeline not loaded")
        
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
            ).images[0]
            
            return result
            
        except Exception as e:
            raise ImageProcessingError(f"Aged style failed: {e}") from e

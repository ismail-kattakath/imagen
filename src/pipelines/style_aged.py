from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

from src.pipelines.base import BasePipeline
from src.core.exceptions import ImageProcessingError


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

        from src.core.config import settings

        def _load():
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                cache_dir=settings.model_cache_dir,
            ).to(self.device)
            pipeline.enable_attention_slicing()
            return pipeline

        self._pipeline = self._load_with_error_handling(
            f"aged style model: {self.model_id}", _load
        )
        self._loaded = True
    
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
            image = self._ensure_rgb(image)

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

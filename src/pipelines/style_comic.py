from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

from src.pipelines.base import BasePipeline
from src.core.exceptions import ImageProcessingError


class ComicStylePipeline(BasePipeline):
    """Convert images to comic/cartoon style."""
    
    def __init__(
        self,
        model_id: str = "nitrosocke/Ghibli-Diffusion",
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
            f"comic style model: {self.model_id}", _load
        )
        self._loaded = True
    
    def process(
        self,
        image: Image.Image,
        prompt: str = "ghibli style, comic, cartoon, vibrant colors",
        strength: float = 0.6,
        guidance_scale: float = 8.0,
        **kwargs,
    ) -> Image.Image:
        """Convert image to comic style."""
        if not self._loaded:
            raise ImageProcessingError("Pipeline not loaded")
        
        try:
            image = self._ensure_rgb(image)

            result = self._pipeline(
                prompt=prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
            ).images[0]
            
            return result
            
        except Exception as e:
            raise ImageProcessingError(f"Comic style failed: {e}") from e

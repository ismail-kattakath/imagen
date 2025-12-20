from PIL import Image
from transformers import pipeline

from src.pipelines.base import BasePipeline
from src.core.exceptions import ImageProcessingError


class BackgroundRemovePipeline(BasePipeline):
    """Remove background from images using segmentation."""
    
    def __init__(
        self,
        model_id: str = "briaai/RMBG-1.4",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
    
    def load(self) -> None:
        if self._loaded:
            return

        from src.core.config import settings

        def _load():
            return pipeline(
                "image-segmentation",
                model=self.model_id,
                trust_remote_code=True,
                device=0 if self.device == "cuda" else -1,
                model_kwargs={"cache_dir": settings.model_cache_dir},
            )

        self._pipeline = self._load_with_error_handling(
            f"background removal: {self.model_id}", _load
        )
        self._loaded = True
    
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        """Remove background from image."""
        if not self._loaded:
            raise ImageProcessingError("Pipeline not loaded")

        try:
            image = self._ensure_rgb(image)

            # The RMBG model returns a mask, apply it to create transparent image
            result = self._pipeline(image)

            # Handle the segmentation result
            if isinstance(result, list) and len(result) > 0:
                # Get the mask from the first result
                mask = result[0]["mask"]

                # Convert original image to RGBA
                rgba_image = image.convert("RGBA")

                # Convert mask to L mode if needed
                if mask.mode != "L":
                    mask = mask.convert("L")

                # Apply mask as alpha channel
                rgba_image.putalpha(mask)
                return rgba_image
            else:
                # Fallback if format is unexpected
                return image.convert("RGBA")

        except Exception as e:
            raise ImageProcessingError(f"Background removal failed: {e}") from e

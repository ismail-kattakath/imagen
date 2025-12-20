from pydantic import BaseModel, Field


class UpscaleParams(BaseModel):
    """Parameters for upscale operation."""
    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)


class EnhanceParams(BaseModel):
    """Parameters for enhance operation."""
    prompt: str = "high quality, detailed, sharp"
    strength: float = Field(default=0.3, ge=0.1, le=1.0)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)


class StyleParams(BaseModel):
    """Parameters for style transfer operations."""
    prompt: str = ""
    strength: float = Field(default=0.6, ge=0.1, le=1.0)
    guidance_scale: float = Field(default=8.0, ge=1.0, le=20.0)


class BackgroundRemoveParams(BaseModel):
    """Parameters for background removal."""
    pass  # No additional params needed

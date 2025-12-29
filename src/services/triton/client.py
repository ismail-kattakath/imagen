"""
Triton Inference Server Client

This module provides a high-level client for calling Triton models.
Workers use this instead of loading models directly.
"""

from typing import Optional
import numpy as np
from PIL import Image
import io
import os

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from src.core.logging import logger


class TritonClient:
    """
    Client for Triton Inference Server.
    
    Usage:
        client = TritonClient()
        result = client.upscale(image_bytes)
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize Triton client.
        
        Args:
            url: Triton server URL (default: triton:8001 in-cluster)
            verbose: Enable verbose logging
        """
        self.url = url or os.getenv("TRITON_URL", "triton:8001")
        self.verbose = verbose
        self._client = None
    
    @property
    def client(self) -> grpcclient.InferenceServerClient:
        """Lazy-initialize gRPC client."""
        if self._client is None:
            self._client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=self.verbose,
            )
        return self._client
    
    def is_ready(self) -> bool:
        """Check if Triton server is ready."""
        try:
            return self.client.is_server_ready()
        except InferenceServerException as e:
            logger.warning(f"Triton not ready: {e}")
            return False
    
    def is_model_ready(self, model_name: str) -> bool:
        """Check if a specific model is ready."""
        try:
            return self.client.is_model_ready(model_name)
        except InferenceServerException as e:
            logger.warning(f"Model {model_name} not ready: {e}")
            return False
    
    def _image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    def _bytes_to_image(self, data: bytes) -> Image.Image:
        """Convert bytes to PIL Image."""
        return Image.open(io.BytesIO(data))
    
    def _infer(
        self,
        model_name: str,
        inputs: list[grpcclient.InferInput],
        outputs: list[grpcclient.InferRequestedOutput],
    ) -> grpcclient.InferResult:
        """
        Run inference on Triton.
        
        Args:
            model_name: Name of the model
            inputs: List of input tensors
            outputs: List of requested outputs
        
        Returns:
            Inference result
        """
        try:
            result = self.client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
            )
            return result
        except InferenceServerException as e:
            logger.error(f"Triton inference failed: {e}")
            raise
    
    def upscale(
        self,
        image: Image.Image | bytes,
        scale: float = 4.0,
    ) -> Image.Image:
        """
        Upscale an image using Real-ESRGAN.
        
        Args:
            image: PIL Image or bytes
            scale: Upscale factor (default: 4x)
        
        Returns:
            Upscaled PIL Image
        """
        # Convert to bytes if needed
        if isinstance(image, Image.Image):
            image_bytes = self._image_to_bytes(image)
        else:
            image_bytes = image
        
        # Prepare inputs
        image_input = grpcclient.InferInput("IMAGE", [len(image_bytes)], "UINT8")
        image_input.set_data_from_numpy(np.frombuffer(image_bytes, dtype=np.uint8))
        
        scale_input = grpcclient.InferInput("SCALE", [1], "FP32")
        scale_input.set_data_from_numpy(np.array([scale], dtype=np.float32))
        
        # Request output
        output = grpcclient.InferRequestedOutput("OUTPUT_IMAGE")
        
        # Run inference
        result = self._infer("upscale", [image_input, scale_input], [output])
        
        # Parse result
        output_bytes = result.as_numpy("OUTPUT_IMAGE").tobytes()
        return self._bytes_to_image(output_bytes)
    
    def enhance(
        self,
        image: Image.Image | bytes,
        prompt: Optional[str] = None,
        strength: float = 0.3,
    ) -> Image.Image:
        """
        Enhance an image using SDXL Refiner.
        
        Args:
            image: PIL Image or bytes
            prompt: Optional enhancement prompt
            strength: Enhancement strength (0-1)
        
        Returns:
            Enhanced PIL Image
        """
        if isinstance(image, Image.Image):
            image_bytes = self._image_to_bytes(image)
        else:
            image_bytes = image
        
        # Prepare inputs
        inputs = []
        
        image_input = grpcclient.InferInput("IMAGE", [len(image_bytes)], "UINT8")
        image_input.set_data_from_numpy(np.frombuffer(image_bytes, dtype=np.uint8))
        inputs.append(image_input)
        
        if prompt:
            prompt_bytes = prompt.encode("utf-8")
            prompt_input = grpcclient.InferInput("PROMPT", [1], "BYTES")
            prompt_input.set_data_from_numpy(np.array([prompt_bytes], dtype=object))
            inputs.append(prompt_input)
        
        strength_input = grpcclient.InferInput("STRENGTH", [1], "FP32")
        strength_input.set_data_from_numpy(np.array([strength], dtype=np.float32))
        inputs.append(strength_input)
        
        # Request output
        output = grpcclient.InferRequestedOutput("OUTPUT_IMAGE")
        
        # Run inference
        result = self._infer("enhance", inputs, [output])
        
        # Parse result
        output_bytes = result.as_numpy("OUTPUT_IMAGE").tobytes()
        return self._bytes_to_image(output_bytes)
    
    def remove_background(
        self,
        image: Image.Image | bytes,
    ) -> Image.Image:
        """
        Remove background from an image using RMBG.
        
        Args:
            image: PIL Image or bytes
        
        Returns:
            Image with transparent background (RGBA)
        """
        if isinstance(image, Image.Image):
            image_bytes = self._image_to_bytes(image)
        else:
            image_bytes = image
        
        # Prepare input
        image_input = grpcclient.InferInput("IMAGE", [len(image_bytes)], "UINT8")
        image_input.set_data_from_numpy(np.frombuffer(image_bytes, dtype=np.uint8))
        
        # Request output
        output = grpcclient.InferRequestedOutput("OUTPUT_IMAGE")
        
        # Run inference
        result = self._infer("background_remove", [image_input], [output])
        
        # Parse result
        output_bytes = result.as_numpy("OUTPUT_IMAGE").tobytes()
        return self._bytes_to_image(output_bytes)
    
    def style_comic(
        self,
        image: Image.Image | bytes,
    ) -> Image.Image:
        """Apply comic style to an image."""
        if isinstance(image, Image.Image):
            image_bytes = self._image_to_bytes(image)
        else:
            image_bytes = image
        
        image_input = grpcclient.InferInput("IMAGE", [len(image_bytes)], "UINT8")
        image_input.set_data_from_numpy(np.frombuffer(image_bytes, dtype=np.uint8))
        
        output = grpcclient.InferRequestedOutput("OUTPUT_IMAGE")
        result = self._infer("style_comic", [image_input], [output])
        
        output_bytes = result.as_numpy("OUTPUT_IMAGE").tobytes()
        return self._bytes_to_image(output_bytes)
    
    def style_aged(
        self,
        image: Image.Image | bytes,
    ) -> Image.Image:
        """Apply aged/vintage style to an image."""
        if isinstance(image, Image.Image):
            image_bytes = self._image_to_bytes(image)
        else:
            image_bytes = image
        
        image_input = grpcclient.InferInput("IMAGE", [len(image_bytes)], "UINT8")
        image_input.set_data_from_numpy(np.frombuffer(image_bytes, dtype=np.uint8))
        
        output = grpcclient.InferRequestedOutput("OUTPUT_IMAGE")
        result = self._infer("style_aged", [image_input], [output])
        
        output_bytes = result.as_numpy("OUTPUT_IMAGE").tobytes()
        return self._bytes_to_image(output_bytes)


# Singleton instance
_triton_client: Optional[TritonClient] = None


def get_triton_client() -> TritonClient:
    """Get or create Triton client singleton."""
    global _triton_client
    if _triton_client is None:
        _triton_client = TritonClient()
    return _triton_client

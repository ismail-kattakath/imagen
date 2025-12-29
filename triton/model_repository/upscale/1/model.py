"""
Triton Python Backend for Real-ESRGAN Upscaling

This model is loaded by Triton Inference Server and handles
batched inference requests for image upscaling.
"""

import numpy as np
import torch
from PIL import Image
import io
import json

# Triton Python backend
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Triton Python model for Real-ESRGAN upscaling."""

    def initialize(self, args):
        """
        Called once when the model is loaded.
        Load the Real-ESRGAN model into GPU memory.
        """
        self.model_config = json.loads(args["model_config"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get model path from config
        model_path = "/models/huggingface/realesrgan"
        for param in self.model_config.get("parameters", []):
            if param.get("key") == "model_path":
                model_path = param["value"]["string_value"]
        
        # Load Real-ESRGAN model
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        # Initialize model architecture
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        
        # Create upscaler
        self.upscaler = RealESRGANer(
            scale=4,
            model_path=f"{model_path}/RealESRGAN_x4plus.pth",
            model=model,
            tile=0,  # No tiling for batch processing
            tile_pad=10,
            pre_pad=0,
            half=True,  # FP16 for memory efficiency
            device=self.device,
        )
        
        print(f"Upscale model loaded on {self.device}")

    def execute(self, requests):
        """
        Called for each batch of inference requests.
        Triton handles batching automatically based on config.pbtxt.
        """
        responses = []
        
        # Collect all images in the batch
        batch_images = []
        batch_scales = []
        
        for request in requests:
            # Get input image bytes
            image_bytes = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            image_bytes = image_bytes.as_numpy().tobytes()
            
            # Decode image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            batch_images.append(np.array(image))
            
            # Get optional scale parameter
            scale_tensor = pb_utils.get_input_tensor_by_name(request, "SCALE")
            if scale_tensor is not None:
                batch_scales.append(float(scale_tensor.as_numpy()[0]))
            else:
                batch_scales.append(4.0)  # Default 4x upscale
        
        # Process batch
        batch_results = []
        for img, scale in zip(batch_images, batch_scales):
            try:
                # Run upscaling
                output, _ = self.upscaler.enhance(img, outscale=scale)
                
                # Encode result as PNG bytes
                result_image = Image.fromarray(output)
                buffer = io.BytesIO()
                result_image.save(buffer, format="PNG")
                result_bytes = buffer.getvalue()
                
                batch_results.append(result_bytes)
            except Exception as e:
                # On error, return empty result
                print(f"Error processing image: {e}")
                batch_results.append(b"")
        
        # Create responses
        for result_bytes in batch_results:
            output_tensor = pb_utils.Tensor(
                "OUTPUT_IMAGE",
                np.frombuffer(result_bytes, dtype=np.uint8),
            )
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses

    def finalize(self):
        """Called when the model is unloaded."""
        print("Upscale model unloaded")

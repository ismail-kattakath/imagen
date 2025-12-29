"""
Triton Python Backend for Aged/Vintage Style Transfer

Uses Stable Diffusion v1.5 model for vintage/aged photo effect.
"""

import numpy as np
import torch
from PIL import Image
import io
import json

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Triton Python model for aged/vintage style transfer."""

    def initialize(self, args):
        """Load the Stable Diffusion model."""
        self.model_config = json.loads(args["model_config"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get model ID from config
        model_id = "runwayml/stable-diffusion-v1-5"
        for param in self.model_config.get("parameters", []):
            if param.get("key") == "model_id":
                model_id = param["value"]["string_value"]
        
        # Load Stable Diffusion Img2Img pipeline
        from diffusers import StableDiffusionImg2ImgPipeline
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(self.device)
        
        # Enable memory optimizations
        self.pipe.enable_attention_slicing()
        
        # Default prompts for aged style
        self.default_prompt = (
            "aged photograph, vintage, sepia tones, old photo, "
            "faded colors, film grain, nostalgic"
        )
        self.default_negative = "modern, digital, sharp, vibrant colors"
        
        print(f"Aged style model loaded on {self.device}")

    def execute(self, requests):
        """Process batch of aged style requests."""
        responses = []
        
        # Collect batch
        batch_images = []
        batch_prompts = []
        batch_strengths = []
        
        for request in requests:
            # Get input image
            image_bytes = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            image_bytes = image_bytes.as_numpy().tobytes()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            batch_images.append(image)
            
            # Get optional prompt
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            if prompt_tensor is not None:
                prompt = prompt_tensor.as_numpy()[0].decode("utf-8")
            else:
                prompt = self.default_prompt
            batch_prompts.append(prompt)
            
            # Get optional strength
            strength_tensor = pb_utils.get_input_tensor_by_name(request, "STRENGTH")
            if strength_tensor is not None:
                strength = float(strength_tensor.as_numpy()[0])
            else:
                strength = 0.5  # Default strength for aged style
            batch_strengths.append(strength)
        
        # Process each image
        batch_results = []
        for image, prompt, strength in zip(batch_images, batch_prompts, batch_strengths):
            try:
                # Run style transfer
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.default_negative,
                    image=image,
                    strength=strength,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                ).images[0]
                
                # Encode result
                buffer = io.BytesIO()
                result.save(buffer, format="PNG")
                batch_results.append(buffer.getvalue())
            except Exception as e:
                print(f"Error processing aged style: {e}")
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
        """Cleanup."""
        del self.pipe
        torch.cuda.empty_cache()
        print("Aged style model unloaded")

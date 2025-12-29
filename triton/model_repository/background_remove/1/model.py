"""
Triton Python Backend for Background Removal

Uses RMBG-1.4 for efficient background removal with true batching support.
"""

import numpy as np
import torch
from PIL import Image
import io
import json

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Triton Python model for background removal."""

    def initialize(self, args):
        """Load the RMBG model."""
        self.model_config = json.loads(args["model_config"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get model ID
        model_id = "briaai/RMBG-1.4"
        for param in self.model_config.get("parameters", []):
            if param.get("key") == "model_id":
                model_id = param["value"]["string_value"]
        
        # Load RMBG model
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        print(f"Background remove model loaded on {self.device}")

    def execute(self, requests):
        """Process batch of background removal requests."""
        responses = []
        
        # Collect and preprocess batch
        batch_tensors = []
        original_images = []
        original_sizes = []
        
        for request in requests:
            image_bytes = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            image_bytes = image_bytes.as_numpy().tobytes()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            original_images.append(image)
            original_sizes.append(image.size)
            
            # Preprocess
            tensor = self.transform(image)
            batch_tensors.append(tensor)
        
        # Stack into batch tensor
        batch = torch.stack(batch_tensors).to(self.device)
        
        # Run inference on entire batch at once
        with torch.no_grad():
            preds = self.model(batch)[-1].sigmoid()
        
        # Post-process each result
        batch_results = []
        for i, (pred, orig_img, orig_size) in enumerate(
            zip(preds, original_images, original_sizes)
        ):
            try:
                # Resize mask to original size
                pred_pil = transforms.ToPILImage()(pred.squeeze().cpu())
                mask = pred_pil.resize(orig_size, Image.BILINEAR)
                
                # Apply mask to original image
                result = Image.new("RGBA", orig_size, (0, 0, 0, 0))
                result.paste(orig_img, mask=mask)
                
                # Encode result
                buffer = io.BytesIO()
                result.save(buffer, format="PNG")
                batch_results.append(buffer.getvalue())
            except Exception as e:
                print(f"Error processing image {i}: {e}")
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
        del self.model
        torch.cuda.empty_cache()
        print("Background remove model unloaded")

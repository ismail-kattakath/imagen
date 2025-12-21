#!/usr/bin/env python3
"""Quick script to process user's image."""
import requests
from PIL import Image
import io

# The image from the conversation
image_data = None  # Will use the test API with direct upload

def process_image():
    """Process the user's image via API."""
    url = "http://localhost:8000/api/v1/images/background/remove"

    # For now, let's use curl to upload the image directly
    print("Please upload your image using:")
    print(f"curl -X POST -F 'file=@/path/to/your/image.jpg' {url}")

if __name__ == "__main__":
    process_image()

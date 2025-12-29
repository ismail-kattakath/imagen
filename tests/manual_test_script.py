#!/usr/bin/env python3
"""Test script for the Imagen API."""

import sys
from pathlib import Path

import requests


def test_background_removal(image_path: str):
    """Test the background removal endpoint."""
    url = "http://localhost:8000/api/v1/images/background/remove"

    print(f"Testing background removal with: {image_path}")
    print(f"Uploading to: {url}")

    # Open and upload the image
    filename = Path(image_path).name
    with open(image_path, "rb") as f:
        files = {"file": (filename, f, "image/png")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()
        print("\n✓ Job created successfully!")
        print(f"  Job ID: {result['job_id']}")
        print(f"  Status: {result['status']}")
        return result["job_id"]
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(f"  Response: {response.text}")
        return None


if __name__ == "__main__":
    # Default to test fixtures if no argument provided
    if len(sys.argv) < 2:
        image_path = "tests/fixtures/test_image.png"
    else:
        image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}")
        print("Usage: python tests/test_api.py [image_path]")
        sys.exit(1)

    job_id = test_background_removal(image_path)

    if job_id:
        print("\nMonitor the worker logs to see the job being processed!")
        print(f"Job ID: {job_id}")
        print(f"\nCheck storage/outputs/{job_id}/ for the result!")

#!/usr/bin/env python3
"""
Model downloader for Imagen platform.
Pre-downloads all ML models to avoid slow startup.
"""

import os
import sys
from pathlib import Path


def download_models(cache_dir: str = "./models"):
    """Download all models to cache directory."""
    try:
        from diffusers import StableDiffusionUpscalePipeline, StableDiffusionImg2ImgPipeline
        from transformers import pipeline as hf_pipeline
        import torch
    except ImportError as e:
        print(f"❌ Error: Missing dependencies. Run: pip install -e '.[dev]'")
        print(f"   Details: {e}")
        sys.exit(1)

    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Imagen Model Downloader")
    print("=" * 70)
    print(f"\nCache directory: {cache_path.absolute()}")
    print(f"Total download size: ~14GB")
    print(f"Estimated time: 5-20 minutes (depends on connection)\n")
    
    models = [
        {
            "name": "Upscale (4x)",
            "id": "stabilityai/stable-diffusion-x4-upscaler",
            "size": "~2.5GB",
            "loader": StableDiffusionUpscalePipeline,
        },
        {
            "name": "Enhance (SDXL Refiner)",
            "id": "stabilityai/stable-diffusion-xl-refiner-1.0",
            "size": "~6GB",
            "loader": StableDiffusionImg2ImgPipeline,
        },
        {
            "name": "Comic Style (Ghibli)",
            "id": "nitrosocke/Ghibli-Diffusion",
            "size": "~4GB",
            "loader": StableDiffusionImg2ImgPipeline,
        },
    ]
    
    # Download diffusers models
    for i, model in enumerate(models, 1):
        print(f"[{i}/4] Downloading {model['name']} ({model['size']})...")
        print(f"     Model ID: {model['id']}")
        try:
            model['loader'].from_pretrained(
                model['id'],
                cache_dir=str(cache_path),
                torch_dtype=torch.float16,
            )
            print(f"     ✓ Downloaded successfully\n")
        except Exception as e:
            print(f"     ✗ Error: {e}\n")
            continue
    
    # Download background removal model
    print(f"[4/4] Downloading Background Removal (~1GB)...")
    print(f"     Model ID: briaai/RMBG-1.4")
    try:
        hf_pipeline(
            "image-segmentation",
            model="briaai/RMBG-1.4",
            trust_remote_code=True,
            cache_dir=str(cache_path),
        )
        print(f"     ✓ Downloaded successfully\n")
    except Exception as e:
        print(f"     ✗ Error: {e}\n")
    
    print("=" * 70)
    print("✅ Model download complete!")
    print("=" * 70)
    print(f"\nModels cached in: {cache_path.absolute()}")
    print(f"Disk usage: ", end="")
    
    # Calculate total size
    try:
        import subprocess
        result = subprocess.run(
            ["du", "-sh", str(cache_path)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            size = result.stdout.split()[0]
            print(f"{size}")
        else:
            print("(unable to calculate)")
    except:
        print("(unable to calculate)")
    
    print("\nWorkers will now load models from cache (much faster!).")
    print("To start a worker: make worker-upscale")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download all Imagen AI models for offline use"
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("MODEL_CACHE_DIR", "./models"),
        help="Directory to cache models (default: ./models or MODEL_CACHE_DIR env var)"
    )
    
    args = parser.parse_args()
    
    try:
        download_models(args.cache_dir)
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)

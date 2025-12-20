#!/usr/bin/env python3
"""
Setup validation script for Imagen platform.
Checks that all files are in place and configuration is ready.
"""

import os
import sys
from pathlib import Path


def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists."""
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {path}")
    return exists


def check_directory_exists(path: str, description: str) -> bool:
    """Check if a directory exists."""
    exists = Path(path).is_dir()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {path}")
    return exists


def main():
    print("=" * 60)
    print("Imagen Platform Setup Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Check core files
    print("\n📁 Core Configuration Files:")
    all_passed &= check_file_exists(".env", ".env file")
    all_passed &= check_file_exists("pyproject.toml", "Python project config")
    all_passed &= check_file_exists("Makefile", "Makefile")
    all_passed &= check_file_exists("README.md", "README")
    all_passed &= check_file_exists("DEPLOYMENT_GUIDE.md", "Deployment guide")
    
    # Check source structure
    print("\n📦 Source Code Structure:")
    all_passed &= check_directory_exists("src/api", "API module")
    all_passed &= check_directory_exists("src/pipelines", "Pipelines module")
    all_passed &= check_directory_exists("src/workers", "Workers module")
    all_passed &= check_directory_exists("src/services", "Services module")
    all_passed &= check_directory_exists("src/core", "Core module")
    
    # Check API files
    print("\n🌐 API Files:")
    all_passed &= check_file_exists("src/api/main.py", "API main")
    all_passed &= check_file_exists("src/api/routes/health.py", "Health routes")
    all_passed &= check_file_exists("src/api/routes/jobs.py", "Jobs routes")
    all_passed &= check_file_exists("src/api/routes/images.py", "Images routes")
    
    # Check pipeline files
    print("\n🔧 Pipeline Files:")
    all_passed &= check_file_exists("src/pipelines/base.py", "Base pipeline")
    all_passed &= check_file_exists("src/pipelines/upscale.py", "Upscale pipeline")
    all_passed &= check_file_exists("src/pipelines/enhance.py", "Enhance pipeline")
    all_passed &= check_file_exists("src/pipelines/style_comic.py", "Comic style pipeline")
    all_passed &= check_file_exists("src/pipelines/background_remove.py", "Background removal pipeline")
    
    # Check worker files
    print("\n👷 Worker Files:")
    all_passed &= check_file_exists("src/workers/base.py", "Base worker")
    all_passed &= check_file_exists("src/workers/upscale.py", "Upscale worker")
    all_passed &= check_file_exists("src/workers/enhance.py", "Enhance worker")
    all_passed &= check_file_exists("src/workers/style_comic.py", "Comic style worker")
    all_passed &= check_file_exists("src/workers/background_remove.py", "Background removal worker")
    
    # Check service files
    print("\n☁️  Service Files:")
    all_passed &= check_file_exists("src/services/storage.py", "Storage service")
    all_passed &= check_file_exists("src/services/queue.py", "Queue service")
    all_passed &= check_file_exists("src/services/jobs.py", "Jobs service")
    
    # Check Docker files
    print("\n🐳 Docker Files:")
    all_passed &= check_file_exists("docker/Dockerfile.api", "API Dockerfile")
    all_passed &= check_file_exists("docker/Dockerfile.worker", "Worker Dockerfile")
    all_passed &= check_file_exists("docker/docker-compose.yml", "Docker Compose")
    
    # Check Kubernetes manifests
    print("\n☸️  Kubernetes Manifests:")
    all_passed &= check_file_exists("k8s/base/namespace.yaml", "Namespace")
    all_passed &= check_file_exists("k8s/base/configmap.yaml", "ConfigMap")
    all_passed &= check_file_exists("k8s/base/pvc.yaml", "PVC")
    all_passed &= check_file_exists("k8s/base/workload-identity.yaml", "Workload Identity")
    all_passed &= check_file_exists("k8s/workers/upscale-worker.yaml", "Upscale worker")
    all_passed &= check_file_exists("k8s/workers/enhance-worker.yaml", "Enhance worker")
    all_passed &= check_file_exists("k8s/workers/comic-worker.yaml", "Comic worker")
    all_passed &= check_file_exists("k8s/workers/background-remove-worker.yaml", "Background removal worker")
    
    # Check Terraform
    print("\n🏗️  Terraform Files:")
    all_passed &= check_file_exists("terraform/main.tf", "Main Terraform config")
    all_passed &= check_file_exists("terraform/variables.tf", "Terraform variables")
    all_passed &= check_file_exists("terraform/outputs.tf", "Terraform outputs")
    
    # Check .env configuration
    print("\n⚙️  Environment Configuration:")
    if Path(".env").exists():
        with open(".env") as f:
            env_content = f.read()
            has_project = "GOOGLE_CLOUD_PROJECT=" in env_content
            has_bucket = "GCS_BUCKET=" in env_content
            
            if "your-project-id" in env_content:
                print("  ⚠ Warning: GOOGLE_CLOUD_PROJECT still set to 'your-project-id'")
                print("    Please update .env with your actual GCP project ID")
            else:
                print("  ✓ GOOGLE_CLOUD_PROJECT configured")
                
            if "your-bucket-name" in env_content:
                print("  ⚠ Warning: GCS_BUCKET still set to 'your-bucket-name'")
                print("    Please update .env with your actual GCS bucket name")
            else:
                print("  ✓ GCS_BUCKET configured")
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All structure checks passed!")
        print("\nNext steps:")
        print("  1. Update .env with your GCP project details")
        print("  2. Install dependencies: pip install -e '.[dev]'")
        print("  3. Start local services: make dev")
        print("  4. Run API: make api")
        print("  5. See DEPLOYMENT_GUIDE.md for production deployment")
    else:
        print("❌ Some checks failed. Please review the errors above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()

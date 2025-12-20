# Docker Configuration

This directory contains Docker configurations for local development and production deployment.

---

## Files

```
docker/
├── Dockerfile.api       # API container (FastAPI)
├── Dockerfile.worker    # GPU worker container (parameterized)
├── docker-compose.yml   # Local development environment
└── README.md            # This file
```

---

## Quick Start

### API Only (No GPU)

```bash
# Start API with dependencies (no GPU workers)
docker compose up api redis minio pubsub firestore
```

### With GPU Workers

```bash
# Start everything including GPU workers
docker compose --profile gpu up
```

### CPU Workers (Testing without GPU)

```bash
# Start with CPU-only workers (slow but works without GPU)
docker compose --profile cpu up
```

---

## Services

### Core Services

| Service | Description | Port |
|---------|-------------|------|
| `api` | FastAPI application | 8000 |
| `redis` | Cache and message broker | 6379 |
| `pubsub` | Pub/Sub emulator | 8085 |
| `firestore` | Firestore emulator | 8080 |
| `minio` | S3-compatible storage (replaces GCS) | 9000, 9001 |

### GPU Workers (Profile: `gpu`)

| Service | Worker Module | Model |
|---------|---------------|-------|
| `upscale-worker` | `src/workers/upscale.py` | SD x4 Upscaler |
| `enhance-worker` | `src/workers/enhance.py` | SDXL Refiner |
| `comic-worker` | `src/workers/style_comic.py` | Ghibli Diffusion |
| `background-remove-worker` | `src/workers/background_remove.py` | RMBG-1.4 |

### CPU Workers (Profile: `cpu`)

| Service | Description |
|---------|-------------|
| `upscale-worker-cpu` | Same as GPU but runs on CPU (very slow) |

---

## Docker Compose Profiles

Profiles allow selective service startup:

```bash
# Default services only (api, redis, minio, pubsub, firestore)
docker compose up

# Include GPU workers
docker compose --profile gpu up

# Include CPU workers (for testing without GPU)
docker compose --profile cpu up

# Multiple profiles
docker compose --profile gpu --profile cpu up
```

---

## Building Images

### API Image

```bash
# From project root
docker build -t imagen-api -f docker/Dockerfile.api .
```

### Worker Images

The worker Dockerfile is **parameterized** using build arguments:

```bash
# From project root

# Build specific worker
docker build --build-arg WORKER=upscale -t imagen-worker-upscale -f docker/Dockerfile.worker .
docker build --build-arg WORKER=enhance -t imagen-worker-enhance -f docker/Dockerfile.worker .
docker build --build-arg WORKER=style_comic -t imagen-worker-comic -f docker/Dockerfile.worker .
docker build --build-arg WORKER=background_remove -t imagen-worker-bg -f docker/Dockerfile.worker .

# Or build generic worker (select at runtime)
docker build -t imagen-worker -f docker/Dockerfile.worker .
docker run -e WORKER=enhance imagen-worker
```

### Available Worker Types

| WORKER Value | Module | Description |
|--------------|--------|-------------|
| `upscale` | `src/workers/upscale.py` | 4x image upscaling |
| `enhance` | `src/workers/enhance.py` | Quality enhancement |
| `style_comic` | `src/workers/style_comic.py` | Comic/cartoon style |
| `style_aged` | `src/workers/style_aged.py` | Vintage effect |
| `background_remove` | `src/workers/background_remove.py` | Remove background |

---

## Environment Variables

### API

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `false` |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | - |
| `GCS_BUCKET` | Storage bucket name | - |
| `PUBSUB_EMULATOR_HOST` | Pub/Sub emulator address | - |
| `FIRESTORE_EMULATOR_HOST` | Firestore emulator address | - |

### Workers

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKER` | Worker type to run | `upscale` |
| `DEVICE` | PyTorch device (`cuda`/`cpu`) | `cuda` |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | - |
| `GCS_BUCKET` | Storage bucket name | - |
| `HF_HOME` | HuggingFace cache directory | `/models/huggingface` |

---

## Volumes

| Volume | Purpose | Mount Point |
|--------|---------|-------------|
| `imagen-models` | ML model cache | `/models` |
| `redis-data` | Redis persistence | `/data` |
| `minio-data` | MinIO storage | `/data` |

### Model Cache

Models are downloaded once and cached in the `imagen-models` volume:

```bash
# Check volume size
docker volume inspect imagen-models

# Clear model cache (force re-download)
docker volume rm imagen-models
```

---

## GPU Support

### Requirements

- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed
- Docker Compose v2+

### Install NVIDIA Docker Runtime

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access

```bash
# Check GPU is visible
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

---

## Local Development Workflow

### 1. Start Infrastructure

```bash
cd docker
docker compose up -d redis minio pubsub firestore
```

### 2. Start API (with hot reload)

```bash
# Option A: Docker
docker compose up api

# Option B: Local Python (faster for development)
cd ..
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start Worker (optional, for testing)

```bash
# GPU worker
docker compose --profile gpu up upscale-worker

# Or run locally
python -m src.workers.upscale
```

### 4. Test API

```bash
# Health check
curl http://localhost:8000/health

# Submit job
curl -X POST http://localhost:8000/api/v1/images/upscale \
  -F "file=@test.jpg"
```

---

## Production Images

For production, images are built by Cloud Build and pushed to Artifact Registry:

```bash
# API
us-central1-docker.pkg.dev/PROJECT_ID/imagen/api:latest

# Worker
us-central1-docker.pkg.dev/PROJECT_ID/imagen/worker:latest
```

See [CI_CD.md](../CI_CD.md) for automated deployment.

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker info | grep -i runtime

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# If not working, reinstall nvidia-container-toolkit
```

### Out of Memory (OOM)

```bash
# Check GPU memory
nvidia-smi

# Reduce batch size or use smaller model
# Or use CPU for testing
docker compose --profile cpu up
```

### Slow Model Download

```bash
# Pre-download models
docker compose run --rm upscale-worker python -c "
from src.pipelines.upscale import UpscalePipeline
UpscalePipeline().load()
"

# Models cached in imagen-models volume
```

### Container Won't Start

```bash
# Check logs
docker compose logs upscale-worker

# Common issues:
# - GPU not available → Use --profile cpu
# - Port in use → Change port mapping
# - Volume permissions → docker volume rm and recreate
```

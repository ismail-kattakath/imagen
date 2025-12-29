# Docker Configuration (Triton Architecture)

This directory contains Docker configurations for local development using the **Triton-based architecture** deployed in production.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   TRITON ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────┘

    Pub/Sub Queues                Workers (CPU-only)
    ┌──────────┐                  ┌──────────┐
    │ upscale  │───────────────▶  │ upscale  │
    │ enhance  │───────────────▶  │ enhance  │
    │ comic    │───────────────▶  │ comic    │
    │ aged     │───────────────▶  │ aged     │
    │ bg-remove│───────────────▶  │ bg-remove│
    └──────────┘                  └─────┬────┘
                                        │
                                        │ gRPC
                                        ▼
                              ┌─────────────────┐
                              │ Triton Server   │
                              │ (1× T4 GPU)     │
                              │ • All 5 models  │
                              │ • Auto batching │
                              └─────────────────┘
```

**Key Benefits:**
- **60-80% cost reduction** - 1 GPU serves all 5 models
- **Faster startup** - Workers are thin (no model loading)
- **Auto batching** - Triton batches 2-4 images automatically
- **Production parity** - Matches GKE deployment architecture

---

## Files

```
docker/
├── Dockerfile.api       # FastAPI API container
├── Dockerfile.worker    # Thin worker container (CPU-only)
├── docker-compose.yml   # Local development environment (Triton architecture)
└── README.md            # This file
```

**Note:** Triton Dockerfile is at `../triton/Dockerfile`

---

## Quick Start

### API Only (No GPU Required)

```bash
# Start API with infrastructure (no workers)
docker compose up api redis minio pubsub firestore jaeger
```

### With Triton + Workers (Requires NVIDIA GPU)

```bash
# Start everything including Triton and all workers
docker compose --profile all up
```

### Selective Startup

```bash
# Infrastructure only
docker compose up

# API + infrastructure
docker compose up api

# Add Triton (GPU required)
docker compose --profile triton up

# Add workers (requires Triton running)
docker compose --profile workers up

# Specific worker only
docker compose up upscale-worker
```

---

## Services

### Core Services

| Service | Description | Port |
|---------|-------------|------|
| `api` | FastAPI application | 8000 |
| `redis` | Cache and rate limiting | 6379 |
| `pubsub` | Pub/Sub emulator | 8085 |
| `firestore` | Firestore emulator | 8080 |
| `minio` | S3-compatible storage (replaces GCS) | 9000, 9001 |
| `jaeger` | Distributed tracing UI | 16686 |

### Triton Inference Server (Profile: `triton` or `all`)

| Service | Description | Ports |
|---------|-------------|-------|
| `triton` | NVIDIA Triton with all 5 models | 8000 (HTTP), 8001 (gRPC), 8002 (metrics) |

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed
- ~18GB disk space for models

### Thin Workers (Profile: `workers` or `all`)

| Service | Model | Description |
|---------|-------|-------------|
| `upscale-worker` | `upscale` | 4x image upscaling |
| `enhance-worker` | `enhance` | Quality enhancement |
| `comic-worker` | `style_comic` | Comic style transfer |
| `style-aged-worker` | `style_aged` | Vintage effect |
| `background-remove-worker` | `background_remove` | Background removal |

**Note:** All workers are CPU-only and call Triton via gRPC (no GPU required for workers).

---

## Docker Compose Profiles

Profiles allow selective service startup:

```bash
# Default (infrastructure only)
docker compose up

# API + infrastructure
docker compose up api

# Add Triton (requires GPU)
docker compose --profile triton up

# Add workers (requires Triton)
docker compose --profile workers up

# Everything (requires GPU)
docker compose --profile all up
```

---

## Building Images

### API Image

```bash
# From project root
docker build -t imagen-api -f docker/Dockerfile.api .
```

### Worker Image

The worker Dockerfile builds a **generic thin worker** that accepts `MODEL_NAME` at runtime:

```bash
# From project root
docker build -t imagen-worker -f docker/Dockerfile.worker .

# Run with specific model
docker run -e MODEL_NAME=upscale -e TRITON_URL=triton:8001 imagen-worker
docker run -e MODEL_NAME=enhance -e TRITON_URL=triton:8001 imagen-worker
docker run -e MODEL_NAME=style_comic -e TRITON_URL=triton:8001 imagen-worker
```

**Available MODEL_NAME values:**
- `upscale` - 4x upscaling
- `enhance` - Quality enhancement
- `style_comic` - Comic style
- `style_aged` - Vintage effect
- `background_remove` - Background removal

### Triton Image

```bash
# From project root
docker build -t triton-imagen -f triton/Dockerfile .
```

See `triton/README.md` for Triton-specific documentation.

---

## Environment Variables

### API

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `true` |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | `local-dev` |
| `GCS_BUCKET` | Storage bucket name | `local-bucket` |
| `PUBSUB_EMULATOR_HOST` | Pub/Sub emulator address | `pubsub:8085` |
| `FIRESTORE_EMULATOR_HOST` | Firestore emulator address | `firestore:8080` |
| `STORAGE_EMULATOR_HOST` | MinIO address | `http://minio:9000` |
| `OTEL_ENABLED` | Enable OpenTelemetry | `true` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Jaeger endpoint | `http://jaeger:4317` |

### Workers

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Model to process (required) | - |
| `TRITON_URL` | Triton gRPC endpoint | `triton:8001` |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | `local-dev` |
| `GCS_BUCKET` | Storage bucket name | `local-bucket` |
| `PUBSUB_SUBSCRIPTION_*` | Pub/Sub subscription name | `*-sub` |
| `OTEL_ENABLED` | Enable OpenTelemetry | `true` |

### Triton

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_HOME` | HuggingFace cache directory | `/models/huggingface` |
| `TRANSFORMERS_CACHE` | Transformers cache | `/models/huggingface` |
| `CUDA_VISIBLE_DEVICES` | GPU device ID | `0` |

---

## Volumes

| Volume | Purpose | Mount Point |
|--------|---------|-------------|
| `imagen-models` | ML model cache (shared by Triton) | `/models/huggingface` |
| `redis-data` | Redis persistence | `/data` |
| `minio-data` | MinIO storage | `/data` |

### Model Cache

Models are downloaded once by Triton and cached in the `imagen-models` volume (~18GB):

```bash
# Check volume size
docker volume inspect imagen-models

# Clear model cache (force re-download)
docker volume rm imagen-models
```

---

## GPU Support

### Requirements

- NVIDIA GPU with CUDA support (Compute Capability 7.0+)
- NVIDIA Docker runtime installed
- Docker Compose v2+

### Install NVIDIA Docker Runtime

**Ubuntu/Debian:**
```bash
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
docker compose up -d redis minio pubsub firestore jaeger
```

### 2. Start API (with hot reload)

```bash
# Option A: Docker
docker compose up api

# Option B: Local Python (faster for development)
cd ..
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start Triton (optional, for testing workers)

```bash
# Requires NVIDIA GPU
docker compose --profile triton up -d
```

### 4. Start Worker (optional, for end-to-end testing)

```bash
# Start specific worker
docker compose up upscale-worker

# Or all workers
docker compose --profile workers up
```

### 5. Test API

```bash
# Health check
curl http://localhost:8000/health

# Submit job
curl -X POST http://localhost:8000/api/v1/images/upscale \
  -F "file=@test.jpg"
```

### 6. View Traces

Open Jaeger UI: http://localhost:16686

---

## Production Images

For production, images are built by Cloud Build and pushed to Artifact Registry:

```bash
# API
us-central1-docker.pkg.dev/PROJECT_ID/imagen/api:latest

# Worker (thin, CPU-only)
us-central1-docker.pkg.dev/PROJECT_ID/imagen/worker:latest

# Triton
us-central1-docker.pkg.dev/PROJECT_ID/imagen/triton:latest
```

See `cloudbuild.yaml` for CI/CD pipeline details.

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

### Triton Fails to Start

```bash
# Check logs
docker compose logs triton

# Common issues:
# - GPU not available → Check nvidia-smi
# - Models not loading → Check volume and disk space
# - Out of memory → Reduce batch sizes in triton/model_repository/*/config.pbtxt
```

### Worker Can't Connect to Triton

```bash
# Check Triton is running
docker compose ps triton

# Check Triton health
curl http://localhost:8000/v2/health/ready

# Check worker logs
docker compose logs upscale-worker

# Verify network connectivity
docker compose exec upscale-worker ping triton
```

### Out of Memory (OOM)

```bash
# Check GPU memory
nvidia-smi

# Solutions:
# 1. Reduce Triton batch sizes in config.pbtxt
# 2. Scale down number of concurrent workers
# 3. Use smaller models
```

### Slow Model Download

Models are ~18GB total. First startup takes 10-20 minutes to download.

```bash
# Pre-download models (speeds up Triton startup)
docker compose run --rm triton \
  bash -c "cd /models/repository && ls -lh"

# Models cached in imagen-models volume
```

### Port Conflicts

```bash
# If ports 8000/8001 are in use, modify docker-compose.yml:
ports:
  - "8010:8000"  # API
  - "8011:8001"  # Triton gRPC
```

---

## Architecture Comparison

### Old (Pre-Triton)

- 5 GPU workers (one per model)
- Each worker loads model directly
- No batching
- ~$365/mo GPU cost

### New (Triton-based) ✅

- 1 Triton GPU server (all 5 models)
- 5 CPU-only thin workers
- Automatic batching
- ~$100/mo total cost
- **~73% cost savings**

---

## References

- **Production deployment:** `cloudbuild.yaml`, `k8s/triton/`, `k8s/workers/`
- **Worker implementation:** `src/workers/triton_worker.py`
- **Triton setup:** `triton/README.md`
- **System design:** `docs/core-concepts/SYSTEM_DESIGN.md`

# Imagen

AI-powered image processing microservices platform built for Google Cloud Platform.

## Features

- **Upscale** - 4x image upscaling using Real-ESRGAN
- **Enhance** - Image quality enhancement using SDXL Refiner
- **Comic Style** - Convert images to comic/cartoon style
- **Aged Style** - Apply aged/vintage effect to images
- **Background Remove** - Remove backgrounds using RMBG-1.4

## Architecture

Uses **NVIDIA Triton Inference Server** for efficient GPU utilization with dynamic batching.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Client    │────▶│  Cloud Run   │────▶│   Pub/Sub       │
└─────────────┘     │  (FastAPI)   │     │   (Job Queue)   │
                    └──────────────┘     └────────┬────────┘
                                                  │
                    ┌─────────────────────────────┼───────────────┐
                    │           GKE Autopilot     ▼               │
                    │                                             │
                    │   ┌─────────────────────────────────────┐   │
                    │   │     Workers (CPU-only)             │   │
                    │   │  upscale │ enhance │ comic │ ...    │   │
                    │   └─────────────────┬───────────────────┘   │
                    │                     │ gRPC                  │
                    │                     ▼                       │
                    │   ┌─────────────────────────────────────┐   │
                    │   │   Triton Inference Server (T4 GPU)  │   │
                    │   │   • Dynamic batching (2-4 images)   │   │
                    │   │   • All 5 models on single GPU      │   │
                    │   │   • ~4x throughput vs old approach  │   │
                    │   └─────────────────────────────────────┘   │
                    └─────────────────────────────────────────────┘
```

**Cost Savings:** 60-80% reduction by consolidating 5 GPU workers → 1-3 Triton instances.

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- GCP account with billing enabled (for production)
- gcloud CLI configured (for production)

### Local Development

```bash
# Setup environment
cd imagen
cp .env.example .env
nano .env  # Update GOOGLE_CLOUD_PROJECT and GCS_BUCKET

# Install dependencies
pip install -e ".[dev]"

# Start local services (Redis, MinIO)
make dev

# Run API server
make api

# Run Triton locally (requires NVIDIA GPU + Docker)
make triton-local

# Run thin workers (connects to local Triton)
make worker-triton-upscale
make worker-triton-enhance
# ... etc

# Run tests
make test
```

### Model Management

Models are downloaded automatically by Triton on first startup (~18.5GB total).

For faster cold starts, pre-download models:

```bash
make download-models
```

See [docs/models/MODEL_MANAGEMENT.md](docs/models/MODEL_MANAGEMENT.md) for details.

### API Endpoints

```
POST /api/v1/images/upscale           - Upscale image 4x
POST /api/v1/images/enhance           - Enhance image quality
POST /api/v1/images/style/comic       - Convert to comic style
POST /api/v1/images/style/aged        - Apply aged/vintage effect
POST /api/v1/images/background/remove - Remove background

GET  /api/v1/jobs/{job_id}            - Get job status
```

### Example Usage

```bash
# Submit an upscale job
curl -X POST "http://localhost:8000/api/v1/images/upscale" \
  -F "file=@image.jpg" \
  -F "prompt=high quality"

# Response: {"job_id": "abc-123", "status": "queued"}

# Check job status
curl "http://localhost:8000/api/v1/jobs/abc-123"
```

## Deployment

### Quick Deploy

```bash
# Deploy to dev
kubectl apply -k k8s/overlays/dev

# Deploy to production
kubectl apply -k k8s/overlays/prod

# Verify deployment
kubectl get pods -n imagen
```

### Infrastructure Setup

```bash
# Initialize Terraform
cd terraform
terraform init

# Plan and apply
terraform plan -var-file=environments/prod.tfvars
terraform apply -var-file=environments/prod.tfvars
```

### CI/CD

Push to `main` triggers automatic deployment via Cloud Build:
1. Builds API, Worker, and Triton images
2. Deploys to GKE
3. Waits for Triton to be ready
4. Deploys thin workers

See [cloudbuild.yaml](cloudbuild.yaml) for details.

## Project Structure

```
imagen/
├── src/
│   ├── api/           # FastAPI application
│   ├── pipelines/     # ML pipelines (legacy, for local dev)
│   ├── workers/       # Thin workers (triton_worker.py)
│   ├── services/      # GCP + Triton client integrations
│   └── core/          # Configuration & utilities
├── triton/            # Triton model repository
│   ├── Dockerfile
│   └── model_repository/
│       ├── upscale/
│       ├── enhance/
│       ├── background_remove/
│       ├── style_comic/
│       └── style_aged/
├── docker/            # Dockerfiles
├── k8s/               # Kubernetes manifests
│   ├── base/
│   ├── triton/        # Triton deployment
│   ├── workers/       # CPU-only workers
│   ├── autoscaling/
│   └── overlays/
├── terraform/         # Infrastructure as Code
└── docs/              # Documentation
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | - |
| `GCS_BUCKET` | Storage bucket name | - |
| `TRITON_URL` | Triton gRPC endpoint | `triton:8001` |
| `DEBUG` | Enable debug mode | `false` |

## Documentation

📚 **[Complete Documentation Index](docs/README.md)**

| Guide | Description |
|-------|-------------|
| [⚡ Quickstart](docs/getting-started/QUICKSTART.md) | Get running locally |
| [🚀 First Deployment](docs/getting-started/FIRST_DEPLOYMENT.md) | Deploy to GKE |
| [📖 System Design](docs/core-concepts/SYSTEM_DESIGN.md) | Architecture deep-dive |
| [🔧 Triton Setup](triton/README.md) | Triton configuration |

## License

MIT

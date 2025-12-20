# Imagen

AI-powered image processing microservices platform built for Google Cloud Platform.

## Features

- **Upscale** - 4x image upscaling using Stable Diffusion
- **Enhance** - Image quality enhancement
- **Comic Style** - Convert images to comic/cartoon style
- **Aged Style** - Apply aged/vintage effect to images
- **Background Remove** - Remove backgrounds from images

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Client    │────▶│  Cloud Run   │────▶│   Pub/Sub       │
└─────────────┘     │  (FastAPI)   │     │   (Job Queue)   │
                    └──────────────┘     └────────┬────────┘
                                                  │
                    ┌─────────────────────────────┼───────┐
                    │                             ▼       │
                    │           GKE Autopilot             │
                    │   ┌─────────┐  ┌─────────┐          │
                    │   │Upscale  │  │Enhance  │  ...     │
                    │   │Worker   │  │Worker   │          │
                    │   │(T4 GPU) │  │(T4 GPU) │          │
                    │   └─────────┘  └─────────┘          │
                    └─────────────────────────────────────┘
```

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

# .env file is already created - edit with your details
nano .env  # Update GOOGLE_CLOUD_PROJECT and GCS_BUCKET

# Install dependencies
pip install -e ".[dev]"

# Start local services (Redis, MinIO)
make dev

# Run API server
make api

# In separate terminals, run workers:
make worker-upscale
make worker-enhance
make worker-comic
make worker-aged
make worker-background-remove

# Run tests
make test
```

**Note:** For local development without GCP, the API runs in debug mode and skips GCP validation.

### Model Management

Models are downloaded automatically on first use (~18.5GB total). To speed up startup:

```bash
# Pre-download all models (recommended)
make download-models

# Or let them download automatically when workers start
make worker-upscale  # Downloads on first run
```

**See [docs/models/MODEL_MANAGEMENT.md](docs/models/MODEL_MANAGEMENT.md) for complete model management guide.**

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

**See [docs/getting-started/FIRST_DEPLOYMENT.md](docs/getting-started/FIRST_DEPLOYMENT.md) for step-by-step deployment or [docs/deployment/PRODUCTION_DEPLOYMENT.md](docs/deployment/PRODUCTION_DEPLOYMENT.md) for complete deployment guide.**

### Infrastructure Setup

```bash
# Initialize Terraform
cd terraform
terraform init

# Plan and apply
terraform plan -var-file=environments/prod.tfvars
terraform apply -var-file=environments/prod.tfvars
```

### Deploy Application

```bash
# Build and push images using Cloud Build (recommended)
gcloud builds submit --config=cloudbuild.yaml

# Deploy workers to GKE
kubectl apply -f k8s/base/
kubectl apply -f k8s/workers/

# Verify deployment
kubectl get pods -n imagen
```

### Cost Warning

GPU workers are expensive (~$0.35/hour per T4 GPU = ~$250/month per worker).
Set up billing alerts before deploying to production!

## Project Structure

```
imagen/
├── src/
│   ├── api/           # FastAPI application
│   ├── pipelines/     # ML processing pipelines
│   ├── workers/       # Pub/Sub workers
│   ├── services/      # GCP service integrations
│   └── core/          # Configuration & utilities
├── docker/            # Dockerfiles
├── terraform/         # Infrastructure as Code
├── k8s/              # Kubernetes manifests
└── tests/            # Test suite
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | - |
| `GCS_BUCKET` | Storage bucket name | - |
| `DEVICE` | PyTorch device | `cuda` |
| `DEBUG` | Enable debug mode | `false` |

## Documentation

📚 **[Complete Documentation Index](docs/README.md)**

### Quick Start Guides

| Guide | Description |
|-------|-------------|
| [⚡ Quickstart](docs/getting-started/QUICKSTART.md) | Get running locally in 10 minutes |
| [🚀 First Deployment](docs/getting-started/FIRST_DEPLOYMENT.md) | Deploy to production step-by-step |
| [📖 System Design](docs/core-concepts/SYSTEM_DESIGN.md) | Complete architecture and design |

### Key Documentation

| Document | Description |
|----------|-------------|
| [Quick Reference](docs/reference/QUICK_REFERENCE.md) | One-page cheat sheet for common commands |
| [Configuration Reference](docs/reference/CONFIGURATION_REFERENCE.md) | All environment variables explained |
| [Model Management](docs/models/MODEL_MANAGEMENT.md) | ML model lifecycle and caching |
| [Infrastructure Guide](docs/infrastructure/INFRASTRUCTURE_OVERVIEW.md) | Terraform and Kubernetes basics |
| [CI/CD Pipeline](docs/deployment/CICD_PIPELINE.md) | Automated build and deployment |
| [Git Workflow](docs/development/GIT_WORKFLOW.md) | Git branching and commit conventions |
| [Changelog](CHANGELOG.md) | Project history and changes |

## License

MIT

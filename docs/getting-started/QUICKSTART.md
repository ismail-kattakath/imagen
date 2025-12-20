# Quickstart Guide

Get Imagen running locally in **10 minutes** or less.

## Prerequisites

- Python 3.11+
- Docker (optional, for local services)
- 15GB free disk space for ML models

## Quick Setup (Local Development)

###

 1. Clone and Install

```bash
cd imagen
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
# The .env file already exists with defaults
# For local development, you can use it as-is
cat .env  # Review the settings
```

### 3. Start API Server

```bash
# Start in development mode (no GCP required)
make api

# Or manually:
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**API is now running at**: `http://localhost:8000`

### 4. Test the API

```bash
# Check health
curl http://localhost:8000/health

# Response: {"status":"healthy"}
```

### 5. Start a Worker (Optional)

Workers process the actual image transformations. Start one to test end-to-end:

```bash
# Upscale worker (downloads ~2.5GB on first run)
make worker-upscale

# This will:
# 1. Auto-detect your device (GPU or CPU)
# 2. Download model on first run (2-10 minutes)
# 3. Start listening for jobs
```

### 6. Submit a Test Job

```bash
# Submit an image for upscaling
curl -X POST "http://localhost:8000/api/v1/images/upscale" \
  -F "file=@/path/to/image.jpg" \
  -F "prompt=high quality"

# Response: {"job_id": "abc-123", "status": "queued"}
```

## What's Next?

### For Development

- **[Local Setup](../development/LOCAL_SETUP.md)** - Complete local development guide
- **[Development Workflow](../development/DEVELOPMENT_WORKFLOW.md)** - Daily development practices

### For Deployment

- **[First Deployment](FIRST_DEPLOYMENT.md)** - Deploy to production step-by-step
- **[Production Deployment](../deployment/PRODUCTION_DEPLOYMENT.md)** - Full deployment guide

### Learn More

- **[Architecture Overview](../core-concepts/ARCHITECTURE_OVERVIEW.md)** - How Imagen works
- **[Model Management](../models/MODEL_MANAGEMENT.md)** - Understanding ML models

## Common Issues

### Port already in use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Models downloading slowly

```bash
# Pre-download all models
make download-models
```

### Worker not starting

```bash
# Check if you have GPU
python -c "import torch; print(torch.cuda.is_available())"

# Workers auto-fallback to CPU if no GPU
```

## Project Structure Overview

```
imagen/
├── src/
│   ├── api/          # FastAPI REST API
│   ├── workers/      # Background job processors
│   ├── pipelines/    # ML model pipelines
│   └── services/     # GCP integrations
├── docker/           # Container definitions
├── k8s/              # Kubernetes manifests
└── docs/             # Documentation (you are here)
```

## Need Help?

- **Quick answers**: See [Quick Reference](../reference/QUICK_REFERENCE.md)
- **Commands**: See [Command Reference](../reference/COMMAND_REFERENCE.md)
- **Issues**: Check [Development Troubleshooting](../development/TROUBLESHOOTING_DEV.md)

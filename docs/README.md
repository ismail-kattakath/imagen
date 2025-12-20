# Imagen Documentation

This directory contains comprehensive documentation for the Imagen project.

## Documentation Index

### Getting Started
- [Main README](../README.md) - Project overview and quick start guide
- [Quick Reference](../QUICK_REFERENCE.md) - Common commands and workflows

### Development
- [Architecture](../ARCHITECTURE.md) - System architecture and design patterns
- [Model Management](../MODEL_MANAGEMENT.md) - Working with ML models
- [Model Quickstart](../MODEL_QUICKSTART.md) - Quick guide to ML models

### Deployment
- [Deployment Guide](../DEPLOYMENT_GUIDE.md) - Production deployment instructions
- [Infrastructure Guide](../INFRASTRUCTURE_GUIDE.md) - Infrastructure setup and management
- [CI/CD](../CI_CD.md) - Continuous integration and deployment

### Project Management
- [Changelog](../CHANGELOG.md) - Version history and changes
- [Completeness Report](../COMPLETENESS_REPORT.md) - Project status and completeness
- [Git Setup](../GIT_SETUP.md) - Git configuration and workflow

## Quick Links

### Common Tasks
- **First Time Setup**: See [README.md](../README.md#quick-start)
- **Deploy to Production**: See [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)
- **Add New Model**: See [MODEL_MANAGEMENT.md](../MODEL_MANAGEMENT.md)
- **CI/CD Setup**: See [CI_CD.md](../CI_CD.md)

### API Documentation
- **Health Check**: `GET /health`
- **Submit Job**: `POST /api/v1/images/{operation}`
- **Check Status**: `GET /api/v1/jobs/{job_id}`

Available operations: `upscale`, `enhance`, `comic-style`, `aged-style`, `background-remove`

### Architecture Overview

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

## Project Structure

```
imagen/
├── src/
│   ├── api/           # FastAPI REST API
│   ├── pipelines/     # ML processing pipelines
│   ├── workers/       # Pub/Sub message handlers
│   ├── services/      # GCP service integrations
│   ├── core/          # Core utilities and config
│   └── utils/         # Helper functions
├── docker/            # Container definitions
├── k8s/               # Kubernetes manifests
├── tests/             # Test suite
└── docs/              # Documentation (this directory)
```

## Configuration

Key environment variables (see `.env.example`):

```bash
# GCP Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GCS_BUCKET=your-bucket-name

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Model Configuration
MODEL_CACHE_DIR=/models
DEVICE=cuda  # or 'cpu' for CPU-only
```

## Development Workflow

1. **Local Development**
   ```bash
   make dev          # Start API server
   make test         # Run tests
   make lint         # Check code quality
   ```

2. **Docker Build**
   ```bash
   make docker-build-api      # Build API container
   make docker-build-worker   # Build worker container
   ```

3. **Deployment**
   ```bash
   make deploy-dev   # Deploy to development
   make deploy-prod  # Deploy to production
   ```

## Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: All documentation is in this repository
- **Architecture Questions**: See [ARCHITECTURE.md](../ARCHITECTURE.md)

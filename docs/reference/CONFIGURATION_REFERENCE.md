# Configuration Reference

Complete reference for all environment variables and configuration options in Imagen.

## Environment Variables

All configuration is done through environment variables, loaded from `.env` file or system environment.

### GCP Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | Yes* | `""` | GCP project ID where resources are deployed |
| `GCS_BUCKET` | Yes* | `""` | Google Cloud Storage bucket name for image storage |

**Note**: These are required for production deployment. In development mode (`DEBUG=true`), validation is skipped.

### API Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_HOST` | No | `0.0.0.0` | API server bind address (use `0.0.0.0` for Docker) |
| `API_PORT` | No | `8000` | API server port |
| `DEBUG` | No | `false` | Enable debug mode (disables GCP validation) |

### Pub/Sub Subscriptions

Each worker type has its own subscription:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PUBSUB_SUBSCRIPTION_UPSCALE` | No | `upscale-jobs-sub` | Subscription for upscale jobs |
| `PUBSUB_SUBSCRIPTION_ENHANCE` | No | `enhance-jobs-sub` | Subscription for enhance jobs |
| `PUBSUB_SUBSCRIPTION_COMIC` | No | `style-comic-jobs-sub` | Subscription for comic style jobs |
| `PUBSUB_SUBSCRIPTION_AGED` | No | `style-aged-jobs-sub` | Subscription for aged style jobs |
| `PUBSUB_SUBSCRIPTION_BACKGROUND_REMOVE` | No | `background-remove-jobs-sub` | Subscription for background removal jobs |

### Model Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_CACHE_DIR` | No | `/models` | Directory where ML models are cached |
| `DEVICE` | No | Auto-detect | PyTorch device: `cuda`, `cpu`, or `None` for auto-detect |
| `TORCH_DTYPE` | No | `float16` | Model precision: `float16` (GPU) or `float32` (CPU) |

**Auto-detection behavior**:
- If `DEVICE=None` or not set: Automatically detects CUDA availability
- Uses `float16` if CUDA available, `float32` if CPU only
- Workers log detected device on startup

### HuggingFace Cache (Auto-configured)

These are set automatically based on `MODEL_CACHE_DIR`:

| Variable | Auto-Value | Description |
|----------|------------|-------------|
| `HF_HOME` | `{MODEL_CACHE_DIR}/huggingface` | HuggingFace cache directory |
| `TRANSFORMERS_CACHE` | `{MODEL_CACHE_DIR}/transformers` | Transformers library cache |

### Local Development (Optional)

These are for local development with alternative services:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | No | `None` | Redis URL for local job queue (not implemented) |
| `MINIO_ENDPOINT` | No | `None` | MinIO endpoint for local storage (not implemented) |
| `MINIO_ACCESS_KEY` | No | `None` | MinIO access key |
| `MINIO_SECRET_KEY` | No | `None` | MinIO secret key |

---

## Configuration Files

### .env File

Example `.env` file:

```bash
# GCP Configuration (Required for production)
GOOGLE_CLOUD_PROJECT=imagen-prod-12345
GCS_BUCKET=imagen-prod-12345-images

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Pub/Sub Subscriptions
PUBSUB_SUBSCRIPTION_UPSCALE=upscale-jobs-sub
PUBSUB_SUBSCRIPTION_ENHANCE=enhance-jobs-sub
PUBSUB_SUBSCRIPTION_COMIC=style-comic-jobs-sub
PUBSUB_SUBSCRIPTION_AGED=style-aged-jobs-sub
PUBSUB_SUBSCRIPTION_BACKGROUND_REMOVE=background-remove-jobs-sub

# Model Configuration
MODEL_CACHE_DIR=/models
DEVICE=cuda
TORCH_DTYPE=float16
```

### Kubernetes ConfigMap

Located at `k8s/base/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: imagen-config
data:
  GOOGLE_CLOUD_PROJECT: "your-project-id"
  GCS_BUCKET: "your-bucket-name"
  MODEL_CACHE_DIR: "/models"
  DEVICE: "cuda"
  # ... other variables
```

---

## Environment-Specific Configuration

### Local Development

```bash
# .env for local development
DEBUG=true                              # Skip GCP validation
GOOGLE_CLOUD_PROJECT=local-dev         # Placeholder (not validated)
GCS_BUCKET=local-dev-bucket            # Placeholder (not validated)
MODEL_CACHE_DIR=./models               # Local directory
DEVICE=cpu                             # Or cuda if you have GPU
```

### Development (GCP)

```bash
DEBUG=false
GOOGLE_CLOUD_PROJECT=imagen-dev-12345
GCS_BUCKET=imagen-dev-12345-images
MODEL_CACHE_DIR=/models
DEVICE=cuda
```

### Production (GCP)

```bash
DEBUG=false
GOOGLE_CLOUD_PROJECT=imagen-prod-12345
GCS_BUCKET=imagen-prod-12345-images
MODEL_CACHE_DIR=/models
DEVICE=cuda
TORCH_DTYPE=float16
```

---

## Validation Rules

### Production Mode Validation

When `DEBUG=false` and `GOOGLE_CLOUD_PROJECT` is set:

1. **GOOGLE_CLOUD_PROJECT** must not be:
   - Empty string
   - "your-project-id"
   - Any placeholder value

2. **GCS_BUCKET** must not be:
   - Empty string
   - "your-bucket-name"
   - Any placeholder value

### Error Messages

If validation fails:

```
Configuration errors:
  - GOOGLE_CLOUD_PROJECT must be set to your actual GCP project ID
  - GCS_BUCKET must be set to your actual GCS bucket name
```

Application will fail to start with clear error message.

---

## Configuration Loading Order

Settings are loaded in this priority order:

1. **System environment variables** (highest priority)
2. **`.env` file** in project root
3. **Default values** in `src/core/config.py`

Example:
```bash
# If you set in terminal:
export DEBUG=true

# This overrides .env file:
DEBUG=false  # This is ignored

# Final value: DEBUG=true
```

---

## Kubernetes Configuration

### Per-Environment Overlays

Imagen uses Kustomize overlays for environment-specific config:

**Base**: `k8s/base/`
- Common resources (namespace, PVC, configmap)

**Dev**: `k8s/overlays/dev/`
- Development-specific patches
- Lower resource limits
- Development image tags

**Prod**: `k8s/overlays/prod/`
- Production-specific patches
- Production resource limits
- Production image tags

### Applying Configuration

```bash
# Development
kubectl apply -k k8s/overlays/dev/

# Production
kubectl apply -k k8s/overlays/prod/
```

---

## Model-Specific Configuration

### Model IDs

Default model IDs (can be overridden in code):

| Pipeline | Model ID | Size |
|----------|----------|------|
| Upscale | `stabilityai/stable-diffusion-x4-upscaler` | 2.5GB |
| Enhance | `stabilityai/stable-diffusion-xl-refiner-1.0` | 6GB |
| Comic Style | `nitrosocke/Ghibli-Diffusion` | 4GB |
| Aged Style | `stabilityai/stable-diffusion-2-1` | 5GB |
| Background Remove | `briaai/RMBG-1.4` | 1GB |

### Cache Directory Structure

```
/models/
├── huggingface/
│   └── hub/
│       ├── models--stabilityai--stable-diffusion-x4-upscaler/
│       ├── models--stabilityai--stable-diffusion-xl-refiner-1.0/
│       ├── models--nitrosocke--Ghibli-Diffusion/
│       ├── models--stabilityai--stable-diffusion-2-1/
│       └── models--briaai--RMBG-1.4/
└── torch/
    └── hub/
        └── checkpoints/
```

---

## Common Configuration Patterns

### Local Development (No GPU)

```bash
DEBUG=true
DEVICE=cpu
TORCH_DTYPE=float32
MODEL_CACHE_DIR=./models
```

### Local Development (With GPU)

```bash
DEBUG=true
DEVICE=cuda
TORCH_DTYPE=float16
MODEL_CACHE_DIR=./models
```

### CI/CD Pipeline

```bash
DEBUG=true
GOOGLE_CLOUD_PROJECT=ci-test-project
GCS_BUCKET=ci-test-bucket
MODEL_CACHE_DIR=/tmp/models
```

### Production (Cost-Optimized)

```bash
DEBUG=false
DEVICE=cuda
TORCH_DTYPE=float16
# Use smaller PVC if models pre-loaded in image
MODEL_CACHE_DIR=/models
```

---

## Troubleshooting Configuration

### Check Current Configuration

```bash
# In Python
from src.core.config import settings
print(settings.model_dump())
```

### Validate Configuration

```bash
# Run validation
python -c "from src.core.config import settings; settings.validate_gcp_config()"
```

### Common Issues

**Issue**: "Configuration validation failed"

**Solution**: Check `GOOGLE_CLOUD_PROJECT` and `GCS_BUCKET` are not placeholders

**Issue**: Models downloading to wrong location

**Solution**: Verify `MODEL_CACHE_DIR` in environment

**Issue**: "CUDA not available" but you have GPU

**Solution**: Check NVIDIA drivers and CUDA installation

---

## Security Best Practices

### Never Commit

❌ Don't commit to git:
- `.env` file with real values
- Service account keys
- API keys or secrets

### Use Secret Management

For production:
- Use **Google Secret Manager** for sensitive values
- Mount secrets as environment variables in Cloud Run/GKE
- Use **Workload Identity** instead of service account keys

### Example with Secret Manager

```bash
# Store secret
gcloud secrets create gcp-project-id --data-file=- <<< "imagen-prod-12345"

# Use in Cloud Run
gcloud run deploy imagen-api \
  --set-secrets="GOOGLE_CLOUD_PROJECT=gcp-project-id:latest"
```

---

## Related Documentation

- **[Quickstart](../getting-started/QUICKSTART.md)** - Get started quickly
- **[Production Deployment](../deployment/PRODUCTION_DEPLOYMENT.md)** - Full deployment guide
- **[Model Management](../models/MODEL_MANAGEMENT.md)** - Model configuration details
- **[Infrastructure Overview](../infrastructure/INFRASTRUCTURE_OVERVIEW.md)** - Infrastructure setup

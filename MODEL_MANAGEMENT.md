# Model Management Guide for Imagen

Complete guide for managing AI models in local development and production environments.

---

## Model Architecture Overview

Imagen uses 4 different ML models from HuggingFace:

| Pipeline | Model ID | Size | Purpose |
|----------|----------|------|---------|
| **Upscale** | `stabilityai/stable-diffusion-x4-upscaler` | ~2.5GB | 4x image upscaling |
| **Enhance** | `stabilityai/stable-diffusion-xl-refiner-1.0` | ~6GB | Image quality enhancement |
| **Comic Style** | `nitrosocke/Ghibli-Diffusion` | ~4GB | Convert to comic/cartoon style |
| **Background Remove** | `briaai/RMBG-1.4` | ~1GB | Remove image backgrounds |

**Total Storage Required:** ~14GB for all models

---

## How Models Are Loaded

### Current Implementation

Models are loaded **on-demand** when workers start:

```python
# In src/pipelines/upscale.py
self._pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id,  # e.g., "stabilityai/stable-diffusion-x4-upscaler"
    torch_dtype=self.dtype,
    cache_dir=settings.model_cache_dir,  # Defaults to /models
).to(self.device)
```

### Model Loading Flow

1. **Worker starts** → `worker.run()` called
2. **Pipeline initialization** → `pipeline.load()` called
3. **HuggingFace download** → Checks cache first
4. **If cached** → Load from disk (~10-30 seconds)
5. **If not cached** → Download from HuggingFace (~2-10 minutes)
6. **GPU transfer** → Model loaded to GPU/CPU
7. **Worker ready** → Starts processing jobs

---

## Local Development

### Option 1: Auto-Download (Simplest)

Models download automatically on first use:

```bash
# Start worker - models will download automatically
make worker-upscale

# First run: Downloads ~2.5GB, takes 2-10 min
# Subsequent runs: Loads from cache, takes 10-30 sec
```

**Pros:**
- No setup required
- Always gets latest model versions

**Cons:**
- Slow first startup
- Requires internet connection
- Downloads every model separately

### Option 2: Pre-Download Models

Pre-download models to avoid slow startup:

```bash
# Create download script
cat > download_models.py << 'SCRIPT'
from diffusers import StableDiffusionUpscalePipeline, StableDiffusionImg2ImgPipeline
from transformers import pipeline
import os

# Set cache directory
cache_dir = os.getenv("MODEL_CACHE_DIR", "./models")
os.makedirs(cache_dir, exist_ok=True)

print("Downloading models...")

# Upscale model
print("1/4 Downloading upscale model...")
StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    cache_dir=cache_dir
)

# Enhance model
print("2/4 Downloading enhance model...")
StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    cache_dir=cache_dir
)

# Comic style model
print("3/4 Downloading comic style model...")
StableDiffusionImg2ImgPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion",
    cache_dir=cache_dir
)

# Background removal model
print("4/4 Downloading background removal model...")
pipeline(
    "image-segmentation",
    model="briaai/RMBG-1.4",
    trust_remote_code=True,
    cache_dir=cache_dir
)

print("✅ All models downloaded successfully!")
SCRIPT

# Run download script
python download_models.py
```

**Pros:**
- Fast worker startup
- Can download once, use many times
- Works offline after download

**Cons:**
- Requires ~14GB disk space
- Initial download still takes time

### Option 3: Use Smaller Models (Development Only)

For testing without GPU or to save space:

```python
# Create src/pipelines/config.py
MODEL_CONFIGS = {
    "dev": {
        "upscale": "stabilityai/sd-x2-latent-upscaler",  # Smaller 2x model
        "enhance": "runwayml/stable-diffusion-v1-5",      # Smaller model
        # ... etc
    },
    "prod": {
        "upscale": "stabilityai/stable-diffusion-x4-upscaler",
        # ... full models
    }
}
```

---

## Production (GKE) - Recommended Approach

### Strategy 1: Persistent Volume (Recommended)

**How it works:**
- Models stored on shared persistent disk
- All workers access same cached models
- Download once, used by all workers

**Setup:**

1. **Persistent Volume Claim (Already Created)**
   ```yaml
   # k8s/base/pvc.yaml
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: models-pvc
     namespace: imagen
   spec:
     accessModes:
       - ReadWriteOnce  # ← One worker at a time
     resources:
       requests:
         storage: 50Gi  # ← Plenty for all models
   ```

2. **Workers Mount PVC**
   ```yaml
   # k8s/workers/upscale-worker.yaml
   volumeMounts:
     - name: models
       mountPath: /models
   volumes:
     - name: models
       persistentVolumeClaim:
         claimName: models-pvc
   ```

3. **First Worker Downloads**
   - First worker to start downloads models
   - Subsequent workers find cached models
   - ~14GB stored on persistent disk

**Pros:**
- ✅ Fast startup after initial download
- ✅ No duplicate downloads
- ✅ Survives pod restarts
- ✅ Cost-effective (pay for storage, not downloads)

**Cons:**
- ⚠️ ReadWriteOnce = one writer at a time
- ⚠️ ~$2/month for 50GB storage

**Best for:** Most production deployments

### Strategy 2: Pre-built Docker Images

**How it works:**
- Bake models into Docker image
- No download at runtime
- Fastest startup time

**Implementation:**

```dockerfile
# docker/Dockerfile.worker-with-models
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y build-essential git
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# PRE-DOWNLOAD MODELS
COPY download_models.py .
RUN python download_models.py

# Copy source
COPY src/ src/

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/huggingface

CMD python -m src.workers.${WORKER}
```

**Build:**
```bash
# Build image with models (takes 10-20 min)
docker build -t imagen-worker-with-models -f docker/Dockerfile.worker-with-models .

# Image size: ~15GB (base 2GB + models 13GB)
```

**Pros:**
- ✅ Fastest worker startup (~10 seconds)
- ✅ No runtime downloads
- ✅ Consistent across environments
- ✅ Works in air-gapped environments

**Cons:**
- ❌ Large images (~15GB)
- ❌ Longer build times
- ❌ Higher registry costs
- ❌ Hard to update models

**Best for:** 
- Air-gapped deployments
- Critical uptime requirements
- Fixed model versions

### Strategy 3: Cloud Storage Bucket

**How it works:**
- Models stored in GCS bucket
- Workers download from GCS (faster than HuggingFace)
- Can use signed URLs or public access

**Setup:**

```bash
# 1. Download models locally
python download_models.py

# 2. Upload to GCS
gsutil -m cp -r ./models/* gs://your-bucket/models/

# 3. Update config
# In .env or ConfigMap:
MODEL_SOURCE=gs://your-bucket/models
```

**Update pipeline code:**
```python
# src/pipelines/base.py
if settings.model_source.startswith("gs://"):
    # Download from GCS first
    download_from_gcs(model_id, cache_dir)
    
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    cache_dir / model_id,
    local_files_only=True
)
```

**Pros:**
- ✅ Faster download than HuggingFace
- ✅ Full control over models
- ✅ Version control
- ✅ Works in restricted networks

**Cons:**
- ⚠️ Additional complexity
- ⚠️ GCS egress costs
- ⚠️ Manual updates

**Best for:**
- Large-scale deployments
- Compliance requirements
- Custom models

---

## Model Caching Configuration

### Environment Variables

```bash
# .env
MODEL_CACHE_DIR=/models              # Where to cache models
HF_HOME=/models/huggingface          # HuggingFace cache
TRANSFORMERS_CACHE=/models/transformers  # Transformers cache
HUGGINGFACE_HUB_CACHE=/models/hub    # Hub cache
```

### Cache Directories Explained

```
/models/
├── huggingface/              # Diffusers models
│   ├── hub/                  # Downloaded files
│   └── diffusers/            # Pipeline caches
├── transformers/             # Transformers models
└── torch/                    # PyTorch model cache
```

### Clear Cache

```bash
# Local
rm -rf ./models/*

# In Kubernetes
kubectl exec -n imagen upscale-worker-xxx -- rm -rf /models/*
```

---

## Performance Optimization

### 1. Model Loading Optimization

Already implemented in `src/pipelines/upscale.py`:

```python
# Enable memory optimizations
self._pipeline.enable_attention_slicing()

# Additional optimizations you can add:
# For CUDA:
if self.device == "cuda":
    self._pipeline.enable_xformers_memory_efficient_attention()
    self._pipeline.enable_vae_slicing()
```

### 2. Pre-warm Workers

Create an init container to pre-download models:

```yaml
# k8s/workers/upscale-worker.yaml
initContainers:
  - name: model-downloader
    image: us-central1-docker.pkg.dev/PROJECT_ID/imagen/worker:latest
    command:
      - python
      - -c
      - |
        from src.pipelines import UpscalePipeline
        pipeline = UpscalePipeline()
        pipeline.load()
        print("Models pre-downloaded")
    volumeMounts:
      - name: models
        mountPath: /models
```

### 3. ReadWriteMany for Multiple Workers

To allow multiple workers to access models simultaneously:

```yaml
# k8s/base/pvc.yaml
spec:
  accessModes:
    - ReadWriteMany  # ← Multiple workers
  storageClassName: standard-rwx  # ← GKE Filestore
  resources:
    requests:
      storage: 50Gi
```

**Note:** ReadWriteMany requires Filestore (~$200/month), only use if scaling >5 workers

---

## Monitoring & Troubleshooting

### Check Model Download Progress

```bash
# Local
ls -lh ./models/

# Kubernetes
kubectl exec -n imagen upscale-worker-xxx -- du -sh /models/*
```

### Worker Logs

```bash
# Watch model loading
kubectl logs -n imagen -l app=upscale-worker --tail=100 -f

# Look for:
# "Loading upscale model: stabilityai/..."
# "Upscale model loaded successfully"
```

### Common Issues

**1. Out of disk space**
```
Error: No space left on device
Solution: Increase PVC size or clean old models
```

**2. Slow downloads**
```
Taking >10 minutes to download
Solution: 
- Use GCS bucket strategy
- Pre-build Docker images
- Check network bandwidth
```

**3. OOM (Out of Memory)**
```
Error: CUDA out of memory
Solution:
- Increase worker memory limits
- Enable memory optimizations
- Use smaller batch sizes
- Switch to CPU for testing
```

---

## Cost Analysis

### Storage Costs (GCP)

| Strategy | Storage Cost/Month | Download Cost | Total |
|----------|-------------------|---------------|-------|
| **PVC (50GB)** | ~$2 | Free (within GCP) | ~$2 |
| **Docker Images (15GB/image × 4)** | ~$20 (Artifact Registry) | $0 | ~$20 |
| **Filestore (ReadWriteMany)** | ~$200 | $0 | ~$200 |
| **GCS Bucket (14GB)** | ~$0.30 | ~$0.12/GB egress | ~$2 |

**Recommendation:** Use PVC strategy for cost-effectiveness

---

## Quick Reference

### Local Development
```bash
# Auto-download (easiest)
make worker-upscale

# Pre-download all models
python download_models.py
```

### Production (Recommended)
```bash
# Deploy with PVC (already configured)
kubectl apply -f k8s/base/pvc.yaml
kubectl apply -f k8s/workers/upscale-worker.yaml

# Models auto-download on first worker start
# Shared across all workers via PVC
```

### Check Model Status
```bash
# Local
ls -lh ./models/huggingface/hub/

# Production
kubectl exec -n imagen upscale-worker-xxx -- ls -lh /models/
```

---

## Best Practices

1. **Use PVC in production** - Cost-effective and simple
2. **Pre-download in dev** - Saves startup time
3. **Monitor disk usage** - Models are large
4. **Version control** - Track which models are used
5. **Test with CPU first** - Before spending on GPU
6. **Enable optimizations** - Memory-efficient attention
7. **Set cache locations** - Use environment variables

---

## Summary

**For Local Development:**
- Let models auto-download (simplest)
- Or pre-download with script (faster)

**For Production:**
- Use PVC strategy (recommended, ~$2/month)
- Models download once, shared by all workers
- Fast startup, cost-effective, simple

**Advanced:**
- Pre-built images for air-gap
- GCS bucket for large scale
- Filestore for many concurrent workers

The current implementation already supports the recommended PVC strategy! Just deploy and it works. ✅

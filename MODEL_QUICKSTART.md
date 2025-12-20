# Model Management - Quick Reference

## TL;DR

### Local Development (Fastest Setup)

```bash
# Option 1: Pre-download models (recommended)
make download-models  # Takes 5-20 min, downloads ~14GB

# Option 2: Auto-download on first worker start
make worker-upscale   # Downloads model automatically
```

### Production (Recommended Setup)

Already configured! Models auto-download to PVC on first worker start:

```bash
# Deploy (models auto-download once)
kubectl apply -f k8s/base/pvc.yaml
kubectl apply -f k8s/workers/upscale-worker.yaml

# All workers share the same cached models via PVC
```

---

## Models Used

| Pipeline | Model | Size | Purpose |
|----------|-------|------|---------|
| Upscale | `stabilityai/stable-diffusion-x4-upscaler` | 2.5GB | 4x upscaling |
| Enhance | `stabilityai/stable-diffusion-xl-refiner-1.0` | 6GB | Quality enhancement |
| Comic | `nitrosocke/Ghibli-Diffusion` | 4GB | Cartoon style |
| Background | `briaai/RMBG-1.4` | 1GB | Remove background |

**Total:** ~14GB

---

## Local Dev Strategies

### 1. Auto-Download (Easiest)
```bash
make worker-upscale  # Downloads on first run
```
✅ No setup  
⚠️ Slow first startup (2-10 min)

### 2. Pre-Download (Recommended)
```bash
make download-models  # Run once
make worker-upscale   # Fast startup
```
✅ Fast subsequent startups  
✅ Works offline  
⚠️ Uses 14GB disk

### 3. Custom Cache Location
```bash
# In .env
MODEL_CACHE_DIR=./my-models

make download-models
```

---

## Production Strategies

### PVC (Default & Recommended) ✅
- Models stored on persistent disk
- Shared across all workers
- Auto-download on first worker start
- Cost: ~$2/month for 50GB

**Already configured - just deploy!**

### Pre-built Images (Advanced)
- Bake models into Docker image
- Fastest startup (~10 sec)
- Image size: ~15GB
- Cost: ~$20/month storage

### GCS Bucket (Enterprise)
- Models in Cloud Storage
- Version control
- Fast downloads within GCP
- Cost: ~$2/month

---

## Troubleshooting

### Slow downloads?
```bash
# Use pre-download script
make download-models
```

### Out of space?
```bash
# Check usage
du -sh ./models/

# Clean cache
rm -rf ./models/*
```

### Workers crashing?
```bash
# Check if models loaded
kubectl logs -n imagen upscale-worker-xxx | grep "model loaded"

# Check disk space
kubectl exec -n imagen upscale-worker-xxx -- df -h /models
```

---

## Commands

```bash
# Download all models
make download-models

# Check download status
ls -lh ./models/huggingface/hub/

# Start worker (auto-downloads if needed)
make worker-upscale

# Clean cache
rm -rf ./models/*
```

---

## FAQs

**Q: Do I need to download models manually?**  
A: No, they auto-download. But pre-downloading is faster.

**Q: Where are models stored?**  
A: Local: `./models/` or `MODEL_CACHE_DIR` env var  
A: Production: `/models` mounted from PVC

**Q: How much disk space needed?**  
A: ~14GB for all 4 models

**Q: Can I use different models?**  
A: Yes, change `model_id` in pipeline files

**Q: Do workers share cached models?**  
A: Local: Yes, if same cache directory  
A: Production: Yes, via PVC

**Q: What if download fails?**  
A: Retry or check internet connection. Models download from HuggingFace.

---

For complete details, see **[MODEL_MANAGEMENT.md](MODEL_MANAGEMENT.md)**

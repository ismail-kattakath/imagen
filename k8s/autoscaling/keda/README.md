# KEDA - Scale to Zero

KEDA (Kubernetes Event-Driven Autoscaling) enables GKE workers to scale to zero when idle.

---

## Why KEDA?

Standard HPA has a limitation:

```
Standard HPA: Can scale N → 0, but NOT 0 → N
KEDA:         Can scale 0 → N → 0
```

KEDA monitors Pub/Sub queue depth and wakes pods when messages arrive.

---

## Cost Savings

| Configuration | Monthly Cost | Notes |
|---------------|--------------|-------|
| Current (min: 1 each) | ~$1,400 | 4 GPUs 24/7 |
| KEDA (all scale-to-zero) | ~$170 | Assuming 4 hrs/day usage |
| Hybrid (1 hot, 3 cold) | ~$350 | Balance cost/latency |

---

## Trade-offs

### Pros
- Massive cost savings for bursty workloads
- Pay only when processing

### Cons
- Cold start delay: 2-5 minutes for GPU workloads
  - Node provisioning: ~60-120s
  - Image pull: ~30-60s
  - Model loading: ~30-60s

---

## Installation

```bash
# Install KEDA
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda \
  --namespace keda \
  --create-namespace

# Verify
kubectl get pods -n keda
```

---

## Configuration Options

### Option 1: Full Scale-to-Zero (Maximum Savings)

All workers scale to 0 when queues are empty.

```bash
kubectl apply -f k8s/autoscaling/keda/scaledobjects.yaml
```

**When to use:** Low traffic, cost-sensitive, cold starts acceptable

### Option 2: Hybrid (Recommended)

Primary pipeline stays hot, others scale to zero.

```bash
kubectl apply -f k8s/autoscaling/keda/hybrid-scaledobjects.yaml
```

**When to use:** Need low latency for main use case, save on secondary features

---

## Switching from Standard HPA

```bash
# Remove existing HPAs (they conflict with KEDA)
kubectl delete hpa -n imagen --all

# Apply KEDA ScaledObjects
kubectl apply -f k8s/autoscaling/keda/scaledobjects.yaml

# Verify
kubectl get scaledobject -n imagen
```

---

## Monitoring

```bash
# Check ScaledObject status
kubectl get scaledobject -n imagen

# Watch scaling events
kubectl describe scaledobject upscale-worker-keda -n imagen

# Check KEDA operator logs
kubectl logs -n keda -l app=keda-operator
```

---

## Mitigating Cold Starts

### 1. Use Hybrid Configuration
Keep your most-used pipeline hot.

### 2. Pre-bake Models in Image
```dockerfile
# Dockerfile.worker
COPY models/ /models/
# Models included in image, faster startup
```

### 3. Use PVC for Model Cache
Models persist on disk, only load to GPU on start.

### 4. Warm-up Endpoint
Add endpoint that pre-loads models without processing:
```python
@app.post("/warmup")
async def warmup():
    model.to("cuda")  # Load to GPU
    return {"status": "warm"}
```

---

## Files

```
k8s/autoscaling/keda/
├── scaledobjects.yaml        # Full scale-to-zero
├── hybrid-scaledobjects.yaml # 1 hot, rest cold
└── README.md                 # This file
```

---

## Rollback to Standard HPA

```bash
# Remove KEDA ScaledObjects
kubectl delete scaledobject -n imagen --all

# Re-apply standard HPAs
kubectl apply -f k8s/autoscaling/upscale-hpa.yaml
kubectl apply -f k8s/autoscaling/enhance-hpa.yaml
kubectl apply -f k8s/autoscaling/comic-hpa.yaml
kubectl apply -f k8s/autoscaling/background-remove-hpa.yaml
```

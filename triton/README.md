# Triton Inference Server Architecture

This directory contains the configuration for NVIDIA Triton Inference Server, which provides centralized GPU inference for all image processing pipelines.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRITON ARCHITECTURE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────────────┐
                    │              Pub/Sub Queues                           │
                    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
                    │  │upscale  │ │enhance  │ │comic   │ │bg-remove│ ... │
                    │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘     │
                    └───────┼───────────┼───────────┼───────────┼──────────┘
                            │           │           │           │
                            ▼           ▼           ▼           ▼
                    ┌──────────────────────────────────────────────────────┐
                    │              Thin Workers (CPU-only)                  │
                    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
                    │  │upscale  │ │enhance  │ │comic   │ │bg-remove│ ... │
                    │  │worker   │ │worker   │ │worker  │ │worker   │     │
                    │  │(CPU)    │ │(CPU)    │ │(CPU)   │ │(CPU)    │     │
                    │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘     │
                    └───────┼───────────┼───────────┼───────────┼──────────┘
                            │           │           │           │
                            └───────────┴─────┬─────┴───────────┘
                                              │
                                         gRPC │ :8001
                                              ▼
                    ┌──────────────────────────────────────────────────────┐
                    │              Triton Inference Server                  │
                    │                                                       │
                    │  ┌─────────────────────────────────────────────────┐ │
                    │  │              Dynamic Batcher                     │ │
                    │  │         (collects requests, batches)             │ │
                    │  └─────────────────────┬───────────────────────────┘ │
                    │                        │                              │
                    │    ┌───────────┬───────┴───────┬───────────┐         │
                    │    ▼           ▼               ▼           ▼         │
                    │ ┌──────┐  ┌──────┐       ┌──────┐    ┌──────┐       │
                    │ │upscale│ │enhance│      │comic │    │rmbg  │       │
                    │ │model │  │model │       │model │    │model │       │
                    │ └──────┘  └──────┘       └──────┘    └──────┘       │
                    │                                                       │
                    │                    T4 GPU (16GB)                      │
                    └──────────────────────────────────────────────────────┘
```

---

## Cost Comparison

| Architecture | GPU Pods | GPU Cost | CPU Cost | Total |
|--------------|----------|----------|----------|-------|
| **Old (5 GPU workers)** | 5× T4 | ~$365/mo | - | ~$365/mo |
| **New (Triton)** | 1× T4 | ~$73/mo | ~$25/mo | ~$100/mo |
| **Savings** | | | | **~73%** |

*Costs based on GKE Autopilot Spot pricing*

---

## Benefits

| Aspect | Old Architecture | Triton Architecture |
|--------|------------------|---------------------|
| **GPU Pods** | 5 (one per model) | 1 (centralized) |
| **Batching** | None | Automatic (2-4 images) |
| **Throughput** | ~0.07 jobs/s/GPU | ~0.3 jobs/s/GPU |
| **Cold Start** | 2-5 min (model load) | <30s (workers are thin) |
| **Memory** | Separate per worker | Shared, optimized |
| **Scaling** | 5 HPAs, 5 node pools | 1 Triton HPA + CPU workers |

---

## Directory Structure

```
triton/
├── README.md                      # This file
├── Dockerfile                     # Custom Triton image
└── model_repository/              # Model configurations
    ├── upscale/
    │   ├── config.pbtxt           # Triton model config
    │   └── 1/
    │       └── model.py           # Python backend
    ├── enhance/
    │   ├── config.pbtxt
    │   └── 1/
    │       └── model.py
    ├── background_remove/
    │   ├── config.pbtxt
    │   └── 1/
    │       └── model.py
    ├── style_comic/
    │   ├── config.pbtxt
    │   └── 1/
    │       └── model.py
    └── style_aged/
        ├── config.pbtxt
        └── 1/
            └── model.py
```

---

## Deployment

### Build Custom Triton Image

```bash
# Build locally
docker build -t triton-imagen -f triton/Dockerfile .

# Or use Cloud Build (automatic on git push)
gcloud builds submit --config=cloudbuild.yaml
```

### Deploy to GKE

```bash
# Deploy everything (Triton + thin workers)
kubectl apply -k k8s/

# Or deploy Triton separately
kubectl apply -f k8s/triton/
```

### Verify Deployment

```bash
# Check Triton pod
kubectl get pods -n imagen -l app=triton

# Check Triton health
kubectl port-forward -n imagen svc/triton 8000:8000
curl localhost:8000/v2/health/ready

# Check model status
curl localhost:8000/v2/models/upscale
curl localhost:8000/v2/models/enhance
```

---

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 8000 | HTTP | REST API |
| 8001 | gRPC | High-performance inference (used by workers) |
| 8002 | HTTP | Prometheus metrics |

---

## Dynamic Batching Configuration

Each model has configurable batching parameters in `config.pbtxt`:

```protobuf
max_batch_size: 4

dynamic_batching {
  preferred_batch_size: [2, 4]
  max_queue_delay_microseconds: 100000  # 100ms
  preserve_ordering: true
}
```

- **max_batch_size**: Maximum images batched together
- **preferred_batch_size**: Optimal batch sizes for GPU efficiency
- **max_queue_delay_microseconds**: How long to wait for batch to fill

---

## Monitoring

Triton exposes Prometheus metrics at `:8002/metrics`:

```bash
# Port forward metrics
kubectl port-forward -n imagen svc/triton 8002:8002

# View metrics
curl localhost:8002/metrics | grep triton
```

Key metrics:
- `nv_inference_request_success` - Successful inferences
- `nv_inference_request_failure` - Failed inferences
- `nv_inference_queue_duration_us` - Queue wait time
- `nv_inference_compute_infer_duration_us` - Inference time

---

## Troubleshooting

### Model Not Loading

```bash
# Check Triton logs
kubectl logs -n imagen deployment/triton-inference-server

# Common issues:
# - Missing model files in /models/repository
# - Python dependencies not installed
# - Out of GPU memory
```

### Slow Inference

```bash
# Check if batching is working
curl localhost:8002/metrics | grep queue_duration

# If queue_duration is high, reduce max_queue_delay_microseconds
# If batch sizes are always 1, check traffic patterns
```

### Worker Connection Issues

```bash
# Verify Triton service
kubectl get svc triton -n imagen

# Test gRPC connectivity from worker
kubectl exec -n imagen deployment/upscale-worker -- \
  python -c "from src.services.triton import get_triton_client; print(get_triton_client().is_ready())"
```

---

*Last updated: December 2024*

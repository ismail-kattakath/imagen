# Imagen Platform Architecture

Comprehensive architecture documentation for the AI-powered image processing platform.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Triton Architecture](#triton-architecture)
3. [Request Flow](#request-flow)
4. [Project Structure](#project-structure)
5. [API Endpoints](#api-endpoints)
6. [GCP Services Used](#gcp-services-used)
7. [Kubernetes Resources](#kubernetes-resources)
8. [Auto-Scaling](#auto-scaling)
9. [Model Management](#model-management)
10. [Key Design Decisions](#key-design-decisions)
11. [Cost Estimates](#cost-estimates)

---

## High-Level Overview

```
                                    ┌─────────────────┐
                                    │   Clients       │
                                    │  (Web/Mobile)   │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Cloud Load     │
                                    │  Balancer       │
                                    └────────┬────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              │                             │
                     ┌────────▼────────┐          ┌────────▼────────┐
                     │   Cloud Run     │          │   Cloud Run     │
                     │   (FastAPI)     │          │   (Webhooks)    │
                     └────────┬────────┘          └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐    ┌───────▼───────┐    ┌───────▼───────┐
│  Cloud        │    │  Cloud        │    │  Firestore    │
│  Pub/Sub      │    │  Storage      │    │  (Job State)  │
│  (5 Topics)   │    │  (Images)     │    │               │
└───────┬───────┘    └───────────────┘    └───────────────┘
        │
        │ Pull Subscriptions
        │
┌───────▼─────────────────────────────────────────────────────────┐
│                        GKE Autopilot                            │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              THIN WORKERS (CPU-only, no GPU)               │ │
│  │                                                            │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │ │
│  │  │ Upscale  │ │ Enhance  │ │  Comic   │ │ BG Remove│ ...  │ │
│  │  │ Worker   │ │ Worker   │ │  Worker  │ │  Worker  │      │ │
│  │  │ (CPU)    │ │ (CPU)    │ │ (CPU)    │ │ (CPU)    │      │ │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘      │ │
│  │       │            │            │            │             │ │
│  │       └────────────┴─────┬──────┴────────────┘             │ │
│  │                          │ gRPC                            │ │
│  │                          ▼                                 │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │           TRITON INFERENCE SERVER (T4 GPU)           │  │ │
│  │  │                                                      │  │ │
│  │  │  ┌─────────────────────────────────────────────────┐ │  │ │
│  │  │  │              Dynamic Batcher                    │ │  │ │
│  │  │  │         (batches 2-4 requests)                  │ │  │ │
│  │  │  └──────────────────────┬──────────────────────────┘ │  │ │
│  │  │                         │                            │  │ │
│  │  │    ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │  │ │
│  │  │    │upscale│ │enhance│ │comic │ │ aged │ │rmbg  │     │  │ │
│  │  │    │model │ │model │ │model │ │model │ │model │     │  │ │
│  │  │    └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │  │ │
│  │  │                                                      │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    PVC (50Gi)                            │   │
│  │              Shared Model Storage                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Gateway | Cloud Run + FastAPI | Receives requests, queues jobs |
| Job Queue | Cloud Pub/Sub | Decouples API from workers |
| Job State | Firestore | Tracks job status |
| Image Storage | Cloud Storage (GCS) | Stores input/output images |
| Thin Workers | GKE (CPU-only pods) | Downloads images, calls Triton, uploads results |
| Triton Server | NVIDIA Triton + T4 GPU | Centralized ML inference with dynamic batching |
| Model Storage | Persistent Volume | Shared model cache for Triton |
| Auto-Scaling | HPA + Metrics Adapter | Scales workers and Triton based on load |

---

## Triton Architecture

### Why Triton?

The previous architecture had **5 separate GPU workers**, each requiring its own T4 GPU:

| Problem | Impact |
|---------|--------|
| 5× T4 GPUs minimum | ~$365/month (Spot) or ~$1,260/month (On-demand) |
| No batching | 1 image at a time per GPU |
| Model loading per worker | Slow cold starts, wasted memory |
| Independent scaling | Over-provisioned GPUs for bursty traffic |

### Triton Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRITON BENEFITS                                      │
└─────────────────────────────────────────────────────────────────────────────┘

OLD: 5 GPU Workers (1 GPU each)          NEW: Triton + Thin Workers
─────────────────────────────            ─────────────────────────────

┌─────────┐ ┌─────────┐ ┌─────────┐      ┌────────────────────────────┐
│Worker 1 │ │Worker 2 │ │Worker 3 │      │    Thin Workers (CPU)      │
│ T4 GPU  │ │ T4 GPU  │ │ T4 GPU  │      │  • No GPU required         │
│ upscale │ │ enhance │ │  comic  │      │  • Cheap to scale (20+)    │
└─────────┘ └─────────┘ └─────────┘      │  • Fast startup (<10s)     │
                                         └──────────────┬─────────────┘
┌─────────┐ ┌─────────┐                                │ gRPC
│Worker 4 │ │Worker 5 │                                ▼
│ T4 GPU  │ │ T4 GPU  │                  ┌────────────────────────────┐
│  aged   │ │  rmbg   │                  │   Triton Server (1-3 GPU)  │
└─────────┘ └─────────┘                  │  • Dynamic batching (2-4x) │
                                         │  • Multi-model on 1 GPU    │
Cost: 5 GPUs = $365+/mo                  │  • Built-in metrics        │
Throughput: ~0.07 jobs/s/GPU             └────────────────────────────┘
                                         
                                         Cost: 1-3 GPUs = $100-250/mo
                                         Throughput: ~0.3 jobs/s/GPU
```

### Thin Worker Pattern

Workers are now **CPU-only** and act as orchestrators:

```python
class TritonUpscaleWorker(TritonWorker):
    model_name = "upscale"
    subscription_name = "upscale-jobs-sub"
    
    def process_with_triton(self, image, params: dict):
        # No local model - just call Triton via gRPC
        return self.triton.upscale(image, scale=params.get("scale", 4.0))
```

**Worker responsibilities:**
1. Pull message from Pub/Sub
2. Download image from GCS
3. Send gRPC request to Triton
4. Upload result to GCS
5. Update job status in Firestore

**Triton responsibilities:**
1. Load all 5 models into GPU memory
2. Queue incoming requests
3. Batch requests (2-4 images)
4. Run inference
5. Return results

---

## Request Flow

### Async Processing Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        REQUEST LIFECYCLE                          │
└──────────────────────────────────────────────────────────────────┘

1. CLIENT UPLOADS IMAGE
   │
   │  POST /api/v1/images/upscale
   │  Body: image file + parameters
   ▼
2. FASTAPI RECEIVES REQUEST (Cloud Run)
   │
   ├──▶ Validates image format
   ├──▶ Generates job_id (UUID)
   ├──▶ Uploads image to GCS: inputs/{job_id}/filename.jpg
   ├──▶ Creates Firestore record: {status: "queued"}
   └──▶ Publishes message to Pub/Sub: upscale-jobs topic
   │
   │  Returns immediately: {"job_id": "abc-123", "status": "queued"}
   ▼
3. THIN WORKER PULLS MESSAGE (GKE - CPU only)
   │
   ├──▶ Receives message from upscale-jobs-sub
   ├──▶ Updates Firestore: {status: "processing"}
   ├──▶ Downloads image from GCS
   ├──▶ Sends gRPC request to Triton ──────────────────┐
   │                                                    │
   │    ┌───────────────────────────────────────────────▼───────┐
   │    │ TRITON INFERENCE SERVER                               │
   │    │                                                       │
   │    │  1. Request queued in dynamic batcher                 │
   │    │  2. Batched with other requests (2-4 images)          │
   │    │  3. GPU inference runs on batch                       │
   │    │  4. Results returned via gRPC                         │
   │    └───────────────────────────────────────────────────────┘
   │                                                    │
   ├──▶ Receives result from Triton ◄──────────────────┘
   ├──▶ Uploads result to GCS: outputs/{job_id}/result.png
   ├──▶ Updates Firestore: {status: "completed", output_path: ...}
   └──▶ ACKs message (removes from queue)
   │
   ▼
4. CLIENT POLLS STATUS
   │
   │  GET /api/v1/jobs/{job_id}
   │
   ├──▶ If status == "completed": returns signed URL
   └──▶ Client downloads result from GCS
```

### Timeline Comparison

| Stage | Old (GPU Worker) | New (Triton) |
|-------|------------------|--------------|
| Worker startup | 60-120s (model load) | 5-10s (no model) |
| Image download | 1s | 1s |
| Inference | 5-30s | 3-20s (batched) |
| Image upload | 1s | 1s |
| **Total** | **67-152s** | **10-32s** |

---

## Project Structure

```
imagen/
├── src/
│   ├── api/                        # FastAPI Application
│   │   ├── main.py                 # App entrypoint, middleware
│   │   ├── routes/                 # API endpoints
│   │   ├── middleware/             # Auth, rate limiting, metrics
│   │   └── schemas/                # Request/response models
│   │
│   ├── pipelines/                  # ML Pipelines (legacy, for local dev)
│   │   ├── base.py
│   │   ├── upscale.py
│   │   └── ...
│   │
│   ├── workers/                    # Pub/Sub Consumers
│   │   ├── base.py                 # Legacy GPU worker base
│   │   ├── triton_worker.py        # ⭐ Thin worker base + all workers
│   │   └── ...
│   │
│   ├── services/
│   │   ├── storage.py              # GCS operations
│   │   ├── queue.py                # Pub/Sub operations
│   │   ├── jobs.py                 # Firestore operations
│   │   └── triton/                 # ⭐ Triton client
│   │       ├── __init__.py
│   │       └── client.py           # gRPC client for Triton
│   │
│   └── core/                       # Configuration & Utilities
│
├── triton/                         # ⭐ Triton Inference Server
│   ├── Dockerfile                  # Custom Triton image
│   ├── README.md
│   └── model_repository/           # Model configs + Python backends
│       ├── upscale/
│       │   ├── config.pbtxt
│       │   └── 1/model.py
│       ├── enhance/
│       ├── background_remove/
│       ├── style_comic/
│       └── style_aged/
│
├── k8s/
│   ├── base/                       # Namespace, ConfigMap, PVC
│   ├── triton/                     # ⭐ Triton deployment + service + HPA
│   ├── workers/       # Workers (CPU-only, calls Triton)
│   ├── autoscaling/                # HPAs for workers
│   ├── monitoring/                 # PodMonitoring for GMP
│   └── overlays/
│       ├── dev/
│       └── prod/
│
├── terraform/                      # Infrastructure as Code
├── docker/                         # Dockerfiles
└── docs/                           # Documentation
```

---


## API Endpoints

### Image Processing

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `POST` | `/api/v1/images/upscale` | 4x image upscaling | `prompt`, `guidance_scale` |
| `POST` | `/api/v1/images/enhance` | Quality enhancement | `strength`, `prompt` |
| `POST` | `/api/v1/images/style/comic` | Comic/cartoon style | `strength`, `guidance_scale` |
| `POST` | `/api/v1/images/style/aged` | Vintage/aged look | `strength`, `prompt` |
| `POST` | `/api/v1/images/background/remove` | Remove background | - |

### Job Management

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `GET` | `/api/v1/jobs/{job_id}` | Get job status | Status, output URL |

---

## GCP Services Used

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **Cloud Run** | API hosting | Auto-scales 0→10, 512MB memory |
| **GKE Autopilot** | Triton + Workers | T4 GPUs for Triton, CPU for workers |
| **Cloud Pub/Sub** | Job queue | 5 topics, 10 min ACK deadline |
| **Cloud Storage** | Image storage | Regional, 7-day lifecycle |
| **Firestore** | Job state | Native mode, auto-scaling |
| **Artifact Registry** | Docker images | API, Worker, Triton images |
| **Cloud Build** | CI/CD | Builds & deploys on push |

---

## Kubernetes Resources

### Resource Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    Namespace: imagen                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ ConfigMap: imagen-config                                   │ │
│  │ - GOOGLE_CLOUD_PROJECT, GCS_BUCKET, TRITON_URL             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Triton Inference Server (GPU)                              │ │
│  │                                                             │ │
│  │  Deployment: triton-inference-server                        │ │
│  │  - replicas: 1-3 (HPA)                                     │ │
│  │  - GPU: nvidia-tesla-t4                                     │ │
│  │  - Memory: 24Gi                                             │ │
│  │  - Ports: 8000 (HTTP), 8001 (gRPC), 8002 (metrics)         │ │
│  │                                                             │ │
│  │  Service: triton (ClusterIP)                                │ │
│  │  HPA: triton-hpa (scales on queue latency)                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Thin Workers (CPU-only)                                     │ │
│  │                                                             │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │ │
│  │  │upscale-worker│  │enhance-worker│  │comic-worker  │ ...  │ │
│  │  │              │  │              │  │              │      │ │
│  │  │replicas: 0-20│  │replicas: 0-20│  │replicas: 0-20│      │ │
│  │  │CPU: 250m-1   │  │CPU: 250m-1   │  │CPU: 250m-1   │      │ │
│  │  │Mem: 512Mi-2Gi│  │Mem: 512Mi-2Gi│  │Mem: 512Mi-2Gi│      │ │
│  │  │NO GPU        │  │NO GPU        │  │NO GPU        │      │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │ │
│  │                                                             │ │
│  │  HPAs scale based on Pub/Sub queue depth                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ PVC: models-pvc (50Gi) - Mounted by Triton only            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Auto-Scaling

### Two-Tier Scaling

```
┌─────────────────────────────────────────────────────────────────┐
│                      AUTO-SCALING STRATEGY                       │
└─────────────────────────────────────────────────────────────────┘

TIER 1: Thin Workers (fast, cheap)
──────────────────────────────────
Metric: Pub/Sub queue depth (undelivered messages)
Target: 2 messages per worker
Range: 0-20 replicas per worker type

Queue fills up → Workers scale up → More gRPC calls to Triton


TIER 2: Triton Server (slow, expensive)
───────────────────────────────────────
Metric: Triton inference queue latency
Target: 100ms average queue delay
Range: 1-3 replicas

Triton queue backs up → Triton scales up → More GPU capacity
```

### Scaling Timeline Example

```
Time    Queue    Workers (5 types)    Triton    Action
──────────────────────────────────────────────────────────────
0:00    0        5 (1 each)           1         Idle
0:05    100      5→25                 1         Traffic spike
0:10    100      25→50                1→2       Workers saturating Triton
0:15    50       50                   2         Draining
0:20    0        50                   2         Queue empty
0:25    0        50→25                2         Scale down workers
0:35    0        25→5                 2→1       Back to baseline
```

---

## Model Management

### Model Registry

| Pipeline | Model | Size | Triton Backend |
|----------|-------|------|----------------|
| upscale | Real-ESRGAN | 2.5GB | Python |
| enhance | SDXL Refiner | 6GB | Python |
| comic | Ghibli-Diffusion | 4GB | Python |
| aged | SD 2.1 | 5GB | Python |
| background_remove | RMBG-1.4 | 1GB | Python |
| **Total** | | **~18.5GB** | |

### Triton Model Configuration

Each model has a `config.pbtxt`:

```protobuf
name: "upscale"
backend: "python"
max_batch_size: 4

dynamic_batching {
  preferred_batch_size: [2, 4]
  max_queue_delay_microseconds: 100000  # 100ms
}

instance_group [{
  count: 1
  kind: KIND_GPU
  gpus: [0]
}]
```

---

## Key Design Decisions

### 1. Triton for Centralized Inference

| Decision | Triton Inference Server |
|----------|------------------------|
| **Why** | Consolidate 5 GPUs → 1-3 GPUs with batching |
| **Alternative** | Keep separate GPU workers |
| **Benefit** | 60-80% cost reduction, 4x throughput per GPU |

### 2. Thin Workers (CPU-only)

| Decision | CPU-only job orchestrators |
|----------|---------------------------|
| **Why** | GPUs are expensive; workers just move data |
| **Alternative** | GPU workers with local models |
| **Benefit** | Scale to 20+ workers cheaply, fast startup |

### 3. gRPC for Triton Communication

| Decision | gRPC over HTTP |
|----------|----------------|
| **Why** | Lower latency, efficient binary protocol |
| **Alternative** | HTTP REST API |
| **Benefit** | ~30% faster than HTTP for large payloads |

### 4. Dynamic Batching

| Decision | Triton dynamic batcher |
|----------|----------------------|
| **Why** | Batch 2-4 images for GPU efficiency |
| **Alternative** | Process 1 image at a time |
| **Benefit** | 2-4x throughput improvement |

---

## Cost Estimates

### Old Architecture (5 GPU Workers)

| Component | Specification | Cost/Month (Spot) |
|-----------|---------------|-------------------|
| 5× upscale/enhance/etc workers | 5× T4 GPU | ~$365 |
| Cloud Run API | 1 vCPU | ~$50 |
| Storage, Pub/Sub, Firestore | - | ~$20 |
| **Total** | **5 T4 GPUs** | **~$435** |

### New Architecture (Triton + Thin Workers)

| Component | Specification | Cost/Month (Spot) |
|-----------|---------------|-------------------|
| Triton (1-3 replicas) | 1-3× T4 GPU | ~$73-219 |
| 5× thin workers | CPU only | ~$25 |
| Cloud Run API | 1 vCPU | ~$50 |
| Storage, Pub/Sub, Firestore | - | ~$20 |
| **Total** | **1-3 T4 GPUs** | **~$168-314** |

### Savings

| Scenario | Old Cost | New Cost | Savings |
|----------|----------|----------|---------|
| Low traffic | $435/mo | $168/mo | **61%** |
| High traffic | $2,600/mo | $600/mo | **77%** |

---

## Quick Reference

### Deployment Commands

```bash
# Deploy with Kustomize
kubectl apply -k k8s/overlays/dev   # Development
kubectl apply -k k8s/overlays/prod  # Production

# Check Triton status
kubectl get pods -n imagen -l app=triton
kubectl logs -f deployment/triton-inference-server -n imagen

# Check thin workers
kubectl get pods -n imagen -l app.kubernetes.io/component=worker

# Check HPAs
kubectl get hpa -n imagen

# Port-forward Triton for testing
kubectl port-forward -n imagen svc/triton 8000:8000 8001:8001 8002:8002
```

### Triton Health Checks

```bash
# Server ready
curl http://localhost:8000/v2/health/ready

# Model status
curl http://localhost:8000/v2/models/upscale

# Metrics
curl http://localhost:8002/metrics
```

---

*Architecture updated: December 2024 - Migrated to Triton Inference Server*

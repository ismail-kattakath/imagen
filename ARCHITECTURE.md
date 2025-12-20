# Imagen Platform Architecture

Comprehensive architecture documentation for the AI-powered image processing platform.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Request Flow](#request-flow)
3. [Project Structure](#project-structure)
4. [API Endpoints](#api-endpoints)
5. [GCP Services Used](#gcp-services-used)
6. [Kubernetes Resources](#kubernetes-resources)
7. [Auto-Scaling](#auto-scaling)
8. [Model Management](#model-management)
9. [Key Design Decisions](#key-design-decisions)
10. [Cost Estimates](#cost-estimates)

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
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ Upscale  │ │ Enhance  │ │  Comic   │ │ BG Remove│ │  Aged  │ │
│  │ Worker   │ │ Worker   │ │  Worker  │ │  Worker  │ │ Worker │ │
│  │ (T4 GPU) │ │ (T4 GPU) │ │ (T4 GPU) │ │ (T4 GPU) │ │(T4 GPU)│ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
│       │            │            │            │           │      │
│       └────────────┴────────────┼────────────┴───────────┘      │
│                                 │                               │
│  ┌──────────────────────────────▼───────────────────────────┐   │
│  │                    PVC (50Gi)                            │   │
│  │              Shared Model Storage                        │   │
│  │    ┌─────────────────────────────────────────────────┐   │   │
│  │    │ /models/huggingface/hub/                        │   │   │
│  │    │   ├── stabilityai--sd-x4-upscaler    (~2.5GB)   │   │   │
│  │    │   ├── stabilityai--sdxl-refiner      (~6GB)     │   │   │
│  │    │   ├── nitrosocke--Ghibli-Diffusion   (~4GB)     │   │   │
│  │    │   ├── stabilityai--sd-2-1            (~5GB)     │   │   │
│  │    │   └── briaai--RMBG-1.4               (~1GB)     │   │   │
│  │    └─────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Auto-Scaling (HPA)                    │   │
│  │                                                          │   │
│  │   Metrics ──▶ HPA ──▶ Deployment ──▶ GKE provisions      │   │
│  │   Adapter       │      replicas       GPU nodes          │   │
│  │                 │                                        │   │
│  │   Queue depth > 2 msgs/worker? Scale up (max 10)         │   │
│  │   Queue empty for 5 min? Scale down (min 1)              │   │
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
| GPU Workers | GKE + T4 GPUs | Runs ML pipelines |
| Model Storage | Persistent Volume | Shared model cache |
| Auto-Scaling | HPA + Metrics Adapter | Scales workers based on queue |

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
3. CLIENT RECEIVES RESPONSE (~1-2 seconds)
   │
   │  Client now polls for status
   ▼
4. GPU WORKER PULLS MESSAGE (GKE)
   │
   ├──▶ Receives message from upscale-jobs-sub
   ├──▶ Updates Firestore: {status: "processing"}
   ├──▶ Downloads image from GCS
   ├──▶ Loads ML model (if not cached)
   ├──▶ Runs inference (5-30 seconds)
   ├──▶ Uploads result to GCS: outputs/{job_id}/result.png
   ├──▶ Updates Firestore: {status: "completed", output_path: ...}
   └──▶ ACKs message (removes from queue)
   │
   ▼
5. CLIENT POLLS STATUS
   │
   │  GET /api/v1/jobs/{job_id}
   │
   ├──▶ If status == "queued": keep polling
   ├──▶ If status == "processing": keep polling
   ├──▶ If status == "completed": returns signed URL
   └──▶ If status == "failed": returns error message
   │
   ▼
6. CLIENT DOWNLOADS RESULT
   │
   │  GET {signed_url}  (direct GCS download)
   │
   ▼
   Done!
```

### Timeline Example

```
Time    Event                                   Status
─────────────────────────────────────────────────────────────
0s      Client uploads image                    -
1s      API responds with job_id                queued
1s      Message published to Pub/Sub            queued
2s      Worker picks up message                 processing
2s      Worker downloads image                  processing
3s      Worker runs ML pipeline                 processing
25s     Worker uploads result                   processing
26s     Worker updates Firestore                completed
27s     Client polls, gets signed URL           completed
28s     Client downloads result                 -
```

---

## Project Structure

```
imagen/
├── src/
│   ├── api/                        # FastAPI Application
│   │   ├── main.py                 # App entrypoint, middleware
│   │   ├── routes/
│   │   │   ├── health.py           # GET /health, /ready
│   │   │   ├── images.py           # POST /images/* endpoints
│   │   │   └── jobs.py             # GET /jobs/{id}
│   │   └── schemas/
│   │       ├── images.py           # Request params (UpscaleParams, etc.)
│   │       └── jobs.py             # Response models (JobResponse, etc.)
│   │
│   ├── pipelines/                  # ML Pipelines (Diffusers)
│   │   ├── base.py                 # BasePipeline abstract class
│   │   ├── upscale.py              # 4x upscaling (SD Upscaler)
│   │   ├── enhance.py              # Quality enhancement (SDXL Refiner)
│   │   ├── style_comic.py          # Comic style (Ghibli Diffusion)
│   │   ├── style_aged.py           # Vintage look (SD 2.1)
│   │   └── background_remove.py    # BG removal (RMBG-1.4)
│   │
│   ├── workers/                    # Pub/Sub Consumers
│   │   ├── base.py                 # BaseWorker with common logic
│   │   ├── upscale.py              # Upscale worker
│   │   ├── enhance.py              # Enhance worker
│   │   ├── style_comic.py          # Comic style worker
│   │   └── background_remove.py    # Background removal worker
│   │
│   ├── services/                   # GCP Service Integrations
│   │   ├── storage.py              # GCS: upload, download, signed URLs
│   │   ├── queue.py                # Pub/Sub: publish, subscribe
│   │   └── jobs.py                 # Firestore: job CRUD operations
│   │
│   ├── core/                       # Configuration & Utilities
│   │   ├── config.py               # Settings from environment
│   │   ├── logging.py              # Structured logging
│   │   └── exceptions.py           # Custom exceptions
│   │
│   └── utils/
│       └── image.py                # Image helpers (resize, convert)
│
├── docker/
│   ├── Dockerfile.api              # API container (slim Python)
│   ├── Dockerfile.worker           # Worker container (PyTorch + CUDA)
│   └── docker-compose.yml          # Local development setup
│
├── k8s/
│   ├── base/                       # Foundation resources
│   │   ├── namespace.yaml          # 'imagen' namespace
│   │   ├── configmap.yaml          # Environment variables
│   │   ├── pvc.yaml                # 50Gi model storage
│   │   └── workload-identity.yaml  # GCP IAM binding
│   │
│   ├── workers/                    # Worker deployments
│   │   ├── upscale-worker.yaml
│   │   ├── enhance-worker.yaml
│   │   ├── comic-worker.yaml
│   │   └── background-remove-worker.yaml
│   │
│   └── autoscaling/                # Auto-scaling configuration
│       ├── custom-metrics-adapter.yaml  # Reads Pub/Sub metrics
│       ├── upscale-hpa.yaml        # HPA for upscale worker
│       ├── enhance-hpa.yaml        # HPA for enhance worker
│       ├── comic-hpa.yaml          # HPA for comic worker
│       ├── background-remove-hpa.yaml
│       └── README.md               # Auto-scaling documentation
│
├── terraform/                      # Infrastructure as Code
│   ├── main.tf                     # All GCP resources
│   ├── variables.tf                # Input variables
│   ├── outputs.tf                  # Output values
│   ├── modules/
│   │   └── autoscaling/            # Metrics adapter IAM
│   └── environments/
│       ├── dev.tfvars              # Development config
│       └── prod.tfvars             # Production config
│
├── tests/
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
│
├── models/                         # Model cache (git-ignored)
│
└── Documentation
    ├── README.md                   # Project overview
    ├── ARCHITECTURE.md             # This file
    ├── INFRASTRUCTURE_GUIDE.md     # Terraform & K8s explained
    ├── MODEL_MANAGEMENT.md         # Model loading & caching
    ├── DEPLOYMENT_GUIDE.md         # Production deployment
    ├── QUICK_REFERENCE.md          # Command cheat sheet
    └── CHANGELOG.md                # Project history
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

### Health Checks

| Method | Endpoint | Description | Used By |
|--------|----------|-------------|---------|
| `GET` | `/health` | Liveness check | Load balancer |
| `GET` | `/ready` | Readiness check | Kubernetes |

### Example Request/Response

```bash
# Submit upscale job
curl -X POST "https://api.example.com/api/v1/images/upscale" \
  -F "file=@photo.jpg" \
  -F "prompt=high quality, detailed"

# Response
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Job queued successfully"
}

# Check status
curl "https://api.example.com/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000"

# Response (completed)
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "upscale",
  "status": "completed",
  "output_url": "https://storage.googleapis.com/...",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:25Z"
}
```

---

## GCP Services Used

### Service Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        GCP Project                               │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Compute                               │    │
│  │                                                          │    │
│  │   ┌─────────────┐          ┌─────────────────────────┐  │    │
│  │   │ Cloud Run   │          │     GKE Autopilot       │  │    │
│  │   │ (API)       │          │     (GPU Workers)       │  │    │
│  │   │             │          │                         │  │    │
│  │   │ - Serverless│          │ - T4 GPUs              │  │    │
│  │   │ - Auto-scale│          │ - Auto-provisions nodes│  │    │
│  │   │ - $0 at idle│          │ - Spot instances       │  │    │
│  │   └─────────────┘          └─────────────────────────┘  │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Messaging                             │    │
│  │                                                          │    │
│  │   ┌─────────────────────────────────────────────────┐   │    │
│  │   │                 Cloud Pub/Sub                    │   │    │
│  │   │                                                  │   │    │
│  │   │  Topics:              Subscriptions:             │   │    │
│  │   │  - upscale-jobs       - upscale-jobs-sub        │   │    │
│  │   │  - enhance-jobs       - enhance-jobs-sub        │   │    │
│  │   │  - style-comic-jobs   - style-comic-jobs-sub    │   │    │
│  │   │  - style-aged-jobs    - style-aged-jobs-sub     │   │    │
│  │   │  - background-remove  - background-remove-sub   │   │    │
│  │   │                                                  │   │    │
│  │   └─────────────────────────────────────────────────┘   │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Storage                               │    │
│  │                                                          │    │
│  │   ┌─────────────────┐    ┌─────────────────────────┐    │    │
│  │   │ Cloud Storage   │    │     Firestore           │    │    │
│  │   │ (GCS)           │    │                         │    │    │
│  │   │                 │    │  Collection: jobs       │    │    │
│  │   │ /inputs/{id}/   │    │  - job_id               │    │    │
│  │   │ /outputs/{id}/  │    │  - status               │    │    │
│  │   │                 │    │  - input_path           │    │    │
│  │   │ Auto-delete: 7d │    │  - output_path          │    │    │
│  │   └─────────────────┘    │  - created_at           │    │    │
│  │                          └─────────────────────────┘    │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Security & DevOps                     │    │
│  │                                                          │    │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │   │     IAM     │  │  Artifact   │  │   Cloud     │     │    │
│  │   │             │  │  Registry   │  │   Build     │     │    │
│  │   │ Service     │  │             │  │             │     │    │
│  │   │ Accounts:   │  │ Docker      │  │ CI/CD       │     │    │
│  │   │ - worker    │  │ images      │  │ pipeline    │     │    │
│  │   │ - metrics   │  │             │  │             │     │    │
│  │   └─────────────┘  └─────────────┘  └─────────────┘     │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Service Details

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **Cloud Run** | API hosting | Auto-scales 0→10, 512MB memory |
| **GKE Autopilot** | GPU workers | T4 GPUs, auto-provisions nodes |
| **Cloud Pub/Sub** | Job queue | 5 topics, 10 min ACK deadline |
| **Cloud Storage** | Image storage | Regional, 7-day lifecycle |
| **Firestore** | Job state | Native mode, auto-scaling |
| **Artifact Registry** | Docker images | Stores API & worker images |
| **IAM** | Security | Service accounts with minimal perms |
| **Cloud Build** | CI/CD | Builds & deploys on push |
| **Cloud Monitoring** | Metrics | HPA reads queue depth |

---

## Kubernetes Resources

### Resource Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    Namespace: imagen                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ ConfigMap: imagen-config                                   │ │
│  │ - GOOGLE_CLOUD_PROJECT                                     │ │
│  │ - GCS_BUCKET                                               │ │
│  │ - DEVICE                                                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ PVC: models-pvc (50Gi)                                     │ │
│  │ - Shared model storage                                     │ │
│  │ - ReadWriteOnce (or ReadWriteMany with Filestore)         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Deployments                                                 │ │
│  │                                                             │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │ │
│  │  │upscale-worker│  │enhance-worker│  │comic-worker  │      │ │
│  │  │              │  │              │  │              │      │ │
│  │  │replicas: 1-10│  │replicas: 1-10│  │replicas: 1-10│      │ │
│  │  │GPU: T4       │  │GPU: T4       │  │GPU: T4       │      │ │
│  │  │Mem: 16Gi     │  │Mem: 16Gi     │  │Mem: 16Gi     │      │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │ │
│  │                                                             │ │
│  │  ┌──────────────┐                                          │ │
│  │  │bg-remove-    │                                          │ │
│  │  │worker        │                                          │ │
│  │  │              │                                          │ │
│  │  │replicas: 1-10│                                          │ │
│  │  └──────────────┘                                          │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ HorizontalPodAutoscalers                                   │ │
│  │                                                             │ │
│  │  ┌─────────────────┐  ┌─────────────────┐                  │ │
│  │  │upscale-hpa      │  │enhance-hpa      │  ...             │ │
│  │  │                 │  │                 │                  │ │
│  │  │min: 1, max: 10  │  │min: 1, max: 10  │                  │ │
│  │  │metric: queue    │  │metric: queue    │                  │ │
│  │  │target: 2/worker │  │target: 2/worker │                  │ │
│  │  └─────────────────┘  └─────────────────┘                  │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 Namespace: custom-metrics                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Deployment: custom-metrics-stackdriver-adapter             │ │
│  │                                                             │ │
│  │ Reads Pub/Sub metrics from Cloud Monitoring                │ │
│  │ Exposes them to HPA via External Metrics API               │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Auto-Scaling

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                      AUTO-SCALING CHAIN                          │
└─────────────────────────────────────────────────────────────────┘

Step 1: Queue fills up
        │
        │  100 messages in upscale-jobs-sub
        ▼
Step 2: Metrics Adapter reads Cloud Monitoring
        │
        │  pubsub.googleapis.com/subscription/num_undelivered_messages
        ▼
Step 3: HPA calculates desired replicas
        │
        │  100 messages / 2 per worker = 50 desired
        │  max is 10, so target = 10
        ▼
Step 4: HPA updates Deployment
        │
        │  replicas: 1 → 10
        ▼
Step 5: GKE Autopilot provisions nodes
        │
        │  "10 pods need T4 GPUs, creating 10 nodes"
        ▼
Step 6: Workers start processing
        │
        │  Queue drains 10x faster
        ▼
Step 7: Queue empties, scale down
        │
        │  After 5 min idle: replicas: 10 → 1
        │
        Done!
```

### Scaling Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `minReplicas` | 1 | Always keep 1 worker running |
| `maxReplicas` | 10 | Cost protection limit |
| `averageValue` | 2 | Scale when >2 msgs per worker |
| Scale up delay | 60s | Prevent flapping |
| Scale down delay | 300s | Ensure traffic is truly gone |

### Example Timeline

```
Time    Queue Depth    Workers    Action
──────────────────────────────────────────────────
0:00    0              1          Idle
0:05    50             1          Traffic spike detected
0:06    50             1→5        Scaling up
0:08    50             5→10       Still scaling
0:10    50             10         Max reached
0:15    25             10         Draining queue
0:20    0              10         Queue empty
0:25    0              10         Waiting (5 min window)
0:30    0              10→9       Scaling down
0:35    0              9→5        Gradual reduction
0:45    0              5→1        Back to minimum
```

---

## Model Management

### Model Registry

| Pipeline | Model | Size | Source |
|----------|-------|------|--------|
| Upscale | `stabilityai/stable-diffusion-x4-upscaler` | 2.5GB | HuggingFace |
| Enhance | `stabilityai/stable-diffusion-xl-refiner-1.0` | 6GB | HuggingFace |
| Comic | `nitrosocke/Ghibli-Diffusion` | 4GB | HuggingFace |
| Aged | `stabilityai/stable-diffusion-2-1` | 5GB | HuggingFace |
| BG Remove | `briaai/RMBG-1.4` | 1GB | HuggingFace |
| **Total** | | **~18.5GB** | |

### Storage Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL STORAGE STRATEGY                        │
└─────────────────────────────────────────────────────────────────┘

LOCAL DEVELOPMENT:
┌─────────────────────────────────────────────────────────────────┐
│  ~/.cache/huggingface/                                          │
│  └── hub/                                                       │
│      └── models--stabilityai--...                              │
│                                                                  │
│  Models download automatically on first use                     │
│  Cached locally for subsequent runs                             │
└─────────────────────────────────────────────────────────────────┘

PRODUCTION (GKE):
┌─────────────────────────────────────────────────────────────────┐
│  PersistentVolumeClaim (50Gi)                                   │
│  └── /models/huggingface/hub/                                  │
│                                                                  │
│  - First worker downloads models                                │
│  - Stored on persistent disk                                    │
│  - Shared across all workers                                    │
│  - Survives pod restarts                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Async Processing

| Decision | Async with Pub/Sub |
|----------|-------------------|
| **Why** | GPU inference takes 5-30 seconds |
| **Alternative** | Sync processing (would timeout) |
| **Benefit** | API responds instantly, workers scale independently |

### 2. Pub/Sub over Redis

| Decision | Cloud Pub/Sub |
|----------|---------------|
| **Why** | Managed, auto-scales, integrates with HPA |
| **Alternative** | Redis + Celery |
| **Benefit** | No infrastructure to manage, native GCP |

### 3. GKE Autopilot

| Decision | Autopilot (not Standard GKE) |
|----------|----------------------------|
| **Why** | Auto-provisions GPU nodes on demand |
| **Alternative** | Standard GKE with node pools |
| **Benefit** | Pay only when workers run, no node management |

### 4. Cloud Run for API

| Decision | Cloud Run (not GKE) |
|----------|---------------------|
| **Why** | Scales to zero, cheap for API workload |
| **Alternative** | API in GKE |
| **Benefit** | $0 when idle, auto-scales instantly |

### 5. Firestore for Job State

| Decision | Firestore |
|----------|-----------|
| **Why** | Serverless, fast reads, auto-scales |
| **Alternative** | Cloud SQL (Postgres) |
| **Benefit** | No connection pooling, no DB management |

### 6. PVC for Models

| Decision | Shared Persistent Volume |
|----------|-------------------------|
| **Why** | Models are large (18GB), expensive to download |
| **Alternative** | Bake into Docker image |
| **Benefit** | Download once, share across workers |

### 7. Separate Workers per Pipeline

| Decision | One worker type per pipeline |
|----------|----------------------------|
| **Why** | Each model needs different GPU memory |
| **Alternative** | Single worker with all models |
| **Benefit** | Independent scaling, isolated failures |

### 8. T4 Spot Instances

| Decision | T4 GPUs with Spot VMs |
|----------|----------------------|
| **Why** | Best price/performance for inference |
| **Alternative** | A100 or V100 |
| **Benefit** | ~70% cost savings vs on-demand |

---

## Cost Estimates

### Per-Component Costs

| Component | Specification | Cost/Month |
|-----------|---------------|------------|
| **Cloud Run (API)** | 1 vCPU, 512MB | ~$50 |
| **GKE Workers (1 T4 Spot)** | 1 GPU, 16GB | ~$250 |
| **GKE Workers (10 T4 Spot)** | 10 GPUs | ~$2,500 |
| **Cloud Storage (100GB)** | Regional | ~$2 |
| **Pub/Sub** | 1M messages | ~$10 |
| **Firestore** | 100K reads/day | ~$5 |
| **Artifact Registry** | 20GB | ~$5 |

### Scenario Estimates

| Scenario | Workers | Monthly Cost |
|----------|---------|--------------|
| **Development** | 0 (local) | ~$0 |
| **Low Traffic** | 1 always | ~$320 |
| **Medium Traffic** | 1-5 auto-scale | ~$500-1,000 |
| **High Traffic** | 1-10 auto-scale | ~$1,000-2,600 |

### Cost Optimization Tips

1. **Use Spot VMs** — 70% savings on GPU nodes
2. **Scale to zero** — Set `minReplicas: 0` if cold start is acceptable
3. **Right-size max replicas** — Don't over-provision
4. **Set billing alerts** — Avoid surprise bills
5. **Use lifecycle rules** — Auto-delete old images from GCS

---

## Quick Reference

### Deployment Commands

```bash
# Infrastructure
cd terraform && terraform apply -var-file=environments/dev.tfvars

# Connect to GKE
gcloud container clusters get-credentials imagen-cluster --region us-central1

# Deploy all
make k8s-deploy-all

# Check status
kubectl get pods -n imagen
kubectl get hpa -n imagen
```

### Useful Commands

```bash
# Watch scaling
make k8s-watch

# View logs
kubectl logs -f deployment/upscale-worker -n imagen

# Check queue depth
gcloud pubsub subscriptions describe upscale-jobs-sub

# Force scale
kubectl scale deployment/upscale-worker --replicas=5 -n imagen
```

---

*Last updated: December 2024*

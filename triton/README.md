# Triton Inference Server Setup

This directory contains configuration for NVIDIA Triton Inference Server integration.

---

## Overview

Triton provides:
- **Automatic dynamic batching** — Collects requests into optimal batches
- **Multi-model serving** — All 5 models on one GPU with smart scheduling
- **Model versioning** — A/B testing, canary deployments
- **Built-in metrics** — Prometheus-compatible out of the box
- **Multiple backends** — Python, ONNX, TensorRT, PyTorch

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRITON ARCHITECTURE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                                   ┌──────────────────────────────────────────┐
                                   │         Triton Inference Server          │
                                   │                                          │
 Pub/Sub ──► Worker ──► gRPC ────► │  ┌─────────────────────────────────┐    │
                                   │  │      Dynamic Batcher            │    │
                                   │  │  (collects requests, batches)   │    │
                                   │  └───────────────┬─────────────────┘    │
                                   │                  │                       │
                                   │    ┌─────────────┴─────────────┐        │
                                   │    ▼             ▼             ▼        │
                                   │ ┌──────┐    ┌──────┐    ┌──────┐       │
                                   │ │upscale│   │enhance│   │rmbg  │  ...  │
                                   │ │model │    │model │    │model │       │
                                   │ └──────┘    └──────┘    └──────┘       │
                                   │                                          │
                                   │              T4 GPU (16GB)               │
                                   └──────────────────────────────────────────┘
                                                     ▲
                                                     │
                                            Model Repository
                                            (GCS or local PVC)
```

---

## Directory Structure

```
triton/
├── README.md                      # This file
├── model_repository/              # Model configurations
│   ├── upscale/
│   │   ├── config.pbtxt           # Model config
│   │   └── 1/
│   │       └── model.py           # Python backend
│   ├── enhance/
│   │   ├── config.pbtxt
│   │   └── 1/
│   │       └── model.py
│   ├── background_remove/
│   │   ├── config.pbtxt
│   │   └── 1/
│   │       └── model.py
│   ├── style_comic/
│   │   ├── config.pbtxt
│   │   └── 1/
│   │       └── model.py
│   └── style_aged/
│       ├── config.pbtxt
│       └── 1/
│           └── model.py
├── Dockerfile                     # Custom Triton image
└── scripts/
    ├── export_models.py           # Export models to Triton format
    └── test_client.py             # Test Triton endpoints
```

---

## Quick Start

### 1. Build Custom Triton Image

```bash
cd triton
docker build -t triton-imagen .
```

### 2. Deploy to GKE

```bash
kubectl apply -k k8s/triton/
```

### 3. Test Locally

```bash
# Port forward
kubectl port-forward -n imagen svc/triton 8000:8000 8001:8001 8002:8002

# Health check
curl localhost:8000/v2/health/ready

# Model status
curl localhost:8000/v2/models/upscale
```

---

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 8000 | HTTP | REST API |
| 8001 | gRPC | High-performance inference |
| 8002 | HTTP | Prometheus metrics |

---

## Benefits Over Current Setup

| Aspect | Current (Python Workers) | With Triton |
|--------|--------------------------|-------------|
| Batching | None (1 job at a time) | Automatic dynamic batching |
| Models per GPU | 1 | Multiple (smart scheduling) |
| Throughput | ~0.07 jobs/s/GPU | ~0.3 jobs/s/GPU |
| Memory management | Manual | Automatic |
| Metrics | Custom Prometheus | Built-in |
| Model loading | On worker start | Hot-reload capable |

---

*See individual model configs for batching parameters.*

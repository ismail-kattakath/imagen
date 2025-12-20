# CI/CD Pipeline

Automated build and deployment using Google Cloud Build.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Pipeline Files](#pipeline-files)
4. [Triggers](#triggers)
5. [Setup Guide](#setup-guide)
6. [Deployment Flow](#deployment-flow)
7. [Commands](#commands)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The CI/CD pipeline automatically builds and deploys the Imagen platform whenever code is pushed to GitHub.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   git push  ──▶  Cloud Build  ──▶  Cloud Run + GKE              │
│                                                                  │
│   • Push to main     → Deploy to production                     │
│   • Push to develop  → Deploy to development                    │
│   • Pull request     → Build & test only (no deploy)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Benefits

| Feature | Description |
|---------|-------------|
| **Automatic** | No manual deployment steps |
| **Fast** | ~10-15 minutes for full deployment |
| **Safe** | PR checks before merge |
| **Consistent** | Same process every time |
| **Traceable** | Full build logs in Cloud Console |

---

## Architecture

### Complete CI/CD Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTOMATED CI/CD FLOW                          │
└─────────────────────────────────────────────────────────────────┘

Developer pushes code
        │
        ▼
┌─────────────────┐
│    GitHub       │
│                 │
│  main branch ───┼──▶ Production deployment
│  develop branch─┼──▶ Development deployment  
│  pull request ──┼──▶ Build & test only
│                 │
└────────┬────────┘
         │
         │ Webhook triggers Cloud Build
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CLOUD BUILD                                  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ STAGE 1: BUILD (parallel)                                  │ │
│  │                                                             │ │
│  │  ┌─────────────────┐    ┌─────────────────┐               │ │
│  │  │ Build API image │    │ Build Worker    │               │ │
│  │  │ (Dockerfile.api)│    │ image           │               │ │
│  │  └─────────────────┘    └─────────────────┘               │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                         │                                        │
│                         ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ STAGE 2: PUSH (parallel)                                   │ │
│  │                                                             │ │
│  │  ┌─────────────────┐    ┌─────────────────┐               │ │
│  │  │ Push API image  │    │ Push Worker     │               │ │
│  │  │ to Artifact Reg │    │ image           │               │ │
│  │  └─────────────────┘    └─────────────────┘               │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                         │                                        │
│                         ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ STAGE 3: DEPLOY API                                        │ │
│  │                                                             │ │
│  │  gcloud run deploy imagen-api --image api:${SHORT_SHA}     │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                         │                                        │
│                         ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ STAGE 4: DEPLOY WORKERS                                    │ │
│  │                                                             │ │
│  │  kubectl apply -f k8s/base/                                │ │
│  │  kubectl apply -f k8s/workers/                             │ │
│  │  kubectl apply -f k8s/autoscaling/                         │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                         │                                        │
│                         ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ STAGE 5: VERIFY                                            │ │
│  │                                                             │ │
│  │  kubectl get pods -n imagen                                │ │
│  │  kubectl get hpa -n imagen                                 │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYED!                                     │
│                                                                  │
│  ✓ API running on Cloud Run                                     │
│  ✓ Workers running on GKE with new image                        │
│  ✓ Auto-scaling configured                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### What Gets Deployed Where

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   SOURCE CODE                                                   │
│   src/api/  ─────────────────┐                                  │
│   src/pipelines/             │                                  │
│   src/workers/               │                                  │
│   src/services/              │                                  │
│                              ▼                                  │
│                    ┌─────────────────┐                          │
│                    │  Docker Build   │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│              ┌──────────────┴──────────────┐                    │
│              ▼                             ▼                    │
│   ┌─────────────────────┐      ┌─────────────────────┐         │
│   │ Dockerfile.api      │      │ Dockerfile.worker   │         │
│   │                     │      │                     │         │
│   │ • FastAPI           │      │ • PyTorch + CUDA    │         │
│   │ • Lightweight       │      │ • ML models         │         │
│   │ • ~500MB            │      │ • GPU support       │         │
│   │                     │      │ • ~10GB             │         │
│   └──────────┬──────────┘      └──────────┬──────────┘         │
│              │                             │                    │
│              ▼                             ▼                    │
│   ┌─────────────────────┐      ┌─────────────────────┐         │
│   │ Artifact Registry   │      │ Artifact Registry   │         │
│   │ imagen/api:latest   │      │ imagen/worker:latest│         │
│   └──────────┬──────────┘      └──────────┬──────────┘         │
│              │                             │                    │
│              ▼                             ▼                    │
│   ┌─────────────────────┐      ┌─────────────────────┐         │
│   │     CLOUD RUN       │      │        GKE          │         │
│   │                     │      │                     │         │
│   │ • API endpoints     │      │ • GPU workers       │         │
│   │ • Auto-scales 0→10  │      │ • HPA auto-scaling  │         │
│   │ • Serverless        │      │ • T4 GPUs           │         │
│   └─────────────────────┘      └─────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Files

### File Structure

```
imagen/
├── cloudbuild.yaml          # Full deployment pipeline
├── cloudbuild-pr.yaml       # PR validation pipeline
│
└── terraform/
    └── modules/
        └── cloud-build/
            └── main.tf      # Trigger definitions + IAM
```

### cloudbuild.yaml (Full Deployment)

Runs on push to `main` or `develop` branch.

| Step | Name | Description | Duration |
|------|------|-------------|----------|
| 1 | `build-api` | Build API Docker image | ~2 min |
| 2 | `build-worker` | Build Worker Docker image | ~5 min |
| 3 | `push-api` | Push API to Artifact Registry | ~1 min |
| 4 | `push-worker` | Push Worker to Artifact Registry | ~2 min |
| 5 | `deploy-api` | Deploy API to Cloud Run | ~1 min |
| 6 | `gke-credentials` | Get GKE cluster credentials | ~30 sec |
| 7 | `update-k8s-manifests` | Update image tags in YAML | ~10 sec |
| 8 | `deploy-k8s-base` | Apply namespace, configmap, PVC | ~30 sec |
| 9 | `deploy-k8s-workers` | Deploy worker pods | ~1 min |
| 10 | `deploy-k8s-autoscaling` | Deploy HPAs | ~30 sec |
| 11 | `verify-deployment` | Show deployment status | ~10 sec |

**Total: ~10-15 minutes**

### cloudbuild-pr.yaml (PR Check)

Runs on pull requests to `main` branch.

| Step | Name | Description | Duration |
|------|------|-------------|----------|
| 1 | `build-api` | Build API Docker image | ~2 min |
| 2 | `build-worker` | Build Worker Docker image | ~5 min |
| 3 | `lint` | Run Ruff linter | ~1 min |
| 4 | `test` | Run unit tests | ~2 min |
| 5 | `summary` | Print results | ~10 sec |

**Total: ~5 minutes** (steps run in parallel)

---

## Triggers

### Trigger Configuration

| Trigger Name | Branch/Event | Pipeline | Action |
|--------------|--------------|----------|--------|
| `imagen-deploy-production` | Push to `main` | `cloudbuild.yaml` | Full deploy |
| `imagen-deploy-development` | Push to `develop` | `cloudbuild.yaml` | Full deploy |
| `imagen-pr-check` | PR to `main` | `cloudbuild-pr.yaml` | Build + test |

### Trigger Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRIGGER MATRIX                              │
└─────────────────────────────────────────────────────────────────┘

EVENT                    TRIGGER                  RESULT
─────────────────────────────────────────────────────────────────

Push to main      ──▶   imagen-deploy-production
                               │
                               ├──▶ Build images
                               ├──▶ Push to registry
                               ├──▶ Deploy API (Cloud Run)
                               └──▶ Deploy Workers (GKE prod)


Push to develop   ──▶   imagen-deploy-development
                               │
                               ├──▶ Build images
                               ├──▶ Push to registry
                               ├──▶ Deploy API (Cloud Run)
                               └──▶ Deploy Workers (GKE dev)


PR to main        ──▶   imagen-pr-check
                               │
                               ├──▶ Build images (validate)
                               ├──▶ Run linting
                               └──▶ Run tests
                               
                               ❌ NO DEPLOYMENT
```

---

## Setup Guide

### Prerequisites

- GCP project with billing enabled
- GitHub repository
- Terraform installed locally
- gcloud CLI configured

### Step 1: Deploy Infrastructure

```bash
cd terraform

# Initialize Terraform
terraform init -backend-config=backend.tfvars

# Deploy infrastructure (without CI/CD triggers)
terraform apply -var-file=environments/dev.tfvars
```

### Step 2: Connect GitHub to Cloud Build

#### Option A: Cloud Console (Recommended)

1. Go to [Cloud Build Triggers](https://console.cloud.google.com/cloud-build/triggers)
2. Click **"Connect Repository"**
3. Select **"GitHub (Cloud Build GitHub App)"**
4. Authorize and select your repository
5. Click **"Connect"**

#### Option B: Install GitHub App Directly

1. Go to [Cloud Build GitHub App](https://github.com/apps/google-cloud-build)
2. Click **"Install"**
3. Select your repository
4. Follow the prompts

### Step 3: Update Terraform with GitHub Info

```hcl
# terraform/environments/dev.tfvars

# Add these lines:
github_owner = "your-username"   # GitHub username or org
github_repo  = "imagen"          # Repository name
```

### Step 4: Apply Terraform to Create Triggers

```bash
cd terraform
terraform apply -var-file=environments/dev.tfvars
```

### Step 5: Verify Setup

```bash
# Check triggers were created
gcloud builds triggers list

# Expected output:
# NAME                         STATUS
# imagen-deploy-production     ENABLED
# imagen-deploy-development    ENABLED
# imagen-pr-check              ENABLED
```

### Step 6: Push Code!

```bash
git add .
git commit -m "Initial deployment"
git push origin main
```

Cloud Build will automatically trigger and deploy.

---

## Deployment Flow

### What Happens on Each Event

#### Push to `main` (Production)

```
┌─────────────────────────────────────────────────────────────────┐
│ PRODUCTION DEPLOYMENT                                            │
│                                                                  │
│ 1. Build Docker images                                          │
│    • imagen/api:abc123                                          │
│    • imagen/worker:abc123                                       │
│                                                                  │
│ 2. Push to Artifact Registry                                    │
│    • us-central1-docker.pkg.dev/PROJECT/imagen/api:abc123       │
│    • us-central1-docker.pkg.dev/PROJECT/imagen/worker:abc123    │
│                                                                  │
│ 3. Deploy API to Cloud Run                                      │
│    • imagen-api service updated                                 │
│    • New revision: imagen-api-abc123                            │
│                                                                  │
│ 4. Deploy Workers to GKE (imagen-prod cluster)                  │
│    • All workers updated to new image                           │
│    • Rolling update (zero downtime)                             │
│                                                                  │
│ 5. Verify                                                       │
│    • Check pod status                                           │
│    • Check HPA status                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Push to `develop` (Development)

Same as production, but deploys to `imagen-dev` GKE cluster.

#### Pull Request

```
┌─────────────────────────────────────────────────────────────────┐
│ PR VALIDATION                                                    │
│                                                                  │
│ 1. Build Docker images (validates Dockerfiles)                  │
│    • Does NOT push to registry                                  │
│                                                                  │
│ 2. Run linting (Ruff)                                           │
│    • Code style checks                                          │
│                                                                  │
│ 3. Run tests (pytest)                                           │
│    • Unit tests only                                            │
│                                                                  │
│ 4. Report results                                               │
│    • Success = ready to merge                                   │
│    • Failure = needs fixes                                      │
│                                                                  │
│ ❌ NO DEPLOYMENT                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Image Tagging Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                      IMAGE TAGS                                  │
└─────────────────────────────────────────────────────────────────┘

Each build creates two tags:

1. Commit SHA (immutable):
   imagen/api:abc123f
   imagen/worker:abc123f

2. Latest (mutable):
   imagen/api:latest
   imagen/worker:latest

Benefits:
• SHA tags enable rollback to specific commits
• Latest tag always points to most recent build
• Kubernetes manifests use SHA for reproducibility
```

---

## Commands

### Makefile Commands

```bash
# Trigger manual build (without git push)
make build-manual

# Run PR check locally
make build-pr-check

# View recent builds
make build-history

# Check configured triggers
make build-triggers

# Full first-time deployment
make deploy-all
```

### gcloud Commands

```bash
# List recent builds
gcloud builds list --limit=10

# View build details
gcloud builds describe BUILD_ID

# Stream build logs
gcloud builds log BUILD_ID --stream

# List triggers
gcloud builds triggers list

# Run trigger manually
gcloud builds triggers run imagen-deploy-production

# Cancel running build
gcloud builds cancel BUILD_ID
```

### Monitoring Builds

```bash
# Watch build in real-time
watch gcloud builds list --limit=5

# Open Cloud Console
open https://console.cloud.google.com/cloud-build/builds
```

---

## Troubleshooting

### Common Issues

#### Build Fails: "Permission Denied"

```
ERROR: permission denied to access Artifact Registry
```

**Solution:** Ensure Cloud Build service account has required IAM roles.

```bash
# Check current roles
gcloud projects get-iam-policy PROJECT_ID \
  --filter="bindings.members:cloudbuild" \
  --format="table(bindings.role)"

# Required roles:
# - roles/run.admin
# - roles/container.developer
# - roles/artifactregistry.writer
# - roles/iam.serviceAccountUser
# - roles/storage.admin
```

These are automatically set by Terraform. If missing, re-run:

```bash
terraform apply -var-file=environments/dev.tfvars
```

#### Build Fails: "Cluster Not Found"

```
ERROR: cluster "imagen-cluster" not found
```

**Solution:** Check cluster name matches in:
- `cloudbuild.yaml` (`_GKE_CLUSTER` substitution)
- `terraform/environments/*.tfvars` (`gke_cluster_name`)

#### Build Fails: "Image Not Found"

```
ERROR: image not found: imagen/api:latest
```

**Solution:** First deployment needs images to exist. Run manual build first:

```bash
make build-manual
```

#### Build Slow: Worker Image Takes Forever

Worker image includes PyTorch + CUDA (~10GB). This is normal.

**Optimization tips:**
- Use Cloud Build machine type `E2_HIGHCPU_8` (already configured)
- Consider layer caching
- Pre-build base images

#### Trigger Not Firing

```bash
# Check trigger is enabled
gcloud builds triggers list

# Check GitHub connection
gcloud builds triggers describe TRIGGER_NAME

# Verify webhook in GitHub
# Settings → Webhooks → Should see google.com webhook
```

### Viewing Logs

#### Cloud Console

1. Go to [Cloud Build History](https://console.cloud.google.com/cloud-build/builds)
2. Click on a build
3. Click on a step to see logs

#### CLI

```bash
# List builds
gcloud builds list

# View specific build
gcloud builds log BUILD_ID

# Stream logs (real-time)
gcloud builds log BUILD_ID --stream
```

### Rollback

#### Rollback API (Cloud Run)

```bash
# List revisions
gcloud run revisions list --service imagen-api

# Rollback to previous revision
gcloud run services update-traffic imagen-api \
  --to-revisions=imagen-api-PREVIOUS=100
```

#### Rollback Workers (GKE)

```bash
# Get previous image SHA from build history
gcloud builds list --limit=5

# Update deployment
kubectl set image deployment/upscale-worker \
  worker=us-central1-docker.pkg.dev/PROJECT/imagen/worker:PREVIOUS_SHA \
  -n imagen
```

---

## Security

### Service Account Permissions

Cloud Build uses a service account with these roles:

| Role | Purpose |
|------|---------|
| `roles/run.admin` | Deploy to Cloud Run |
| `roles/container.developer` | Deploy to GKE |
| `roles/artifactregistry.writer` | Push Docker images |
| `roles/iam.serviceAccountUser` | Act as service accounts |
| `roles/storage.admin` | Access GCS (build artifacts) |

### Secrets Management

For sensitive values (API keys, etc.):

```yaml
# cloudbuild.yaml

steps:
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        gcloud secrets versions access latest --secret=MY_SECRET
    secretEnv: ['MY_SECRET']

availableSecrets:
  secretManager:
    - versionName: projects/PROJECT/secrets/MY_SECRET/versions/latest
      env: MY_SECRET
```

---

## Cost

### Cloud Build Pricing

| Resource | Free Tier | Cost After |
|----------|-----------|------------|
| Build minutes | 120 min/day | $0.003/min |
| Storage | 50 GB | $0.026/GB |

### Typical Costs

| Build Type | Duration | Cost |
|------------|----------|------|
| Full deployment | ~15 min | ~$0.05 |
| PR check | ~5 min | ~$0.02 |
| Daily (10 deploys) | 150 min | ~$0.45 |
| Monthly (estimate) | | ~$15-30 |

---

## Quick Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                      CI/CD QUICK REFERENCE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PUSH TO MAIN       →  Full production deployment (~15 min)     │
│  PUSH TO DEVELOP    →  Full dev deployment (~15 min)            │
│  PULL REQUEST       →  Build + test only (~5 min)               │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  COMMANDS:                                                       │
│                                                                  │
│  make build-manual     # Trigger build without git push         │
│  make build-history    # View recent builds                     │
│  make build-triggers   # List configured triggers               │
│                                                                  │
│  gcloud builds list                    # List builds            │
│  gcloud builds log BUILD_ID            # View logs              │
│  gcloud builds triggers run NAME       # Manual trigger         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FILES:                                                          │
│                                                                  │
│  cloudbuild.yaml       # Full deployment pipeline               │
│  cloudbuild-pr.yaml    # PR validation pipeline                 │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CONSOLE:                                                        │
│                                                                  │
│  https://console.cloud.google.com/cloud-build/builds            │
│  https://console.cloud.google.com/cloud-build/triggers          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

*Last updated: December 2024*

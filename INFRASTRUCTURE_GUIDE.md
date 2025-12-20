# Infrastructure & Kubernetes Guide

A beginner-friendly guide to understanding how Imagen's infrastructure works.

---

## Table of Contents

1. [Two-Layer Architecture](#two-layer-architecture)
2. [What Terraform Does](#what-terraform-does)
3. [What Kubernetes Does](#what-kubernetes-does)
4. [How kubectl Works](#how-kubectl-works)
5. [Folder Structure](#folder-structure)
6. [Auto-Scaling Explained](#auto-scaling-explained)
7. [Deployment Sequence](#deployment-sequence)
8. [Common Commands](#common-commands)

---

## Two-Layer Architecture

Imagen uses two separate layers of configuration:

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   LAYER 1: INFRASTRUCTURE (Terraform)                                │
│   "Build the house"                                                  │
│                                                                      │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│   │   GKE   │ │ Pub/Sub │ │   GCS   │ │Firestore│ │   IAM   │        │
│   │ Cluster │ │ Topics  │ │ Bucket  │ │Database │ │ Roles   │        │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │
│                                                                      │
│----------------------------------------------------------------------│
│                                                                      │
│   LAYER 2: APPLICATIONS (kubectl)                                    │
│   "Put furniture in the house"                                       │
│                                                                      │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                    │
│   │ Worker  │ │ Worker  │ │  HPA    │ │ConfigMap│                    │
│   │  Pods   │ │  Pods   │ │         │ │         │                    │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘                    │
│                                                                      │
│   Running INSIDE GKE cluster                                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Quick Comparison

| Aspect       | Terraform              | kubectl               |
| ------------ | ---------------------- | --------------------- |
| Creates      | GCP resources          | K8s resources         |
| Where        | Google Cloud           | Inside GKE cluster    |
| Config files | `*.tf`                 | `*.yaml`              |
| Run when     | First (infrastructure) | Second (applications) |
| Analogy      | Construction company   | Interior designer     |

---

## What Terraform Does

Terraform creates the **GCP infrastructure** — the foundation everything runs on.

### Resources Created by Terraform

```
terraform/main.tf creates:

┌─────────────────────────────────────────────────────────────────┐
│ GCP Project                                                     │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ GKE Autopilot   │    │ Cloud Storage   │                     │
│  │ Cluster         │    │ Bucket          │                     │
│  │ (empty, no apps)│    │ (for images)    │                     │
│  └─────────────────┘    └─────────────────┘                     │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ Pub/Sub Topics  │    │ Firestore       │                     │
│  │ & Subscriptions │    │ Database        │                     │
│  └─────────────────┘    └─────────────────┘                     │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ Cloud Run       │    │ IAM             │                     │
│  │ (API hosting)   │    │ Service Accounts│                     │
│  └─────────────────┘    └─────────────────┘                     │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │ Artifact Registry│                                           │
│  │ (Docker images)  │                                           │
│  └──────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Point

After Terraform runs, you have:

- A GKE cluster (but it's **empty** — no applications running)
- Pub/Sub topics (but **no workers** listening)
- A storage bucket (but **no images** yet)

**Terraform builds the infrastructure. kubectl deploys the applications.**

---

## What Kubernetes Does

Kubernetes (K8s) **orchestrates containers**. It decides:

- Where to run your containers (which node)
- How many copies to run (replicas)
- What to do if a container crashes (restart it)

### Resources Created by kubectl

```
k8s/*.yaml creates:

┌─────────────────────────────────────────────────────────────────┐
│ Inside GKE Cluster                                              │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ Namespace       │    │ ConfigMap       │                     │
│  │ (imagen)        │    │ (env vars)      │                     │
│  └─────────────────┘    └─────────────────┘                     │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ Deployments     │    │ PVC             │                     │
│  │ (worker pods)   │    │ (model storage) │                     │
│  └─────────────────┘    └─────────────────┘                     │
│                                                                 │
│  ┌─────────────────┐                                            │
│  │ HPAs            │                                            │
│  │ (auto-scaling)  │                                            │
│  └─────────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key K8s Concepts

| Concept        | What It Is            | Our Usage                                |
| -------------- | --------------------- | ---------------------------------------- |
| **Namespace**  | Isolated environment  | `imagen` namespace for all our resources |
| **Deployment** | Defines what to run   | Worker containers with GPU               |
| **Pod**        | Running container(s)  | One worker instance                      |
| **ConfigMap**  | Environment variables | GCP project ID, bucket names             |
| **PVC**        | Persistent storage    | Model cache (50Gi)                       |
| **HPA**        | Auto-scaling rules    | Scale based on queue depth               |

---

## How kubectl Works

**Important: GKE doesn't automatically know about your YAML files.**

You must explicitly push them using `kubectl apply`.

### The Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Your Machine                           GKE Cluster            │
│                                                                 │
│   k8s/                                                          │
│   ├── base/                                                     │
│   ├── workers/                  ────────────────────────────    │
│   └── autoscaling/              GKE has NO idea these exist!    │
│                                 ────────────────────────────    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

You must PUSH the config to GKE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   $ kubectl apply -f k8s/base/namespace.yaml                    │
│                        │                                        │
│                        │  "Here, take this YAML"                │
│                        ▼                                        │
│                   ┌─────────┐                                   │
│                   │   GKE   │  "OK, I'll create a Namespace"    │
│                   └─────────┘                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### kubectl Commands

```bash
# Apply single file
kubectl apply -f k8s/base/namespace.yaml

# Apply entire folder (all YAML files in it)
kubectl apply -f k8s/base/

# Apply multiple folders
kubectl apply -f k8s/base/ -f k8s/workers/

# Apply everything recursively
kubectl apply -f k8s/ --recursive
```

---

## Folder Structure

**Kubernetes doesn't care about folder names.** The structure is purely for human organization.

### What K8s Sees

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Your Folder Structure          What Kubernetes Sees           │
│   (for humans)                   (just YAML content)            │
│                                                                 │
│   k8s/                                                          │
│   ├── base/                      kubectl apply -f file.yaml     │
│   │   └── namespace.yaml    ───▶ "Oh, a Namespace resource"     │
│   ├── workers/                                                  │
│   │   └── upscale.yaml      ───▶ "Oh, a Deployment resource"    │
│   └── autoscaling/                                              │
│       └── hpa.yaml          ───▶ "Oh, an HPA resource"          │
│                                                                 │
│   Kubernetes ignores folder names completely!                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Our Structure

```
k8s/
├── base/              # Foundation - namespace, config, storage
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   └── workload-identity.yaml
│
├── workers/           # Workloads - the actual applications
│   ├── upscale-worker.yaml
│   ├── enhance-worker.yaml
│   ├── comic-worker.yaml
│   └── background-remove-worker.yaml
│
└── autoscaling/       # Scaling - dynamic scaling rules
    ├── custom-metrics-adapter.yaml
    ├── upscale-hpa.yaml
    ├── enhance-hpa.yaml
    ├── comic-hpa.yaml
    ├── background-remove-hpa.yaml
    └── README.md
```

### Common Patterns

Other projects might organize differently:

```
# By environment
k8s/
├── base/
├── dev/
├── staging/
└── prod/

# By application
k8s/
├── api/
├── upscale-worker/
└── enhance-worker/

# Flat (simple projects)
k8s/
├── namespace.yaml
├── api-deployment.yaml
└── worker-deployment.yaml
```

**All are valid!** Choose what makes sense for your team.

---

## Auto-Scaling Explained

### Two Levels of Scaling

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCALING LAYERS                               │
└─────────────────────────────────────────────────────────────────┘

LAYER 1: NODE SCALING (GKE Autopilot handles this ✅)
─────────────────────────────────────────────────────
"Do we have enough machines (VMs) to run the pods?"

LAYER 2: POD SCALING (HPA handles this ✅)
─────────────────────────────────────────────────────
"How many copies of our worker should run?"
```

### How They Work Together

```
┌─────────────────────────────────────────────────────────────────┐
│                      WITH HPA + GKE AUTOPILOT                   │
└─────────────────────────────────────────────────────────────────┘

100 messages in queue
        │
        ▼
┌─────────────────┐
│ Metrics Adapter │  ← Reads queue depth from Cloud Monitoring
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      HPA        │  ← "100 msgs / 2 per worker = need 50"
│                 │     "max is 10, so scale to 10"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Deployment    │  ← replicas: 1 → 10
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GKE Autopilot   │  ← "10 pods need GPUs, provisioning 10 nodes"
└─────────────────┘
```

### Without HPA

```
100 messages in queue
        │
        ▼
Deployment: replicas: 1 (hardcoded)
        │
        ▼
GKE: "You want 1 pod, here's 1 node"
        │
        ▼
┌────────────┐
│   1 Node   │
│   1 Pod    │  ← Processes 100 messages one by one
│   1 GPU    │
└────────────┘

Time to process: 100 × 20s = 2000s (33 minutes!) 😱
```

### With HPA

```
100 messages in queue
        │
        ▼
HPA: "Scale to 10 replicas"
        │
        ▼
Deployment: replicas: 1 → 10
        │
        ▼
GKE Autopilot: "10 pods, provisioning 10 GPU nodes"
        │
        ▼
┌──────┐ ┌──────┐ ┌──────┐ ... (10 workers)
│Pod 1 │ │Pod 2 │ │Pod 3 │
└──────┘ └──────┘ └──────┘

Time to process: (100/10) × 20s = 200s (3 min!) ✅
```

### Key Point

**GKE Autopilot is ready to scale, but it waits for you to ask.**

- Without HPA: Deployment always says "1 pod please"
- With HPA: HPA tells Deployment "scale to N pods"
- GKE Autopilot responds by provisioning N nodes

---

## Deployment Sequence

### Complete Deployment Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT SEQUENCE                        │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Terraform (create infrastructure)
        │
        │  $ cd terraform
        │  $ terraform init
        │  $ terraform apply -var-file=environments/dev.tfvars
        │
        ▼
        Creates: GKE cluster, Pub/Sub, GCS, IAM, etc.
        ─────────────────────────────────────────────


STEP 2: Build & Push Docker Images
        │
        │  $ gcloud builds submit --config=cloudbuild.yaml
        │
        ▼
        Pushes: API and worker images to Artifact Registry
        ─────────────────────────────────────────────────


STEP 3: Connect to GKE
        │
        │  $ gcloud container clusters get-credentials imagen-cluster --region us-central1
        │
        ▼
        Now: kubectl can talk to your cluster
        ─────────────────────────────────────


STEP 4: kubectl (deploy applications)
        │
        │  $ kubectl apply -f k8s/base/
        │  $ kubectl apply -f k8s/workers/
        │  $ kubectl apply -f k8s/autoscaling/
        │
        ▼
        Creates: Pods, HPAs, ConfigMaps inside GKE
        ─────────────────────────────────────────


STEP 5: Verify
        │
        │  $ kubectl get pods -n imagen
        │  $ kubectl get hpa -n imagen
        │
        ▼
        Done! 🎉
```

### Using Makefile

```bash
# Step 1: Infrastructure
make tf-init
make tf-apply

# Step 2: Build images
# (handled by cloudbuild.yaml)
gcloud builds submit --config=cloudbuild.yaml

# Step 3: Connect to GKE
gcloud container clusters get-credentials imagen-cluster --region us-central1

# Step 4: Deploy applications
make k8s-deploy-all

# Step 5: Verify
make k8s-hpa-status
```

---

## Common Commands

### Terraform

```bash
# Initialize (first time only)
terraform init

# Preview changes
terraform plan -var-file=environments/dev.tfvars

# Apply changes
terraform apply -var-file=environments/dev.tfvars

# Destroy everything (careful!)
terraform destroy -var-file=environments/dev.tfvars
```

### kubectl Basics

```bash
# View resources
kubectl get pods -n imagen          # List pods
kubectl get deployments -n imagen   # List deployments
kubectl get hpa -n imagen           # List HPAs
kubectl get all -n imagen           # List everything

# Describe (detailed info)
kubectl describe pod <pod-name> -n imagen
kubectl describe hpa <hpa-name> -n imagen

# Logs
kubectl logs <pod-name> -n imagen
kubectl logs -f <pod-name> -n imagen  # Follow logs

# Apply configs
kubectl apply -f k8s/base/
kubectl apply -f k8s/workers/
kubectl apply -f k8s/autoscaling/

# Delete resources
kubectl delete -f k8s/workers/upscale-worker.yaml
```

### Debugging

```bash
# Pod not starting?
kubectl describe pod <pod-name> -n imagen
kubectl logs <pod-name> -n imagen

# HPA not scaling?
kubectl describe hpa <hpa-name> -n imagen

# Check events
kubectl get events -n imagen --sort-by='.lastTimestamp'

# Shell into a pod
kubectl exec -it <pod-name> -n imagen -- /bin/bash
```

### Monitoring

```bash
# Watch pods in real-time
watch kubectl get pods -n imagen

# Watch HPA scaling
watch kubectl get hpa -n imagen

# Combined view
make k8s-watch
```

---

## Analogy Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Building a Restaurant                                         │
│                                                                 │
│   TERRAFORM = Construction Company                              │
│   - Builds the building                                         │
│   - Installs plumbing (Pub/Sub)                                 │
│   - Sets up electricity (IAM)                                   │
│   - Creates parking lot (GCS bucket)                            │
│                                                                 │
│   KUBECTL = Interior Designer + Staff Manager                   │
│   - Arranges tables and chairs (Deployments)                    │
│   - Hires chefs (Worker pods)                                   │
│   - Sets up kitchen equipment (ConfigMaps)                      │
│   - Decides how many chefs per shift (HPA)                      │
│                                                                 │
│   GKE AUTOPILOT = Building Management                           │
│   - "You need 10 chefs? I'll open 10 kitchen stations"          │
│   - Provisions GPU nodes on demand                              │
│                                                                 │
│   You need ALL of them:                                         │
│   - Without building → nowhere to put kitchen                   │
│   - Without furniture → empty building                          │
│   - Without management → can't scale                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

| Task                  | Command                           |
| --------------------- | --------------------------------- |
| Create infrastructure | `make tf-apply`                   |
| Deploy everything     | `make k8s-deploy-all`             |
| Check pods            | `kubectl get pods -n imagen`      |
| Check scaling         | `kubectl get hpa -n imagen`       |
| View logs             | `kubectl logs -f <pod> -n imagen` |
| Watch scaling         | `make k8s-watch`                  |

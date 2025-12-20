# Imagen Quick Reference

## Architecture Overview

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client        │────▶│  Cloud Run   │────▶│  Pub/Sub    │
│                 │     │  (FastAPI)   │     │  (Queue)    │
└─────────────────┘     └──────────────┘     └──────┬──────┘
                                                    │
                        ┌───────────────────────────┼───────┐
                        │          GKE Autopilot    │       │
                        │   ┌───────┐ ┌───────┐ ┌───────┐   │
                        │   │Worker │ │Worker │ │Worker │   │
                        │   │(GPU)  │ │(GPU)  │ │(GPU)  │   │
                        │   └───────┘ └───────┘ └───────┘   │
                        │            ▲                      │
                        │     HPA scales these              │
                        └───────────────────────────────────┘
```

## Two Layers

| Layer | Tool | Creates | Files |
|-------|------|---------|-------|
| Infrastructure | Terraform | GKE, Pub/Sub, GCS, IAM | `terraform/*.tf` |
| Applications | kubectl | Pods, HPAs, ConfigMaps | `k8s/*.yaml` |

## Deployment Commands

```bash
# 1. Infrastructure (once)
cd terraform
terraform init -backend-config=backend.tfvars
terraform apply -var-file=environments/dev.tfvars

# 2. Connect to GKE
gcloud container clusters get-credentials imagen-cluster --region us-central1

# 3. Deploy apps (automatic via CI/CD, or manual)
make k8s-deploy-all

# 4. Verify
kubectl get pods -n imagen
kubectl get hpa -n imagen
```

## CI/CD (Automatic Deployment)

```bash
# Push to deploy
git push origin main      # → Production deployment
git push origin develop   # → Development deployment

# Manual build
make build-manual         # Trigger build without git push
make build-history        # View recent builds
```

## Key kubectl Commands

```bash
kubectl get pods -n imagen           # List pods
kubectl get hpa -n imagen            # Check auto-scaling
kubectl logs -f <pod> -n imagen      # View logs
kubectl describe pod <pod> -n imagen # Debug pod
kubectl apply -f k8s/workers/        # Deploy workers
```

## Auto-Scaling Flow

```
Queue fills → Metrics Adapter reads → HPA scales Deployment → GKE adds nodes
```

## Folder Structure

```
imagen/
├── src/               # Application code
├── k8s/
│   ├── base/          # Namespace, ConfigMap, PVC
│   ├── workers/       # Worker Deployments
│   └── autoscaling/   # HPAs + Metrics Adapter
├── terraform/
│   ├── main.tf        # Module orchestration
│   ├── modules/       # Reusable modules (10)
│   └── environments/  # Dev/Prod configs
├── docker/            # Dockerfiles
├── cloudbuild.yaml    # CI/CD pipeline
└── *.md               # Documentation
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/images/upscale` | 4x upscaling |
| POST | `/api/v1/images/enhance` | Quality enhancement |
| POST | `/api/v1/images/style/comic` | Comic style |
| POST | `/api/v1/images/style/aged` | Vintage look |
| POST | `/api/v1/images/background/remove` | Remove background |
| GET | `/api/v1/jobs/{job_id}` | Get job status |

## Cost Estimate

| Component | Cost/Month |
|-----------|------------|
| GKE (1-10 T4 GPUs) | $250-2500 |
| Cloud Run (API) | ~$50 |
| Pub/Sub + GCS + Firestore | ~$20 |
| Cloud Build | ~$15 |
| **Total** | $335-2600 |

## Documentation

| Doc | Contents |
|-----|----------|
| `ARCHITECTURE.md` | Complete platform design |
| `CI_CD.md` | Automated deployment pipeline |
| `INFRASTRUCTURE_GUIDE.md` | Terraform & K8s explained |
| `MODEL_MANAGEMENT.md` | ML model caching |
| `terraform/README.md` | Terraform modules |

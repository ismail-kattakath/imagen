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
terraform init
terraform apply -var-file=environments/dev.tfvars

# 2. Connect to GKE
gcloud container clusters get-credentials imagen-cluster --region us-central1

# 3. Deploy apps
make k8s-deploy-all

# 4. Verify
kubectl get pods -n imagen
kubectl get hpa -n imagen
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
k8s/
├── base/          # Namespace, ConfigMap, PVC
├── workers/       # Worker Deployments
└── autoscaling/   # HPAs + Metrics Adapter

terraform/
├── main.tf        # All GCP resources
└── environments/  # Dev/Prod configs
```

## Cost Estimate

| Component | Cost/Month |
|-----------|------------|
| GKE (1-10 T4 GPUs) | $250-2500 |
| Cloud Run (API) | ~$50 |
| Pub/Sub + GCS | ~$15 |
| **Total** | $300-2600 |

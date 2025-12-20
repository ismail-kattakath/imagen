# Terraform Infrastructure

This directory contains modular Terraform configuration for the Imagen platform.

## Structure

```
terraform/
├── main.tf                 # Module orchestration
├── variables.tf            # Input variables
├── outputs.tf              # Output values
├── backend.tfvars          # Remote state configuration
│
├── environments/           # Environment-specific configs
│   ├── dev.tfvars
│   └── prod.tfvars
│
└── modules/                # Reusable modules
    ├── apis/               # Enable GCP APIs
    ├── storage/            # Cloud Storage bucket
    ├── pubsub/             # Pub/Sub topics & subscriptions
    ├── firestore/          # Firestore database
    ├── iam/                # Service accounts & permissions
    ├── artifact-registry/  # Docker image repository
    ├── gke/                # GKE Autopilot cluster
    ├── cloud-run/          # Cloud Run API service
    └── autoscaling/        # HPA metrics adapter IAM
```

## Module Dependency Graph

```
                    ┌─────────┐
                    │  apis   │
                    └────┬────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   ┌─────────┐     ┌──────────┐     ┌───────────┐
   │ storage │     │  pubsub  │     │ firestore │
   └────┬────┘     └──────────┘     └───────────┘
        │                │                │
        │    ┌───────────┼────────────────┤
        │    │           │                │
        ▼    ▼           ▼                ▼
   ┌──────────────┐  ┌─────────┐  ┌───────────────────┐
   │   artifact   │  │   iam   │  │                   │
   │   registry   │  └────┬────┘  │                   │
   └──────┬───────┘       │       │                   │
          │               │       │                   │
          └───────────────┼───────┘                   │
                          │                           │
                ┌─────────┴─────────┐                 │
                │                   │                 │
                ▼                   ▼                 │
          ┌───────────┐       ┌─────────┐            │
          │ cloud-run │       │   gke   │            │
          └───────────┘       └────┬────┘            │
                                   │                 │
                                   ▼                 │
                            ┌─────────────┐          │
                            │ autoscaling │          │
                            └─────────────┘          │
```

## Quick Start

### 1. First-time Setup

```bash
# Create state bucket (one time)
PROJECT_ID="your-project-id"
gsutil mb -l us-central1 gs://${PROJECT_ID}-terraform-state
gsutil versioning set on gs://${PROJECT_ID}-terraform-state

# Update backend.tfvars with your bucket name
```

### 2. Initialize

```bash
cd terraform

# Initialize with backend
terraform init -backend-config=backend.tfvars
```

### 3. Deploy

```bash
# Development
terraform plan -var-file=environments/dev.tfvars
terraform apply -var-file=environments/dev.tfvars

# Production
terraform plan -var-file=environments/prod.tfvars
terraform apply -var-file=environments/prod.tfvars
```

### 4. Get Outputs

```bash
# Show all outputs
terraform output

# Get specific output
terraform output api_url
terraform output gke_get_credentials
```

## Modules

### apis

Enables required GCP APIs. Must run first.

| API | Purpose |
|-----|---------|
| `run.googleapis.com` | Cloud Run |
| `pubsub.googleapis.com` | Pub/Sub |
| `storage.googleapis.com` | Cloud Storage |
| `firestore.googleapis.com` | Firestore |
| `container.googleapis.com` | GKE |
| `artifactregistry.googleapis.com` | Artifact Registry |
| `monitoring.googleapis.com` | Cloud Monitoring |

### storage

Cloud Storage bucket for images.

| Variable | Default | Description |
|----------|---------|-------------|
| `lifecycle_age_days` | 7 | Auto-delete after N days |

### pubsub

Pub/Sub topics and subscriptions for job queues.

| Variable | Default | Description |
|----------|---------|-------------|
| `job_types` | 5 types | List of pipeline names |
| `ack_deadline_seconds` | 600 | 10 min for GPU jobs |

### firestore

Firestore database for job state.

| Variable | Default | Description |
|----------|---------|-------------|
| `database_name` | (default) | Firestore database name |

### iam

Service accounts and IAM bindings.

| Service Account | Purpose |
|-----------------|---------|
| `imagen-worker` | GPU workers in GKE |
| `imagen-api` | API in Cloud Run |

### artifact-registry

Docker image repository.

### gke

GKE Autopilot cluster for GPU workers.

| Variable | Default | Description |
|----------|---------|-------------|
| `cluster_name` | imagen-cluster | Cluster name |

### cloud-run

Cloud Run API service.

| Variable | Default | Description |
|----------|---------|-------------|
| `min_instances` | 0 | Scale to zero |
| `max_instances` | 10 | Max instances |

### autoscaling

IAM for HPA metrics adapter.

## Variables Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_id` | string | required | GCP project ID |
| `region` | string | us-central1 | GCP region |
| `environment` | string | dev | Environment name |
| `storage_lifecycle_days` | number | 7 | Image retention days |
| `job_types` | list | 5 types | Pipeline names |
| `gke_cluster_name` | string | imagen-cluster | GKE cluster name |
| `api_min_instances` | number | 0 | Min API instances |
| `api_max_instances` | number | 10 | Max API instances |

## Outputs Reference

| Output | Description |
|--------|-------------|
| `api_url` | Cloud Run API URL |
| `gcs_bucket` | Storage bucket name |
| `pubsub_topics` | List of topic names |
| `gke_cluster_name` | GKE cluster name |
| `gke_get_credentials` | kubectl config command |
| `docker_repository` | Artifact Registry URL |
| `worker_service_account` | Worker SA email |
| `quick_start` | Post-deployment instructions |

## Best Practices

### State Management

- Always use remote state (GCS backend)
- Enable versioning on state bucket
- Never commit `.tfstate` files

### Environment Isolation

- Use separate `.tfvars` for each environment
- Consider separate GCP projects for prod
- Use different cluster names per environment

### Security

- Use Workload Identity (not service account keys)
- Minimal IAM permissions per service account
- Enable deletion protection in prod

### Cost Control

- Set `api_min_instances = 0` for dev
- Use Spot VMs for GPU workers
- Set appropriate `maxReplicas` in HPA

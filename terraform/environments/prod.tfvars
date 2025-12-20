# =============================================================================
# PRODUCTION ENVIRONMENT
# =============================================================================

# Project
project_id  = "your-project-id"  # TODO: Replace with your GCP project ID
region      = "us-central1"
environment = "prod"

# Storage
storage_lifecycle_days = 30  # Keep images longer in prod

# Job types (pipelines)
job_types = [
  "upscale",
  "enhance",
  "style-comic",
  "style-aged",
  "background-remove"
]

# GKE
gke_cluster_name = "imagen-prod"

# Cloud Run API
api_min_instances = 1   # Always keep 1 instance warm
api_max_instances = 10  # Higher limit for prod

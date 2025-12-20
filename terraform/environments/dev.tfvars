# =============================================================================
# DEVELOPMENT ENVIRONMENT
# =============================================================================

# Project
project_id  = "your-project-id"  # TODO: Replace with your GCP project ID
region      = "us-central1"
environment = "dev"

# Storage
storage_lifecycle_days = 7  # Delete images after 7 days

# Job types (pipelines)
job_types = [
  "upscale",
  "enhance",
  "style-comic",
  "style-aged",
  "background-remove"
]

# GKE
gke_cluster_name = "imagen-dev"

# Cloud Run API
api_min_instances = 0   # Scale to zero when idle
api_max_instances = 5   # Lower limit for dev

# =============================================================================
# IMAGEN PLATFORM - OUTPUTS
# =============================================================================

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------

output "api_url" {
  description = "Cloud Run API URL"
  value       = module.cloud_run.service_url
}

# -----------------------------------------------------------------------------
# STORAGE
# -----------------------------------------------------------------------------

output "gcs_bucket" {
  description = "GCS bucket name for images"
  value       = module.storage.bucket_name
}

# -----------------------------------------------------------------------------
# PUB/SUB
# -----------------------------------------------------------------------------

output "pubsub_topics" {
  description = "Pub/Sub topic names"
  value       = module.pubsub.topic_names
}

output "pubsub_subscriptions" {
  description = "Pub/Sub subscription names"
  value       = module.pubsub.subscription_names
}

# -----------------------------------------------------------------------------
# GKE
# -----------------------------------------------------------------------------

output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = module.gke.cluster_name
}

output "gke_get_credentials" {
  description = "Command to configure kubectl"
  value       = module.gke.get_credentials_command
}

# -----------------------------------------------------------------------------
# DOCKER REGISTRY
# -----------------------------------------------------------------------------

output "docker_repository" {
  description = "Artifact Registry repository URL"
  value       = module.artifact_registry.repository_url
}

output "docker_api_image" {
  description = "Docker image path for API"
  value       = module.artifact_registry.api_image
}

output "docker_worker_image" {
  description = "Docker image path for workers"
  value       = module.artifact_registry.worker_image
}

# -----------------------------------------------------------------------------
# SERVICE ACCOUNTS
# -----------------------------------------------------------------------------

output "worker_service_account" {
  description = "Worker service account email"
  value       = module.iam.worker_service_account_email
}

output "api_service_account" {
  description = "API service account email"
  value       = module.iam.api_service_account_email
}

output "metrics_adapter_service_account" {
  description = "Metrics adapter service account email"
  value       = module.autoscaling.metrics_adapter_service_account
}

# -----------------------------------------------------------------------------
# QUICK START
# -----------------------------------------------------------------------------

output "quick_start" {
  description = "Quick start commands"
  value       = <<-EOT

    ============================================================
    IMAGEN PLATFORM DEPLOYED!
    ============================================================

    1. Configure kubectl:
       ${module.gke.get_credentials_command}

    2. Build and push Docker images:
       docker build -t ${module.artifact_registry.api_image}:latest -f docker/Dockerfile.api .
       docker push ${module.artifact_registry.api_image}:latest

       docker build -t ${module.artifact_registry.worker_image}:latest -f docker/Dockerfile.worker .
       docker push ${module.artifact_registry.worker_image}:latest

    3. Deploy Kubernetes resources:
       kubectl apply -f k8s/base/
       kubectl apply -f k8s/workers/
       kubectl apply -f k8s/autoscaling/

    4. Test the API:
       curl ${module.cloud_run.service_url}/health

    ============================================================
  EOT
}

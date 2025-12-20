# Artifact Registry for Docker images

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "repository_id" {
  description = "Repository name"
  type        = string
  default     = "imagen"
}

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = var.repository_id
  format        = "DOCKER"
  description   = "Docker images for Imagen platform"
}

output "repository_id" {
  description = "Artifact Registry repository ID"
  value       = google_artifact_registry_repository.images.repository_id
}

output "repository_url" {
  description = "Full repository URL for docker push/pull"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "api_image" {
  description = "Full path for API image"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/api"
}

output "worker_image" {
  description = "Full path for worker image"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/worker"
}

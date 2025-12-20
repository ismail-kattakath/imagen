# Cloud Run API service

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "imagen-api"
}

variable "image" {
  description = "Docker image URL"
  type        = string
}

variable "service_account_email" {
  description = "Service account for the API"
  type        = string
}

variable "gcs_bucket" {
  description = "GCS bucket name for images"
  type        = string
}

variable "min_instances" {
  description = "Minimum instances (0 = scale to zero)"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum instances"
  type        = number
  default     = 10
}

variable "cpu" {
  description = "CPU limit"
  type        = string
  default     = "1"
}

variable "memory" {
  description = "Memory limit"
  type        = string
  default     = "512Mi"
}

variable "allow_unauthenticated" {
  description = "Allow public access"
  type        = bool
  default     = true
}

# =============================================================================
# CLOUD RUN SERVICE
# =============================================================================

resource "google_cloud_run_v2_service" "api" {
  name     = var.service_name
  location = var.region

  template {
    service_account = var.service_account_email

    containers {
      image = var.image

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GCS_BUCKET"
        value = var.gcs_bucket
      }
      env {
        name  = "DEBUG"
        value = "false"
      }

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
      }

      # Health check
      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }
  }

  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
}

# =============================================================================
# PUBLIC ACCESS (optional)
# =============================================================================

resource "google_cloud_run_v2_service_iam_member" "api_public" {
  count = var.allow_unauthenticated ? 1 : 0

  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# =============================================================================
# OUTPUTS
# =============================================================================

output "service_name" {
  description = "Cloud Run service name"
  value       = google_cloud_run_v2_service.api.name
}

output "service_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_v2_service.api.uri
}

output "service_id" {
  description = "Cloud Run service ID"
  value       = google_cloud_run_v2_service.api.id
}

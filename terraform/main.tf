terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    # Configure in environments/*.tfvars
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "pubsub.googleapis.com",
    "storage.googleapis.com",
    "firestore.googleapis.com",
    "container.googleapis.com",
    "artifactregistry.googleapis.com",
  ])
  
  service            = each.value
  disable_on_destroy = false
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "imagen"
  format        = "DOCKER"
  
  depends_on = [google_project_service.apis]
}

# Cloud Storage bucket for images
resource "google_storage_bucket" "images" {
  name     = "${var.project_id}-imagen-images"
  location = var.region
  
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 7  # Auto-delete after 7 days
    }
    action {
      type = "Delete"
    }
  }
  
  cors {
    origin          = ["*"]
    method          = ["GET", "PUT", "POST"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

# Pub/Sub topics and subscriptions
locals {
  job_types = ["upscale", "enhance", "style-comic", "background-remove"]
}

resource "google_pubsub_topic" "jobs" {
  for_each = toset(local.job_types)
  name     = "${each.key}-jobs"
  
  depends_on = [google_project_service.apis]
}

resource "google_pubsub_subscription" "jobs" {
  for_each = google_pubsub_topic.jobs
  
  name  = "${each.key}-sub"
  topic = each.value.name
  
  ack_deadline_seconds       = 600  # 10 min for GPU jobs
  message_retention_duration = "86400s"  # 24 hours
  
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
}

# Firestore database
resource "google_firestore_database" "main" {
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
  
  depends_on = [google_project_service.apis]
}

# Service account for workers
resource "google_service_account" "worker" {
  account_id   = "imagen-worker"
  display_name = "Imagen GPU Worker"
}

# IAM bindings for worker
resource "google_project_iam_member" "worker_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_project_iam_member" "worker_pubsub" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_project_iam_member" "worker_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

# Cloud Run API service
resource "google_cloud_run_v2_service" "api" {
  name     = "imagen-api"
  location = var.region
  
  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/imagen/api:latest"
      
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GCS_BUCKET"
        value = google_storage_bucket.images.name
      }
      
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }
    
    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }
  }
  
  depends_on = [google_project_service.apis]
}

# Allow unauthenticated access to API
resource "google_cloud_run_v2_service_iam_member" "api_public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Cloud Storage bucket for images

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "bucket_suffix" {
  description = "Suffix for bucket name"
  type        = string
  default     = "imagen-images"
}

variable "lifecycle_age_days" {
  description = "Days before auto-deletion"
  type        = number
  default     = 7
}

resource "google_storage_bucket" "images" {
  name     = "${var.project_id}-${var.bucket_suffix}"
  location = var.region

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = var.lifecycle_age_days
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

output "bucket_name" {
  description = "Name of the storage bucket"
  value       = google_storage_bucket.images.name
}

output "bucket_url" {
  description = "URL of the storage bucket"
  value       = google_storage_bucket.images.url
}

# IAM: Service accounts and role bindings

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

# =============================================================================
# WORKER SERVICE ACCOUNT
# =============================================================================

resource "google_service_account" "worker" {
  account_id   = "imagen-worker"
  display_name = "Imagen GPU Worker"
  description  = "Service account for GPU workers running in GKE"
}

# Storage: Upload/download images
resource "google_project_iam_member" "worker_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

# Pub/Sub: Pull messages from subscriptions
resource "google_project_iam_member" "worker_pubsub" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

# Firestore: Read/write job status
resource "google_project_iam_member" "worker_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

# =============================================================================
# API SERVICE ACCOUNT (Cloud Run)
# =============================================================================

resource "google_service_account" "api" {
  account_id   = "imagen-api"
  display_name = "Imagen API"
  description  = "Service account for API running in Cloud Run"
}

# Storage: Upload input images
resource "google_project_iam_member" "api_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.api.email}"
}

# Pub/Sub: Publish messages to topics
resource "google_project_iam_member" "api_pubsub" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.api.email}"
}

# Firestore: Create/read job records
resource "google_project_iam_member" "api_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.api.email}"
}

# =============================================================================
# WORKLOAD IDENTITY BINDINGS (GKE)
# =============================================================================

# Allow GKE service account to impersonate worker service account
resource "google_service_account_iam_member" "worker_workload_identity" {
  service_account_id = google_service_account.worker.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[imagen/imagen-worker]"
}

# =============================================================================
# OUTPUTS
# =============================================================================

output "worker_service_account_email" {
  description = "Worker service account email"
  value       = google_service_account.worker.email
}

output "worker_service_account_name" {
  description = "Worker service account full name"
  value       = google_service_account.worker.name
}

output "api_service_account_email" {
  description = "API service account email"
  value       = google_service_account.api.email
}

output "api_service_account_name" {
  description = "API service account full name"
  value       = google_service_account.api.name
}

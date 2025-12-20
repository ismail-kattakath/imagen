# Custom Metrics Adapter IAM for HPA auto-scaling
# Allows HPA to read Pub/Sub queue depth from Cloud Monitoring

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

# =============================================================================
# METRICS ADAPTER SERVICE ACCOUNT
# =============================================================================

resource "google_service_account" "metrics_adapter" {
  account_id   = "metrics-adapter"
  display_name = "Custom Metrics Stackdriver Adapter"
  description  = "Allows HPA to read Pub/Sub metrics for auto-scaling"
}

# Allow reading monitoring metrics
resource "google_project_iam_member" "metrics_adapter_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.viewer"
  member  = "serviceAccount:${google_service_account.metrics_adapter.email}"
}

# Workload Identity binding
# Links Kubernetes ServiceAccount to GCP ServiceAccount
resource "google_service_account_iam_member" "metrics_adapter_workload_identity" {
  service_account_id = google_service_account.metrics_adapter.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[custom-metrics/custom-metrics-stackdriver-adapter]"
}

# =============================================================================
# OUTPUTS
# =============================================================================

output "metrics_adapter_service_account" {
  description = "Metrics adapter service account email"
  value       = google_service_account.metrics_adapter.email
}

output "metrics_adapter_service_account_name" {
  description = "Metrics adapter service account full name"
  value       = google_service_account.metrics_adapter.name
}

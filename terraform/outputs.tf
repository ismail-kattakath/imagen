output "api_url" {
  description = "Cloud Run API URL"
  value       = google_cloud_run_v2_service.api.uri
}

output "storage_bucket" {
  description = "GCS bucket for images"
  value       = google_storage_bucket.images.name
}

output "artifact_registry" {
  description = "Artifact Registry URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/imagen"
}

output "worker_service_account" {
  description = "Worker service account email"
  value       = google_service_account.worker.email
}

output "pubsub_topics" {
  description = "Pub/Sub topic names"
  value       = [for topic in google_pubsub_topic.jobs : topic.name]
}

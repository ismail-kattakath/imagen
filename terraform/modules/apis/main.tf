# Enable required GCP APIs

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "apis" {
  description = "List of APIs to enable"
  type        = list(string)
  default = [
    "run.googleapis.com",
    "pubsub.googleapis.com",
    "storage.googleapis.com",
    "firestore.googleapis.com",
    "container.googleapis.com",
    "artifactregistry.googleapis.com",
    "monitoring.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(var.apis)

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

output "enabled_apis" {
  description = "List of enabled APIs"
  value       = [for api in google_project_service.apis : api.service]
}

output "apis_ready" {
  description = "Dependency marker for other modules"
  value       = true
  depends_on  = [google_project_service.apis]
}

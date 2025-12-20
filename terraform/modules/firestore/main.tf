# Firestore database for job state tracking

variable "region" {
  description = "GCP region"
  type        = string
}

variable "database_name" {
  description = "Firestore database name"
  type        = string
  default     = "(default)"
}

variable "database_type" {
  description = "Firestore mode: FIRESTORE_NATIVE or DATASTORE_MODE"
  type        = string
  default     = "FIRESTORE_NATIVE"
}

resource "google_firestore_database" "main" {
  name        = var.database_name
  location_id = var.region
  type        = var.database_type
}

output "database_name" {
  description = "Firestore database name"
  value       = google_firestore_database.main.name
}

output "database_id" {
  description = "Firestore database ID"
  value       = google_firestore_database.main.id
}

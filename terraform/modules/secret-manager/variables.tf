variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "jwt_secret" {
  description = "JWT secret for authentication"
  type        = string
  sensitive   = true
}

variable "api_keys_json" {
  description = "API keys as JSON string"
  type        = string
  sensitive   = true
}

variable "cors_origins" {
  description = "Comma-separated list of allowed CORS origins"
  type        = string
  default     = ""
}

variable "cloudrun_service_account" {
  description = "Cloud Run service account email"
  type        = string
  default     = null
}

variable "gke_workload_identity" {
  description = "GKE workload identity service account email"
  type        = string
  default     = null
}

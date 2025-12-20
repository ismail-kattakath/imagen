# GKE Autopilot cluster for GPU workers

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "imagen-cluster"
}

variable "network" {
  description = "VPC network name"
  type        = string
  default     = "default"
}

variable "subnetwork" {
  description = "VPC subnetwork name"
  type        = string
  default     = "default"
}

# =============================================================================
# GKE AUTOPILOT CLUSTER
# =============================================================================

resource "google_container_cluster" "main" {
  name     = var.cluster_name
  location = var.region

  # Enable Autopilot mode
  enable_autopilot = true

  # Network configuration
  network    = var.network
  subnetwork = var.subnetwork

  # IP allocation policy (required for Autopilot)
  ip_allocation_policy {
    # Use default ranges
  }

  # Workload Identity (required for Autopilot)
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Release channel
  release_channel {
    channel = "REGULAR"
  }

  # Deletion protection (disable for dev, enable for prod)
  deletion_protection = false
}

# =============================================================================
# OUTPUTS
# =============================================================================

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.main.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.main.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.main.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.main.location
}

output "workload_identity_pool" {
  description = "Workload Identity pool for service account binding"
  value       = "${var.project_id}.svc.id.goog"
}

# Command to get credentials
output "get_credentials_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.main.name} --region ${var.region} --project ${var.project_id}"
}

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

variable "pods_range_name" {
  description = "Secondary range name for pods"
  type        = string
  default     = "gke-pods"
}

variable "services_range_name" {
  description = "Secondary range name for services"
  type        = string
  default     = "gke-services"
}

variable "master_ipv4_cidr_block" {
  description = "CIDR block for GKE master (must be /28)"
  type        = string
  default     = "172.16.0.0/28"
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

  # IP allocation policy (use secondary ranges from VPC subnet)
  ip_allocation_policy {
    cluster_secondary_range_name  = var.pods_range_name
    services_secondary_range_name = var.services_range_name
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false # Keep public endpoint for kubectl access
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }

  # Master authorized networks (restrict access to control plane)
  master_authorized_networks_config {
    # Allow access from anywhere (can be restricted to specific IPs)
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All networks"
    }
  }

  # Workload Identity (required for Autopilot)
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Release channel
  release_channel {
    channel = "REGULAR"
  }

  # Security features
  security_posture_config {
    mode = "BASIC"
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

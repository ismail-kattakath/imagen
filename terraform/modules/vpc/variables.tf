variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gke_subnet_cidr" {
  description = "CIDR range for GKE subnet"
  type        = string
  default     = "10.0.0.0/20"
}

variable "gke_pods_cidr" {
  description = "CIDR range for GKE pods"
  type        = string
  default     = "10.4.0.0/14"
}

variable "gke_services_cidr" {
  description = "CIDR range for GKE services"
  type        = string
  default     = "10.8.0.0/20"
}

variable "vpc_connector_cidr" {
  description = "CIDR range for VPC connector (must be /28)"
  type        = string
  default     = "10.8.16.0/28"
}

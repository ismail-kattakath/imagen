# =============================================================================
# IMAGEN PLATFORM - INPUT VARIABLES
# =============================================================================

# -----------------------------------------------------------------------------
# REQUIRED VARIABLES
# -----------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

# -----------------------------------------------------------------------------
# STORAGE
# -----------------------------------------------------------------------------

variable "storage_lifecycle_days" {
  description = "Days before auto-deleting images from GCS"
  type        = number
  default     = 7
}

# -----------------------------------------------------------------------------
# PUB/SUB
# -----------------------------------------------------------------------------

variable "job_types" {
  description = "List of job types (creates topic + subscription per type)"
  type        = list(string)
  default     = [
    "upscale",
    "enhance",
    "style-comic",
    "style-aged",
    "background-remove"
  ]
}

# -----------------------------------------------------------------------------
# GKE
# -----------------------------------------------------------------------------

variable "gke_cluster_name" {
  description = "GKE Autopilot cluster name"
  type        = string
  default     = "imagen-cluster"
}

# -----------------------------------------------------------------------------
# CLOUD RUN
# -----------------------------------------------------------------------------

variable "api_min_instances" {
  description = "Minimum API instances (0 = scale to zero)"
  type        = number
  default     = 0
}

variable "api_max_instances" {
  description = "Maximum API instances"
  type        = number
  default     = 10
}

# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# -----------------------------------------------------------------------------
# CLOUD BUILD (CI/CD)
# -----------------------------------------------------------------------------

variable "repository_type" {
  description = "Repository type: GITHUB or CLOUD_SOURCE_REPOSITORIES"
  type        = string
  default     = "GITHUB"
}

variable "github_owner" {
  description = "GitHub repository owner (username or org). Leave empty to skip trigger creation."
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
  default     = "imagen"
}

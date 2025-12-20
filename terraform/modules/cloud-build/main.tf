# =============================================================================
# CLOUD BUILD - CI/CD TRIGGERS
# =============================================================================
#
# Automatically triggers builds on git push.
#
# Supports:
#   - GitHub (Cloud Build GitHub App)
#   - Cloud Source Repositories
#
# =============================================================================

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "repository_type" {
  description = "Repository type: GITHUB or CLOUD_SOURCE_REPOSITORIES"
  type        = string
  default     = "GITHUB"
}

variable "github_owner" {
  description = "GitHub repository owner (username or org)"
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
  default     = ""
}

variable "branch_main" {
  description = "Main branch name for production deployments"
  type        = string
  default     = "main"
}

variable "branch_develop" {
  description = "Development branch name"
  type        = string
  default     = "develop"
}

variable "gke_cluster_name" {
  description = "GKE cluster name for worker deployments"
  type        = string
}

# =============================================================================
# CLOUD BUILD SERVICE ACCOUNT PERMISSIONS
# =============================================================================

# Get the Cloud Build service account
data "google_project" "project" {
  project_id = var.project_id
}

locals {
  cloud_build_sa = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

# Cloud Run Admin - Deploy API
resource "google_project_iam_member" "cloudbuild_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = local.cloud_build_sa
}

# Service Account User - Act as service accounts
resource "google_project_iam_member" "cloudbuild_sa_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = local.cloud_build_sa
}

# GKE Developer - Deploy to Kubernetes
resource "google_project_iam_member" "cloudbuild_gke_developer" {
  project = var.project_id
  role    = "roles/container.developer"
  member  = local.cloud_build_sa
}

# Artifact Registry Writer - Push images
resource "google_project_iam_member" "cloudbuild_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = local.cloud_build_sa
}

# Storage Admin - Access GCS
resource "google_project_iam_member" "cloudbuild_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = local.cloud_build_sa
}

# =============================================================================
# TRIGGER: PRODUCTION (main branch)
# =============================================================================

resource "google_cloudbuild_trigger" "production" {
  count = var.repository_type == "GITHUB" && var.github_owner != "" ? 1 : 0

  name        = "imagen-deploy-production"
  description = "Deploy to production on push to ${var.branch_main}"
  location    = var.region

  github {
    owner = var.github_owner
    name  = var.github_repo

    push {
      branch = "^${var.branch_main}$"
    }
  }

  filename = "cloudbuild.yaml"

  substitutions = {
    _REGION      = var.region
    _GKE_CLUSTER = var.gke_cluster_name
    _NAMESPACE   = "imagen"
    _ENV         = "prod"
  }

  include_build_logs = "INCLUDE_BUILD_LOGS_WITH_STATUS"
}

# =============================================================================
# TRIGGER: DEVELOPMENT (develop branch)
# =============================================================================

resource "google_cloudbuild_trigger" "development" {
  count = var.repository_type == "GITHUB" && var.github_owner != "" ? 1 : 0

  name        = "imagen-deploy-development"
  description = "Deploy to development on push to ${var.branch_develop}"
  location    = var.region

  github {
    owner = var.github_owner
    name  = var.github_repo

    push {
      branch = "^${var.branch_develop}$"
    }
  }

  filename = "cloudbuild.yaml"

  substitutions = {
    _REGION      = var.region
    _GKE_CLUSTER = "${var.gke_cluster_name}-dev"
    _NAMESPACE   = "imagen"
    _ENV         = "dev"
  }

  include_build_logs = "INCLUDE_BUILD_LOGS_WITH_STATUS"
}

# =============================================================================
# TRIGGER: PULL REQUEST (build only, no deploy)
# =============================================================================

resource "google_cloudbuild_trigger" "pull_request" {
  count = var.repository_type == "GITHUB" && var.github_owner != "" ? 1 : 0

  name        = "imagen-pr-check"
  description = "Build and test on pull request"
  location    = var.region

  github {
    owner = var.github_owner
    name  = var.github_repo

    pull_request {
      branch = "^${var.branch_main}$"
    }
  }

  filename = "cloudbuild-pr.yaml"

  include_build_logs = "INCLUDE_BUILD_LOGS_WITH_STATUS"
}

# =============================================================================
# OUTPUTS
# =============================================================================

output "production_trigger_id" {
  description = "Production trigger ID"
  value       = length(google_cloudbuild_trigger.production) > 0 ? google_cloudbuild_trigger.production[0].id : null
}

output "development_trigger_id" {
  description = "Development trigger ID"
  value       = length(google_cloudbuild_trigger.development) > 0 ? google_cloudbuild_trigger.development[0].id : null
}

output "cloud_build_sa" {
  description = "Cloud Build service account"
  value       = local.cloud_build_sa
}

output "setup_instructions" {
  description = "Setup instructions for GitHub integration"
  value       = <<-EOT
    
    ============================================================
    GITHUB INTEGRATION SETUP
    ============================================================
    
    1. Install Cloud Build GitHub App:
       https://github.com/apps/google-cloud-build
    
    2. Connect your repository in Cloud Console:
       https://console.cloud.google.com/cloud-build/triggers/connect
    
    3. Update terraform.tfvars with:
       github_owner = "your-username"
       github_repo  = "imagen"
    
    4. Re-run: terraform apply
    
    ============================================================
  EOT
}

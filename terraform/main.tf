# =============================================================================
# IMAGEN PLATFORM - MAIN TERRAFORM CONFIGURATION
# =============================================================================
#
# This file orchestrates all infrastructure modules.
# Each module is responsible for a specific GCP service.
#
# Deployment order (handled by depends_on):
# 1. APIs (enable required GCP services)
# 2. Storage, Pub/Sub, Firestore, Artifact Registry (in parallel)
# 3. IAM (service accounts)
# 4. GKE, Cloud Run (compute)
# 5. Autoscaling (HPA metrics)
#
# =============================================================================

terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Remote state in GCS (configure bucket in backend.tfvars)
  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# =============================================================================
# MODULE: APIs
# Enable required GCP services
# =============================================================================

module "apis" {
  source = "./modules/apis"

  project_id = var.project_id
}

# =============================================================================
# MODULE: STORAGE
# Cloud Storage bucket for input/output images
# =============================================================================

module "storage" {
  source = "./modules/storage"

  project_id         = var.project_id
  region             = var.region
  lifecycle_age_days = var.storage_lifecycle_days

  depends_on = [module.apis]
}

# =============================================================================
# MODULE: PUB/SUB
# Job queues (topics + subscriptions)
# =============================================================================

module "pubsub" {
  source = "./modules/pubsub"

  job_types            = var.job_types
  ack_deadline_seconds = 600 # 10 min for GPU jobs

  depends_on = [module.apis]
}

# =============================================================================
# MODULE: FIRESTORE
# Job state database
# =============================================================================

module "firestore" {
  source = "./modules/firestore"

  region = var.region

  depends_on = [module.apis]
}

# =============================================================================
# MODULE: ARTIFACT REGISTRY
# Docker image repository
# =============================================================================

module "artifact_registry" {
  source = "./modules/artifact-registry"

  project_id = var.project_id
  region     = var.region

  depends_on = [module.apis]
}

# =============================================================================
# MODULE: IAM
# Service accounts and permissions
# =============================================================================

module "iam" {
  source = "./modules/iam"

  project_id = var.project_id

  depends_on = [module.apis]
}

# =============================================================================
# MODULE: GKE
# Kubernetes cluster for GPU workers
# =============================================================================

module "gke" {
  source = "./modules/gke"

  project_id   = var.project_id
  region       = var.region
  cluster_name = var.gke_cluster_name

  depends_on = [module.apis]
}

# =============================================================================
# MODULE: CLOUD RUN
# API service
# =============================================================================

module "cloud_run" {
  source = "./modules/cloud-run"

  project_id            = var.project_id
  region                = var.region
  service_name          = "imagen-api"
  image                 = "${module.artifact_registry.api_image}:latest"
  service_account_email = module.iam.api_service_account_email
  gcs_bucket            = module.storage.bucket_name
  min_instances         = var.api_min_instances
  max_instances         = var.api_max_instances

  depends_on = [module.iam, module.storage, module.artifact_registry]
}

# =============================================================================
# MODULE: AUTOSCALING
# HPA metrics adapter IAM
# =============================================================================

module "autoscaling" {
  source = "./modules/autoscaling"

  project_id = var.project_id

  depends_on = [module.gke]
}


# =============================================================================
# MODULE: CLOUD BUILD
# CI/CD triggers for automatic deployment
# =============================================================================

module "cloud_build" {
  source = "./modules/cloud-build"

  project_id       = var.project_id
  region           = var.region
  repository_type  = var.repository_type
  github_owner     = var.github_owner
  github_repo      = var.github_repo
  gke_cluster_name = var.gke_cluster_name

  depends_on = [module.gke, module.artifact_registry, module.cloud_run]
}

# =============================================================================
# TERRAFORM BACKEND CONFIGURATION
# =============================================================================
#
# This configures where Terraform stores its state file.
# Using GCS ensures state is shared across team members and CI/CD.
#
# Usage:
#   terraform init -backend-config=backend.tfvars
#
# First time setup:
#   1. Create the bucket manually:
#      gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-terraform-state
#
#   2. Enable versioning:
#      gsutil versioning set on gs://YOUR_PROJECT_ID-terraform-state
#
# =============================================================================

bucket = "your-project-id-terraform-state"  # TODO: Replace
prefix = "imagen"

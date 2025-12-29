# =============================================================================
# SECRET MANAGER MODULE
# =============================================================================
#
# Manages secrets using Google Cloud Secret Manager instead of environment variables
#
# =============================================================================

# JWT Secret for authentication
resource "google_secret_manager_secret" "jwt_secret" {
  secret_id = "jwt-secret"
  project   = var.project_id

  replication {
    auto {}
  }

  labels = {
    purpose = "authentication"
  }
}

resource "google_secret_manager_secret_version" "jwt_secret" {
  secret      = google_secret_manager_secret.jwt_secret.id
  secret_data = var.jwt_secret
}

# API Keys (stored as JSON)
resource "google_secret_manager_secret" "api_keys" {
  secret_id = "api-keys"
  project   = var.project_id

  replication {
    auto {}
  }

  labels = {
    purpose = "api-authentication"
  }
}

resource "google_secret_manager_secret_version" "api_keys" {
  secret      = google_secret_manager_secret.api_keys.id
  secret_data = var.api_keys_json
}

# CORS Origins (stored as comma-separated list)
resource "google_secret_manager_secret" "cors_origins" {
  secret_id = "cors-origins"
  project   = var.project_id

  replication {
    auto {}
  }

  labels = {
    purpose = "api-security"
  }
}

resource "google_secret_manager_secret_version" "cors_origins" {
  secret      = google_secret_manager_secret.cors_origins.id
  secret_data = var.cors_origins
}

# =============================================================================
# IAM ACCESS
# =============================================================================

# Grant Cloud Run service account access to secrets
resource "google_secret_manager_secret_iam_member" "cloudrun_jwt_access" {
  count = var.cloudrun_service_account != null ? 1 : 0

  secret_id = google_secret_manager_secret.jwt_secret.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudrun_service_account}"
}

resource "google_secret_manager_secret_iam_member" "cloudrun_apikeys_access" {
  count = var.cloudrun_service_account != null ? 1 : 0

  secret_id = google_secret_manager_secret.api_keys.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudrun_service_account}"
}

resource "google_secret_manager_secret_iam_member" "cloudrun_cors_access" {
  count = var.cloudrun_service_account != null ? 1 : 0

  secret_id = google_secret_manager_secret.cors_origins.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.cloudrun_service_account}"
}

# Grant GKE workload identity access to secrets (for workers)
resource "google_secret_manager_secret_iam_member" "gke_jwt_access" {
  count = var.gke_workload_identity != null ? 1 : 0

  secret_id = google_secret_manager_secret.jwt_secret.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.gke_workload_identity}"
}

resource "google_secret_manager_secret_iam_member" "gke_apikeys_access" {
  count = var.gke_workload_identity != null ? 1 : 0

  secret_id = google_secret_manager_secret.api_keys.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.gke_workload_identity}"
}

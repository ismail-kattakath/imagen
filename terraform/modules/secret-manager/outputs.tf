output "jwt_secret_id" {
  description = "JWT secret ID"
  value       = google_secret_manager_secret.jwt_secret.secret_id
}

output "api_keys_secret_id" {
  description = "API keys secret ID"
  value       = google_secret_manager_secret.api_keys.secret_id
}

output "cors_origins_secret_id" {
  description = "CORS origins secret ID"
  value       = google_secret_manager_secret.cors_origins.secret_id
}

output "jwt_secret_name" {
  description = "JWT secret full resource name"
  value       = google_secret_manager_secret.jwt_secret.name
}

output "api_keys_secret_name" {
  description = "API keys secret full resource name"
  value       = google_secret_manager_secret.api_keys.name
}

output "cors_origins_secret_name" {
  description = "CORS origins secret full resource name"
  value       = google_secret_manager_secret.cors_origins.name
}

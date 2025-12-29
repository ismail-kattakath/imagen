output "security_policy_name" {
  description = "Cloud Armor security policy name"
  value       = google_compute_security_policy.waf_policy.name
}

output "security_policy_id" {
  description = "Cloud Armor security policy ID"
  value       = google_compute_security_policy.waf_policy.id
}

output "security_policy_self_link" {
  description = "Cloud Armor security policy self link"
  value       = google_compute_security_policy.waf_policy.self_link
}

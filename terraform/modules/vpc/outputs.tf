output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.vpc.name
}

output "network_id" {
  description = "VPC network ID"
  value       = google_compute_network.vpc.id
}

output "network_self_link" {
  description = "VPC network self link"
  value       = google_compute_network.vpc.self_link
}

output "gke_subnet_name" {
  description = "GKE subnet name"
  value       = google_compute_subnetwork.gke_subnet.name
}

output "gke_subnet_self_link" {
  description = "GKE subnet self link"
  value       = google_compute_subnetwork.gke_subnet.self_link
}

output "gke_pods_range_name" {
  description = "GKE pods secondary range name"
  value       = "gke-pods"
}

output "gke_services_range_name" {
  description = "GKE services secondary range name"
  value       = "gke-services"
}

output "vpc_connector_id" {
  description = "VPC connector ID"
  value       = google_vpc_access_connector.connector.id
}

output "vpc_connector_name" {
  description = "VPC connector name"
  value       = google_vpc_access_connector.connector.name
}

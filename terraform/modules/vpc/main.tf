# =============================================================================
# VPC MODULE - Private Networking
# =============================================================================
#
# Creates a private VPC with:
# - Private subnets
# - Cloud NAT for outbound traffic
# - VPC firewall rules
# - VPC Flow Logs
# - VPC connector for Cloud Run
#
# =============================================================================

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "${var.project_id}-vpc"
  auto_create_subnetworks = false
  project                 = var.project_id
}

# Private Subnet for GKE
resource "google_compute_subnetwork" "gke_subnet" {
  name          = "${var.project_id}-gke-subnet"
  ip_cidr_range = var.gke_subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc.id
  project       = var.project_id

  # Secondary IP ranges for GKE pods and services
  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = var.gke_pods_cidr
  }

  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = var.gke_services_cidr
  }

  # Enable VPC Flow Logs
  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }

  private_ip_google_access = true
}

# Private Subnet for Cloud Run VPC Connector
resource "google_compute_subnetwork" "vpc_connector_subnet" {
  name          = "${var.project_id}-vpc-connector-subnet"
  ip_cidr_range = var.vpc_connector_cidr
  region        = var.region
  network       = google_compute_network.vpc.id
  project       = var.project_id

  private_ip_google_access = true
}

# Cloud Router for Cloud NAT
resource "google_compute_router" "router" {
  name    = "${var.project_id}-router"
  region  = var.region
  network = google_compute_network.vpc.id
  project = var.project_id

  bgp {
    asn = 64514
  }
}

# Cloud NAT for outbound traffic
resource "google_compute_router_nat" "nat" {
  name                               = "${var.project_id}-nat"
  router                             = google_compute_router.router.name
  region                             = google_compute_router.router.region
  project                            = var.project_id
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# VPC Connector for Cloud Run
resource "google_vpc_access_connector" "connector" {
  name          = "${var.project_id}-vpc-connector"
  region        = var.region
  project       = var.project_id
  network       = google_compute_network.vpc.name
  ip_cidr_range = var.vpc_connector_cidr

  depends_on = [google_compute_subnetwork.vpc_connector_subnet]
}

# Firewall Rules

# Allow internal traffic within VPC
resource "google_compute_firewall" "allow_internal" {
  name    = "${var.project_id}-allow-internal"
  network = google_compute_network.vpc.name
  project = var.project_id

  allow {
    protocol = "tcp"
  }

  allow {
    protocol = "udp"
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [
    var.gke_subnet_cidr,
    var.gke_pods_cidr,
    var.gke_services_cidr,
    var.vpc_connector_cidr,
  ]

  priority = 1000
}

# Allow SSH from IAP (for debugging)
resource "google_compute_firewall" "allow_iap_ssh" {
  name    = "${var.project_id}-allow-iap-ssh"
  network = google_compute_network.vpc.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  # IAP's IP range
  source_ranges = ["35.235.240.0/20"]

  priority = 1000
}

# Allow health checks from Google Cloud load balancers
resource "google_compute_firewall" "allow_health_checks" {
  name    = "${var.project_id}-allow-health-checks"
  network = google_compute_network.vpc.name
  project = var.project_id

  allow {
    protocol = "tcp"
  }

  # Google Cloud health check IP ranges
  source_ranges = [
    "35.191.0.0/16",
    "130.211.0.0/22",
  ]

  priority = 1000
}

# Deny all other ingress traffic (default deny)
resource "google_compute_firewall" "deny_all_ingress" {
  name    = "${var.project_id}-deny-all-ingress"
  network = google_compute_network.vpc.name
  project = var.project_id

  deny {
    protocol = "all"
  }

  source_ranges = ["0.0.0.0/0"]
  priority      = 65534
}

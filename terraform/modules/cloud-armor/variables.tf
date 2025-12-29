variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "rate_limit_threshold" {
  description = "Requests per minute before rate limiting kicks in"
  type        = number
  default     = 100
}

variable "blocked_ip_ranges" {
  description = "List of IP ranges to block"
  type        = list(string)
  default     = []
}

variable "enable_adaptive_protection" {
  description = "Enable adaptive DDoS protection"
  type        = bool
  default     = true
}

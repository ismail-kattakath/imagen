# Pub/Sub topics and subscriptions for job queues

variable "job_types" {
  description = "List of job types (creates topic + subscription per type)"
  type        = list(string)
  default     = ["upscale", "enhance", "style-comic", "style-aged", "background-remove"]
}

variable "ack_deadline_seconds" {
  description = "ACK deadline for subscriptions (GPU jobs need longer)"
  type        = number
  default     = 600 # 10 minutes
}

variable "message_retention_duration" {
  description = "How long to retain unacked messages"
  type        = string
  default     = "86400s" # 24 hours
}

# Topics
resource "google_pubsub_topic" "jobs" {
  for_each = toset(var.job_types)
  name     = "${each.key}-jobs"
}

# Subscriptions
resource "google_pubsub_subscription" "jobs" {
  for_each = google_pubsub_topic.jobs

  name  = "${each.key}-jobs-sub"
  topic = each.value.name

  ack_deadline_seconds       = var.ack_deadline_seconds
  message_retention_duration = var.message_retention_duration

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  # Dead letter policy (optional - uncomment to enable)
  # dead_letter_policy {
  #   dead_letter_topic     = google_pubsub_topic.dead_letter.id
  #   max_delivery_attempts = 5
  # }
}

# Optional: Dead letter topic for failed messages
# resource "google_pubsub_topic" "dead_letter" {
#   name = "imagen-dead-letter"
# }

output "topics" {
  description = "Map of topic names to IDs"
  value       = { for k, v in google_pubsub_topic.jobs : k => v.id }
}

output "subscriptions" {
  description = "Map of subscription names to IDs"
  value       = { for k, v in google_pubsub_subscription.jobs : k => v.id }
}

output "topic_names" {
  description = "List of topic names"
  value       = [for t in google_pubsub_topic.jobs : t.name]
}

output "subscription_names" {
  description = "List of subscription names"
  value       = [for s in google_pubsub_subscription.jobs : s.name]
}

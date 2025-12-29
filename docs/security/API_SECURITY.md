# API Security & Observability

This document covers the production-ready security and observability features implemented in the Imagen API.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    REQUEST FLOW                                  │
└─────────────────────────────────────────────────────────────────┘

Client Request
      │
      ▼
┌─────────────────┐
│ Cloud Armor     │ → DDoS protection, WAF rules
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Load Balancer   │ → SSL/TLS termination
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                           │
│                                                                  │
│  1. RequestLoggingMiddleware → Request ID, timing, logging      │
│  2. MetricsMiddleware        → Prometheus metrics               │
│  3. RateLimitMiddleware      → Per-user rate limiting           │
│  4. SizeLimitMiddleware      → File/body size limits            │
│  5. CORS                     → Cross-origin requests            │
│                                                                  │
│  Route Handler:                                                  │
│  • Authentication (get_api_key / get_jwt_user)                  │
│  • File validation                                               │
│  • Business logic                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Authentication

### API Key Authentication

API keys are the primary authentication method for machine-to-machine access.

**Request:**
```bash
curl -X POST https://api.imagen.example/api/v1/images/upscale \
  -H "X-API-Key: sk-your-api-key" \
  -F "file=@image.jpg"
```

**Tiers:**
| Tier | Rate Limit | Max File Size | Daily Limit |
|------|------------|---------------|-------------|
| free | 10/min | 5 MB | 500 |
| pro | 60/min | 25 MB | 10,000 |
| enterprise | 300/min | 100 MB | 100,000 |

### JWT Authentication

JWT tokens are for user-based authentication (mobile/web apps).

**Request:**
```bash
curl -X POST https://api.imagen.example/api/v1/images/upscale \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -F "file=@image.jpg"
```

### Configuration

```bash
# Environment variables
JWT_SECRET=your-jwt-secret-min-32-chars
API_KEYS='[{"key": "sk-prod-key", "name": "Production", "tier": "enterprise"}]'
```

---

## Rate Limiting

### How It Works

Rate limiting uses a sliding window algorithm with Redis backend.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLIDING WINDOW                                │
└─────────────────────────────────────────────────────────────────┘

Time:     |-------- 60 seconds --------|
Requests: [x] [x] [x]    [x]  [x] [x] → 6 requests in window

Limit: 10/min
Result: ALLOWED (6 < 10)

Next request:
Requests: [x] [x] [x]    [x]  [x] [x] [x] → 7 requests
Result: ALLOWED (7 < 10)
```

### Response Headers

Every response includes rate limit information:

```
X-RateLimit-Limit: 60        # Max requests per window
X-RateLimit-Remaining: 45    # Requests remaining
X-RateLimit-Reset: 42        # Seconds until reset
```

### Rate Limit Exceeded

```json
HTTP/1.1 429 Too Many Requests
Retry-After: 30

{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please try again later.",
    "request_id": "abc123"
  }
}
```

### Custom Endpoint Limits

```python
from src.api.middleware import rate_limit

@router.post("/expensive-operation")
@rate_limit(requests=5, window=60)  # 5 per minute (stricter)
async def expensive_operation():
    ...
```

---

## Request Size Limits

### File Upload Limits

| Tier | Max File Size |
|------|---------------|
| anonymous | 2 MB |
| free | 5 MB |
| pro | 25 MB |
| enterprise | 100 MB |

### Image Dimension Limits

- Maximum: 8192 x 8192 pixels
- Minimum: 32 x 32 pixels

### Allowed File Types

- image/jpeg
- image/png
- image/webp
- image/gif

---

## Logging

### Structured Logging

All logs are JSON-formatted for Cloud Logging:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "severity": "INFO",
  "message": "Request completed",
  "request_id": "abc-123-def",
  "method": "POST",
  "path": "/api/v1/images/upscale",
  "status_code": 200,
  "duration_ms": 245.5,
  "logging.googleapis.com/trace": "projects/my-project/traces/abc-123-def"
}
```

### Request Tracing

Every request gets a unique ID:

1. Check `X-Request-ID` header (from client/load balancer)
2. Check `X-Cloud-Trace-Context` (from GCP)
3. Generate new UUID if none provided

Response includes:
```
X-Request-ID: abc-123-def
X-Response-Time: 245.50ms
```

### Using the Logger

```python
from src.api.middleware import ctx_logger

# Automatically includes request_id
ctx_logger.info("Processing image", job_id="123", size=1024)
```

---

## Metrics

### Prometheus Endpoint

```bash
GET /metrics
```

### Available Metrics

**Request Metrics:**
```
# Total requests
imagen_http_requests_total{method="POST", endpoint="/api/v1/images/upscale", status_code="200"}

# Latency histogram
imagen_http_request_duration_seconds{method="POST", endpoint="/api/v1/images/upscale"}

# In-progress requests
imagen_http_requests_in_progress{method="POST", endpoint="/api/v1/images/upscale"}
```

**Business Metrics:**
```
# Jobs created
imagen_jobs_created_total{job_type="upscale"}

# Jobs completed
imagen_jobs_completed_total{job_type="upscale", status="success"}

# Processing time
imagen_job_processing_seconds{job_type="upscale"}

# Queue depth
imagen_queue_depth{queue_name="upscale-jobs"}

# File sizes
imagen_upload_file_size_bytes{job_type="upscale"}
```

### Recording Business Metrics

```python
from src.api.middleware import metrics

# Record job creation
metrics.job_created("upscale")

# Record completion
metrics.job_completed("upscale", "success", duration=12.5)

# Record file upload
metrics.file_uploaded("upscale", size_bytes=1024000)
```

### Grafana Dashboard

Import the dashboard from `monitoring/grafana/imagen-dashboard.json`.

---

## Error Handling

### Standardized Error Format

All errors follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "request_id": "abc-123",
    "details": {
      "errors": [
        {
          "field": "file",
          "message": "File too large",
          "type": "value_error"
        }
      ]
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 422 | Request validation failed |
| `INVALID_IMAGE` | 400 | Invalid image file |
| `JOB_NOT_FOUND` | 404 | Job not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `QUOTA_EXCEEDED` | 429 | Usage quota exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |

### Custom Exceptions

```python
from src.api.middleware import JobNotFoundException, InvalidImageException

# Raise custom exception
raise JobNotFoundException(job_id="123")

# Raises:
# {
#   "error": {
#     "code": "JOB_NOT_FOUND",
#     "message": "Job not found: 123",
#     "details": {"job_id": "123"}
#   }
# }
```

---

## Configuration

### Environment Variables

```bash
# Authentication
JWT_SECRET=your-secret-key-min-32-chars
API_KEYS='[{"key":"sk-xxx","name":"Prod","tier":"enterprise"}]'

# Rate Limiting
REDIS_URL=redis://redis:6379

# CORS
CORS_ORIGINS='["https://app.example.com"]'

# Logging
DEBUG=false  # Set to true for development logging
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: imagen-api-config
  namespace: imagen
data:
  REDIS_URL: "redis://redis:6379"
  CORS_ORIGINS: '["https://app.example.com"]'
```

### Kubernetes Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: imagen-api-secrets
  namespace: imagen
type: Opaque
stringData:
  JWT_SECRET: "your-secret-key"
  API_KEYS: '[{"key":"sk-xxx","name":"Prod","tier":"enterprise"}]'
```

---

## Cloud Armor (GCP WAF)

### Terraform Configuration

```hcl
# terraform/modules/security/cloud-armor.tf

resource "google_compute_security_policy" "api_policy" {
  name = "imagen-api-policy"

  # Default rule - allow
  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
  }

  # Block known bad IPs
  rule {
    action   = "deny(403)"
    priority = "1000"
    match {
      expr {
        expression = "evaluatePreconfiguredExpr('xss-stable')"
      }
    }
  }

  # Rate limiting at edge
  rule {
    action   = "rate_based_ban"
    priority = "2000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action   = "allow"
      exceed_action    = "deny(429)"
      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }
    }
  }
}
```

---

## Testing

### Test Authentication

```bash
# Valid API key
curl -H "X-API-Key: dev-key-12345" http://localhost:8000/api/v1/images/limits

# Invalid API key
curl -H "X-API-Key: invalid" http://localhost:8000/api/v1/images/limits
# Returns: 401 Unauthorized

# Missing API key
curl http://localhost:8000/api/v1/images/upscale -F "file=@test.jpg"
# Returns: 401 Unauthorized
```

### Test Rate Limiting

```bash
# Hammer the endpoint
for i in {1..20}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -H "X-API-Key: dev-key-12345" \
    http://localhost:8000/api/v1/images/limits
done

# After 10 requests: 429 Too Many Requests
```

### Test Metrics

```bash
curl http://localhost:8000/metrics

# Output:
# imagen_http_requests_total{method="GET",endpoint="/api/v1/images/limits",status_code="200"} 10.0
# imagen_http_request_duration_seconds_bucket{...}
```

---

## Files

```
src/api/
├── main.py                 # Application with all middleware
├── middleware/
│   ├── __init__.py         # Package exports
│   ├── auth.py             # API key & JWT authentication
│   ├── rate_limit.py       # Rate limiting (Redis/in-memory)
│   ├── validation.py       # File & size validation
│   ├── logging.py          # Structured logging & tracing
│   ├── metrics.py          # Prometheus metrics
│   └── errors.py           # Error handling
└── routes/
    └── images.py           # Updated with auth
```

---

## Summary

| Feature | Implementation | Backend |
|---------|---------------|---------|
| Authentication | API Keys, JWT | In-memory / Database |
| Rate Limiting | Sliding window | Redis |
| Size Limits | Middleware | Application |
| Logging | Structured JSON | Cloud Logging |
| Metrics | Prometheus | /metrics endpoint |
| Tracing | Request ID | X-Request-ID header |
| Error Handling | Standardized | Application |
| WAF | Cloud Armor | GCP Edge |

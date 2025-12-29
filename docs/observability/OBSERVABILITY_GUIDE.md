# Observability Guide

This guide covers the complete observability setup for the Imagen platform.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OBSERVABILITY ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                              METRICS LAYER                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Cloud Run (API)                  GKE (Workers)                              │
│  ┌─────────────────┐             ┌─────────────────┐                        │
│  │ FastAPI App     │             │ GPU Workers     │                        │
│  │                 │             │                 │                        │
│  │ /metrics        │             │ /metrics        │◄── PodMonitoring       │
│  │ (prometheus)    │             │ (prometheus)    │    (GMP scrapes)       │
│  └────────┬────────┘             └────────┬────────┘                        │
│           │                               │                                  │
│           │ Built-in                      │ GMP Collector                    │
│           │ Cloud Run                     │                                  │
│           │ metrics                       ▼                                  │
│           │                    ┌─────────────────────┐                      │
│           └───────────────────▶│  Cloud Monitoring   │◄─────────────────────│
│                                │  (Managed Prom)     │                      │
│                                └─────────┬───────────┘                      │
│                                          │                                   │
│                          ┌───────────────┼───────────────┐                  │
│                          ▼               ▼               ▼                  │
│                    ┌──────────┐   ┌───────────┐   ┌──────────┐             │
│                    │ PromQL   │   │ Alerting  │   │ Grafana  │             │
│                    │ Queries  │   │ Policies  │   │ (opt)    │             │
│                    └──────────┘   └───────────┘   └──────────┘             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                              LOGGING LAYER                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  All Services → Structured JSON Logs → Cloud Logging → Log-based Metrics    │
│                                                                              │
│  Features:                                                                   │
│  • Request ID correlation across services                                    │
│  • JSON format for Cloud Logging integration                                 │
│  • Severity levels mapped to GCP                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                              TRACING LAYER                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Request ID propagated through:                                              │
│  HTTP Headers → Pub/Sub Message Attributes → Firestore Documents            │
│                                                                              │
│  Future: OpenTelemetry integration for distributed tracing                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Metrics Reference

### API Metrics (Cloud Run)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `imagen_http_requests_total` | Counter | method, endpoint, status_code | Total HTTP requests |
| `imagen_http_request_duration_seconds` | Histogram | method, endpoint | Request latency |
| `imagen_http_requests_in_progress` | Gauge | method, endpoint | Active requests |
| `imagen_errors_total` | Counter | type, endpoint | Total errors |
| `imagen_jobs_created_total` | Counter | job_type | Jobs submitted |
| `imagen_upload_file_size_bytes` | Histogram | job_type | Uploaded file sizes |

### Worker Metrics (GKE)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `imagen_jobs_completed_total` | Counter | job_type, status | Jobs processed |
| `imagen_job_processing_seconds` | Histogram | job_type | Processing duration |
| `imagen_jobs_in_progress` | Gauge | job_type | Active jobs |
| `imagen_model_load_seconds` | Histogram | model_name | Model load time |
| `imagen_worker_errors_total` | Counter | job_type, error_type | Worker errors |
| `imagen_job_slo_met_total` | Counter | job_type | Jobs within SLO |
| `imagen_job_slo_violated_total` | Counter | job_type | Jobs exceeding SLO |

### Recording Rules (Pre-computed)

| Rule | Expression | Description |
|------|------------|-------------|
| `imagen:jobs_completed:rate5m` | `rate(imagen_jobs_completed_total[5m])` | Jobs/second |
| `imagen:jobs_success_rate:ratio5m` | success / total | Success percentage |
| `imagen:job_processing:p50` | `histogram_quantile(0.50, ...)` | Median latency |
| `imagen:job_processing:p95` | `histogram_quantile(0.95, ...)` | P95 latency |
| `imagen:job_processing:p99` | `histogram_quantile(0.99, ...)` | P99 latency |

---

## Quick Start

### 1. Deploy Monitoring

```bash
# Deploy PodMonitoring and alerting rules
make monitoring-deploy

# Verify deployment
make monitoring-status
```

### 2. Check GMP is Running

```bash
# GMP pods should be running in gmp-system namespace
make monitoring-check-gmp
```

### 3. View Metrics in Cloud Console

1. Go to **Cloud Console** → **Monitoring** → **Metrics Explorer**
2. Click **PromQL** tab
3. Enter query: `imagen_jobs_completed_total`

---

## PromQL Query Examples

### Error Rate

```promql
# Error rate percentage (last 5 minutes)
sum(rate(imagen_jobs_completed_total{status="failed"}[5m])) 
/ 
sum(rate(imagen_jobs_completed_total[5m])) 
* 100
```

### Latency Percentiles

```promql
# P95 processing time by job type
histogram_quantile(0.95, 
  sum(rate(imagen_job_processing_seconds_bucket[5m])) by (le, job_type)
)
```

### Throughput

```promql
# Jobs processed per minute by type
sum(rate(imagen_jobs_completed_total[5m])) by (job_type) * 60
```

### SLO Compliance

```promql
# SLO compliance percentage
sum(imagen_job_slo_met_total) 
/ 
(sum(imagen_job_slo_met_total) + sum(imagen_job_slo_violated_total)) 
* 100
```

### Queue Health

```promql
# Current queue depth
imagen_queue_depth
```

---

## Alerting

### Configured Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| `ImagenHighErrorRate` | Error rate > 5% for 5m | Critical |
| `ImagenWorkerDown` | No healthy workers for 5m | Critical |
| `ImagenSlowProcessing` | P95 > 60s for 10m | Warning |
| `ImagenQueueBacklog` | Queue > 100 messages for 5m | Warning |
| `ImagenSLOBurnRateFast` | SLO violations > 2% for 15m | Warning |

### Setting Up Notification Channels

```bash
# Create email notification channel
gcloud alpha monitoring channels create \
  --display-name="Imagen Alerts" \
  --type=email \
  --channel-labels=email_address=team@example.com

# Create Slack notification channel
gcloud alpha monitoring channels create \
  --display-name="Imagen Slack" \
  --type=slack \
  --channel-labels=channel_name=#alerts
```

---

## Dashboards

### Create Dashboard in Cloud Console

1. Go to **Monitoring** → **Dashboards** → **Create Dashboard**
2. Add widgets with these PromQL queries:

**Widget 1: Request Rate**
```promql
sum(rate(imagen_http_requests_total[1m])) by (endpoint)
```

**Widget 2: Error Rate**
```promql
sum(rate(imagen_jobs_completed_total{status="failed"}[5m])) / sum(rate(imagen_jobs_completed_total[5m])) * 100
```

**Widget 3: P95 Latency**
```promql
histogram_quantile(0.95, sum(rate(imagen_job_processing_seconds_bucket[5m])) by (le, job_type))
```

**Widget 4: Jobs In Progress**
```promql
sum(imagen_jobs_in_progress) by (job_type)
```

---

## Troubleshooting

### Metrics Not Appearing

```bash
# 1. Check PodMonitoring exists
kubectl get podmonitoring -n imagen

# 2. Check pods have correct labels
kubectl get pods -n imagen --show-labels | grep upscale

# 3. Check metrics endpoint directly
kubectl port-forward -n imagen deployment/upscale-worker 8080
curl http://localhost:8080/metrics

# 4. Check GMP collector logs
kubectl logs -n gmp-system -l app.kubernetes.io/name=collector
```

### High Cardinality Warnings

If you see cardinality warnings:
1. Check for unbounded labels (user IDs, request IDs)
2. Review `_normalize_path()` in metrics middleware
3. Reduce histogram bucket count if needed

### Missing Worker Metrics

```bash
# Verify workers expose port 8080
kubectl get pods -n imagen -o jsonpath='{.items[*].spec.containers[*].ports}'

# Check worker is running and healthy
kubectl describe pod -n imagen -l app=upscale-worker
```

---

## SLO Configuration

### Current SLO Thresholds

| Job Type | Latency SLO | Current Target |
|----------|-------------|----------------|
| upscale | 30 seconds | 99% |
| enhance | 45 seconds | 99% |
| style-comic | 60 seconds | 99% |
| style-aged | 60 seconds | 99% |
| background-remove | 15 seconds | 99% |

### Adjusting SLOs

Edit `src/workers/base.py`:

```python
SLO_THRESHOLDS = {
    "upscale": 30,        # Change these values
    "enhance": 45,
    "style-comic": 60,
    "style-aged": 60,
    "background-remove": 15,
}
```

---

## Files

```
k8s/monitoring/
├── kustomization.yaml       # Kustomize configuration
├── pod-monitoring.yaml      # PodMonitoring for all workers
├── rules.yaml               # Recording & alerting rules
└── README.md                # Detailed documentation

src/api/middleware/
├── metrics.py               # API Prometheus metrics
└── logging.py               # Structured logging

src/workers/
└── base.py                  # Worker metrics (updated)
```

---

*Last updated: December 2024*

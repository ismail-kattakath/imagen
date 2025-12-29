# Google Managed Prometheus (GMP) Monitoring

This directory contains Kubernetes manifests for Google Managed Prometheus integration.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GMP ARCHITECTURE                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  GKE Cluster                                        Google Cloud
┌─────────────────────────────────────────┐      ┌─────────────────────────┐
│                                         │      │                         │
│  ┌─────────────────┐                    │      │   Cloud Monitoring      │
│  │ upscale-worker  │◄──┐                │      │   (Managed Backend)     │
│  │   :8080/metrics │   │                │      │                         │
│  └─────────────────┘   │                │      │   ┌─────────────────┐   │
│                        │  PodMonitoring │      │   │ Prometheus      │   │
│  ┌─────────────────┐   │  (scrape)      │      │   │ Data Store      │   │
│  │ enhance-worker  │◄──┤                │      │   │                 │   │
│  │   :8080/metrics │   │                │ ────▶│   │ - 24mo retention│   │
│  └─────────────────┘   │                │      │   │ - Auto-scaled   │   │
│                        │                │      │   │ - PromQL API    │   │
│  ┌─────────────────┐   │                │      │   └─────────────────┘   │
│  │ comic-worker    │◄──┤                │      │           │             │
│  │   :8080/metrics │   │                │      │           ▼             │
│  └─────────────────┘   │                │      │   ┌─────────────────┐   │
│                        │                │      │   │ Cloud Console   │   │
│  ┌─────────────────┐   │                │      │   │ Dashboards      │   │
│  │ GMP Collector   │◄──┘                │      │   └─────────────────┘   │
│  │ (auto-deployed) │                    │      │           │             │
│  └─────────────────┘                    │      │           ▼             │
│                                         │      │   ┌─────────────────┐   │
└─────────────────────────────────────────┘      │   │ Grafana         │   │
                                                 │   │ (optional)      │   │
                                                 │   └─────────────────┘   │
                                                 │                         │
                                                 └─────────────────────────┘
```

---

## Components

### PodMonitoring Resources

Each worker has a `PodMonitoring` resource that tells GMP to scrape its `/metrics` endpoint:

| Resource | Target | Port | Interval |
|----------|--------|------|----------|
| `upscale-worker-monitoring` | upscale-worker pods | 8080 | 30s |
| `enhance-worker-monitoring` | enhance-worker pods | 8080 | 30s |
| `comic-worker-monitoring` | comic-worker pods | 8080 | 30s |
| `style-aged-worker-monitoring` | style-aged-worker pods | 8080 | 30s |
| `background-remove-worker-monitoring` | background-remove-worker pods | 8080 | 30s |

### Metrics Exposed

Workers expose these Prometheus metrics at `/metrics`:

```
# Business Metrics
imagen_jobs_completed_total{job_type, status}
imagen_job_processing_seconds{job_type}
imagen_queue_depth{queue_name}

# Worker Health
imagen_model_load_seconds{model_name}
imagen_gpu_memory_used_bytes
imagen_worker_errors_total{error_type}
```

---

## Prerequisites

### 1. Enable GMP on GKE Cluster

GMP is enabled by default on GKE Autopilot. For Standard GKE:

```bash
gcloud container clusters update CLUSTER_NAME \
  --region=REGION \
  --enable-managed-prometheus
```

### 2. Verify GMP is Running

```bash
kubectl get pods -n gmp-system

# Expected output:
# NAME                              READY   STATUS
# collector-xxxxx                   2/2     Running
# gmp-operator-xxxxx                1/1     Running
```

---

## Deployment

### Apply All Monitoring Resources

```bash
kubectl apply -k k8s/monitoring/
```

### Verify PodMonitoring

```bash
kubectl get podmonitoring -n imagen

# Expected output:
# NAME                              AGE
# upscale-worker-monitoring         1m
# enhance-worker-monitoring         1m
# ...
```

### Check Targets Being Scraped

```bash
# Port-forward to GMP UI
kubectl -n gmp-system port-forward svc/frontend 9090

# Open http://localhost:9090/targets
```

---

## Querying Metrics

### Using PromQL in Cloud Console

1. Go to Cloud Console → Monitoring → Metrics Explorer
2. Click "PromQL" tab
3. Enter query:

```promql
# Job completion rate by type
sum(rate(imagen_jobs_completed_total[5m])) by (job_type)

# P95 processing time
histogram_quantile(0.95, 
  sum(rate(imagen_job_processing_seconds_bucket[5m])) by (le, job_type)
)

# Error rate
sum(rate(imagen_jobs_completed_total{status="failed"}[5m])) 
/ 
sum(rate(imagen_jobs_completed_total[5m])) * 100
```

### Using gcloud CLI

```bash
# Query via API
gcloud monitoring query --project=PROJECT_ID \
  'fetch prometheus_target
   | metric "prometheus.googleapis.com/imagen_jobs_completed_total/counter"
   | every 1m'
```

---

## Alerting

### Create Alert Policy

```bash
# Example: Alert when error rate > 5%
gcloud alpha monitoring policies create \
  --display-name="Imagen High Error Rate" \
  --condition-display-name="Error rate > 5%" \
  --condition-filter='
    resource.type="prometheus_target" AND
    metric.type="prometheus.googleapis.com/imagen_jobs_completed_total/counter" AND
    metric.labels.status="failed"
  ' \
  --condition-threshold-value=0.05 \
  --notification-channels=CHANNEL_ID
```

### Recommended Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Error Rate | error_rate > 5% for 5m | Critical |
| Slow Processing | p95_latency > 60s for 10m | Warning |
| Queue Backlog | queue_depth > 100 for 5m | Warning |
| Worker Down | absent(up{app=~".*-worker"}) | Critical |

---

## Grafana Integration (Optional)

GMP can be used as a Grafana data source:

### 1. Create Service Account

```bash
gcloud iam service-accounts create grafana-gmp \
  --display-name="Grafana GMP Reader"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:grafana-gmp@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/monitoring.viewer"
```

### 2. Configure Grafana Data Source

```yaml
# Grafana data source configuration
apiVersion: 1
datasources:
  - name: Google Managed Prometheus
    type: prometheus
    access: proxy
    url: https://monitoring.googleapis.com/v1/projects/PROJECT_ID/location/global/prometheus
    jsonData:
      httpMethod: POST
      authenticationType: gce
```

---

## Cloud Run API Metrics

The Cloud Run API service cannot use PodMonitoring (it's not in GKE). 

**Options for Cloud Run metrics:**

1. **Built-in Cloud Run metrics** (automatic)
   - Request count, latency, instance count
   - Available in Cloud Monitoring without configuration

2. **Custom metrics via OpenTelemetry** (requires code changes)
   - Add `opentelemetry-exporter-gcp-monitoring` package
   - Export Prometheus metrics to Cloud Monitoring

3. **Prometheus Pushgateway** (additional infrastructure)
   - Deploy pushgateway in GKE
   - Configure API to push metrics on shutdown

See `docs/observability/CLOUD_RUN_METRICS.md` for implementation details.

---

## Troubleshooting

### Metrics Not Appearing

```bash
# 1. Check PodMonitoring status
kubectl describe podmonitoring upscale-worker-monitoring -n imagen

# 2. Check collector logs
kubectl logs -n gmp-system -l app.kubernetes.io/name=collector

# 3. Verify pods have correct labels
kubectl get pods -n imagen -l app=upscale-worker --show-labels

# 4. Test metrics endpoint directly
kubectl port-forward -n imagen deployment/upscale-worker 8080
curl http://localhost:8080/metrics
```

### High Cardinality Warning

If you see cardinality warnings:

1. Check for unbounded labels (user IDs, request IDs)
2. Review `_normalize_path()` in metrics middleware
3. Consider dropping high-cardinality metrics

---

## Cost Considerations

GMP pricing (as of 2024):

| Tier | Samples/Month | Price |
|------|---------------|-------|
| Free | 0-50 billion | $0 |
| Standard | 50B+ | $0.15/million samples |

**Cost estimation for Imagen:**
- 5 workers × 100 metrics × 2 samples/min = 1,000 samples/min
- Monthly: ~43 million samples
- Cost: **Free tier** (well under 50B)

---

## Files

```
k8s/monitoring/
├── kustomization.yaml              # Kustomize configuration
├── pod-monitoring.yaml             # PodMonitoring for all workers
├── rules.yaml                      # Recording & alerting rules (optional)
└── README.md                       # This file
```

---

*Last updated: December 2024*

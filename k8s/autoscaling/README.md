# Auto-Scaling Configuration

This directory contains Kubernetes manifests for auto-scaling GPU workers based on Pub/Sub queue depth.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTO-SCALING FLOW                             │
└─────────────────────────────────────────────────────────────────┘

   Pub/Sub Queue              Metrics Adapter           HPA
   (num_undelivered           (reads metrics)        (scales pods)
    _messages)
        │                          │                      │
        │    "50 messages"         │                      │
        ├─────────────────────────▶│                      │
        │                          │   "50 messages"      │
        │                          ├─────────────────────▶│
        │                          │                      │
        │                          │              ┌───────┴───────┐
        │                          │              │ 50 msgs / 2   │
        │                          │              │ = 25 replicas │
        │                          │              │ (max 10)      │
        │                          │              └───────┬───────┘
        │                          │                      │
        │                          │                      ▼
        │                          │              Deployment
        │                          │              replicas: 1→10
        │                          │                      │
        │                          │                      ▼
        │                          │              GKE Autopilot
        │                          │              provisions 10
        │                          │              GPU nodes
```

## Components

### 1. Custom Metrics Adapter (`custom-metrics-adapter.yaml`)

Bridges Google Cloud Monitoring metrics to Kubernetes HPA.

- Reads Pub/Sub metrics from Cloud Monitoring
- Exposes them via Kubernetes External Metrics API
- HPA can then use these metrics for scaling decisions

### 2. HorizontalPodAutoscaler (HPA)

One HPA per worker type:
- `upscale-hpa.yaml`
- `enhance-hpa.yaml`
- `comic-hpa.yaml`
- `background-remove-hpa.yaml`

Each HPA watches its corresponding Pub/Sub subscription.

## Configuration

### Scaling Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `minReplicas` | 1 | Minimum workers (always running) |
| `maxReplicas` | 10 | Maximum workers (cost protection) |
| `averageValue` | 2 | Target messages per worker |

### Scaling Behavior

**Scale Up:**
- Stabilization: 60 seconds (wait before scaling up)
- Max pods added: 4 at a time
- Period: Every 60 seconds

**Scale Down:**
- Stabilization: 300 seconds (wait 5 min before scaling down)
- Max pods removed: 1 at a time
- Period: Every 120 seconds

### Why These Settings?

```
Scale Up (Aggressive):
- Fast response to traffic spikes
- 60s stabilization prevents flapping
- Add up to 4 pods quickly

Scale Down (Conservative):
- GPU nodes are expensive to provision
- 5 min wait ensures traffic is truly gone
- Remove 1 at a time to avoid over-scaling down
```

## Deployment

### Step 1: Deploy Metrics Adapter

```bash
# Replace PROJECT_ID in the manifest
sed -i 's/PROJECT_ID/your-project-id/g' k8s/autoscaling/custom-metrics-adapter.yaml

# Apply
kubectl apply -f k8s/autoscaling/custom-metrics-adapter.yaml
```

### Step 2: Deploy Terraform IAM

```bash
cd terraform

# Add module to main.tf:
# module "autoscaling" {
#   source     = "./modules/autoscaling"
#   project_id = var.project_id
# }

terraform apply -var-file=environments/dev.tfvars
```

### Step 3: Deploy HPAs

```bash
kubectl apply -f k8s/autoscaling/upscale-hpa.yaml
kubectl apply -f k8s/autoscaling/enhance-hpa.yaml
kubectl apply -f k8s/autoscaling/comic-hpa.yaml
kubectl apply -f k8s/autoscaling/background-remove-hpa.yaml
```

## Verification

### Check HPA Status

```bash
kubectl get hpa -n imagen

# Expected output:
# NAME                          REFERENCE                    TARGETS   MINPODS   MAXPODS   REPLICAS
# upscale-worker-hpa            Deployment/upscale-worker    0/2       1         10        1
# enhance-worker-hpa            Deployment/enhance-worker    0/2       1         10        1
```

### Check Metrics

```bash
# Verify metrics adapter is working
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1" | jq .

# Check specific Pub/Sub metric
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/imagen/pubsub.googleapis.com|subscription|num_undelivered_messages" | jq .
```

### Watch Scaling in Action

```bash
# Terminal 1: Watch HPA
watch kubectl get hpa -n imagen

# Terminal 2: Watch pods
watch kubectl get pods -n imagen

# Terminal 3: Send test requests
for i in {1..20}; do
  curl -X POST http://API_URL/api/v1/images/upscale -F "file=@test.jpg"
done
```

## Cost Optimization

### Scale to Zero (Optional)

By default, `minReplicas: 1` keeps one worker always running. To scale to zero:

```yaml
# In HPA manifest
spec:
  minReplicas: 0  # ← Allows scale to zero
```

**Tradeoff:**
- Pro: No cost when idle
- Con: Cold start when first request arrives (~2-5 min for GPU node + model load)

### Max Replicas Limit

Adjust based on budget:

```yaml
spec:
  maxReplicas: 5   # Lower = lower max cost
  maxReplicas: 20  # Higher = handle more traffic
```

### Cost Estimates

| Workers | GPU Type | Cost/Hour | Max/Month |
|---------|----------|-----------|-----------|
| 1 | T4 Spot | $0.35 | ~$250 |
| 5 | T4 Spot | $1.75 | ~$1,250 |
| 10 | T4 Spot | $3.50 | ~$2,500 |

## Troubleshooting

### HPA Shows "Unknown" Metrics

```bash
# Check metrics adapter is running
kubectl get pods -n custom-metrics

# Check adapter logs
kubectl logs -n custom-metrics -l app=custom-metrics-stackdriver-adapter

# Verify IAM permissions
gcloud projects get-iam-policy PROJECT_ID --filter="bindings.members:metrics-adapter"
```

### HPA Not Scaling

```bash
# Describe HPA for events
kubectl describe hpa upscale-worker-hpa -n imagen

# Check if metric exists
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/imagen/pubsub.googleapis.com|subscription|num_undelivered_messages?labelSelector=resource.labels.subscription_id=upscale-jobs-sub"
```

### Pods Pending (No GPU Nodes)

```bash
# Check node pool status
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-tesla-t4

# Check pod events
kubectl describe pod -n imagen -l app=upscale-worker

# GKE Autopilot may take 2-5 min to provision GPU nodes
```

## Files

```
k8s/autoscaling/
├── custom-metrics-adapter.yaml    # Metrics adapter deployment
├── upscale-hpa.yaml               # Upscale worker HPA
├── enhance-hpa.yaml               # Enhance worker HPA
├── comic-hpa.yaml                 # Comic worker HPA
├── background-remove-hpa.yaml     # Background remove worker HPA
└── README.md                      # This file

terraform/modules/autoscaling/
└── main.tf                        # IAM for metrics adapter
```

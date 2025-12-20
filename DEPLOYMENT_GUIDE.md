# Imagen Deployment Guide

Complete guide for deploying the Imagen AI image processing platform.

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- GCP account with billing enabled
- gcloud CLI configured
- kubectl installed
- Terraform 1.0+

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
cd imagen

# Install Python dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
# The .env file has been created from .env.example
# Edit it with your actual GCP project details:
nano .env

# Update these values:
# GOOGLE_CLOUD_PROJECT=your-actual-project-id
# GCS_BUCKET=your-actual-bucket-name
```

### 3. Start Local Services

```bash
# Start Redis and MinIO (for local development)
make dev

# Or manually:
docker compose -f docker/docker-compose.yml up -d
```

### 4. Run API (Development Mode)

```bash
# The API will run without GCP validation in debug mode
make api

# Or manually:
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test API

```bash
# Health check
curl http://localhost:8000/health

# Should return: {"status":"healthy"}
```

## Workers (Local Development)

Workers require GPU and can run without GCP in development mode:

```bash
# Terminal 1 - Upscale worker
make worker-upscale

# Terminal 2 - Enhance worker
make worker-enhance

# Terminal 3 - Comic style worker
make worker-comic

# Terminal 4 - Background removal worker
make worker-background-remove
```

**Note:** Workers will fail without GPU unless `DEVICE=cpu` is set in `.env`.

## Production Deployment to GCP

### Step 1: Set Up Infrastructure with Terraform

```bash
cd terraform

# Initialize Terraform
terraform init

# Review the plan
terraform plan -var-file=environments/prod.tfvars

# Apply infrastructure
terraform apply -var-file=environments/prod.tfvars
```

This creates:
- Cloud Storage bucket for images
- Pub/Sub topics and subscriptions
- Firestore database
- Cloud Run service for API
- Artifact Registry repository
- Service accounts with IAM roles

### Step 2: Build and Push Docker Images

```bash
# Using Cloud Build (recommended)
gcloud builds submit --config=cloudbuild.yaml

# Or manually
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1

# Build images
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/imagen/api:latest -f docker/Dockerfile.api .
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/imagen/worker:latest -f docker/Dockerfile.worker .

# Push images
docker push $REGION-docker.pkg.dev/$PROJECT_ID/imagen/api:latest
docker push $REGION-docker.pkg.dev/$PROJECT_ID/imagen/worker:latest
```

### Step 3: Deploy Workers to GKE

```bash
# Update ConfigMap with your project details
nano k8s/base/configmap.yaml
# Change GOOGLE_CLOUD_PROJECT and GCS_BUCKET

# Update worker manifests with your project ID
sed -i '' "s/PROJECT_ID/$PROJECT_ID/g" k8s/workers/*.yaml

# Apply Kubernetes manifests
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/workload-identity.yaml
kubectl apply -f k8s/base/configmap.yaml
kubectl apply -f k8s/base/pvc.yaml

# Deploy workers
kubectl apply -f k8s/workers/upscale-worker.yaml
kubectl apply -f k8s/workers/enhance-worker.yaml
kubectl apply -f k8s/workers/comic-worker.yaml
kubectl apply -f k8s/workers/background-remove-worker.yaml

# Check deployment status
kubectl get pods -n imagen
```

### Step 4: Verify Deployment

```bash
# Get Cloud Run API URL
gcloud run services describe imagen-api --region us-central1 --format 'value(status.url)'

# Test the API
export API_URL=$(gcloud run services describe imagen-api --region us-central1 --format 'value(status.url)')
curl $API_URL/health
```

## Testing the Full Pipeline

### Submit a Job

```bash
# Upload an image for upscaling
curl -X POST "$API_URL/api/v1/images/upscale" \
  -F "file=@test_image.jpg" \
  -F "prompt=high quality, detailed"

# Response will include job_id
# {"job_id":"abc-123-def","status":"queued","message":"Job queued successfully"}
```

### Check Job Status

```bash
# Replace JOB_ID with the actual job ID from above
curl "$API_URL/api/v1/jobs/JOB_ID"

# When completed, you'll get a signed URL to download the result
```

## Available Endpoints

- `POST /api/v1/images/upscale` - 4x image upscaling
- `POST /api/v1/images/enhance` - Enhance image quality
- `POST /api/v1/images/style/comic` - Convert to comic style
- `POST /api/v1/images/background/remove` - Remove background
- `GET /api/v1/jobs/{job_id}` - Get job status
- `GET /health` - Health check
- `GET /ready` - Readiness check

## Cost Monitoring

**IMPORTANT:** GPU workers are expensive!

- T4 GPU: ~$0.35/hour = ~$250/month per worker
- Total estimated cost: $300-500/month with 1-2 workers

### Set up cost alerts:

```bash
# Create a budget alert
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Imagen Monthly Budget" \
  --budget-amount=500USD \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90
```

## Troubleshooting

### Workers not processing jobs

```bash
# Check worker logs
kubectl logs -n imagen -l app=upscale-worker --tail=100

# Check Pub/Sub subscriptions
gcloud pubsub subscriptions list

# Check if messages are stuck
gcloud pubsub subscriptions pull upscale-jobs-sub --limit=5
```

### API errors

```bash
# Check Cloud Run logs
gcloud run services logs read imagen-api --limit=50

# Check IAM permissions
gcloud projects get-iam-policy $PROJECT_ID
```

### Out of memory errors

Increase worker memory in `k8s/workers/*.yaml`:

```yaml
resources:
  limits:
    memory: "24Gi"  # Increase from 16Gi
```

## Development Tips

### Run tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Code quality

```bash
# Lint code
make lint

# Format code
make format
```

### Clean up

```bash
# Remove Python cache files
make clean

# Stop local services
docker compose -f docker/docker-compose.yml down

# Remove volumes
docker compose -f docker/docker-compose.yml down -v
```

## Scaling

### Scale workers

```bash
# Scale up upscale workers to 3 replicas
kubectl scale deployment upscale-worker -n imagen --replicas=3

# Or edit the deployment
kubectl edit deployment upscale-worker -n imagen
```

### Scale API

Cloud Run auto-scales, but you can adjust:

```bash
gcloud run services update imagen-api \
  --region us-central1 \
  --min-instances=1 \
  --max-instances=20
```

## Security Best Practices

1. **Never commit .env file** - Already in .gitignore
2. **Use Workload Identity** - Configured in k8s/base/workload-identity.yaml
3. **Enable VPC** - Consider deploying in VPC for production
4. **Restrict API access** - Add authentication middleware
5. **Rotate credentials** - Regularly rotate service account keys

## Monitoring (Recommended)

Add Cloud Monitoring integration:

```python
# In src/core/logging.py
from google.cloud import logging as cloud_logging

client = cloud_logging.Client()
client.setup_logging()
```

## Next Steps

- [ ] Set up monitoring dashboards
- [ ] Configure alerting for errors
- [ ] Add rate limiting to API
- [ ] Implement authentication
- [ ] Set up CI/CD pipeline
- [ ] Add integration tests
- [ ] Configure backup policies

## Support

For issues, check:
- Logs: `kubectl logs -n imagen <pod-name>`
- Events: `kubectl get events -n imagen`
- GCP Console: Cloud Run, GKE, Pub/Sub pages

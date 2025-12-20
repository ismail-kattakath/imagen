# Your First Production Deployment

Step-by-step guide to deploying Imagen to Google Cloud Platform for the first time.

## Before You Begin

⚠️ **IMPORTANT**: Complete the [Setup Checklist](SETUP_CHECKLIST.md) first to ensure all configuration files are updated with your project ID!

### Prerequisites Checklist

- [ ] GCP account with billing enabled
- [ ] `gcloud` CLI installed and authenticated
- [ ] `kubectl` installed
- [ ] `terraform` installed (v1.0+)
- [ ] Project cloned locally
- [ ] **[Setup Checklist](SETUP_CHECKLIST.md) completed** ⭐

### Estimated Time

- **Initial setup**: 30-45 minutes
- **Infrastructure deployment**: 15-20 minutes
- **Application deployment**: 10-15 minutes
- **Total**: ~1-1.5 hours

### Estimated Cost

**Monthly costs** (minimal production deployment with 1 T4 GPU worker):

| Component | Cost/Month |
|-----------|------------|
| Cloud Run API (1 vCPU, 512MB) | ~$50 |
| GKE + T4 GPU Worker (Spot) | ~$250 |
| Cloud Storage + Firestore | ~$7 |
| Pub/Sub + Networking | ~$10 |
| Artifact Registry | ~$5 |
| **Total** | **~$320/month** |

**Note**: Using Spot VMs can reduce GPU costs by ~70%. See [Cost Optimization](../infrastructure/COST_OPTIMIZATION.md) for savings strategies.

⚠️ **Set up billing alerts (recommended: $400/month) before proceeding!**

---

## Step 1: GCP Project Setup

### Create New Project

```bash
# Set your project ID
export PROJECT_ID="imagen-prod-$(date +%s)"
echo $PROJECT_ID

# Create project
gcloud projects create $PROJECT_ID --name="Imagen Production"

# Set as default
gcloud config set project $PROJECT_ID

# Link billing account (replace BILLING_ACCOUNT_ID)
gcloud billing accounts list
gcloud billing projects link $PROJECT_ID \
  --billing-account=BILLING_ACCOUNT_ID
```

### Enable Required APIs

```bash
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  storage.googleapis.com \
  firestore.googleapis.com \
  pubsub.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com
```

### Set Up Billing Alerts

```bash
# Go to: https://console.cloud.google.com/billing/budgets
# Create budget alert for $200/month with alerts at 50%, 75%, 90%, 100%
```

---

## Step 2: Create GCS Bucket

```bash
# Create bucket for image storage
export BUCKET_NAME="${PROJECT_ID}-images"
gsutil mb -p $PROJECT_ID -l us-central1 gs://$BUCKET_NAME

# Verify
gsutil ls
```

---

## Step 3: Update Configuration Files

⚠️ **CRITICAL**: You must update configuration files with your actual project ID before deployment!

Follow the manual steps below to update all configuration files:

#### 1. Update Terraform Variables

**File**: `terraform/environments/prod.tfvars`

```bash
# Edit the file
nano terraform/environments/prod.tfvars

# BEFORE (placeholder - WILL NOT WORK)
project_id = "your-project-id"

# AFTER (your actual project ID from Step 1)
project_id = "imagen-prod-1703094123"  # Use YOUR actual PROJECT_ID
```

#### 2. Update .env File

```bash
nano .env

# BEFORE
GOOGLE_CLOUD_PROJECT=your-project-id
GCS_BUCKET=your-bucket-name

# AFTER (use values from Step 1 and Step 2)
GOOGLE_CLOUD_PROJECT=imagen-prod-1703094123
GCS_BUCKET=imagen-prod-1703094123-images
```

#### 3. Update Kubernetes Overlays

**File**: `k8s/overlays/prod/kustomization.yaml`

```bash
nano k8s/overlays/prod/kustomization.yaml

# Replace ALL occurrences of YOUR_PROJECT_ID (3 places)
# Find: YOUR_PROJECT_ID
# Replace with: imagen-prod-1703094123  # Your actual PROJECT_ID
```

**File**: `k8s/base/workload-identity.yaml`

```bash
nano k8s/base/workload-identity.yaml

# Replace YOUR_PROJECT_ID with your actual project ID
```

### Verify All Placeholders Replaced

```bash
# This should return NO results
grep -r "your-project-id" . --exclude-dir=.git --exclude-dir=docs
grep -r "YOUR_PROJECT_ID" . --exclude-dir=.git --exclude-dir=docs

# If you see any results, update those files!
```

✅ **Checkpoint**: All configuration files updated with actual project ID

---

## Step 4: Deploy Infrastructure with Terraform

Now that configuration is updated, deploy the infrastructure:

```bash
cd terraform

# Initialize
terraform init

# Create workspace (optional but recommended)
terraform workspace new prod

# Plan - using the prod.tfvars file we just updated!
terraform plan -var-file=environments/prod.tfvars -out=tfplan

# Review the plan carefully!
# Verify it's using YOUR project ID, not "your-project-id"

# Apply
terraform apply tfplan
```

⚠️ **Important**: The `project_id` comes from `terraform/environments/prod.tfvars` which you updated in Step 3!

### What This Creates

- ✅ GKE Autopilot cluster
- ✅ Pub/Sub topics and subscriptions (5 total)
- ✅ Firestore database
- ✅ Service accounts with appropriate permissions
- ✅ Artifact Registry for container images

---

## Step 5: Configure kubectl

```bash
# Get cluster credentials
gcloud container clusters get-credentials imagen-cluster \
  --region=us-central1 \
  --project=$PROJECT_ID

# Verify connection
kubectl get nodes
```

---

## Step 6: Build and Push Container Images

### Option A: Using Cloud Build (Recommended)

```bash
cd ..  # Back to project root

# Submit build
gcloud builds submit \
  --config=cloudbuild.yaml \
  --project=$PROJECT_ID

# This builds both API and worker images
# Takes ~10-15 minutes
```

### Option B: Build Locally

```bash
# Build API image
docker build -f docker/Dockerfile.api \
  -t gcr.io/$PROJECT_ID/imagen-api:latest .

# Build worker image
docker build -f docker/Dockerfile.worker \
  -t gcr.io/$PROJECT_ID/imagen-worker:latest .

# Push to GCR
docker push gcr.io/$PROJECT_ID/imagen-api:latest
docker push gcr.io/$PROJECT_ID/imagen-worker:latest
```

---

## Step 7: Deploy to Kubernetes

### Deploy Base Resources

```bash
# Create namespace and base resources
kubectl apply -k k8s/base/

# Verify
kubectl get namespace imagen
kubectl get pvc -n imagen
kubectl get configmap -n imagen
```

### Deploy Workers

```bash
# Deploy all workers
kubectl apply -k k8s/workers/

# Verify workers
kubectl get pods -n imagen
kubectl get deployments -n imagen
```

Expected output:
```
NAME                              READY   STATUS    RESTARTS   AGE
upscale-worker-xxx                0/1     Init      0          10s
enhance-worker-xxx                0/1     Init      0          10s
comic-worker-xxx                  0/1     Init      0          10s
background-remove-worker-xxx      0/1     Init      0          10s
```

### Wait for Models to Download

First worker to start will download models to PVC (~14GB, 10-20 minutes):

```bash
# Watch logs
kubectl logs -f -n imagen upscale-worker-xxx

# You'll see:
# "Loading upscale model: stabilityai/stable-diffusion-x4-upscaler"
# "Model loaded successfully"
```

---

## Step 8: Deploy API to Cloud Run

```bash
# Deploy API
gcloud run deploy imagen-api \
  --image gcr.io/$PROJECT_ID/imagen-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCS_BUCKET=$BUCKET_NAME" \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 10

# Get API URL
gcloud run services describe imagen-api \
  --platform managed \
  --region us-central1 \
  --format='value(status.url)'
```

Save this URL - this is your API endpoint!

---

## Step 9: Test Your Deployment

### Test Health Endpoint

```bash
export API_URL=$(gcloud run services describe imagen-api \
  --platform managed --region us-central1 \
  --format='value(status.url)')

curl $API_URL/health
# Expected: {"status":"healthy"}
```

### Submit Test Job

```bash
# Submit upscale job
curl -X POST "$API_URL/api/v1/images/upscale" \
  -F "file=@test-image.jpg" \
  -F "prompt=high quality, detailed"

# Response: {"job_id": "abc-123", "status": "queued"}
```

### Check Job Status

```bash
JOB_ID="abc-123"  # Use actual job ID from above
curl "$API_URL/api/v1/jobs/$JOB_ID"

# Monitor until status changes to "completed"
```

### Monitor Workers

```bash
# Watch worker processing
kubectl logs -f -n imagen upscale-worker-xxx

# Check all worker statuses
kubectl get pods -n imagen
```

---

## Step 10: Set Up Monitoring

### Cloud Console

1. Go to **GKE** → **Workloads** → View your workers
2. Go to **Pub/Sub** → **Subscriptions** → See message counts
3. Go to **Cloud Run** → **imagen-api** → View metrics

### Set Up Alerts

Create alerts for:
- Worker pod restarts
- High Pub/Sub message age
- API error rate > 5%
- GPU utilization > 80%

---

## Verification Checklist

After deployment, verify:

- [ ] All 4 workers are running: `kubectl get pods -n imagen`
- [ ] Models downloaded to PVC: `kubectl exec -n imagen upscale-worker-xxx -- du -sh /models`
- [ ] API responds: `curl $API_URL/health`
- [ ] Can submit job: Test with curl
- [ ] Job completes: Check job status
- [ ] Images stored in GCS: `gsutil ls gs://$BUCKET_NAME/outputs/`
- [ ] Firestore has job records: Check Cloud Console
- [ ] Billing alerts configured
- [ ] Monitoring dashboards accessible

---

## What's Next?

### Immediate Actions

1. **[Set up CI/CD](../deployment/CICD_PIPELINE.md)** - Automate deployments
2. **[Configure monitoring](../deployment/MONITORING.md)** - Set up observability
3. **[Review security](../deployment/SECURITY.md)** - Harden your deployment

### Ongoing Operations

- **[Scaling Operations](../deployment/SCALING.md)** - Handle growth
- **[Cost Optimization](../infrastructure/COST_OPTIMIZATION.md)** - Reduce expenses
- **[Troubleshooting](../deployment/TROUBLESHOOTING_PROD.md)** - Fix issues

---

## Common Issues

### Workers stuck in "Init" state

**Cause**: Downloading models for first time

**Solution**: Wait 10-20 minutes. Check logs:
```bash
kubectl logs -n imagen upscale-worker-xxx
```

### "Permission denied" errors

**Cause**: Service account permissions

**Solution**: Verify Terraform applied correctly:
```bash
terraform show | grep google_service_account
```

### API can't connect to Pub/Sub

**Cause**: Firewall or service account issues

**Solution**: Check Cloud Run service account permissions

### Out of disk space on PVC

**Cause**: PVC too small (should be 50GB)

**Solution**: Check PVC size:
```bash
kubectl get pvc -n imagen
```

---

## Cost Optimization Tips

After first deployment:

1. **Scale down** workers when not in use
2. **Use Spot VMs** in GKE for workers
3. **Set up autoscaling** based on Pub/Sub queue depth
4. **Review billing** weekly for first month

See [Cost Optimization Guide](../infrastructure/COST_OPTIMIZATION.md)

---

## Getting Help

- **Production issues**: See [Production Troubleshooting](../deployment/TROUBLESHOOTING_PROD.md)
- **Architecture questions**: See [System Design](../core-concepts/SYSTEM_DESIGN.md)
- **Further reading**: See [Complete Documentation](../README.md)

🎉 **Congratulations!** Your Imagen platform is now running in production.

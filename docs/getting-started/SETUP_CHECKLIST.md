# Pre-Deployment Setup Checklist

Complete this checklist before deploying Imagen to ensure all configuration is correct.

## ✅ Prerequisites

### Accounts & Access

- [ ] GCP account created
- [ ] Billing enabled on GCP account
- [ ] gcloud CLI installed (`gcloud --version`)
- [ ] gcloud authenticated (`gcloud auth login`)
- [ ] kubectl installed (`kubectl version --client`)
- [ ] terraform installed (`terraform --version`)
- [ ] Git repository cloned locally

### Project Setup

- [ ] GCP project created (or create new one)
- [ ] Project ID noted (format: `your-project-name-123456`)
- [ ] Billing account linked to project
- [ ] Required APIs will be enabled (done via script or Terraform)

---

## 📝 Configuration Files to Update

### 1. Terraform Variables

**File**: `terraform/environments/prod.tfvars`

```bash
# BEFORE (placeholder)
project_id = "your-project-id"

# AFTER (your actual project ID)
project_id = "imagen-prod-20251220"
```

**Required changes**:
- [ ] Update `project_id` to your actual GCP project ID
- [ ] Update `region` if not using `us-central1`
- [ ] Update `github_owner` and `github_repo` (if using CI/CD)

**File**: `terraform/environments/dev.tfvars` (if deploying dev environment)

- [ ] Update `project_id` to your dev GCP project ID
- [ ] Verify `region` setting

---

### 2. Environment Variables

**File**: `.env` (in project root)

```bash
# BEFORE (placeholders)
GOOGLE_CLOUD_PROJECT=your-project-id
GCS_BUCKET=your-bucket-name

# AFTER (your actual values)
GOOGLE_CLOUD_PROJECT=imagen-prod-20251220
GCS_BUCKET=imagen-prod-20251220-images
```

**Required changes**:
- [ ] Update `GOOGLE_CLOUD_PROJECT` to match Terraform `project_id`
- [ ] Update `GCS_BUCKET` to your bucket name (Terraform will create this)

---

### 3. Kubernetes Overlays

**File**: `k8s/overlays/prod/kustomization.yaml`

```yaml
# BEFORE (placeholder)
images:
  - name: imagen-api
    newName: gcr.io/YOUR_PROJECT_ID/imagen-api
  - name: imagen-worker
    newName: gcr.io/YOUR_PROJECT_ID/imagen-worker

# AFTER (your actual project ID)
images:
  - name: imagen-api
    newName: gcr.io/imagen-prod-20251220/imagen-api
  - name: imagen-worker
    newName: gcr.io/imagen-prod-20251220/imagen-worker
```

**Required changes**:
- [ ] Replace all `YOUR_PROJECT_ID` with your actual project ID (3 occurrences)

**File**: `k8s/overlays/dev/kustomization.yaml` (if deploying dev)

- [ ] Replace all `YOUR_PROJECT_ID` with your dev project ID

**File**: `k8s/base/workload-identity.yaml`

```yaml
# BEFORE
iam.gke.io/gcp-service-account: imagen-worker@YOUR_PROJECT_ID.iam.gserviceaccount.com

# AFTER
iam.gke.io/gcp-service-account: imagen-worker@imagen-prod-20251220.iam.gserviceaccount.com
```

- [ ] Replace `YOUR_PROJECT_ID` with your actual project ID

**File**: `k8s/autoscaling/custom-metrics-adapter.yaml`

- [ ] Replace `PROJECT_ID` placeholder with your actual project ID

---

### 4. Cloud Build Configuration

**File**: `cloudbuild.yaml`

Check for any hardcoded project IDs:
- [ ] Verify substitutions use `$PROJECT_ID` variable (should be automatic)
- [ ] No hardcoded project IDs in image names

---

## 🔍 Validation Before Deployment

### Check All Placeholders Replaced

Run this command to find any remaining placeholders:

```bash
# Search for common placeholders
grep -r "your-project-id" . --exclude-dir=.git --exclude-dir=docs
grep -r "YOUR_PROJECT_ID" . --exclude-dir=.git --exclude-dir=docs
grep -r "your-bucket-name" . --exclude-dir=.git --exclude-dir=docs

# Should return: no matches found
```

- [ ] No placeholders found in configuration files

### Verify Terraform Variables

```bash
cd terraform
terraform init
terraform validate
terraform plan -var-file=environments/prod.tfvars

# Check for errors before proceeding
```

- [ ] Terraform validation passes
- [ ] Terraform plan shows expected resources
- [ ] No errors about missing variables

### Verify GCP Access

```bash
# Check current project
gcloud config get-value project

# Set project if needed
gcloud config set project YOUR_PROJECT_ID

# Verify access
gcloud projects describe YOUR_PROJECT_ID
```

- [ ] Can access GCP project
- [ ] Correct project is set as default

---

## 📋 Deployment Checklist

### Infrastructure

- [ ] Terraform `init` completed
- [ ] Terraform `plan` reviewed
- [ ] Terraform `apply` will be executed
- [ ] GCS bucket created
- [ ] Pub/Sub topics created
- [ ] GKE cluster created
- [ ] Firestore database initialized

### Application

- [ ] Container images built
- [ ] Images pushed to GCR/Artifact Registry
- [ ] Kubernetes manifests applied
- [ ] Workers deployed
- [ ] API deployed to Cloud Run

### Verification

- [ ] Health endpoint responds: `/health`
- [ ] Can submit test job
- [ ] Job completes successfully
- [ ] Images stored in GCS
- [ ] Firestore has job records
- [ ] Worker logs show model loading
- [ ] No errors in Cloud Logging

---

## 🚨 Cost Protection

### Billing Alerts

- [ ] Billing budget created
- [ ] Alert thresholds set (50%, 75%, 90%, 100%)
- [ ] Email notifications configured
- [ ] SMS alerts configured (optional)

**Recommended budget**: $200/month for testing

### Resource Limits

- [ ] GKE node pool max size set
- [ ] Cloud Run max instances set
- [ ] Worker replica counts reasonable (1-2 per type)
- [ ] Autoscaling limits in place

---

## 📝 Configuration Summary

Create this file for your records:

**`DEPLOYMENT_CONFIG.txt`** (DO NOT COMMIT TO GIT)

```
===========================================
IMAGEN DEPLOYMENT CONFIGURATION
===========================================

GCP Project ID: ___________________________
GCP Region: us-central1
Environment: prod

Bucket Name: ___________________________
GKE Cluster: imagen-prod
Cloud Run Service: imagen-api

GitHub Owner: ___________________________
GitHub Repo: imagen

Deployment Date: ___________________________
Deployed By: ___________________________

===========================================
ENDPOINTS
===========================================

API URL: https://___________________________
Health Check: https://___________________________/health

===========================================
COST MONITORING
===========================================

Monthly Budget: $___________
Alert Threshold: $___________
Billing Admin: ___________________________

===========================================
```

---

## 🔒 Security Checklist

### Secrets Management

- [ ] No secrets in Git
- [ ] `.env` in `.gitignore`
- [ ] Service account keys stored securely
- [ ] Workload Identity configured (no key files needed)

### Access Control

- [ ] IAM permissions reviewed
- [ ] Service accounts have minimum required permissions
- [ ] Cloud Run allows authenticated requests only (or public if intended)
- [ ] GCS bucket access restricted

---

## 🎯 Quick Reference

### Your Configuration (fill in)

```bash
# GCP
export PROJECT_ID="___________________________"
export REGION="us-central1"
export BUCKET_NAME="___________________________"

# GitHub (if using CI/CD)
export GITHUB_OWNER="___________________________"
export GITHUB_REPO="imagen"

# Verify
echo "Project: $PROJECT_ID"
echo "Bucket: $BUCKET_NAME"
```

---

## ✅ Final Verification

Before running `terraform apply`:

1. [ ] All configuration files updated
2. [ ] No placeholders remaining
3. [ ] Terraform plan reviewed
4. [ ] Billing alerts configured
5. [ ] Team notified of deployment
6. [ ] Rollback plan documented

**Ready to deploy?** → Proceed to [First Deployment](FIRST_DEPLOYMENT.md)

---

## 🆘 Common Issues

### "Invalid project_id" error

**Cause**: Placeholder not replaced in `.tfvars`

**Fix**: Update `terraform/environments/prod.tfvars` with actual project ID

### "Permission denied" errors

**Cause**: Not authenticated or wrong project

**Fix**:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Can't find configuration files

**Cause**: Not in project root

**Fix**:
```bash
cd /path/to/imagen
ls terraform/environments/
```

### Terraform "variable not set" error

**Cause**: Using wrong `.tfvars` file

**Fix**:
```bash
terraform plan -var-file=environments/prod.tfvars
```

---

## 📚 Related Documentation

- **[First Deployment](FIRST_DEPLOYMENT.md)** - Step-by-step deployment guide
- **[Configuration Reference](../reference/CONFIGURATION_REFERENCE.md)** - All configuration options
- **[Production Deployment](../deployment/PRODUCTION_DEPLOYMENT.md)** - Complete deployment guide
- **[Infrastructure Overview](../infrastructure/INFRASTRUCTURE_OVERVIEW.md)** - Understanding Terraform

---

**Last Updated**: 2025-12-20

**Tip**: Print this checklist and check off items as you complete them!

.PHONY: dev api worker-upscale worker-enhance test lint format clean docker-build

# Development
dev:
	docker compose -f docker/docker-compose.yml up -d redis minio
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Workers
worker-upscale:
	python -m src.workers.upscale

worker-enhance:
	python -m src.workers.enhance

worker-comic:
	python -m src.workers.style_comic

worker-aged:
	python -m src.workers.style_aged

worker-background-remove:
	python -m src.workers.background_remove

# Models
download-models:
	python download_models.py

# Testing
test:
	pytest tests/ -v --cov=src

test-unit:
	pytest tests/unit -v

# Code Quality
lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Docker
docker-build:
	docker build -t imagen-api -f docker/Dockerfile.api .
	docker build -t imagen-worker -f docker/Dockerfile.worker .

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache

# GCP Deployment
deploy-api:
	gcloud run deploy imagen-api --source . --region us-central1

tf-init:
	cd terraform && terraform init

tf-plan:
	cd terraform && terraform plan -var-file=environments/dev.tfvars

tf-apply:
	cd terraform && terraform apply -var-file=environments/dev.tfvars


# Kubernetes Deployment

# Using Kustomize (recommended)
k8s-deploy-dev:
	@echo "Deploying to development..."
	@echo "Make sure to update PROJECT_ID in k8s/overlays/dev/kustomization.yaml first!"
	kubectl apply -k k8s/overlays/dev

k8s-deploy-prod:
	@echo "Deploying to production..."
	@echo "Make sure to update PROJECT_ID in k8s/overlays/prod/kustomization.yaml first!"
	kubectl apply -k k8s/overlays/prod

k8s-preview:
	@echo "Preview what will be deployed (dev):"
	kubectl kustomize k8s/overlays/dev

# Legacy manual deployment (use kustomize instead)
k8s-namespace:
	kubectl apply -f k8s/base/namespace.yaml

k8s-base:
	kubectl apply -f k8s/base/

k8s-workers:
	kubectl apply -f k8s/workers/

# Auto-Scaling
k8s-autoscaling-adapter:
	@echo "Deploying Custom Metrics Adapter..."
	kubectl apply -f k8s/autoscaling/custom-metrics-adapter.yaml

k8s-autoscaling-hpa:
	@echo "Deploying HPAs..."
	kubectl apply -f k8s/autoscaling/upscale-hpa.yaml
	kubectl apply -f k8s/autoscaling/enhance-hpa.yaml
	kubectl apply -f k8s/autoscaling/comic-hpa.yaml
	kubectl apply -f k8s/autoscaling/background-remove-hpa.yaml

k8s-autoscaling: k8s-autoscaling-adapter k8s-autoscaling-hpa
	@echo "Auto-scaling configured!"

# KEDA (Scale to Zero)
keda-install:
	helm repo add kedacore https://kedacore.github.io/charts
	helm repo update
	helm install keda kedacore/keda --namespace keda --create-namespace

keda-deploy:
	@echo "Deploying KEDA ScaledObjects (full scale-to-zero)..."
	kubectl delete hpa -n imagen --all 2>/dev/null || true
	kubectl apply -f k8s/autoscaling/keda/scaledobjects.yaml

keda-deploy-hybrid:
	@echo "Deploying KEDA ScaledObjects (hybrid - 1 hot pod)..."
	kubectl delete hpa -n imagen --all 2>/dev/null || true
	kubectl apply -f k8s/autoscaling/keda/hybrid-scaledobjects.yaml

keda-status:
	kubectl get scaledobject -n imagen

keda-remove:
	@echo "Removing KEDA, restoring standard HPAs..."
	kubectl delete scaledobject -n imagen --all 2>/dev/null || true
	kubectl apply -f k8s/autoscaling/

# Check HPA status
k8s-hpa-status:
	kubectl get hpa -n imagen

# Watch scaling in real-time
k8s-watch:
	watch -n 2 'kubectl get hpa -n imagen && echo "" && kubectl get pods -n imagen'

# Full deployment
k8s-deploy-all: k8s-deploy-dev
	@echo "Full deployment complete!"


# =============================================================================
# MONITORING (Google Managed Prometheus)
# =============================================================================

# Deploy GMP PodMonitoring and rules
monitoring-deploy:
	@echo "Deploying Google Managed Prometheus monitoring..."
	kubectl apply -k k8s/monitoring/
	@echo "Monitoring deployed! Metrics will appear in Cloud Monitoring within 2-3 minutes."

# Check GMP is running
monitoring-check-gmp:
	@echo "Checking GMP system pods..."
	kubectl get pods -n gmp-system

# Check PodMonitoring resources
monitoring-status:
	@echo "PodMonitoring resources:"
	kubectl get podmonitoring -n imagen
	@echo ""
	@echo "Rules:"
	kubectl get rules -n imagen

# View worker metrics locally (port-forward)
monitoring-port-forward:
	@echo "Port-forwarding worker metrics to localhost:8080..."
	@echo "Visit http://localhost:8080/metrics to see Prometheus metrics"
	kubectl port-forward -n imagen deployment/upscale-worker 8080:8080

# Query metrics via gcloud (example)
monitoring-query-example:
	@echo "Example PromQL query via Cloud Monitoring API:"
	@echo ""
	@echo "gcloud monitoring query --project=YOUR_PROJECT_ID \\"
	@echo "  'fetch prometheus_target"
	@echo "   | metric \"prometheus.googleapis.com/imagen_jobs_completed_total/counter\""
	@echo "   | every 1m'"

# Delete monitoring resources
monitoring-delete:
	kubectl delete -k k8s/monitoring/


# =============================================================================
# CI/CD (Cloud Build)
# =============================================================================

# Trigger a manual build (without git push)
build-manual:
	gcloud builds submit --config=cloudbuild.yaml

# Trigger PR check build
build-pr-check:
	gcloud builds submit --config=cloudbuild-pr.yaml

# View build history
build-history:
	gcloud builds list --limit=10

# View build logs
build-logs:
	@echo "Enter build ID:"
	@read BUILD_ID && gcloud builds log $$BUILD_ID

# Check Cloud Build triggers
build-triggers:
	gcloud builds triggers list

# =============================================================================
# FULL DEPLOYMENT (first time)
# =============================================================================

# Complete first-time deployment
deploy-all: tf-apply build-manual monitoring-deploy
	@echo ""
	@echo "=============================================="
	@echo "  DEPLOYMENT COMPLETE!"
	@echo "=============================================="
	@echo ""
	@echo "  API URL: $$(terraform -chdir=terraform output -raw api_url)"
	@echo ""
	@echo "  Next steps:"
	@echo "  1. Connect GitHub for automatic deployments"
	@echo "  2. Update terraform with github_owner/github_repo"
	@echo "  3. Push to main branch to trigger deployment"
	@echo "  4. Check metrics: make monitoring-status"
	@echo ""


# =============================================================================
# TRITON INFERENCE SERVER
# =============================================================================

# Build custom Triton image
triton-build:
	@echo "Building custom Triton image..."
	docker build -t triton-imagen -f triton/Dockerfile .

# Push Triton image to Artifact Registry
triton-push:
	@echo "Pushing Triton image to Artifact Registry..."
	docker tag triton-imagen us-central1-docker.pkg.dev/$(PROJECT_ID)/imagen/triton:latest
	docker push us-central1-docker.pkg.dev/$(PROJECT_ID)/imagen/triton:latest

# Deploy Triton to GKE
triton-deploy:
	@echo "Deploying Triton Inference Server..."
	kubectl apply -k k8s/triton/

# Check Triton status
triton-status:
	@echo "Triton pods:"
	kubectl get pods -n imagen -l app=triton
	@echo ""
	@echo "Triton service:"
	kubectl get svc -n imagen triton

# Port-forward to Triton (for local testing)
triton-port-forward:
	@echo "Port-forwarding Triton..."
	@echo "HTTP: localhost:8000, gRPC: localhost:8001, Metrics: localhost:8002"
	kubectl port-forward -n imagen svc/triton 8000:8000 8001:8001 8002:8002

# Check Triton health
triton-health:
	@echo "Checking Triton health..."
	curl -s http://localhost:8000/v2/health/ready && echo " Ready!" || echo " Not ready"

# List loaded models
triton-models:
	@echo "Loaded models:"
	curl -s http://localhost:8000/v2/models | jq

# Check specific model status
triton-model-status:
	@echo "Enter model name (upscale, enhance, background_remove, style_comic, style_aged):"
	@read MODEL && curl -s http://localhost:8000/v2/models/$$MODEL | jq

# View Triton metrics
triton-metrics:
	curl -s http://localhost:8002/metrics | grep -E "^nv_inference|^nv_gpu"

# Delete Triton
triton-delete:
	kubectl delete -k k8s/triton/

# Full Triton deployment
triton-deploy-full: triton-build triton-push triton-deploy
	@echo "Triton deployment complete!"


# =============================================================================
# TRITON WORKERS (Alternative to standard workers)
# =============================================================================

# Run Triton-based workers locally (for development)
triton-worker-upscale:
	TRITON_URL=localhost:8001 python -c "from src.workers.triton_worker import TritonUpscaleWorker; TritonUpscaleWorker().run()"

triton-worker-enhance:
	TRITON_URL=localhost:8001 python -c "from src.workers.triton_worker import TritonEnhanceWorker; TritonEnhanceWorker().run()"

triton-worker-background-remove:
	TRITON_URL=localhost:8001 python -c "from src.workers.triton_worker import TritonBackgroundRemoveWorker; TritonBackgroundRemoveWorker().run()"

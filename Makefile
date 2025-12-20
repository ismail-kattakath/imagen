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

# =============================================================================
# CUSTOM TRITON INFERENCE SERVER IMAGE
# =============================================================================
#
# Based on NGC Triton with:
# - Python dependencies for our model backends
# - Pre-configured model repository
#
# Build:
#   docker build -t triton-imagen -f triton/Dockerfile .
#
# =============================================================================

FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Install Python dependencies for model backends
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    transformers \
    diffusers \
    accelerate \
    safetensors \
    basicsr \
    realesrgan \
    Pillow \
    numpy

# Install additional dependencies for specific models
RUN pip install --no-cache-dir \
    opencv-python-headless \
    scikit-image

# Create directories
RUN mkdir -p /models/huggingface /models/repository

# Copy model repository (config.pbtxt + model.py files)
COPY triton/model_repository/ /models/repository/

# Set environment variables
ENV HF_HOME=/models/huggingface
ENV TRANSFORMERS_CACHE=/models/huggingface
ENV TORCH_HOME=/models/torch

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# Default command - use /models/repository for model configs
CMD ["tritonserver", "--model-repository=/models/repository", "--log-verbose=1"]

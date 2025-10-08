# =============================================================================
# Orpheus 3B TTS - Pre-built Production Docker Image
# =============================================================================
# This Dockerfile creates a production-ready image with:
# - All system dependencies pre-installed
# - Python environment with TensorRT-LLM
# - Pre-built INT4-AWQ + INT8 KV cache engine
# - Optimized for rapid deployment (2-5 min startup vs 45 min build)
#
# Build: docker build -t orpheus-3b-tts .
# Run:   docker run --gpus all -p 8000:8000 orpheus-3b-tts
# =============================================================================

# Use NVIDIA's official TensorRT container as base (includes CUDA, cuDNN, TensorRT)
FROM nvcr.io/nvidia/tensorrt:23.12-py3

# =============================================================================
# Build Arguments and Environment Setup
# =============================================================================

# Build-time arguments (set via --build-arg)
ARG HF_TOKEN
ARG DOCKER_USERNAME
ARG DOCKER_PASSWORD
ARG IMAGE_NAME=orpheus-3b-tts

# Set environment variables for the build
ENV HF_TOKEN=${HF_TOKEN}
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# =============================================================================
# System Dependencies Installation
# =============================================================================

# Update package lists and install system dependencies
RUN apt-get update -y && \
    apt-get install -y \
        git wget curl jq \
        python3-venv python3-dev python3.10-venv python3.10-dev \
        libopenmpi-dev openmpi-bin \
        build-essential \
        software-properties-common \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python Environment Setup
# =============================================================================

# Create application directory
WORKDIR /app

# Copy project files
COPY . .

# Set up Python environment variables from our centralized config
ENV PYTHON_VERSION=3.10
ENV VENV_DIR=/app/.venv
ENV MODEL_ID=canopylabs/orpheus-3b-0.1-ft

# Load our centralized environment configuration
RUN chmod +x scripts/environment.sh && \
    . scripts/environment.sh

# Create Python virtual environment
RUN python3.10 -m venv ${VENV_DIR}
ENV PATH="${VENV_DIR}/bin:$PATH"

# Upgrade pip and install core tools
RUN pip install --upgrade --no-cache-dir pip setuptools wheel

# =============================================================================
# PyTorch Installation with CUDA Support
# =============================================================================

# Install PyTorch with CUDA 12.1 support (matches TensorRT container)
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch --only-binary=:all:

# =============================================================================
# Application Dependencies
# =============================================================================

# Install Python dependencies
RUN pip install -r requirements.txt

# Install TensorRT-LLM and related packages
RUN pip install --upgrade --extra-index-url https://pypi.nvidia.com \
    tensorrt-llm==1.0.0 \
    tensorrt-cu12-bindings \
    tensorrt-cu12-libs

# =============================================================================
# HuggingFace Authentication
# =============================================================================

# Login to HuggingFace only if token is provided (skip if public models)
RUN if [ -n "${HF_TOKEN}" ]; then \
      python -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"; \
    else \
      echo "Skipping HuggingFace login (HF_TOKEN not provided)"; \
    fi

# =============================================================================
# TensorRT-LLM Engine Build
# =============================================================================

# Clone TensorRT-LLM repository for build tools (using NVIDIA official repo)
RUN git clone https://github.com/Yap-With-AI/TensorRT-LLM.git /tmp/tensorrt-llm && \
    cd /tmp/tensorrt-llm && \
    git fetch --tags && \
    (git checkout v1.0.0 || git checkout ae8270b713446948246f16fadf4e2a32e35d0f62)

# Set build environment variables (from our environment.sh)
ENV TRTLLM_ENGINE_DIR=/app/models/orpheus-trt-int4-awq
ENV TRTLLM_MAX_BATCH_SIZE=16
ENV TRTLLM_MAX_INPUT_LEN=48
ENV TRTLLM_MAX_OUTPUT_LEN=1024
ENV TRTLLM_DTYPE=float16

# Create models and audio directories
RUN mkdir -p ${TRTLLM_ENGINE_DIR} /app/audio

# Build optimized TensorRT-LLM engine with INT4-AWQ + INT8 KV cache (2-step process)
WORKDIR /tmp/tensorrt-llm/examples/quantization

# Install quantization requirements
RUN pip install -r requirements.txt

# Enable fast HF downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Step 1: Quantize the model using quantize.py
RUN python quantize.py \
    --model_dir ${MODEL_ID} \
    --output_dir ${TRTLLM_ENGINE_DIR}_checkpoint \
    --dtype ${TRTLLM_DTYPE} \
    --qformat int4_awq \
    --awq_block_size 128 \
    --calib_size 256 \
    --kv_cache_dtype int8

# Step 2: Build TensorRT engine from quantized checkpoint
RUN trtllm-build \
    --checkpoint_dir ${TRTLLM_ENGINE_DIR}_checkpoint \
    --output_dir ${TRTLLM_ENGINE_DIR} \
    --gemm_plugin auto \
    --gpt_attention_plugin float16 \
    --context_fmha enable \
    --paged_kv_cache enable \
    --remove_input_padding enable \
    --max_input_len ${TRTLLM_MAX_INPUT_LEN} \
    --max_seq_len $((TRTLLM_MAX_INPUT_LEN + TRTLLM_MAX_OUTPUT_LEN)) \
    --max_batch_size ${TRTLLM_MAX_BATCH_SIZE} \
    --log_level info \
    --workers 1

# Clean up intermediate checkpoint to reduce image size
RUN rm -rf ${TRTLLM_ENGINE_DIR}_checkpoint

# Return to app directory
WORKDIR /app

# Verify engine was built successfully
RUN ls -la ${TRTLLM_ENGINE_DIR}/ && \
    test -f ${TRTLLM_ENGINE_DIR}/rank0.engine || (echo "ERROR: Engine build failed" && exit 1)

# =============================================================================
# Cleanup Build Artifacts
# =============================================================================

# Remove build dependencies and temporary files to reduce image size
RUN rm -rf /tmp/tensorrt-llm && \
    pip cache purge && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# =============================================================================
# Runtime Configuration
# =============================================================================

# Set runtime environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV TRTLLM_ENGINE_DIR=/app/models/orpheus-trt-int4-awq

# Expose the server port
EXPOSE 8000

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash tts && \
    chown -R tts:tts /app
USER tts

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Copy startup script for manual control if needed
COPY docker/start-server.sh /app/start-server.sh
RUN chmod +x /app/start-server.sh

# Default command - wait for API key, then start server
# Server requires YAP_API_KEY to be set at runtime
# For manual control: docker run ... bash (then use /app/start-server.sh)
CMD ["bash", "/app/start-server.sh"]

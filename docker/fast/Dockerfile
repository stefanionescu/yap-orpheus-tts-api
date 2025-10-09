# Fast production image: CUDA 12.1 runtime on Ubuntu 22.04 (matches PyTorch cu121 wheels)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Use bash for RUN lines
SHELL ["/bin/bash", "-lc"]

# Non-interactive APT
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# System bootstrap (essential dependencies only)
# ----------------------------------------------------------------------------
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      git wget curl jq \
      python3.10 python3.10-venv python3.10-dev \
      python3-venv python3-dev \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

## Ensure consistent python and create virtual env
ENV PYTHON_VERSION=3.10
ENV VENV_DIR=/opt/venv
RUN python3.10 -m venv "$VENV_DIR" && \
    source "$VENV_DIR/bin/activate" && \
    python -m ensurepip --upgrade || true && \
    python -m pip install --upgrade --no-cache-dir pip setuptools wheel

# Make venv default
ENV PATH="/opt/venv/bin:${PATH}"

# ----------------------------------------------------------------------------
# PyTorch installation (CUDA 12.1 index)
# ----------------------------------------------------------------------------
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN echo "Using PyTorch index: ${PYTORCH_INDEX_URL}" && \
    pip install --no-cache-dir --index-url "${PYTORCH_INDEX_URL}" torch==2.4.1 --only-binary=:all:

# ----------------------------------------------------------------------------
# App Python dependencies (production runtime only)
# ----------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Install runtime requirements but skip torch (already installed) and mpi4py (needs MPI dev libs)
RUN grep -Ev '^(torch==|mpi4py==)' /app/requirements.txt > /app/requirements-fast.txt && \
    pip install --no-cache-dir -r /app/requirements-fast.txt

# ----------------------------------------------------------------------------
# TensorRT-LLM installation (runtime wheel only, no repo)
# ----------------------------------------------------------------------------
ARG TRTLLM_WHEEL_URL=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl
RUN pip install --upgrade --no-cache-dir --extra-index-url https://pypi.nvidia.com \
      "${TRTLLM_WHEEL_URL}" \
      tensorrt-cu12-bindings \
      tensorrt-cu12-libs

# Validate dependency graph
RUN pip check

# ----------------------------------------------------------------------------
# Copy server code and runtime scripts into image
# ----------------------------------------------------------------------------
COPY server/ /app/server/
COPY tests/ /app/tests/
RUN rm -f /app/tests/client.py || true
COPY docker/fast/scripts/start-server.sh /usr/local/bin/start-server.sh
COPY docker/fast/scripts/environment.sh /usr/local/bin/environment.sh
RUN chmod +x /usr/local/bin/start-server.sh /usr/local/bin/environment.sh

# Ensure environment defaults are loaded in all shells
ENV BASH_ENV=/usr/local/bin/environment.sh
RUN mkdir -p /etc/profile.d && printf 'test -f /usr/local/bin/environment.sh && . /usr/local/bin/environment.sh\n' > /etc/profile.d/orpheus-env.sh

# ----------------------------------------------------------------------------
# Production environment setup
# ----------------------------------------------------------------------------
# Default directories for runtime model and engine mounting
ENV MODELS_DIR=/opt/models
ENV ENGINES_DIR=/opt/engines
RUN mkdir -p "${MODELS_DIR}" "${ENGINES_DIR}"

# Default environment knobs for production
ENV HF_TRANSFER=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Default server configuration
ENV HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Health check for production readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

# Default command: start the server
CMD ["bash", "-lc", "start-server.sh"]

LABEL org.opencontainers.image.title="Yap Orpheus TTS Fast (CUDA 12.1.1, Py310)" \
      org.opencontainers.image.description="Lean production image with dependencies only - models and engines loaded at runtime." \
      org.opencontainers.image.source="https://github.com/Yap-With-AI/yap-orpheus-tts-api"

# Base image: CUDA 12.4 runtime on Ubuntu 22.04 (matches PyTorch cu124 wheels)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Use bash for RUN lines
SHELL ["/bin/bash", "-lc"]

# Non-interactive APT
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# System bootstrap (replicates custom/setup/bootstrap.sh install set)
# ----------------------------------------------------------------------------
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      git wget curl jq \
      python3.10 python3.10-venv python3.10-dev \
      python3-venv python3-dev \
      libopenmpi-dev openmpi-bin && \
    rm -rf /var/lib/apt/lists/*

# Ensure consistent python and create virtual env (replicates venv setup)
ENV PYTHON_VERSION=3.10
ENV VENV_DIR=/opt/venv
RUN python3.10 -m venv "$VENV_DIR" && \
    source "$VENV_DIR/bin/activate" && \
    python -m ensurepip --upgrade || true && \
    python -m pip install --upgrade --no-cache-dir pip setuptools wheel

# Make venv default
ENV PATH="/opt/venv/bin:${PATH}"

# ----------------------------------------------------------------------------
# PyTorch installation (replicates _install_pytorch with CUDA 12.4 index)
# ----------------------------------------------------------------------------
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN echo "Using PyTorch index: ${PYTORCH_INDEX_URL}" && \
    pip install --no-cache-dir --index-url "${PYTORCH_INDEX_URL}" torch==2.4.1 --only-binary=:all:

# ----------------------------------------------------------------------------
# App Python dependencies
# ----------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ----------------------------------------------------------------------------
# TensorRT-LLM from NVIDIA PyPI (replicates _install_tensorrt_llm)
# Default wheel URL matches Python 3.10, TRT-LLM 1.0.0
# ----------------------------------------------------------------------------
ARG TRTLLM_WHEEL_URL=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl
RUN pip install --upgrade --no-cache-dir --extra-index-url https://pypi.nvidia.com \
      "${TRTLLM_WHEEL_URL}" \
      tensorrt-cu12-bindings \
      tensorrt-cu12-libs

# Optional: Validate dependency graph similarly to install-dependencies.sh
RUN pip check

# ----------------------------------------------------------------------------
# Optional: Hugging Face authentication (supply at build time to bake creds)
# WARNING: Embedding tokens in images may be insecure; prefer runtime injection
# ----------------------------------------------------------------------------
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN if [[ -n "${HF_TOKEN}" ]]; then \
      python - <<'PY'
import os
from huggingface_hub import login
token = os.environ.get("HF_TOKEN")
login(token=token, add_to_git_credential=False)
print("\u2713 HuggingFace authentication OK (baked into image)")
PY
    ; else echo "Skipping HF auth during build (no HF_TOKEN provided)"; fi

# Default environment knobs carried from custom/environment.sh (safe defaults)
ENV HF_TRANSFER=1 \
    HF_DOWNLOAD_ESSENTIAL_ONLY=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# This is a base image layer; no entrypoint/cmd so it can be reused downstream.
WORKDIR /app

LABEL org.opencontainers.image.title="Yap Orpheus TTS Base (CUDA 12.4, Py310)" \
      org.opencontainers.image.description="Prebuilt base with system deps, PyTorch CU124, and TensorRT-LLM to speed up cloud provisioning." \
      org.opencontainers.image.source="https://github.com/Yap-With-AI/yap-orpheus-tts-api"



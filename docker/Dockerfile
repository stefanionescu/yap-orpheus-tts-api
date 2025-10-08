# Base image: CUDA 12.1 runtime on Ubuntu 22.04 (matches PyTorch cu121 wheels)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Use bash for RUN lines
SHELL ["/bin/bash", "-lc"]

# Non-interactive APT
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# System bootstrap (replicates scripts/setup/bootstrap.sh install set)
# ----------------------------------------------------------------------------
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      git wget curl jq \
      python3.10 python3.10-venv python3.10-dev \
      python3-venv python3-dev \
      build-essential gfortran \
      libopenmpi-dev openmpi-bin && \
    rm -rf /var/lib/apt/lists/*

## Ensure consistent python and create virtual env (replicates venv setup)
ENV PYTHON_VERSION=3.10
ENV VENV_DIR=/opt/venv
RUN python3.10 -m venv "$VENV_DIR" && \
    source "$VENV_DIR/bin/activate" && \
    python -m ensurepip --upgrade || true && \
    python -m pip install --upgrade --no-cache-dir pip setuptools wheel

# Make venv default
ENV PATH="/opt/venv/bin:${PATH}"

# ----------------------------------------------------------------------------
# PyTorch installation (replicates _install_pytorch with CUDA 12.1 index)
# ----------------------------------------------------------------------------
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN echo "Using PyTorch index: ${PYTORCH_INDEX_URL}" && \
    pip install --no-cache-dir --index-url "${PYTORCH_INDEX_URL}" torch==2.4.1 --only-binary=:all:

# PyTorch already provided by base image (2.4.1 + cu121). No action needed.

# ----------------------------------------------------------------------------
# App Python dependencies
# ----------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Install remaining requirements while keeping the exact CUDA wheel for torch already installed
RUN grep -v '^torch==' /app/requirements.txt > /app/requirements-no-torch.txt && \
    pip install --no-cache-dir -r /app/requirements-no-torch.txt
# ----------------------------------------------------------------------------
# Copy server code and runtime scripts into image
# ----------------------------------------------------------------------------
COPY server/ /app/server/
COPY tests/ /app/tests/
COPY docker/scripts/01-quantize-and-build.sh /usr/local/bin/01-quantize-and-build.sh
COPY docker/scripts/02-start-server.sh /usr/local/bin/02-start-server.sh
COPY docker/scripts/run.sh /usr/local/bin/run.sh
COPY docker/scripts/environment.sh /usr/local/bin/environment.sh
RUN chmod +x /usr/local/bin/01-quantize-and-build.sh /usr/local/bin/02-start-server.sh /usr/local/bin/run.sh /usr/local/bin/environment.sh

# Ensure environment defaults are loaded in all shells
ENV BASH_ENV=/usr/local/bin/environment.sh
RUN mkdir -p /etc/profile.d && printf 'test -f /usr/local/bin/environment.sh && . /usr/local/bin/environment.sh\n' > /etc/profile.d/orpheus-env.sh


# ----------------------------------------------------------------------------
# TensorRT-LLM from NVIDIA PyPI (cp310 wheel to match Python 3.10 venv)
# ----------------------------------------------------------------------------
ARG TRTLLM_WHEEL_URL=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl
RUN pip install --upgrade --no-cache-dir --extra-index-url https://pypi.nvidia.com \
      "${TRTLLM_WHEEL_URL}" \
      tensorrt-cu12-bindings \
      tensorrt-cu12-libs

# Optional: Validate dependency graph similarly to install-dependencies.sh
RUN pip check

# ----------------------------------------------------------------------------
# Hugging Face model pre-download (REQUIRED at build time; token not persisted)
# Usage: DOCKER_BUILDKIT=1 HF_TOKEN=... docker build --secret id=HF_TOKEN,env=HF_TOKEN ...
# ----------------------------------------------------------------------------
ARG MODEL_ID=canopylabs/orpheus-3b-0.1-ft
ENV MODEL_ID=${MODEL_ID}
ENV MODELS_DIR=/opt/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN --mount=type=secret,id=HF_TOKEN,required=true <<'BASH'
set -euo pipefail
python - <<'PY'
import os
from huggingface_hub import snapshot_download

tok_path = "/run/secrets/HF_TOKEN"
if not os.path.exists(tok_path):
    raise SystemExit("HF_TOKEN secret missing (required at build-time)")
token = open(tok_path, "r").read().strip()

model_id = os.environ.get("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")
models_dir = os.environ.get("MODELS_DIR", "/opt/models")
basename = model_id.split("/")[-1]
local_dir = os.path.join(models_dir, f"{basename}-hf")
os.makedirs(local_dir, exist_ok=True)

snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False, token=token)
print(f"\u2713 Downloaded model {model_id} to {local_dir}")
PY
BASH

# ----------------------------------------------------------------------------
# TensorRT-LLM repository: clone and sync to installed wheel version
# ----------------------------------------------------------------------------
ARG TRTLLM_REPO_URL=https://github.com/Yap-With-AI/TensorRT-LLM.git
ENV TRTLLM_REPO_DIR=/opt/TensorRT-LLM
RUN git clone "${TRTLLM_REPO_URL}" "${TRTLLM_REPO_DIR}" && \
    trtllm_ver=$(python -c 'import tensorrt_llm as t; print(t.__version__)') && \
    echo "Detected TensorRT-LLM version: ${trtllm_ver}" && \
    git -C "${TRTLLM_REPO_DIR}" fetch --tags && \
    if ! git -C "${TRTLLM_REPO_DIR}" checkout "v${trtllm_ver}" 2>/dev/null; then \
      if [[ "${trtllm_ver}" == "1.0.0" ]]; then \
        echo "Tag v1.0.0 not found, using known commit ae8270b713446948246f16fadf4e2a32e35d0f62" && \
        git -C "${TRTLLM_REPO_DIR}" checkout ae8270b713446948246f16fadf4e2a32e35d0f62; \
      else \
        echo "ERROR: Could not checkout version ${trtllm_ver}" >&2; exit 1; \
      fi; \
    fi

# Match build flow: do not pre-install repo requirements; install quant deps at runtime
ENV TRTLLM_EXAMPLES_DIR=${TRTLLM_REPO_DIR}/examples/quantization

ARG MODEL_ID=canopylabs/orpheus-3b-0.1-ft
ENV MODEL_ID=${MODEL_ID}
ENV MODELS_DIR=/opt/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Default environment knobs carried from custom/environment.sh (safe defaults)
ENV HF_TRANSFER=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# This is a base image layer; no entrypoint/cmd so it can be reused downstream.
WORKDIR /app

LABEL org.opencontainers.image.title="Yap Orpheus TTS Base (CUDA 12.1, Py310)" \
      org.opencontainers.image.description="Prebuilt base with system deps, PyTorch CU12.1, and TensorRT-LLM to speed up cloud provisioning." \
      org.opencontainers.image.source="https://github.com/Yap-With-AI/yap-orpheus-tts-api"



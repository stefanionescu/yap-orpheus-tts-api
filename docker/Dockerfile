# Base image: CUDA 12.6 runtime on Ubuntu 22.04 (matches PyTorch cu126 wheels)
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

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
# PyTorch installation (replicates _install_pytorch with CUDA 12.6 index)
# ----------------------------------------------------------------------------
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN echo "Using PyTorch index: ${PYTORCH_INDEX_URL}" && \
    pip install --no-cache-dir --index-url "${PYTORCH_INDEX_URL}" torch==2.4.1 --only-binary=:all:

# ----------------------------------------------------------------------------
# App Python dependencies
# ----------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# ----------------------------------------------------------------------------
# Copy server code and runtime scripts into image
# ----------------------------------------------------------------------------
COPY server/ /app/server/
COPY tests/ /app/tests/
COPY docker/scripts/01-quantize-and-build.sh /usr/local/bin/01-quantize-and-build.sh
COPY docker/scripts/02-start-server.sh /usr/local/bin/02-start-server.sh
COPY docker/scripts/run.sh /usr/local/bin/run.sh
RUN chmod +x /usr/local/bin/01-quantize-and-build.sh /usr/local/bin/02-start-server.sh /usr/local/bin/run.sh


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
# Hugging Face authentication (REQUIRED at build-time) — authenticate BEFORE downloads
# ----------------------------------------------------------------------------
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
# Explicitly fail the build if HF_TOKEN is not provided
RUN test -n "${HF_TOKEN}" || (echo "ERROR: HF_TOKEN build-arg is required (export HF_TOKEN=...)" >&2; exit 1)
RUN python - <<'PY'
import os
from huggingface_hub import login
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN missing")
login(token=token, add_to_git_credential=False)
print("\u2713 HuggingFace authentication OK (validated during build)")
PY

# ----------------------------------------------------------------------------
# TensorRT-LLM repository: clone and sync to installed wheel version
# ----------------------------------------------------------------------------
ARG TRTLLM_REPO_URL=https://github.com/Yap-With-AI/TensorRT-LLM.git
ENV TRTLLM_REPO_DIR=/opt/TensorRT-LLM
RUN git clone "${TRTLLM_REPO_URL}" "${TRTLLM_REPO_DIR}" && \
    trtllm_ver=$(python - <<'PY'
import tensorrt_llm as t
print(t.__version__)
PY
) && \
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

# Install TRT-LLM repo dependencies commonly used by build/quant scripts
RUN if [ -f "${TRTLLM_REPO_DIR}/requirements.txt" ]; then \
      pip install --no-cache-dir -r "${TRTLLM_REPO_DIR}/requirements.txt" || true; \
    fi && \
    if [ -f "${TRTLLM_REPO_DIR}/examples/quantization/requirements.txt" ]; then \
      pip install --no-cache-dir -r "${TRTLLM_REPO_DIR}/examples/quantization/requirements.txt" || true; \
    fi

# ----------------------------------------------------------------------------
# Pre-download Orpheus model snapshot from Hugging Face (full repo, no quantization)
# ----------------------------------------------------------------------------
ARG MODEL_ID=canopylabs/orpheus-3b-0.1-ft
ENV MODEL_ID=${MODEL_ID}
ENV MODELS_DIR=/opt/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN mkdir -p "${MODELS_DIR}" && \
    python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get('MODEL_ID', 'canopylabs/orpheus-3b-0.1-ft')
models_dir = os.environ.get('MODELS_DIR', '/opt/models')
basename = model_id.split('/')[-1]
local_dir = os.path.join(models_dir, f"{basename}-hf")
os.makedirs(local_dir, exist_ok=True)

snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)

print(f"\u2713 Downloaded model '{model_id}' to {local_dir}")
PY

# Default environment knobs carried from custom/environment.sh (safe defaults)
ENV HF_TRANSFER=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# This is a base image layer; no entrypoint/cmd so it can be reused downstream.
WORKDIR /app

LABEL org.opencontainers.image.title="Yap Orpheus TTS Base (CUDA 12.6, Py310)" \
      org.opencontainers.image.description="Prebuilt base with system deps, PyTorch CU126, and TensorRT-LLM to speed up cloud provisioning." \
      org.opencontainers.image.source="https://github.com/Yap-With-AI/yap-orpheus-tts-api"



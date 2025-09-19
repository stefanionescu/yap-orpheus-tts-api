#!/usr/bin/env bash
set -euo pipefail
# Common helpers and env
source "scripts/lib/common.sh"
load_env_if_present

# Defaults
: "${PYTHON_VERSION:=3.10}"
: "${VENV_DIR:=$PWD/.venv}"

# Required
require_env HF_TOKEN

echo "[install] Creating venv at ${VENV_DIR}"
BACKEND=${ORPHEUS_BACKEND:-trtllm}
echo "[install] BACKEND=${BACKEND}"

# Resolve Python executable
PY_EXE=$(choose_python_exe) || { echo "[install] ERROR: Python not found. Please install Python ${PYTHON_VERSION}." >&2; exit 1; }

PY_MAJMIN=$($PY_EXE -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')

# Ensure venv module is available (Ubuntu often needs pythonX.Y-venv)
if ! $PY_EXE -m ensurepip --version >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "[install] Installing python venv support via apt-get"
    apt-get update -y || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y python${PY_MAJMIN}-venv || \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv || true
  else
    echo "[install] WARNING: ensurepip missing and apt-get unavailable. venv creation may fail." >&2
  fi
fi

$PY_EXE -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Pick the right PyTorch CUDA wheel channel
if [ -z "${CUDA_VER:-}" ]; then
  CUDA_VER=$(detect_cuda_version)
fi
TORCH_IDX=$(map_torch_index_url "${CUDA_VER:-}")

echo "[install] Installing Torch from ${TORCH_IDX}"
pip install --index-url "${TORCH_IDX}" torch --only-binary=:all:

echo "[install] Requirements (base)"
if [ -f requirements.txt ]; then
  if [ "${BACKEND}" != "vllm" ]; then
    echo "[install] Installing requirements without vLLM (TRT backend)"
    grep -v -E '^\s*vllm(==|\s|$)' requirements.txt > .tmp.req
    pip install -r .tmp.req
    rm -f .tmp.req
  else
    pip install -r requirements.txt
  fi
else
  pip install -r server/requirements.txt
fi

# Install TensorRT-LLM (Dockerless) if requested backend is TRT-LLM
if [ "${BACKEND}" != "vllm" ]; then
  echo "[install] Installing TensorRT-LLM runtime via NVIDIA PyPI"
  set +e
  # Let tensorrt-llm pull a compatible TensorRT (targets CUDA 12.x; prefer 12.6–12.8)
  pip install --extra-index-url https://pypi.nvidia.com \
    tensorrt-llm==0.21.0
  STATUS=$?
  set -e
  if [ $STATUS -ne 0 ]; then
    echo "[install] WARNING: tensorrt-llm install failed. Ensure CUDA 12.8 and driver r535+ are present." >&2
  fi
  echo "[install] Installing mpi4py (requires system MPI runtime)"
  set +e
  pip install --no-binary mpi4py "mpi4py>=4.0.0"
  set -e
  echo "[install] Ensuring correct CUDA 12.8 Python bindings (avoid conflicting 'cuda' packages)"
  set +e
  pip uninstall -y cuda cuda-bindings cuda_pathfinder cuda-pathfinder 2>/dev/null
  set -e
  pip install --upgrade --force-reinstall "cuda-python==12.8.*"
  # Ensure libpython shared library is present and discoverable for TRT-LLM bindings
  if ! ldconfig -p 2>/dev/null | grep -q "libpython${PY_MAJMIN}\.so"; then
    if command -v apt-get >/dev/null 2>&1; then
      echo "[install] Installing libpython${PY_MAJMIN}(-dev) for TRT-LLM"
      DEBIAN_FRONTEND=noninteractive apt-get update -y || true
      DEBIAN_FRONTEND=noninteractive apt-get install -y "libpython${PY_MAJMIN}" "libpython${PY_MAJMIN}-dev" || true
    fi
  fi
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib:${LD_LIBRARY_PATH:-}"
  # Hard check: verify module import resolves in THIS interpreter
  python - <<'PY' || { echo "[install] ERROR: tensorrt-llm not importable in current env" >&2; exit 1; }
import sys
print('[install] python:', sys.executable)
import tensorrt_llm, tensorrt
import cuda as _cuda
print('[install] cuda module:', getattr(_cuda, '__file__', _cuda))
print('[install] tensorrt-llm:', tensorrt_llm.__version__)
print('[install] tensorrt:', tensorrt.__version__)
PY
fi

# Optional: Install FlashAttention 2 prebuilt wheel if available (Linux + NVIDIA)
echo "[install] Checking for FlashAttention prebuilt wheel (optional)"
if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
  PY_INFO=$(python - <<'PY'
import torch, platform
print(f"torch={torch.__version__} cuda={torch.version.cuda} platform={platform.system()}")
PY
)
  echo "[install] ${PY_INFO}"
  set +e
  pip install --no-build-isolation --only-binary=:all: "flash-attn>=2.5.7"
  if [ $? -eq 0 ]; then
    echo "[install] flash-attn installed"
  else
    echo "[install] flash-attn wheel unavailable for torch=$(python -c 'import torch;print(torch.__version__)'); skipping"
  fi
  set -e
else
  echo "[install] Non-Linux or no NVIDIA driver detected; skipping flash-attn"
fi

# Login to HF (non-interactive)
python - <<'PY'
import os
from huggingface_hub import login
tok=os.environ.get("HF_TOKEN")
assert tok, "HF_TOKEN missing"
login(token=tok, add_to_git_credential=False)
print("[install] HF login OK")
PY

if [ "${PREFETCH:-1}" = "1" ]; then
  echo "[install] Pre-fetch model weights to local cache (~/.cache/huggingface)"
  # Enable accelerated downloader if available (speeds up big shards)
  pip install -q hf-transfer || true
  export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
model_id=os.environ.get("MODEL_ID","canopylabs/orpheus-3b-0.1-ft")
tok=os.environ["HF_TOKEN"]
allow = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
    "model.safetensors*",
]
snapshot_download(
    model_id,
    token=tok,
    local_files_only=False,
    allow_patterns=allow,
)
print("[install] Prefetch complete")
PY
else
  echo "[install] PREFETCH=0 → skipping model pre-download"
fi

echo "[install] Done."



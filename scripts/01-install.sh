#!/usr/bin/env bash
set -euo pipefail
# Common helpers and env
source "scripts/lib/common.sh"
load_env_if_present

# Defaults
: "${PYTHON_VERSION:=3.10}"
: "${VENV_DIR:=$PWD/.venv}"

# Speed-friendly pip defaults and cache (persist across venv rebuilds)
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-120}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-"$PWD/.cache/pip"}
mkdir -p "$PIP_CACHE_DIR"

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

# Prefer fast installer if available (uv). Install it quickly if requested.
: "${USE_UV:=1}"
if [ "$USE_UV" = "1" ] && ! command -v uv >/dev/null 2>&1; then
  echo "[install] Installing uv (fast Python package manager)"
  curl -fsSL https://astral.sh/uv/install.sh | sh -s -- -y >/dev/null 2>&1 || true
  export PATH="$HOME/.local/bin:$PATH"
fi
if command -v uv >/dev/null 2>&1; then
  PIP="uv pip"
else
  PIP="pip"
fi

# Pick the right PyTorch CUDA wheel channel
if [ -z "${CUDA_VER:-}" ]; then
  CUDA_VER=$(detect_cuda_version)
fi
TORCH_IDX=$(map_torch_index_url "${CUDA_VER:-}")

echo "[install] Installing Torch + TRT-LLM in one step (faster resolver)"
# Install heavy GPU stack in a single transaction to avoid uninstall/reinstall churn
# Keep Torch pulled from the correct CUDA channel; include torchvision for common ops
${PIP} install \
  --index-url "${TORCH_IDX}" \
  --extra-index-url https://pypi.nvidia.com \
  --only-binary=:all: --prefer-binary \
  torch torchvision "tensorrt-llm==0.21.0"

echo "[install] Requirements (base)"
if [ -f requirements.txt ]; then
  if [ "${BACKEND}" != "vllm" ]; then
    echo "[install] Installing requirements without vLLM (TRT backend)"
    grep -v -E '^\s*vllm(==|\s|$)' requirements.txt > .tmp.req
    ${PIP} install --only-binary=:all: --prefer-binary -r .tmp.req
    rm -f .tmp.req
  else
    ${PIP} install --only-binary=:all: --prefer-binary -r requirements.txt
  fi
else
  ${PIP} install --only-binary=:all: --prefer-binary -r server/requirements.txt
fi

# Install TensorRT-LLM (Dockerless) if requested backend is TRT-LLM
if [ "${BACKEND}" != "vllm" ]; then
  echo "[install] Verifying TRT-LLM runtime and aligning CUDA bindings"
  # Install mpi4py only if needed: multi-GPU or explicitly requested
  GPU_COUNT=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
  fi
  WANT_MPI=false
  case "${TRTLLM_USE_MPI:-auto}" in
    1|true|yes) WANT_MPI=true ;;
    0|false|no) WANT_MPI=false ;;
    auto) if [ "${GPU_COUNT}" -ge 2 ]; then WANT_MPI=true; fi ;;
  esac
  if [ "$WANT_MPI" = true ]; then
    echo "[install] Installing mpi4py for multi-GPU (detected GPUs: ${GPU_COUNT})"
    set +e
    ${PIP} install --prefer-binary mpi4py>=4.0.0
    set -e
  else
    echo "[install] Single-GPU or TRTLLM_USE_MPI=0 — skipping mpi4py"
  fi
  echo "[install] Ensuring correct CUDA 12.8 Python bindings (avoid conflicting 'cuda' packages)"
  WANT_CUDA_PY=12.8
  HAVE_CUDA_PY=$(python - <<'PY'
try:
  import importlib.metadata as m; v=m.version('cuda-python'); print(v)
except Exception:
  print('none')
PY
)
  if [[ "$HAVE_CUDA_PY" != 12.8.* ]]; then
    set +e
    ${PIP} uninstall -y cuda cuda-bindings cuda_pathfinder cuda-pathfinder 2>/dev/null
    set -e
    ${PIP} install --upgrade --force-reinstall --only-binary=:all: --prefer-binary \
      "cuda-python==12.8.*" "cuda-bindings==12.8.*"
  else
    echo "[install] cuda-python already $HAVE_CUDA_PY — skipping reinstall"
  fi

  echo "[install] Removing deprecated pynvml to silence warnings; installing nvidia-ml-py"
  set +e
  ${PIP} uninstall -y pynvml 2>/dev/null
  set -e
  ${PIP} install -U --only-binary=:all: --prefer-binary nvidia-ml-py
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
  ${PIP} install --no-build-isolation --only-binary=:all: --prefer-binary "flash-attn>=2.5.7"
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
    "added_tokens.json",
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



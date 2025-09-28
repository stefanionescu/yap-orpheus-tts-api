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

echo "[install] Requirements (base + backend)"
pip install -r requirements-base.txt

# Backend-specific Python deps
case "${BACKEND:-vllm}" in
  vllm)
    echo "[install] Installing vLLM backend requirements"
    pip install vllm==0.7.3
    ;;
  trtllm)
    echo "[install] Skipping vLLM; TRT backend will be installed in 01-install-trt.sh"
    ;;
  *)
    echo "[install] Unknown BACKEND='${BACKEND:-}', defaulting to vLLM"
    pip install vllm==0.7.3
    ;;
esac

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



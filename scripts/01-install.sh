#!/usr/bin/env bash
set -euo pipefail
# Load env if present
if [ -f ".env" ]; then source ".env"; fi

# Defaults
: "${PYTHON_VERSION:=3.10}"
: "${VENV_DIR:=$PWD/.venv}"

# Required
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[install] ERROR: HF_TOKEN not set. Export HF_TOKEN in the shell (deployment step)." >&2
  echo "           Example: export HF_TOKEN=\"hf_xxx\"" >&2
  exit 1
fi

echo "[install] Creating venv at ${VENV_DIR}"

# Resolve Python executable
if command -v python${PYTHON_VERSION} >/dev/null 2>&1; then
  PY_EXE=python${PYTHON_VERSION}
elif command -v python3 >/dev/null 2>&1; then
  PY_EXE=python3
elif command -v python >/dev/null 2>&1; then
  PY_EXE=python
else
  echo "[install] ERROR: Python not found. Please install Python ${PYTHON_VERSION}." >&2
  exit 1
fi

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
if [ -z "${CUDA_VER:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VER=$(nvidia-smi | grep -o "CUDA Version: [0-9][0-9]*\.[0-9]*" | awk '{print $3}')
fi
CUDA_MINOR=$(echo "${CUDA_VER:-12.1}" | cut -d. -f1-2 | tr -d '.')
# Map CUDA version to PyTorch index URL
# Supported: cu121, cu124, cu126, cu128 (best-effort mapping)
case "$CUDA_MINOR" in
  120|121) TORCH_IDX="https://download.pytorch.org/whl/cu121" ;;
  122|123|124) TORCH_IDX="https://download.pytorch.org/whl/cu124" ;;
  125|126) TORCH_IDX="https://download.pytorch.org/whl/cu126" ;;
  127|128|129) TORCH_IDX="https://download.pytorch.org/whl/cu128" ;;
  *) TORCH_IDX="https://download.pytorch.org/whl/cu124" ;;
esac

echo "[install] Installing Torch from ${TORCH_IDX}"
pip install --index-url "${TORCH_IDX}" torch --only-binary=:all:

echo "[install] Requirements"
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  pip install -r server/requirements.txt
fi

# Orpheus requires pinning vLLM to 0.7.3 due to regressions in newer versions.
echo "[install] Pinning vLLM==0.7.3"
pip install "vllm==0.7.3"

# Optional: Install FlashAttention 2 prebuilt wheel if available (Linux + NVIDIA)
echo "[install] Checking for FlashAttention prebuilt wheel"
if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
  PY_INFO=$(python - <<'PY'
import torch, platform
print(f"torch={torch.__version__} cuda={torch.version.cuda} platform={platform.system()}")
PY
)
  echo "[install] ${PY_INFO}"
  set +e
  pip install --no-build-isolation --only-binary=:all: "flash-attn>=2.5.7" && echo "[install] Installed flash-attn" || echo "[install] flash-attn wheel unavailable; skipping"
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



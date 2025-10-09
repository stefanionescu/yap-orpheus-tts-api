#!/usr/bin/env bash
set -euo pipefail

source "custom/lib/common.sh"
load_env_if_present
load_environment
source "custom/build/helpers.sh"

echo "[step:env] Validating build environment..."

require_env HF_TOKEN

VENV_DIR="${VENV_DIR:-$PWD/.venv}"
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR" >&2
    echo "Run custom/01-install-trt.sh first" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not detected. GPU required for engine build." >&2
    exit 1
fi

echo "[step:env] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "[step:env] Configuring Hugging Face auth..."
_setup_huggingface_auth

echo "[step:env] OK"



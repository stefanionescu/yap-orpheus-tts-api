#!/usr/bin/env bash
set -euo pipefail

: "${VENV_DIR:=$PWD/.venv}"
: "${TRTLLM_WHEEL_URL:=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.20.0-cp310-cp310-linux_x86_64.whl}"

if [ "$(uname -s)" != "Linux" ]; then
  echo "[install-trt] TensorRT-LLM requires Linux with NVIDIA GPUs. Skipping."
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[install-trt] NVIDIA driver / nvidia-smi not detected. Ensure GPU drivers are installed."
fi

[ -d "${VENV_DIR}" ] || { echo "[install-trt] venv missing. Run scripts/01-install.sh first."; exit 1; }
source "${VENV_DIR}/bin/activate"

echo "[install-trt] Installing mpi4py (optional, safe on single GPU)"
pip install --quiet mpi4py || true

echo "[install-trt] Installing base deps (ensures FastAPI==0.115.4)"
pip install -r requirements-base.txt

echo "[install-trt] Installing TensorRT-LLM wheel"
pip install --extra-index-url https://pypi.nvidia.com "${TRTLLM_WHEEL_URL}"

echo "[install-trt] Installing TRT extras"
pip install -r requirements-trt.txt

echo "[install-trt] Verifying env"
pip check || { echo "[install-trt] Dependency conflict detected"; exit 1; }

echo "[install-trt] Done."



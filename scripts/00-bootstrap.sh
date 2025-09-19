#!/usr/bin/env bash
set -euo pipefail

# Common helpers and env
source "scripts/lib/common.sh"
load_env_if_present

echo "[bootstrap] Checking environment…"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
CUDA_VER=$(detect_cuda_version)
echo "[bootstrap] Detected CUDA $CUDA_VER"

# Python (best effort only; on Runpod you are root and Python is preinstalled)
if ! command -v python${PYTHON_VERSION:-3.10} >/dev/null 2>&1; then
  echo "[bootstrap] python${PYTHON_VERSION:-3.10} not found. Skipping install (expected on managed images).";
fi

# System deps (only if apt-get exists). Allow skipping via SKIP_APT=1
if command -v apt-get >/dev/null 2>&1; then
  if [ "${SKIP_APT:-0}" = "1" ]; then
    echo "[bootstrap] SKIP_APT=1 → skipping apt-get installs"
  else
    echo "[bootstrap] Installing minimal system deps via apt-get"
    apt-get update -y || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git wget curl jq || true
    # If TRT backend in use, install an MPI runtime so mpi4py can link
    if [ "${ORPHEUS_BACKEND:-trtllm}" != "vllm" ]; then
      echo "[bootstrap] Installing MPI runtime (openmpi) for TRT-LLM"
      DEBIAN_FRONTEND=noninteractive apt-get install -y openmpi-bin libopenmpi-dev || true
      echo "[bootstrap] Ensuring CUDA runtime/system libs present (common CUDA paths)"
      DEBIAN_FRONTEND=noninteractive apt-get install -y libcudnn9-cuda-12 libnccl2 || true
    fi
  fi
else
  echo "[bootstrap] apt-get not available. Skipping system packages."
  echo "[bootstrap] Note: TRT-LLM pip wheels require recent NVIDIA driver + CUDA runtime present."
fi

# HF token check
require_env HF_TOKEN

echo "[bootstrap] OK"



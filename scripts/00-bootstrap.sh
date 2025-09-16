#!/usr/bin/env bash
set -euo pipefail

# Load env
if [ -f ".env" ]; then source ".env"; fi

echo "[bootstrap] Checking environment…"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
CUDA_VER=$(nvidia-smi | grep -o "CUDA Version: [0-9][0-9]*\.[0-9]*" | awk '{print $3}')
echo "[bootstrap] Detected CUDA $CUDA_VER"

# Python (best effort only; on Runpod you are root and Python is preinstalled)
if ! command -v python${PYTHON_VERSION:-3.10} >/dev/null 2>&1; then
  echo "[bootstrap] python${PYTHON_VERSION:-3.10} not found. Skipping install (expected on managed images).";
fi

# System deps for audio + build (only if apt-get exists)
if command -v apt-get >/dev/null 2>&1; then
  echo "[bootstrap] Installing system deps via apt-get"
  apt-get update -y || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential git wget curl jq ffmpeg libsndfile1 libportaudio2 || true
else
  echo "[bootstrap] apt-get not available. Skipping system packages."
fi

# HF token check
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[bootstrap] ERROR: HF_TOKEN not set. Export it in .env"
  exit 1
fi

echo "[bootstrap] OK"



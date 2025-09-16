#!/usr/bin/env bash
set -euo pipefail

# Load env
if [ -f ".env" ]; then source ".env"; fi

echo "[bootstrap] Checking environment…"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
CUDA_VER=$(nvidia-smi | grep -o "CUDA Version: [0-9][0-9]*\.[0-9]*" | awk '{print $3}')
echo "[bootstrap] Detected CUDA $CUDA_VER"

# Python
if ! command -v python${PYTHON_VERSION:-3.10} >/dev/null 2>&1; then
  echo "[bootstrap] Installing Python ${PYTHON_VERSION:-3.10}…"
  sudo apt-get update -y
  sudo apt-get install -y python${PYTHON_VERSION:-3.10} python${PYTHON_VERSION:-3.10}-venv python3-pip
fi

# System deps for audio + build
sudo apt-get update -y
sudo apt-get install -y build-essential git wget curl jq ffmpeg libsndfile1 libportaudio2

# HF token check
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[bootstrap] ERROR: HF_TOKEN not set. Export it in .env"
  exit 1
fi

echo "[bootstrap] OK"



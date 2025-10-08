#!/usr/bin/env bash
# =============================================================================
# Runpod Direct Build Script (No Docker-in-Docker)
# =============================================================================
# For use inside Runpod containers where Docker-in-Docker doesn't work.
# This builds the TensorRT engine directly, then creates a deployment image.
#
# Usage: bash docker/scripts/runpod-build.sh --hf-token HF_TOKEN
# =============================================================================

set -euo pipefail

echo "=== Runpod Direct Build (No Docker-in-Docker) ==="

HF_TOKEN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token)
      HF_TOKEN="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --hf-token HF_TOKEN"
      echo "This builds the TensorRT engine directly in the current environment."
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: --hf-token is required" >&2
  exit 1
fi

export HF_TOKEN="$HF_TOKEN"

echo "[runpod] Building TensorRT engine directly in current environment..."
echo "[runpod] This will take 30-45 minutes..."

# Use the existing scripts pipeline
echo "[runpod] Running bootstrap..."
bash scripts/00-bootstrap.sh

echo "[runpod] Installing dependencies..."
bash scripts/01-install-trt.sh

echo "[runpod] Building TensorRT engine..."
bash scripts/02-build.sh

echo "[runpod] ✓ Build complete!"
echo "[runpod] Engine location: $(find . -name 'rank0.engine' -type f)"
echo ""
echo "To start the server:"
echo "  export YAP_API_KEY=your_key"
echo "  export TRTLLM_ENGINE_DIR=\$(find \$PWD -name 'rank0.engine' -type f | head -1 | xargs dirname)"
echo "  bash scripts/03-run-server.sh"

#!/usr/bin/env bash
set -euo pipefail

# Build the base image that replicates bootstrap + TRT install

IMAGE_NAME=${IMAGE_NAME:-yapai/orpheus-trtllm-base}
IMAGE_TAG=${IMAGE_TAG:-cu126-py310-trtllm1.0.0}

# Optional: pass HF_TOKEN at build time if you want to bake auth (not recommended)
BUILD_ARGS=(
  --build-arg PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}
  --build-arg TRTLLM_WHEEL_URL=${TRTLLM_WHEEL_URL:-https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl}
  --build-arg HF_TOKEN=${HF_TOKEN:?HF_TOKEN is required: export HF_TOKEN=...}
  --build-arg MODEL_ID=${MODEL_ID:-canopylabs/orpheus-3b-0.1-ft}
  --build-arg TRTLLM_REPO_URL=${TRTLLM_REPO_URL:-https://github.com/Yap-With-AI/TensorRT-LLM.git}
)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$REPO_ROOT"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} (requires HF_TOKEN)"
docker build \
  -f docker/Dockerfile \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  "${BUILD_ARGS[@]}" \
  .

echo "\nBuilt image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Use this as a base in cloud to skip bootstrap and TRT installs."



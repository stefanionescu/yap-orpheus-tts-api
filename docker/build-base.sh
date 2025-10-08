#!/usr/bin/env bash
set -euo pipefail

# Build the base image that replicates bootstrap + TRT install

IMAGE_NAME=${IMAGE_NAME:-yapai/orpheus-trtllm-base}
IMAGE_TAG=${IMAGE_TAG:-cu124-py310-trtllm1.0.0}

# Optional: pass HF_TOKEN at build time if you want to bake auth (not recommended)
BUILD_ARGS=(
  --build-arg PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}
  --build-arg TRTLLM_WHEEL_URL=${TRTLLM_WHEEL_URL:-https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl}
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  BUILD_ARGS+=(--build-arg HF_TOKEN="${HF_TOKEN}")
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$REPO_ROOT"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG}"
docker build \
  -f docker/Dockerfile \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  "${BUILD_ARGS[@]}" \
  .

echo "\nBuilt image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Use this as a base in cloud to skip bootstrap and TRT installs."



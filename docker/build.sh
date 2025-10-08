#!/usr/bin/env bash
set -euo pipefail

# Build the base image that replicates bootstrap + TRT install

IMAGE_NAME=${IMAGE_NAME:-sionescu/orpheus-trtllm}
IMAGE_TAG=${IMAGE_TAG:-cu121-py310}

# Optional: pass HF_TOKEN at build time if you want to bake auth (not recommended)
BUILD_ARGS=(
  --build-arg PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
  --build-arg TRTLLM_WHEEL_URL=${TRTLLM_WHEEL_URL:-https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl}
  ${HF_TOKEN:+--build-arg HF_TOKEN=$HF_TOKEN}
  --build-arg MODEL_ID=${MODEL_ID:-canopylabs/orpheus-3b-0.1-ft}
  --build-arg TRTLLM_REPO_URL=${TRTLLM_REPO_URL:-https://github.com/Yap-With-AI/TensorRT-LLM.git}
)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$REPO_ROOT"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} (platform linux/amd64)"
docker build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  ${HF_TOKEN:+--secret id=HF_TOKEN,env=HF_TOKEN} \
  "${BUILD_ARGS[@]}" \
  .

echo "\nBuilt image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Use this as a base in cloud to skip bootstrap and TRT installs."


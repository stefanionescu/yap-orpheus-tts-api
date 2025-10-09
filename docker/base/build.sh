#!/usr/bin/env bash
set -euo pipefail

# Build the base image that replicates bootstrap + TRT install

IMAGE_NAME=${IMAGE_NAME:-sionescu/orpheus-trtllm}
IMAGE_TAG=${IMAGE_TAG:-cu121-py310}
PUSH_IMAGE=${PUSH_IMAGE:-0}

usage() {
  cat <<'EOF'
Usage: docker/base/build.sh [--push]

Builds the Orpheus TRT-LLM base image. Pass --push (or set PUSH_IMAGE=1) to
push the resulting tag to the configured registry (default docker.io).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)
      PUSH_IMAGE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Optional: pass HF_TOKEN at build time if you want to bake auth (not recommended)
BUILD_ARGS=(
  --build-arg PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
  --build-arg TRTLLM_WHEEL_URL=${TRTLLM_WHEEL_URL:-https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl}
  ${HF_TOKEN:+--build-arg HF_TOKEN=$HF_TOKEN}
  --build-arg MODEL_ID=${MODEL_ID:-canopylabs/orpheus-3b-0.1-ft}
  --build-arg TRTLLM_REPO_URL=${TRTLLM_REPO_URL:-https://github.com/Yap-With-AI/TensorRT-LLM.git}
)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

cd "$REPO_ROOT"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} (platform linux/amd64)"
docker build \
  --platform linux/amd64 \
  -f docker/base/Dockerfile \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  ${HF_TOKEN:+--secret id=HF_TOKEN,env=HF_TOKEN} \
  "${BUILD_ARGS[@]}" \
  .

echo "\nBuilt image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Use this as a base in cloud to skip bootstrap and TRT installs."

if [[ "$PUSH_IMAGE" == "1" ]]; then
  echo "\nPushing ${IMAGE_NAME}:${IMAGE_TAG}..."
  docker push "${IMAGE_NAME}:${IMAGE_TAG}"
  echo "Pushed ${IMAGE_NAME}:${IMAGE_TAG}"
else
  echo "\nTo push: docker push ${IMAGE_NAME}:${IMAGE_TAG}"
  echo "(Or rerun with --push / PUSH_IMAGE=1)"
fi

#!/usr/bin/env bash
set -euo pipefail

# Build the fast production image with dependencies only

IMAGE_NAME=${IMAGE_NAME:-sionescu/orpheus-trtllm-int4-fast}
IMAGE_TAG=${IMAGE_TAG:-cu121-py310}
PUSH_IMAGE=${PUSH_IMAGE:-1}

usage() {
  cat <<'EOF'
Usage: docker/fast/build.sh

Builds the Orpheus TTS fast production image with dependencies only, and pushes
to the configured registry (default docker.io) by default. To skip pushing, set
PUSH_IMAGE=0.
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

# Build arguments for customization
BUILD_ARGS=(
  --build-arg PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
  --build-arg TRTLLM_WHEEL_URL=${TRTLLM_WHEEL_URL:-https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl}
)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

cd "$REPO_ROOT"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} (platform linux/amd64)"
echo "This is a lean production image with dependencies only (~10-15GB)"

docker build \
  --platform linux/amd64 \
  -f docker/fast/Dockerfile \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  "${BUILD_ARGS[@]}" \
  .

echo ""
echo "Built image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Image size: ~10-15GB (lean production build)"
echo "Ready for production deployment with runtime model/engine mounting"

if [[ "$PUSH_IMAGE" == "1" ]]; then
  echo ""
  echo "Pushing ${IMAGE_NAME}:${IMAGE_TAG}..."
  docker push "${IMAGE_NAME}:${IMAGE_TAG}"
  echo "Pushed ${IMAGE_NAME}:${IMAGE_TAG}"
else
  echo ""
  echo "To push: docker push ${IMAGE_NAME}:${IMAGE_TAG}"
  echo "(Or rerun with --push / PUSH_IMAGE=1)"
fi

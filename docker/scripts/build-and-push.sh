#!/usr/bin/env bash
# =============================================================================
# Docker Build and Push Script
# =============================================================================
# Builds the Orpheus 3B TTS Docker image with pre-built TensorRT engine
# and pushes it to Docker Hub for rapid deployment.
#
# Usage: bash docker/scripts/build-and-push.sh
# Environment: Requires HF_TOKEN, DOCKER_USERNAME, DOCKER_PASSWORD
# =============================================================================

set -euo pipefail

echo "=== Orpheus 3B TTS Docker Build & Push ==="

# =============================================================================
# Configuration and Validation
# =============================================================================

# Required environment variables
REQUIRED_VARS=(
    "HF_TOKEN"
    "DOCKER_USERNAME" 
    "DOCKER_PASSWORD"
)

# Optional configuration
IMAGE_NAME="${IMAGE_NAME:-orpheus-3b-tts}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "[build] Validating environment..."

# Check required variables
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var environment variable not set" >&2
        echo "Please export $var before running this script" >&2
        exit 1
    fi
done

# Validate Docker is available
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker not found. Please install Docker first." >&2
    exit 1
fi

# Check GPU availability (required for engine build)
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. GPU required for TensorRT engine build." >&2
    exit 1
fi

echo "[build] Environment validation passed"
echo "[build] Target image: $FULL_IMAGE_NAME"
echo "[build] HF Token: ${HF_TOKEN:0:8}..."

# =============================================================================
# Docker Authentication
# =============================================================================

echo "[build] Authenticating with Docker Hub..."
echo "$DOCKER_PASSWORD" | docker login --username "$DOCKER_USERNAME" --password-stdin

if [ $? -ne 0 ]; then
    echo "ERROR: Docker Hub authentication failed" >&2
    exit 1
fi

echo "[build] ✓ Docker Hub authentication successful"

# =============================================================================
# Docker Build Process
# =============================================================================

echo "[build] Starting Docker build process..."
echo "[build] This will take 30-45 minutes (includes TensorRT engine build)"

# Change to project root for build context
cd "$(dirname "$0")/../.."

# Build Docker image with all required build arguments
docker build \
    --file docker/Dockerfile \
    --tag "$FULL_IMAGE_NAME" \
    --build-arg HF_TOKEN="$HF_TOKEN" \
    --build-arg DOCKER_USERNAME="$DOCKER_USERNAME" \
    --build-arg DOCKER_PASSWORD="$DOCKER_PASSWORD" \
    --build-arg IMAGE_NAME="$IMAGE_NAME" \
    --progress=plain \
    .

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed" >&2
    exit 1
fi

echo "[build] ✓ Docker build completed successfully"

# =============================================================================
# Image Validation
# =============================================================================

echo "[build] Validating built image..."

# Check image exists
if ! docker image inspect "$FULL_IMAGE_NAME" >/dev/null 2>&1; then
    echo "ERROR: Built image not found" >&2
    exit 1
fi

# Get image size
IMAGE_SIZE=$(docker image inspect "$FULL_IMAGE_NAME" --format='{{.Size}}' | numfmt --to=iec)
echo "[build] Image size: $IMAGE_SIZE"

# Quick validation that engine exists in image
echo "[build] Validating TensorRT engine in image..."
docker run --rm "$FULL_IMAGE_NAME" ls -la /app/models/orpheus-trt-int4-awq/rank0.engine

if [ $? -ne 0 ]; then
    echo "ERROR: TensorRT engine not found in built image" >&2
    exit 1
fi

echo "[build] ✓ Image validation passed"

# =============================================================================
# Push to Docker Hub
# =============================================================================

echo "[build] Pushing image to Docker Hub..."
echo "[build] Target: $FULL_IMAGE_NAME"

docker push "$FULL_IMAGE_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: Docker push failed" >&2
    exit 1
fi

echo "[build] ✓ Image pushed successfully to Docker Hub"

# =============================================================================
# Build Summary
# =============================================================================

echo ""
echo "=== Build Summary ==="
echo "Image: $FULL_IMAGE_NAME"
echo "Size: $IMAGE_SIZE"
echo "Registry: Docker Hub"
echo "Status: ✓ Ready for deployment"
echo ""
echo "To deploy on any GPU-enabled machine:"
echo "  docker pull $FULL_IMAGE_NAME"
echo "  docker run --gpus all -p 8000:8000 $FULL_IMAGE_NAME"
echo ""
echo "Or use the deployment script:"
echo "  bash docker/scripts/deploy.sh $FULL_IMAGE_NAME"

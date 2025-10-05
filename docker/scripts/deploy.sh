#!/usr/bin/env bash
# =============================================================================
# Rapid Deployment Script for Pre-built Docker Image
# =============================================================================
# Deploys the pre-built Orpheus 3B TTS Docker image with 2-5 minute startup.
# Uses the pre-built TensorRT engine from Docker Hub for instant deployment.
#
# Usage: bash docker/scripts/deploy.sh [IMAGE_NAME]
# Environment: Requires GPU-enabled Docker host
# =============================================================================

set -euo pipefail

echo "=== Orpheus 3B TTS Rapid Deployment ==="

# =============================================================================
# Configuration
# =============================================================================

# Default image name (can be overridden)
DEFAULT_IMAGE="orpheus-3b-tts:latest"
IMAGE_NAME="${1:-$DEFAULT_IMAGE}"

# Runtime configuration
CONTAINER_NAME="orpheus-tts-server"
HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="8000"

echo "[deploy] Target image: $IMAGE_NAME"
echo "[deploy] Container name: $CONTAINER_NAME"
echo "[deploy] Port mapping: $HOST_PORT:$CONTAINER_PORT"

# =============================================================================
# Environment Validation
# =============================================================================

echo "[deploy] Validating deployment environment..."

# Check Docker availability
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker not found. Please install Docker first." >&2
    exit 1
fi

# Check GPU support
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. GPU required for TTS server." >&2
    exit 1
fi

# Check Docker GPU support (nvidia-container-runtime)
if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: Docker GPU support not available." >&2
    echo "Please install nvidia-container-toolkit:" >&2
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" >&2
    exit 1
fi

echo "[deploy] ✓ Environment validation passed"

# =============================================================================
# Container Cleanup
# =============================================================================

echo "[deploy] Cleaning up any existing containers..."

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[deploy] Stopping existing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# =============================================================================
# Image Pull and Validation
# =============================================================================

echo "[deploy] Pulling latest image from registry..."
docker pull "$IMAGE_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to pull image: $IMAGE_NAME" >&2
    echo "Make sure the image exists and you have access to it." >&2
    exit 1
fi

# Validate image has required components
echo "[deploy] Validating image components..."
docker run --rm "$IMAGE_NAME" test -f /app/models/orpheus-trt-int4-awq/rank0.engine

if [ $? -ne 0 ]; then
    echo "ERROR: Image validation failed - TensorRT engine not found" >&2
    exit 1
fi

echo "[deploy] ✓ Image validation passed"

# =============================================================================
# Container Deployment
# =============================================================================

echo "[deploy] Starting TTS server container..."

# Run container with GPU support and optimized settings
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --restart unless-stopped \
    -p "$HOST_PORT:$CONTAINER_PORT" \
    --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    "$IMAGE_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start container" >&2
    exit 1
fi

echo "[deploy] ✓ Container started successfully"

# =============================================================================
# Health Check and Validation
# =============================================================================

echo "[deploy] Waiting for server to start..."

# Wait for container to be healthy
MAX_WAIT=300  # 5 minutes
WAIT_TIME=0
SLEEP_INTERVAL=5

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if docker ps --format '{{.Names}}\t{{.Status}}' | grep "$CONTAINER_NAME" | grep -q "healthy"; then
        break
    fi
    
    # Check if container is still running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "ERROR: Container stopped unexpectedly" >&2
        echo "Container logs:" >&2
        docker logs "$CONTAINER_NAME" 2>&1 | tail -20
        exit 1
    fi
    
    echo "[deploy] Waiting for health check... (${WAIT_TIME}s/${MAX_WAIT}s)"
    sleep $SLEEP_INTERVAL
    WAIT_TIME=$((WAIT_TIME + SLEEP_INTERVAL))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "ERROR: Server failed to start within $MAX_WAIT seconds" >&2
    echo "Container logs:" >&2
    docker logs "$CONTAINER_NAME" 2>&1 | tail -20
    exit 1
fi

# Final health check via HTTP
echo "[deploy] Performing final health check..."
sleep 5  # Give server a moment to fully initialize

if command -v curl >/dev/null 2>&1; then
    if curl -f -s "http://localhost:$HOST_PORT/healthz" >/dev/null; then
        echo "[deploy] ✓ Health check passed"
    else
        echo "WARNING: Health check endpoint not responding" >&2
    fi
else
    echo "[deploy] curl not available, skipping HTTP health check"
fi

# =============================================================================
# Deployment Summary
# =============================================================================

echo ""
echo "=== Deployment Summary ==="
echo "Container: $CONTAINER_NAME"
echo "Image: $IMAGE_NAME"
echo "Status: ✓ Running"
echo "Health: ✓ Healthy"
echo "Endpoint: http://localhost:$HOST_PORT"
echo "WebSocket: ws://localhost:$HOST_PORT/ws/tts"
echo ""
echo "Quick test:"
echo "  curl http://localhost:$HOST_PORT/healthz"
echo ""
echo "View logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "Stop server:"
echo "  docker stop $CONTAINER_NAME"

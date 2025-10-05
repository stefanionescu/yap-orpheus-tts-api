#!/usr/bin/env bash
# =============================================================================
# Quick Start Script for Runpod/Cloud GPU Deployment
# =============================================================================
# Ultra-fast deployment script for cloud GPU instances (Runpod, etc.).
# Pulls pre-built image and starts server in 2-5 minutes.
#
# Usage: bash <(curl -s https://raw.githubusercontent.com/.../docker/scripts/quick-start.sh)
# Or: wget -O - https://raw.githubusercontent.com/.../docker/scripts/quick-start.sh | bash
# =============================================================================

set -euo pipefail

echo "=== Orpheus 3B TTS Quick Start ==="
echo "Deploying pre-built TTS server on GPU instance..."

# =============================================================================
# Configuration
# =============================================================================

# Default Docker image (update this to your Docker Hub image)
DEFAULT_DOCKER_IMAGE="${DOCKER_IMAGE:-orpheus-3b-tts:latest}"
CONTAINER_NAME="orpheus-tts"
HOST_PORT="${PORT:-8000}"

echo "[quick-start] Using image: $DEFAULT_DOCKER_IMAGE"
echo "[quick-start] Port: $HOST_PORT"

# =============================================================================
# System Preparation
# =============================================================================

echo "[quick-start] Preparing system..."

# Update package lists (non-blocking)
apt-get update -y >/dev/null 2>&1 || true

# Install essential tools if missing
if ! command -v curl >/dev/null 2>&1; then
    echo "[quick-start] Installing curl..."
    apt-get install -y curl >/dev/null 2>&1 || true
fi

if ! command -v wget >/dev/null 2>&1; then
    echo "[quick-start] Installing wget..."
    apt-get install -y wget >/dev/null 2>&1 || true
fi

# =============================================================================
# Docker Installation (if needed)
# =============================================================================

if ! command -v docker >/dev/null 2>&1; then
    echo "[quick-start] Installing Docker..."
    
    # Install Docker using official script
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh >/dev/null 2>&1
    rm get-docker.sh
    
    # Start Docker service
    systemctl start docker
    systemctl enable docker
    
    echo "[quick-start] ✓ Docker installed"
else
    echo "[quick-start] ✓ Docker already available"
fi

# =============================================================================
# NVIDIA Container Toolkit (if needed)
# =============================================================================

if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
    echo "[quick-start] Installing NVIDIA Container Toolkit..."
    
    # Add NVIDIA package repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L "https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list" | \
        tee /etc/apt/sources.list.d/nvidia-docker.list
    
    # Install nvidia-container-toolkit
    apt-get update -y
    apt-get install -y nvidia-container-toolkit
    
    # Restart Docker
    systemctl restart docker
    
    echo "[quick-start] ✓ NVIDIA Container Toolkit installed"
else
    echo "[quick-start] ✓ GPU support already available"
fi

# =============================================================================
# Rapid Deployment
# =============================================================================

echo "[quick-start] Deploying TTS server..."

# Stop any existing container
docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true

# Pull and run the pre-built image
echo "[quick-start] Pulling pre-built image (this may take 2-3 minutes)..."
docker pull "$DEFAULT_DOCKER_IMAGE"

echo "[quick-start] Starting TTS server..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --restart unless-stopped \
    -p "$HOST_PORT:8000" \
    --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "$DEFAULT_DOCKER_IMAGE"

# =============================================================================
# Health Check
# =============================================================================

echo "[quick-start] Waiting for server to start..."

# Wait for health check
MAX_WAIT=120  # 2 minutes
WAIT_TIME=0

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -f -s "http://localhost:$HOST_PORT/healthz" >/dev/null 2>&1; then
        break
    fi
    
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "ERROR: Container failed to start" >&2
        docker logs "$CONTAINER_NAME" 2>&1 | tail -10
        exit 1
    fi
    
    echo "[quick-start] Waiting for server... (${WAIT_TIME}s)"
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "ERROR: Server failed to start within $MAX_WAIT seconds" >&2
    docker logs "$CONTAINER_NAME" 2>&1 | tail -10
    exit 1
fi

# =============================================================================
# Success Summary
# =============================================================================

# Get public IP if available
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "localhost")

echo ""
echo "🚀 === TTS Server Ready! ==="
echo "Status: ✅ Running"
echo "Local: http://localhost:$HOST_PORT"
echo "Public: http://$PUBLIC_IP:$HOST_PORT"
echo "WebSocket: ws://$PUBLIC_IP:$HOST_PORT/ws/tts"
echo ""
echo "🔍 Quick Test:"
echo "curl http://localhost:$HOST_PORT/healthz"
echo ""
echo "📋 Management:"
echo "View logs: docker logs -f $CONTAINER_NAME"
echo "Stop server: docker stop $CONTAINER_NAME"
echo "Restart: docker restart $CONTAINER_NAME"
echo ""
echo "🎉 Ready for TTS requests!"

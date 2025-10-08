#!/usr/bin/env bash
# =============================================================================
# Universal Cleanup Script
# =============================================================================
# Cleans up build artifacts and processes for both Docker and containerized
# environments. Automatically detects environment and cleans appropriately.
# Does NOT uninstall Docker or system components.
#
# Usage: bash docker/scripts/stop.sh [--all]
# Options:
#   --all    Also remove Docker images and built engines
# =============================================================================

set -euo pipefail

echo "=== Orpheus 3B TTS - Universal Cleanup ==="

# Parse arguments
REMOVE_IMAGES=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      REMOVE_IMAGES=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--all]"
      echo "  --all    Also remove Docker images (not just containers)"
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# =============================================================================
# Environment Detection
# =============================================================================

# Check if we're in a containerized environment
IN_CONTAINER=false
if [ -f /.dockerenv ] || [ -n "${RUNPOD_POD_ID:-}" ] || [ -n "${VAST_CONTAINERD:-}" ]; then
    IN_CONTAINER=true
fi

echo "[cleanup] Environment: $([ "$IN_CONTAINER" = true ] && echo "Containerized (Runpod/Vast/Docker)" || echo "Clean machine with Docker")"

# =============================================================================
# Stop TTS Server Processes
# =============================================================================

echo "[cleanup] Stopping TTS server processes..."

# Stop uvicorn/FastAPI processes
UVICORN_PIDS=$(pgrep -f "uvicorn.*server.server:app" || true)
if [ -n "$UVICORN_PIDS" ]; then
    echo "[cleanup] Stopping uvicorn processes: $UVICORN_PIDS"
    echo "$UVICORN_PIDS" | xargs -r kill -TERM 2>/dev/null || true
    sleep 2
    # Force kill if still running
    echo "$UVICORN_PIDS" | xargs -r kill -KILL 2>/dev/null || true
    echo "[cleanup] ✓ TTS server processes stopped"
else
    echo "[cleanup] No TTS server processes found"
fi

# Clean up PID files
if [ -f ".run/server.pid" ]; then
    rm -f .run/server.pid
fi
if [ -f ".run/setup-pipeline.pid" ]; then
    rm -f .run/setup-pipeline.pid
fi

# =============================================================================
# Docker Environment Cleanup
# =============================================================================

if [ "$IN_CONTAINER" = false ] && command -v docker >/dev/null 2>&1; then
    echo "[cleanup] Stopping and removing Orpheus TTS containers..."

    # Stop and remove containers with orpheus in the name
    CONTAINERS=$(docker ps -a --format "{{.Names}}" | grep -E "(orpheus|tts)" || true)
    if [ -n "$CONTAINERS" ]; then
        echo "[cleanup] Found containers: $CONTAINERS"
        echo "$CONTAINERS" | xargs -r docker stop 2>/dev/null || true
        echo "$CONTAINERS" | xargs -r docker rm 2>/dev/null || true
        echo "[cleanup] ✓ Containers stopped and removed"
    else
        echo "[cleanup] No Orpheus TTS containers found"
    fi
else
    echo "[cleanup] Skipping Docker container cleanup (containerized environment or Docker not available)"
fi

# =============================================================================
# Remove Images and Built Engines (if --all specified)
# =============================================================================

if [ "$REMOVE_IMAGES" = true ]; then
    # Remove Docker images (only if not in container and Docker available)
    if [ "$IN_CONTAINER" = false ] && command -v docker >/dev/null 2>&1; then
        echo "[cleanup] Removing Orpheus TTS Docker images..."
        
        # Remove images with orpheus in the name
        IMAGES=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "(orpheus|tts)" || true)
        if [ -n "$IMAGES" ]; then
            echo "[cleanup] Found images: $IMAGES"
            echo "$IMAGES" | xargs -r docker rmi -f 2>/dev/null || true
            echo "[cleanup] ✓ Images removed"
        else
            echo "[cleanup] No Orpheus TTS images found"
        fi
    fi
    
    # Remove built TensorRT engines (both environments)
    echo "[cleanup] Removing built TensorRT engines..."
    ENGINE_DIRS=(
        "models/orpheus-trt-int4-awq"
        "models/orpheus-trtllm-ckpt-int4-awq"
        ".venv"
        ".trtllm-repo"
    )
    
    for dir in "${ENGINE_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            echo "[cleanup] Removing $dir"
            rm -rf "$dir"
        fi
    done
    
    # Remove logs and runtime files
    if [ -d "logs" ]; then
        echo "[cleanup] Removing logs directory"
        rm -rf logs
    fi
    
    if [ -d ".run" ]; then
        echo "[cleanup] Removing .run directory"
        rm -rf .run
    fi
fi

# =============================================================================
# Clean Build Artifacts
# =============================================================================

echo "[cleanup] Cleaning build artifacts..."

# Remove temporary build directories
BUILD_DIRS=(
    "/tmp/yap-orpheus-build"
    "/tmp/tensorrt-llm"
)

for dir in "${BUILD_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "[cleanup] Removing $dir"
        rm -rf "$dir"
    fi
done

# Clean Docker build cache (orpheus-related only)
echo "[cleanup] Cleaning Docker build cache..."
docker builder prune -f --filter "label=orpheus" 2>/dev/null || true

# =============================================================================
# Clean Docker System (safe cleanup)
# =============================================================================

if [ "$IN_CONTAINER" = false ] && command -v docker >/dev/null 2>&1; then
    echo "[cleanup] Running safe Docker system cleanup..."

    # Remove dangling images and unused networks
    docker image prune -f 2>/dev/null || true
    docker network prune -f 2>/dev/null || true

    # Remove unused volumes (be careful - only remove anonymous volumes)
    docker volume prune -f 2>/dev/null || true
else
    echo "[cleanup] Skipping Docker system cleanup (containerized environment or Docker not available)"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=== Cleanup Summary ==="
echo "Environment: $([ "$IN_CONTAINER" = true ] && echo "Containerized" || echo "Clean machine")"
echo "✓ Stopped TTS server processes"

if [ "$IN_CONTAINER" = false ] && command -v docker >/dev/null 2>&1; then
    echo "✓ Stopped and removed Orpheus TTS containers"
    if [ "$REMOVE_IMAGES" = true ]; then
        echo "✓ Removed Orpheus TTS Docker images"
    fi
    echo "✓ Cleaned Docker cache and unused resources"
fi

if [ "$REMOVE_IMAGES" = true ]; then
    echo "✓ Removed built TensorRT engines and models"
    echo "✓ Removed logs and runtime files"
fi

echo "✓ Cleaned build artifacts and temporary files"
echo ""
echo "System components remain installed and ready for use."
echo ""
echo "To rebuild:"
if [ "$IN_CONTAINER" = true ]; then
    echo "  bash docker/scripts/containerized-build.sh --hf-token HF_TOKEN"
else
    echo "  bash docker/scripts/build.sh --docker-username USER --docker-password PASS"
fi

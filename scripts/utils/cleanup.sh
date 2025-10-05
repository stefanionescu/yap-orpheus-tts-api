#!/usr/bin/env bash
# =============================================================================
# System Cleanup Script
# =============================================================================
# Comprehensive cleanup script that stops all processes and optionally removes
# build artifacts, caches, and temporary files. Includes GPU memory cleanup.
#
# Usage: bash scripts/utils/cleanup.sh [OPTIONS]
# Options:
#   --clean-install  Remove Python venv and package caches
#   --clean-system   Remove system package caches
#   --clean-trt      Remove TensorRT-LLM build artifacts
# =============================================================================

set -euo pipefail

echo "=== System Cleanup ==="

# =============================================================================
# Configuration and Argument Parsing
# =============================================================================

# Default paths (must match build scripts)
TRTLLM_REPO_DIR="${TRTLLM_REPO_DIR:-$PWD/.trtllm-repo}"
MODELS_DIR="${MODELS_DIR:-$PWD/models}"

# Parse cleanup options
CLEAN_INSTALL=0
CLEAN_SYSTEM=0
CLEAN_TRT=0

for arg in "$@"; do
    case "$arg" in
        --clean-install)
            CLEAN_INSTALL=1
            echo "[cleanup] Will remove Python environment and caches"
            ;;
        --clean-system)
            CLEAN_SYSTEM=1
            echo "[cleanup] Will remove system package caches"
            ;;
        --clean-trt)
            CLEAN_TRT=1
            echo "[cleanup] Will remove TensorRT-LLM build artifacts"
            ;;
        --help|-h)
            echo "Usage: $0 [--clean-install] [--clean-system] [--clean-trt]"
            echo ""
            echo "Options:"
            echo "  --clean-install  Remove Python venv and package caches"
            echo "  --clean-system   Remove system package caches"
            echo "  --clean-trt      Remove TensorRT-LLM build artifacts"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Process Termination
# =============================================================================

echo "[cleanup] Stopping server processes..."
_stop_server_processes

echo "[cleanup] Stopping background processes..."
_stop_background_processes

echo "[cleanup] Cleaning GPU memory..."
_cleanup_gpu_memory

# =============================================================================
# File System Cleanup
# =============================================================================

echo "[cleanup] Removing runtime files..."
_cleanup_runtime_files

if [ "$CLEAN_INSTALL" = "1" ]; then
    echo "[cleanup] Removing Python environment and caches..."
    _cleanup_python_environment
fi

if [ "$CLEAN_TRT" = "1" ]; then
    echo "[cleanup] Removing TensorRT-LLM artifacts..."
    _cleanup_tensorrt_artifacts
fi

if [ "$CLEAN_SYSTEM" = "1" ]; then
    echo "[cleanup] Removing system caches..."
    _cleanup_system_caches
fi

echo "[cleanup] Removing workspace temporary files..."
_cleanup_workspace_files

echo "[cleanup] ✓ Cleanup completed successfully"

# =============================================================================
# Helper Functions
# =============================================================================

_stop_server_processes() {
    # Stop FastAPI server
    if [ -f .run/server.pid ]; then
        local pid
        pid=$(cat .run/server.pid 2>/dev/null || true)
        if [ -n "$pid" ]; then
            echo "[cleanup] Stopping server (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
        fi
        rm -f .run/server.pid
    fi
    
    # Kill any remaining uvicorn processes
    pkill -f "uvicorn server.server:app" 2>/dev/null || true
    sleep 1
}

_stop_background_processes() {
    # Stop run-all pipeline
    if [ -f .run/run-all.pid ]; then
        local pid
        pid=$(cat .run/run-all.pid 2>/dev/null || true)
        if [ -n "$pid" ]; then
            echo "[cleanup] Stopping pipeline (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f .run/run-all.pid
    fi
    
    # Kill any remaining pipeline processes
    pkill -f "scripts/run-all.sh" 2>/dev/null || true
}

_cleanup_gpu_memory() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    
    echo "[cleanup] Terminating GPU compute processes..."
    
    # Get list of GPU compute processes
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    
    if [ -n "$pids" ]; then
        echo "$pids" | xargs -r -n1 kill 2>/dev/null || true
        sleep 1
        
        # Force kill any remaining processes
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs -r -n1 kill -9 2>/dev/null || true
        fi
    fi
    
    # Kill specific process types
    pkill -9 -f "python.*tensorrt" 2>/dev/null || true
    pkill -9 -f "python.*trtllm" 2>/dev/null || true
    pkill -9 -f "mpirun" 2>/dev/null || true
    pkill -9 -f "mpi4py" 2>/dev/null || true
    sleep 1
    
    # Clean up device files
    if command -v fuser >/dev/null 2>&1; then
        for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia0 /dev/nvidia1; do
            [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
        done
    fi
    
    # Clean up shared memory and temporary files
    rm -rf /dev/shm/nvidia* /dev/shm/cuda* /tmp/cuda* /tmp/.X11-unix/* 2>/dev/null || true
    
    # Clean up IPC resources
    if command -v ipcs >/dev/null 2>&1; then
        local current_user
        current_user=$(whoami)
        ipcs -s | grep "$current_user" | awk '{print $2}' | xargs -r -n1 ipcrm -s 2>/dev/null || true
        ipcs -q | grep "$current_user" | awk '{print $2}' | xargs -r -n1 ipcrm -q 2>/dev/null || true
        ipcs -m | grep "$current_user" | awk '{print $2}' | xargs -r -n1 ipcrm -m 2>/dev/null || true
    fi
    
    sleep 2
    
    # Attempt GPU reset if no processes remain
    local remaining
    remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    if [ -z "$remaining" ]; then
        local gpu_index="${GPU_INDEX:-0}"
        nvidia-smi --gpu-reset -i "$gpu_index" 2>/dev/null || true
        nvidia-smi -i "$gpu_index" -pm 0 2>/dev/null || true
        nvidia-smi -i "$gpu_index" -pm 1 2>/dev/null || true
    fi
}

_cleanup_runtime_files() {
    rm -rf .run logs 2>/dev/null || true
}

_cleanup_python_environment() {
    # Remove virtual environment
    rm -rf .venv || true
    
    # Remove HuggingFace caches
    rm -rf ~/.cache/huggingface ~/.local/share/huggingface || true
    [ -n "${HF_HOME:-}" ] && rm -rf "${HF_HOME}" || true
    [ -n "${HF_HUB_CACHE:-}" ] && rm -rf "${HF_HUB_CACHE}" || true
    
    # Remove PyTorch caches
    rm -rf ~/.cache/torch ~/.cache/torch_extensions ~/.torch || true
    rm -rf /tmp/torch* /tmp/.torch* /dev/shm/torch* || true
    
    # Remove other Python caches
    rm -rf ~/.triton ~/.cache/triton /tmp/triton* || true
    rm -rf ~/.cache/pip ~/.cache/hf_transfer ~/.cache/clip ~/.cache/transformers || true
    rm -rf /tmp/.hf* 2>/dev/null || true
}

_cleanup_tensorrt_artifacts() {
    # Remove TensorRT-LLM repository and models
    rm -rf "$TRTLLM_REPO_DIR" "$MODELS_DIR" || true
    
    # Remove Python cache files
    find "$PWD" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    
    # Remove TensorRT caches
    rm -rf ~/.cache/tensorrt* ~/.cache/nv_tensorrt ~/.cache/nvidia ~/.nv || true
    rm -rf ~/.cache/modelopt ~/.cache/model_optimizer ~/.cache/nvidia_modelopt || true
    
    # Remove temporary build files
    rm -rf /tmp/trt* /tmp/tensorrt* /tmp/modelopt* /tmp/quantiz* /tmp/calib* 2>/dev/null || true
    rm -rf /dev/shm/trt* /dev/shm/quantiz* 2>/dev/null || true
}

_cleanup_system_caches() {
    # Clean package manager caches
    if command -v apt-get >/dev/null 2>&1; then
        apt-get clean || true
    fi
    
    # Remove temporary system files
    rm -rf /var/lib/apt/lists/* /tmp/pip-* /tmp/tmp* /var/tmp/* 2>/dev/null || true
}

_cleanup_workspace_files() {
    # Remove workspace temporary files
    rm -rf audio/*.wav .pytest_cache .mypy_cache .ruff_cache *.egg-info 2>/dev/null || true
}

#!/usr/bin/env bash
# =============================================================================
# System Cleanup Script
# =============================================================================
# Comprehensive cleanup script that stops all processes and optionally removes
# build artifacts, caches, and temporary files. Includes GPU memory cleanup.
# Optimized for cloud/container environments with aggressive cleanup options.
#
# Usage: bash scripts/utils/cleanup.sh [OPTIONS]
# Options:
#   --clean-install  AGGRESSIVELY remove Python venv, ALL caches, and packages  
#   --clean-system   NUCLEAR system cleanup - removes ALL caches, logs, temp files
#   --clean-trt      Remove TensorRT-LLM, CUDA artifacts, force uninstall packages
#   --clean-models   Remove downloaded models, checkpoints, and engines
#   --clean-all      NUCLEAR OPTION - removes EVERYTHING (all of the above)
# =============================================================================

set -euo pipefail

echo "=== System Cleanup ==="

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
    # Stop setup pipeline
    if [ -f .run/setup-pipeline.pid ]; then
        local pid
        pid=$(cat .run/setup-pipeline.pid 2>/dev/null || true)
        if [ -n "$pid" ]; then
            echo "[cleanup] Stopping setup pipeline (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f .run/setup-pipeline.pid
    fi
    
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
    pkill -f "scripts/main.sh" 2>/dev/null || true
    pkill -f "setup-pipeline" 2>/dev/null || true
}

_cleanup_gpu_memory() {
    echo "[cleanup] Terminating GPU compute processes..."
    
    # Kill specific process types (no nvidia-smi bullshit)
    pkill -9 -f "python.*tensorrt" 2>/dev/null || true
    pkill -9 -f "python.*trtllm" 2>/dev/null || true
    pkill -9 -f "python.*modelopt" 2>/dev/null || true
    pkill -9 -f "trtllm-build" 2>/dev/null || true
    pkill -9 -f "quantize.py" 2>/dev/null || true
    pkill -9 -f "mpirun" 2>/dev/null || true
    pkill -9 -f "mpi4py" 2>/dev/null || true
    sleep 1
    
    # Kill any Python processes using CUDA
    pkill -9 -f "python.*cuda" 2>/dev/null || true
    pkill -9 -f ".*\.venv.*python" 2>/dev/null || true
    
    # Clean up device files aggressively
    if command -v fuser >/dev/null 2>&1; then
        for dev in /dev/nvidia* /dev/nvidiactl /dev/nvidia-uvm* /dev/dri/card*; do
            [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
        done
    fi
    
    # Clean up shared memory and temporary files
    rm -rf /dev/shm/nvidia* /dev/shm/cuda* /dev/shm/torch* /dev/shm/tensorrt* 2>/dev/null || true
    rm -rf /tmp/cuda* /tmp/nvidia* /tmp/tensorrt* /tmp/torch* /tmp/.X11-unix/* 2>/dev/null || true
    
    # Clean up IPC resources aggressively
    if command -v ipcs >/dev/null 2>&1; then
        local current_user
        current_user=$(whoami)
        # Clean all IPC resources for current user
        ipcs -s 2>/dev/null | grep "$current_user" | awk '{print $2}' | xargs -r -n1 ipcrm -s 2>/dev/null || true
        ipcs -q 2>/dev/null | grep "$current_user" | awk '{print $2}' | xargs -r -n1 ipcrm -q 2>/dev/null || true  
        ipcs -m 2>/dev/null | grep "$current_user" | awk '{print $2}' | xargs -r -n1 ipcrm -m 2>/dev/null || true
    fi
}

_cleanup_runtime_files() {
    rm -rf .run logs 2>/dev/null || true
}

_cleanup_python_environment() {
    echo "[cleanup] Aggressively removing Python environment and ALL caches..."
    
    # Remove virtual environment completely
    rm -rf .venv || true
    
    # Remove ALL Python-related caches (be fucking thorough)
    rm -rf ~/.cache/* || true
    rm -rf ~/.local/share/* || true
    rm -rf ~/.local/lib/* || true
    rm -rf ~/.local/bin/* || true
    
    # Remove specific ML/AI framework caches
    rm -rf ~/.triton ~/.cache/triton /tmp/triton* || true
    rm -rf ~/.torch ~/.cache/torch ~/.cache/torch_extensions || true
    rm -rf ~/.cache/nvidia ~/.cache/modelopt ~/.cache/quantization || true
    
    # Remove all temporary Python files
    rm -rf /tmp/pip-* /tmp/build-* /tmp/easy_install-* /tmp/python* || true
    rm -rf /tmp/.hf* /tmp/hf-* /tmp/huggingface* || true
    rm -rf /tmp/torch* /tmp/.torch* /tmp/tensorrt* /tmp/trt* || true
    
    # Remove conda/mamba completely if present
    rm -rf ~/.conda ~/.mamba ~/.cache/conda ~/.cache/mamba || true
    rm -rf ~/miniconda* ~/anaconda* ~/mambaforge* || true
    
    # Clean shared memory aggressively
    rm -rf /dev/shm/pip* /dev/shm/python* /dev/shm/torch* /dev/shm/hf* || true
    
    # Remove Python package directories in common locations
    rm -rf /usr/local/lib/python*/site-packages/__pycache__ || true
    rm -rf /usr/lib/python*/site-packages/__pycache__ || true
    
    # Clean up Python bytecode files EVERYWHERE
    find / -name "*.pyc" -delete 2>/dev/null || true
    find / -name "*.pyo" -delete 2>/dev/null || true  
    find / -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    
    # Remove pip cache and wheel cache completely
    rm -rf ~/.pip ~/.cache/pip* ~/.local/share/pip* || true
    
    # Force remove any remaining Python processes
    pkill -9 python 2>/dev/null || true
    pkill -9 python3 2>/dev/null || true
    pkill -9 pip 2>/dev/null || true
}

_cleanup_models_and_artifacts() {
    # Remove downloaded HuggingFace models and local model cache
    rm -rf models/ || true
    
    # Remove quantized checkpoints and TensorRT engines
    [ -n "${CHECKPOINT_DIR:-}" ] && rm -rf "${CHECKPOINT_DIR}" || true
    [ -n "${TRTLLM_ENGINE_DIR:-}" ] && rm -rf "${TRTLLM_ENGINE_DIR}" || true
    
    # Remove default model paths based on environment.sh
    rm -rf "$PWD/models/orpheus-trtllm-ckpt-int4-awq" || true
    rm -rf "$PWD/models/orpheus-trt-int4-awq" || true
    rm -rf "$PWD/models/canopylabs" || true
}

_cleanup_tensorrt_artifacts() {
    echo "[cleanup] Aggressively removing TensorRT and CUDA artifacts..."
    
    # Remove TensorRT-LLM repository and models completely
    rm -rf "$TRTLLM_REPO_DIR" || true
    rm -rf .trtllm-repo || true
    rm -rf models/ || true
    
    # Remove ALL NVIDIA/CUDA/TensorRT caches and files
    rm -rf ~/.cache/nvidia* ~/.cache/tensorrt* ~/.cache/nv* ~/.nv || true
    rm -rf ~/.cache/modelopt* ~/.cache/model_optimizer* ~/.cache/nvidia_modelopt* || true
    rm -rf ~/.cache/cuda* ~/.cache/nvcc* ~/.cache/cutlass* || true
    rm -rf ~/.cache/triton* ~/.triton* || true
    
    # Remove temporary build files EVERYWHERE
    rm -rf /tmp/trt* /tmp/tensorrt* /tmp/TensorRT* /tmp/modelopt* || true
    rm -rf /tmp/quantiz* /tmp/calib* /tmp/nvidia* /tmp/cuda* || true
    rm -rf /tmp/triton* /tmp/torch* /tmp/pytorch* || true
    
    # Clean shared memory aggressively
    rm -rf /dev/shm/trt* /dev/shm/quantiz* /dev/shm/cuda* /dev/shm/nv* || true
    rm -rf /dev/shm/triton* /dev/shm/modelopt* /dev/shm/tensorrt* || true
    
    # Remove NVIDIA installation directories if they exist
    rm -rf /usr/local/tensorrt* /usr/local/cuda* /usr/local/nvidia* || true
    rm -rf /opt/tensorrt* /opt/cuda* /opt/nvidia* || true
    
    # Force uninstall packages (don't depend on venv)
    python3 -m pip uninstall -y nvidia-modelopt nvidia-ammo 2>/dev/null || true
    python3 -m pip uninstall -y tensorrt-cu12-bindings tensorrt-cu12-libs 2>/dev/null || true  
    python3 -m pip uninstall -y tensorrt-llm 2>/dev/null || true
    python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    
    # Clean up any remaining NVIDIA processes
    pkill -9 -f nvidia 2>/dev/null || true
    pkill -9 -f tensorrt 2>/dev/null || true
    pkill -9 -f cuda 2>/dev/null || true
}

_cleanup_system_caches() {
    echo "[cleanup] Nuclear system cache cleanup (aggressive for cloud)..."
    
    # Clean ALL package manager caches aggressively
    if command -v apt-get >/dev/null 2>&1; then
        apt-get clean || true
        apt-get autoclean || true
        apt-get autoremove -y --purge || true
        rm -rf /var/lib/apt/lists/* || true
        rm -rf /var/cache/apt/* || true
        rm -rf /var/lib/dpkg/* || true
    fi
    
    # Clean other package managers aggressively
    if command -v yum >/dev/null 2>&1; then
        yum clean all || true
        rm -rf /var/cache/yum/* || true
    fi
    
    if command -v dnf >/dev/null 2>&1; then
        dnf clean all || true
        rm -rf /var/cache/dnf/* || true
    fi
    
    # Remove ALL temporary files and caches (NUCLEAR option)
    rm -rf /tmp/* /var/tmp/* 2>/dev/null || true
    rm -rf /root/.cache/* /root/.local/* 2>/dev/null || true
    rm -rf /home/*/.cache/* /home/*/.local/* 2>/dev/null || true
    
    # Clean system logs aggressively
    journalctl --vacuum-time=1s 2>/dev/null || true
    rm -rf /var/log/* 2>/dev/null || true
    
    # Clean up ALL build artifacts and development files
    rm -rf /usr/local/src/* /usr/src/* /usr/local/include/* 2>/dev/null || true
    rm -rf /usr/local/lib/python* /usr/local/bin/pip* 2>/dev/null || true
    
    # Clean up system caches
    rm -rf ~/.cache/* ~/.fontconfig ~/.local/* ~/.config/* 2>/dev/null || true
    rm -rf ~/.pki ~/.gnupg ~/.ssh/known_hosts 2>/dev/null || true
    
    # Clean up compiler caches
    rm -rf ~/.ccache ~/.cache/clang* ~/.cache/gcc* 2>/dev/null || true
    
    # Clean docker caches if present
    docker system prune -af 2>/dev/null || true
    rm -rf /var/lib/docker/* 2>/dev/null || true
    
    # Clean snap packages if present
    if command -v snap >/dev/null 2>&1; then
        snap list --all | awk '/disabled/{print $1, $3}' | while read snapname revision; do
            snap remove "$snapname" --revision="$revision" 2>/dev/null || true
        done
    fi
}

_cleanup_workspace_files() {
    echo "[cleanup] Cleaning workspace and additional system areas..."
    
    # Remove workspace temporary files and caches
    rm -rf audio/*.wav .pytest_cache .mypy_cache .ruff_cache *.egg-info 2>/dev/null || true
    rm -rf .git/objects/pack/.tmp* .git/objects/tmp* 2>/dev/null || true
    rm -rf node_modules/ .npm/ .yarn/ 2>/dev/null || true
    
    # Additional comprehensive cleanup for missed areas
    rm -rf ~/.cargo ~/.rustup ~/.gradle ~/.m2 2>/dev/null || true
    rm -rf ~/.npm ~/.nvm ~/.yarn ~/.node-gyp 2>/dev/null || true
    rm -rf ~/.cache/go-build ~/.cache/bazel 2>/dev/null || true
    
    # Clean up any remaining large cache directories
    find /tmp -type f -size +100M -delete 2>/dev/null || true
    find /var/tmp -type f -size +100M -delete 2>/dev/null || true
    find ~/.cache -type f -size +100M -delete 2>/dev/null || true
    
    # Clean up shared libraries and build caches
    rm -rf /usr/local/share/man /usr/local/share/doc 2>/dev/null || true
    rm -rf /usr/share/doc /usr/share/man /usr/share/info 2>/dev/null || true
    
    # Force sync to ensure all deletions are committed
    sync
}

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
CLEAN_MODELS=0

for arg in "$@"; do
    case "$arg" in
        --clean-install)
            CLEAN_INSTALL=1
            echo "[cleanup] Will AGGRESSIVELY remove Python environment and ALL caches"
            ;;
        --clean-system)
            CLEAN_SYSTEM=1
            echo "[cleanup] Will perform NUCLEAR system cleanup (all caches/logs/temp files)"
            ;;
        --clean-trt)
            CLEAN_TRT=1
            echo "[cleanup] Will remove TensorRT-LLM and force uninstall CUDA packages"
            ;;
        --clean-models)
            CLEAN_MODELS=1
            echo "[cleanup] Will remove downloaded models and artifacts"
            ;;
        --clean-all)
            CLEAN_INSTALL=1
            CLEAN_SYSTEM=1
            CLEAN_TRT=1
            CLEAN_MODELS=1
            echo "[cleanup] Will perform NUCLEAR cleanup (EVERYTHING)"
            ;;
        --help|-h)
            echo "Usage: $0 [--clean-install] [--clean-system] [--clean-trt] [--clean-models] [--clean-all]"
            echo ""
            echo "Options:"
            echo "  --clean-install  AGGRESSIVELY remove Python venv and ALL caches"
            echo "  --clean-system   NUCLEAR system cleanup - removes ALL caches/logs/temp files"
            echo "  --clean-trt      Remove TensorRT-LLM and force uninstall CUDA packages"
            echo "  --clean-models   Remove downloaded models, checkpoints, and engines"  
            echo "  --clean-all      NUCLEAR OPTION - removes EVERYTHING (all of the above)"
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

if [ "$CLEAN_MODELS" = "1" ]; then
    echo "[cleanup] Removing models and artifacts..."
    _cleanup_models_and_artifacts
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

echo "[cleanup] ✓ AGGRESSIVE cleanup completed successfully - all specified targets obliterated"

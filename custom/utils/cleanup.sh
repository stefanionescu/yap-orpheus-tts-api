#!/usr/bin/env bash
# =============================================================================
# Orpheus Cleanup Script
# =============================================================================
# Stops the running server and related background jobs. When invoked with
# --clean-all it also removes cached artifacts, models, and local dependencies.
# =============================================================================

set -euo pipefail

TRTLLM_REPO_DIR="${TRTLLM_REPO_DIR:-$PWD/.trtllm-repo}"
MODELS_DIR="${MODELS_DIR:-$PWD/models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
TRTLLM_ENGINE_DIR="${TRTLLM_ENGINE_DIR:-}"

usage() {
    cat <<'EOF_HELP'
Usage: custom/utils/cleanup.sh [--clean-all]

Stops the running Orpheus services. Use --clean-all to additionally remove
downloaded models, virtual environments, and cached dependencies.
EOF_HELP
}

log() {
    echo "[cleanup] $*"
}

_stop_pid_file() {
    local pid_file="$1"
    local label="$2"

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null || true)
        if [ -n "${pid:-}" ]; then
            log "Stopping $label (pid $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

_stop_server_processes() {
    _stop_pid_file ".run/server.pid" "server"
    pkill -f "uvicorn server.server:app" 2>/dev/null || true
}

_stop_background_processes() {
    _stop_pid_file ".run/setup-pipeline.pid" "setup pipeline"
    _stop_pid_file ".run/run-all.pid" "run-all pipeline"
    pkill -f "custom/run-all.sh" 2>/dev/null || true
    pkill -f "custom/main.sh" 2>/dev/null || true
    pkill -f "setup-pipeline" 2>/dev/null || true
}

_release_gpu_processes() {
    pkill -f "python.*tensorrt" 2>/dev/null || true
    pkill -f "python.*trtllm" 2>/dev/null || true
    pkill -f "trtllm-build" 2>/dev/null || true
    pkill -f "quantize.py" 2>/dev/null || true
    pkill -f "mpirun" 2>/dev/null || true
    pkill -f "mpi4py" 2>/dev/null || true
    pkill -f "python.*cuda" 2>/dev/null || true
}

_clear_runtime_state() {
    rm -rf .run 2>/dev/null || true
}

_safe_rm() {
    local target="$1"
    if [ -z "$target" ]; then
        return
    fi
    if [ -e "$target" ] || [ -L "$target" ]; then
        rm -rf "$target" 2>/dev/null || true
    fi
}

_full_cleanup() {
    log "Removing workspace artifacts..."
    local -a workspace_dirs=(
        ".venv"
        "$MODELS_DIR"
        "$TRTLLM_REPO_DIR"
        ".trtllm-repo"
        "logs"
        ".pytest_cache"
        ".mypy_cache"
        ".ruff_cache"
    )

    if [ -n "$CHECKPOINT_DIR" ]; then
        workspace_dirs+=("$CHECKPOINT_DIR")
    fi
    if [ -n "$TRTLLM_ENGINE_DIR" ]; then
        workspace_dirs+=("$TRTLLM_ENGINE_DIR")
    fi

    local dir
    for dir in "${workspace_dirs[@]}"; do
        _safe_rm "$dir"
    done

    if [ -d audio ]; then
        find audio -maxdepth 1 -type f -name '*.wav' -delete 2>/dev/null || true
    fi

    log "Removing cached dependencies..."
    local -a cache_dirs=(
        "$HOME/.cache/huggingface"
        "$HOME/.cache/huggingface_hub"
        "$HOME/.cache/pip"
        "$HOME/.cache/torch"
        "$HOME/.cache/tensorrt"
        "$HOME/.cache/triton"
        "$HOME/.cache/modelopt"
        "$HOME/.cache/nvidia"
        "$HOME/.cache/onnx"
        "$HOME/.pip"
        "$HOME/.torch"
        "$HOME/.triton"
        "$HOME/.nv"
        "$HOME/.local/share/pip"
        "$HOME/.local/share/tensorrt_llm"
        "$HOME/.local/state/pip"
    )

    for dir in "${cache_dirs[@]}"; do
        _safe_rm "$dir"
    done

    log "Clearing temporary build files..."
    rm -rf /tmp/tensorrt* /tmp/trt* /tmp/torch* /tmp/pip-* /tmp/hf* /tmp/cuda* /tmp/nv* 2>/dev/null || true
    rm -rf /dev/shm/tensorrt* /dev/shm/trt* /dev/shm/torch* /dev/shm/nv* /dev/shm/cuda* 2>/dev/null || true
}

CLEAN_ALL=0

for arg in "$@"; do
    case "$arg" in
        --clean-all)
            CLEAN_ALL=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            usage >&2 || true
            exit 1
            ;;
    esac
done

log "Stopping server processes..."
_stop_server_processes

log "Stopping background workers..."
_stop_background_processes

log "Releasing GPU resources..."
_release_gpu_processes

log "Clearing runtime state..."
_clear_runtime_state

if [ "$CLEAN_ALL" -eq 1 ]; then
    log "Performing full cleanup (--clean-all)..."
    _full_cleanup
    log "✓ Full cleanup completed"
else
    log "✓ Server stopped; dependencies and models preserved"
fi

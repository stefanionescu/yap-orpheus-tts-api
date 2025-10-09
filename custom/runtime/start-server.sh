#!/usr/bin/env bash
# =============================================================================
# TTS Server Startup Script
# =============================================================================
# Starts the FastAPI TTS server with proper environment validation and
# configuration display. Runs server in background with log tailing.
#
# Usage: bash custom/runtime/start-server.sh
# Environment: Requires HF_TOKEN, TRTLLM_ENGINE_DIR, optionally HOST, PORT
# =============================================================================

set -euo pipefail

# Load common utilities and environment
source "custom/lib/common.sh"
load_env_if_present
load_environment

echo "=== TTS Server Startup ==="

# =============================================================================
# Environment Validation
# =============================================================================

echo "[server] Validating environment..."

# Check required environment variables
validate_required_env || exit 1

# Check virtual environment
VENV_DIR="${VENV_DIR:-$PWD/.venv}"
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR" >&2
    echo "Run custom/01-install-trt.sh first" >&2
    exit 1
fi

# Check TensorRT-LLM engine (support engines nested under engines/<label>)
if [ ! -f "$TRTLLM_ENGINE_DIR/rank0.engine" ]; then
    if [ -d "$TRTLLM_ENGINE_DIR" ] && [ -f "$TRTLLM_ENGINE_DIR/../build_metadata.json" ]; then
        : # tolerate when engine dir points inside label folder
    else
        echo "ERROR: TensorRT-LLM engine not found at $TRTLLM_ENGINE_DIR/rank0.engine" >&2
        echo "Run custom/02-build.sh first or configure HF_DEPLOY_REPO_ID to pull prebuilt engines." >&2
        exit 1
    fi
fi

# =============================================================================
# Server Startup
# =============================================================================

echo "[server] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "[server] Starting FastAPI server on ${HOST}:${PORT}"
show_config

# Build uvicorn command with optimized settings
CMD=$(build_uvicorn_cmd)

# Start server in background with log tailing
start_background "$CMD" ".run/server.pid" "logs/server.log"

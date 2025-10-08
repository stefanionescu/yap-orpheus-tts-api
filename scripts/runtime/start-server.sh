#!/usr/bin/env bash
# =============================================================================
# TTS Server Startup Script
# =============================================================================
# Starts the FastAPI TTS server with proper environment validation and
# configuration display. Runs server in background with log tailing.
#
# Usage: bash scripts/runtime/start-server.sh
# Environment: Requires HF_TOKEN, TRTLLM_ENGINE_DIR, optionally HOST, PORT
# =============================================================================

set -euo pipefail

# Load common utilities and environment
source "scripts/lib/common.sh"
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
    echo "Run scripts/01-install-trt.sh first" >&2
    exit 1
fi

# Check TensorRT-LLM engine
if [ ! -f "$TRTLLM_ENGINE_DIR/rank0.engine" ]; then
    echo "ERROR: TensorRT-LLM engine not found at $TRTLLM_ENGINE_DIR/rank0.engine" >&2
    echo "Run scripts/02-build.sh first" >&2
    exit 1
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

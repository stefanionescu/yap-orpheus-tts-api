#!/usr/bin/env bash
# =============================================================================
# Complete Setup Pipeline Script
# =============================================================================
# Runs the complete setup pipeline from system bootstrap to server startup:
# 1. System bootstrap (dependencies, CUDA check)
# 2. Python environment setup (venv, packages, TensorRT-LLM)
# 3. Engine build (INT4-AWQ quantization)
# 4. Server startup (FastAPI with TTS endpoints)
#
# Usage: bash scripts/run-complete-setup.sh
# Environment: Requires HF_TOKEN to be set
# =============================================================================

set -euo pipefail

echo "=== Complete TTS Setup Pipeline ==="

# =============================================================================
# Environment Validation
# =============================================================================

# Check required environment variable
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export your HuggingFace token first." >&2
    echo "Example: export HF_TOKEN=\"hf_xxx\"" >&2
    exit 1
fi

echo "[pipeline] HuggingFace token: ${HF_TOKEN:0:8}..."

# =============================================================================
# Pipeline Execution
# =============================================================================

# Create directories for logs and runtime files
mkdir -p logs .run

# Define the complete pipeline command
PIPELINE_CMD='
    echo "[pipeline] === Step 1/4: System Bootstrap ===" && \
    bash scripts/setup/bootstrap.sh && \
    echo "" && \
    echo "[pipeline] === Step 2/4: Install Dependencies ===" && \
    bash scripts/setup/install-dependencies.sh && \
    echo "" && \
    echo "[pipeline] === Step 3/4: Build TensorRT Engine ===" && \
    bash scripts/build/build-engine.sh && \
    echo "" && \
    echo "[pipeline] === Step 4/4: Start TTS Server ===" && \
    export TRTLLM_ENGINE_DIR="${TRTLLM_ENGINE_DIR:-$PWD/models/orpheus-trt-int4-awq}" && \
    bash scripts/runtime/start-server.sh
'

# Run pipeline in background with proper process isolation
echo "[pipeline] Starting complete setup pipeline in background..."
setsid nohup bash -lc "$PIPELINE_CMD" </dev/null > logs/setup-pipeline.log 2>&1 &

# Store background process ID
bg_pid=$!
echo $bg_pid > .run/setup-pipeline.pid

echo "[pipeline] Pipeline started (PID: $bg_pid)"
echo "[pipeline] Logs: logs/setup-pipeline.log"
echo "[pipeline] Server logs: logs/server.log (when server starts)"
echo "[pipeline] To stop: bash scripts/utils/cleanup.sh"
echo ""
echo "[pipeline] Following setup logs (Ctrl-C detaches, pipeline continues)..."

# Tail logs with graceful handling
touch logs/setup-pipeline.log || true
exec tail -n +1 -F logs/setup-pipeline.log

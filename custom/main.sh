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
# Usage: bash custom/run-complete-setup.sh
# Environment: Requires HF_TOKEN to be set
# =============================================================================

set -euo pipefail

echo "=== Complete TTS Setup Pipeline ==="

# =============================================================================
# Environment Validation
# =============================================================================

# Check required environment variables
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export your HuggingFace token first." >&2
    echo "Example: export HF_TOKEN=\"hf_xxx\"" >&2
    exit 1
fi

# Only require GPU_SM_ARCH if we plan to push to HuggingFace
if [ "${HF_PUSH_AFTER_BUILD:-0}" = "1" ] && [ -z "${GPU_SM_ARCH:-}" ]; then
    echo "ERROR: GPU_SM_ARCH not set but HF push is enabled." >&2
    echo "Either disable HF push or set GPU architecture:" >&2
    echo "  export HF_PUSH_AFTER_BUILD=0   # Disable HF push" >&2
    echo "  # OR set GPU architecture:" >&2
    echo "  export GPU_SM_ARCH=\"sm80\"     # A100" >&2
    echo "  export GPU_SM_ARCH=\"sm89\"     # RTX 4090" >&2
    echo "  export GPU_SM_ARCH=\"sm90\"     # H100" >&2
    exit 1
fi

echo "[pipeline] HuggingFace token: ${HF_TOKEN:0:8}..."
if [ -n "${GPU_SM_ARCH:-}" ]; then
    echo "[pipeline] GPU architecture: ${GPU_SM_ARCH}"
else
    echo "[pipeline] GPU architecture: not set (not needed for build-only)"
fi

# =============================================================================
# Pipeline Execution
# =============================================================================

# Create directories for logs and runtime files
mkdir -p logs .run

# Define the complete pipeline command
PIPELINE_CMD='
    echo "[pipeline] === Step 1/4: System Bootstrap ===" && \
    bash custom/00-bootstrap.sh && \
    echo "" && \
    echo "[pipeline] === Step 2/4: Install Dependencies ===" && \
    bash custom/01-install-trt.sh && \
    echo "" && \
    echo "[pipeline] === Step 3/4: Build TensorRT Engine ===" && \
    bash custom/02-build.sh && \
    echo "" && \
    if [ "${HF_PUSH_AFTER_BUILD:-0}" = "1" ]; then \
        echo "[pipeline] === Optional: Push artifacts to Hugging Face ===" && \
        source custom/environment.sh && \
        if [ -z "${HF_PUSH_REPO_ID:-}" ]; then \
            echo "[pipeline] Skipping push: HF_PUSH_REPO_ID not set"; \
        elif [ -z "${GPU_SM_ARCH:-}" ]; then \
            echo "[pipeline] ERROR: HF push enabled but GPU_SM_ARCH not set!" && \
            echo "[pipeline] Set GPU_SM_ARCH (e.g. export GPU_SM_ARCH=sm80 for A100) before pushing." && \
            echo "[pipeline] NEVER push without explicit GPU configuration!" && \
            exit 1; \
        else \
            echo "[pipeline] GPU architecture configured (${GPU_SM_ARCH}) - proceeding with push to HF repo: ${HF_PUSH_REPO_ID}" && \
            # Use venv python if available
            PYTHON_EXEC="${PYTHON_EXEC:-python}"; \
            if [ -x ".venv/bin/python" ]; then PYTHON_EXEC=".venv/bin/python"; fi; \
            "$PYTHON_EXEC" server/hf/push_to_hf.py --repo-id "${HF_PUSH_REPO_ID}" $([ "${HF_PUSH_PRIVATE:-1}" = "1" ] && echo --private) --what "${HF_PUSH_WHAT:-both}" --engine-label "${HF_PUSH_ENGINE_LABEL:-}" $([ "${HF_PUSH_PRUNE:-0}" = "1" ] && echo --prune) $([ "${HF_PUSH_NO_README:-0}" = "1" ] && echo --no-readme); \
        fi; \
        echo ""; \
    fi && \
    echo "[pipeline] === Step 4/4: Start TTS Server ===" && \
    export TRTLLM_ENGINE_DIR="${TRTLLM_ENGINE_DIR:-$PWD/models/orpheus-trt-int4-awq}" && \
    bash custom/03-run-server.sh
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
echo "[pipeline] To stop: bash custom/utils/cleanup.sh"
echo ""
echo "[pipeline] Following setup logs (Ctrl-C detaches, pipeline continues)..."

# Tail logs with graceful handling
touch logs/setup-pipeline.log || true
exec tail -n +1 -F logs/setup-pipeline.log

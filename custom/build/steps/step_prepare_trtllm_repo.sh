#!/usr/bin/env bash
set -euo pipefail

source "custom/lib/common.sh"
load_env_if_present
load_environment
source "custom/build/helpers.sh"

echo "[step:trtllm] Preparing TensorRT-LLM repository..."

TRTLLM_REPO_DIR="${TRTLLM_REPO_DIR:-$PWD/.trtllm-repo}"
PYTHON_EXEC="${PYTHON_EXEC:-python}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"

if [ "$FORCE_REBUILD" = true ] && [ -d "${TRTLLM_REPO_DIR}" ]; then
    echo "[step:trtllm] --force specified: removing existing repository"
    rm -rf "${TRTLLM_REPO_DIR}"
fi

if [ ! -d "${TRTLLM_REPO_DIR}" ]; then
    echo "[step:trtllm] Cloning TensorRT-LLM repo for quantization scripts..."
    git clone https://github.com/Yap-With-AI/TensorRT-LLM.git "${TRTLLM_REPO_DIR}"
fi

echo "[step:trtllm] Syncing repo version with installed wheel..."
trtllm_ver="$(${PYTHON_EXEC} -c 'import tensorrt_llm as t; print(t.__version__)' 2>/dev/null | tail -1 | tr -d '[:space:]')"
echo "[step:trtllm] Detected TensorRT-LLM version: ${trtllm_ver}"
git -C "${TRTLLM_REPO_DIR}" fetch --tags

if ! git -C "${TRTLLM_REPO_DIR}" checkout "v${trtllm_ver}" 2>/dev/null; then
    if [[ "${trtllm_ver}" == "1.0.0" ]]; then
        echo "[step:trtllm] Tag v1.0.0 not found, using known commit ae8270b713446948246f16fadf4e2a32e35d0f62"
        git -C "${TRTLLM_REPO_DIR}" checkout ae8270b713446948246f16fadf4e2a32e35d0f62
    else
        echo "[step:trtllm] ERROR: Could not checkout version ${trtllm_ver}" >&2
        exit 1
    fi
fi

if [ ! -d "$TRTLLM_REPO_DIR/examples/quantization" ]; then
    echo "ERROR: quantization examples not found in $TRTLLM_REPO_DIR/examples/quantization" >&2
    echo "Available examples:" >&2
    ls -la "$TRTLLM_REPO_DIR/examples/" >&2
    exit 1
fi

export TRTLLM_EXAMPLES_DIR="$TRTLLM_REPO_DIR/examples/quantization"
echo "[step:trtllm] OK"



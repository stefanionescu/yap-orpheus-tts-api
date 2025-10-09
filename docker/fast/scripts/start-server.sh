#!/usr/bin/env bash
set -euo pipefail

# Fast image server startup script
# Pulls quantized checkpoint or engines from HF (no TRT repo clone), builds engine if needed, then starts server

echo "Starting Orpheus TTS Server (Fast Image)"

# Source environment if available
if [[ -f /usr/local/bin/environment.sh ]]; then
    source /usr/local/bin/environment.sh
fi

mkdir -p /app/.run /app/logs || true

# If HF_DEPLOY_REPO_ID is set, try to pull engines/checkpoints and build if needed
if [[ -n "${HF_DEPLOY_REPO_ID:-}" ]]; then
    echo "[fast] HF remote deploy enabled: ${HF_DEPLOY_REPO_ID}"
    # Run the same remote deploy step logic as custom build (no repo clone)
    bash -lc "source /usr/local/bin/environment.sh; cd /app; bash custom/build/steps/step_prepare_env.sh; bash custom/build/steps/step_remote_deploy.sh || true"
    if [[ -f /app/.run/remote_result.env ]]; then
        # shellcheck disable=SC1091
        source /app/.run/remote_result.env
        if [[ "${REMOTE_RESULT:-}" == "10" ]]; then
            # Engines ready
            # shellcheck disable=SC1091
            source /app/.run/engine_dir.env || true
            export TRTLLM_ENGINE_DIR="${TRTLLM_ENGINE_DIR:-}"
        elif [[ "${REMOTE_RESULT:-}" == "11" ]]; then
            # Checkpoint ready; build engine locally
            echo "[fast] Building engine from downloaded checkpoint..."
            CHECKPOINT_DIR="${CHECKPOINT_DIR:-/opt/models/_hf_download/trt-llm/checkpoints}" \
            ENGINE_OUTPUT_DIR="${TRTLLM_ENGINE_DIR:-/opt/engines/orpheus-trt-int4-awq}" \
            bash custom/build/steps/step_engine_build.sh
            # shellcheck disable=SC1091
            source /app/.run/engine_dir.env || true
        fi
    fi
fi

# Final validation of engine dir
if [[ -z "${TRTLLM_ENGINE_DIR:-}" || ! -f "${TRTLLM_ENGINE_DIR}/rank0.engine" ]]; then
    echo "ERROR: TensorRT-LLM engine not available at TRTLLM_ENGINE_DIR=$TRTLLM_ENGINE_DIR" >&2
    echo "       Provide HF_DEPLOY_REPO_ID with engines/checkpoints, or mount an engine directory." >&2
    exit 1
fi

# Optional: Download model if MODEL_ID is provided and model doesn't exist
if [[ -n "${MODEL_ID:-}" && -n "${HF_TOKEN:-}" ]]; then
    MODEL_NAME=$(basename "$MODEL_ID")
    MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}-hf"
    
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "Downloading model $MODEL_ID to $MODEL_PATH..."
        python -c "
import os
from huggingface_hub import snapshot_download

model_id = os.environ['MODEL_ID']
token = os.environ['HF_TOKEN']
local_dir = '$MODEL_PATH'

os.makedirs(local_dir, exist_ok=True)
snapshot_download(
    repo_id=model_id, 
    local_dir=local_dir,
    local_dir_use_symlinks=False, 
    token=token
)
print(f'Downloaded {model_id} to {local_dir}')
"
    else
        echo "Model already exists at $MODEL_PATH"
    fi
fi

echo "Configuration:"
echo "   Engine Directory: $TRTLLM_ENGINE_DIR"
echo "   Models Directory: ${MODELS_DIR}"
echo "   Host: ${HOST:-0.0.0.0}"
echo "   Port: ${PORT:-8000}"

# Start the FastAPI server
echo "Starting server..."
exec uvicorn server.server:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --timeout-keep-alive 75 \
    --log-level info

#!/usr/bin/env bash
set -euo pipefail

# Fast image server startup script
# Assumes models and engines are provided at runtime via volumes or downloads

echo "Starting Orpheus TTS Server (Fast Image)"

# Source environment if available
if [[ -f /usr/local/bin/environment.sh ]]; then
    source /usr/local/bin/environment.sh
fi

# Validate required environment variables
if [[ -z "${TRTLLM_ENGINE_DIR:-}" ]]; then
    echo "ERROR: TRTLLM_ENGINE_DIR environment variable is required"
    echo "   Mount your pre-built engine or set path to engine directory"
    exit 1
fi

if [[ ! -d "$TRTLLM_ENGINE_DIR" ]]; then
    echo "ERROR: Engine directory not found: $TRTLLM_ENGINE_DIR"
    echo "   Ensure your TensorRT engine is mounted or available at this path"
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

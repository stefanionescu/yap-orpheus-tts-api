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

# Helper: detect SM arch (e.g., sm80)
_detect_sm() {
    if [[ -n "${GPU_SM_ARCH:-}" ]]; then echo "$GPU_SM_ARCH"; return; fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cap; cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
        if [[ -n "$cap" ]]; then echo "sm${cap/./}"; return; fi
    fi
    echo ""
}

# If HF_DEPLOY_REPO_ID is set, pull engines/checkpoints and optionally build
if [[ -n "${HF_DEPLOY_REPO_ID:-}" ]]; then
    echo "[fast] HF remote deploy: ${HF_DEPLOY_REPO_ID}"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    py_out=$(python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo_id=os.environ.get('HF_DEPLOY_REPO_ID')
use=os.environ.get('HF_DEPLOY_USE','auto').strip().lower()
engine_label=os.environ.get('HF_DEPLOY_ENGINE_LABEL','').strip()
workdir=os.environ.get('HF_DEPLOY_WORKDIR','') or '/opt/models/_hf_download'
gpu_sm=os.environ.get('GPU_SM_ARCH','').strip()

api=HfApi()
try:
    files=api.list_repo_files(repo_id=repo_id, repo_type='model')
except Exception as exc:
    print(f"MODE=error MSG={type(exc).__name__}:{exc}")
    raise SystemExit(0)

engine_labels=set()
for f in files:
    if f.startswith('trt-llm/engines/'):
        parts=f.split('/')
        if len(parts)>=4:
            engine_labels.add(parts[3])
has_ckpt=any(f.startswith('trt-llm/checkpoints/') for f in files)

selected=''
if use in ('engines','auto') and engine_labels:
    if engine_label and engine_label in engine_labels:
        selected=engine_label
    elif len(engine_labels)==1:
        selected=next(iter(engine_labels))
    elif gpu_sm:
        matches=[lab for lab in sorted(engine_labels) if lab.startswith(gpu_sm)]
        if len(matches)==1:
            selected=matches[0]

if selected:
    path=snapshot_download(repo_id=repo_id, local_dir=workdir, local_dir_use_symlinks=False,
                           allow_patterns=[f"trt-llm/engines/{selected}/**", "trt-llm/engines/**/build_metadata.json"])
    eng_dir=str(Path(path)/'trt-llm'/'engines'/selected)
    print(f"MODE=engines")
    print(f"ENGINE_DIR={eng_dir}")
    print(f"ENGINE_LABEL={selected}")
    raise SystemExit(0)

if use in ('checkpoints','auto') and has_ckpt:
    path=snapshot_download(repo_id=repo_id, local_dir=workdir, local_dir_use_symlinks=False,
                           allow_patterns=["trt-llm/checkpoints/**"]) 
    ckpt_dir=str(Path(path)/'trt-llm'/'checkpoints')
    print(f"MODE=checkpoints")
    print(f"CHECKPOINT_DIR={ckpt_dir}")
    raise SystemExit(0)

print("MODE=none")
PY
    )
    mode=$(echo "$py_out" | awk -F= '/^MODE=/{print $2; exit}')
    if [[ "$mode" == "engines" ]]; then
        TRTLLM_ENGINE_DIR=$(echo "$py_out" | awk -F= '/^ENGINE_DIR=/{print $2; exit}')
        if [[ -f "$TRTLLM_ENGINE_DIR/rank0.engine" && -f "$TRTLLM_ENGINE_DIR/config.json" ]]; then
            export TRTLLM_ENGINE_DIR
            echo "[fast] Using prebuilt engine: $TRTLLM_ENGINE_DIR"
        else
            echo "[fast] ERROR: downloaded engines missing required files" >&2
        fi
    elif [[ "$mode" == "checkpoints" ]]; then
        CHECKPOINT_DIR=$(echo "$py_out" | awk -F= '/^CHECKPOINT_DIR=/{print $2; exit}')
        if [[ -f "$CHECKPOINT_DIR/config.json" ]] && ls "$CHECKPOINT_DIR"/rank*.safetensors >/dev/null 2>&1; then
            echo "[fast] Building engine from checkpoint: $CHECKPOINT_DIR"
            : "${TRTLLM_ENGINE_DIR:=${ENGINES_DIR}/orpheus-trt-int4-awq}"
            trtllm-build \
              --checkpoint_dir "$CHECKPOINT_DIR" \
              --output_dir "$TRTLLM_ENGINE_DIR" \
              --gemm_plugin auto \
              --gpt_attention_plugin float16 \
              --context_fmha enable \
              --paged_kv_cache enable \
              --remove_input_padding enable \
              --max_input_len "${TRTLLM_MAX_INPUT_LEN:-48}" \
              --max_seq_len "$(( ${TRTLLM_MAX_INPUT_LEN:-48} + ${TRTLLM_MAX_OUTPUT_LEN:-1024} ))" \
              --max_batch_size "${TRTLLM_MAX_BATCH_SIZE:-16}" \
              --log_level info \
              --workers "$(nproc --all)"
        else
            echo "[fast] ERROR: downloaded checkpoint invalid (missing config or shards)" >&2
        fi
    else
        echo "[fast] No usable artifacts found in HF repo; expecting a mounted engine."
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

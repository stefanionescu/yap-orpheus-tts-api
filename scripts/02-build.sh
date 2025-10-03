#!/usr/bin/env bash
set -euo pipefail

source "scripts/lib/common.sh"
load_env_if_present

: "${VENV_DIR:=$PWD/.venv}"
: "${MODEL_ID:=canopylabs/orpheus-3b-0.1-ft}"
: "${CHECKPOINT_DIR:=$PWD/models/orpheus-trtllm-ckpt-fp16}"
: "${TRTLLM_ENGINE_DIR:=$PWD/models/orpheus-trt-fp16}"
: "${TRTLLM_DTYPE:=float16}"
: "${TRTLLM_MAX_INPUT_LEN:=128}"
: "${TRTLLM_MAX_OUTPUT_LEN:=1024}"
: "${TRTLLM_MAX_BATCH_SIZE:=16}"
: "${PYTHON_EXEC:=python}"
: "${TRTLLM_REPO_DIR:=$PWD/.trtllm-repo}"

usage() {
  cat <<USAGE
Usage: $0 [--model ID_OR_PATH] [--checkpoint-dir DIR] [--engine-dir DIR] [--dtype float16|bfloat16] [--max-input-len N] [--max-output-len N] [--max-batch-size N] [--force]

End-to-end FP16 build (no quantization):
  HF checkpoint → TRT-LLM checkpoint → TRT engine

Defaults:
  --model              ${MODEL_ID}
  --checkpoint-dir     ${CHECKPOINT_DIR}
  --engine-dir         ${TRTLLM_ENGINE_DIR}
  --dtype              ${TRTLLM_DTYPE}
  --max-input-len      ${TRTLLM_MAX_INPUT_LEN}
  --max-output-len     ${TRTLLM_MAX_OUTPUT_LEN}
  --max-batch-size     ${TRTLLM_MAX_BATCH_SIZE}
USAGE
}

ARGS=()
FORCE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ID="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --engine-dir) TRTLLM_ENGINE_DIR="$2"; shift 2 ;;
    --dtype) TRTLLM_DTYPE="$2"; shift 2 ;;
    --max-input-len) TRTLLM_MAX_INPUT_LEN="$2"; shift 2 ;;
    --max-output-len) TRTLLM_MAX_OUTPUT_LEN="$2"; shift 2 ;;
    --max-batch-size) TRTLLM_MAX_BATCH_SIZE="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]:-}"

[ -d "${VENV_DIR}" ] || { echo "[build] venv missing. Run scripts/01-install.sh first." >&2; exit 1; }
source "${VENV_DIR}/bin/activate"

if [ "$(uname -s)" != "Linux" ]; then
  echo "[build] ERROR: TensorRT-LLM builds require Linux with NVIDIA GPUs." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[build] ERROR: nvidia-smi not detected. Ensure the GPU is visible." >&2
  exit 1
fi

source_env_dir "scripts/env"
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
  export HF_HUB_TOKEN="${HF_HUB_TOKEN:-$HF_TOKEN}"
fi

# Ensure TensorRT-LLM repo is available for conversion scripts
if [ ! -d "${TRTLLM_REPO_DIR}" ]; then
  echo "[build] Cloning TensorRT-LLM repo for conversion scripts..."
  git clone https://github.com/NVIDIA/TensorRT-LLM.git "${TRTLLM_REPO_DIR}"
fi

# Pin repo tag to installed wheel version (critical for compatibility)
echo "[build] Syncing repo version with installed wheel..."
# Extract version number, filtering out TRT-LLM log messages
TRTLLM_VER="$(${PYTHON_EXEC} -c 'import tensorrt_llm as t; print(t.__version__)' 2>/dev/null | tail -1 | tr -d '[:space:]')"
echo "[build] Detected TensorRT-LLM version: ${TRTLLM_VER}"
git -C "${TRTLLM_REPO_DIR}" fetch --tags

# Try tag first, fallback to known commit for v1.0.0
if ! git -C "${TRTLLM_REPO_DIR}" checkout "v${TRTLLM_VER}" 2>/dev/null; then
  if [[ "${TRTLLM_VER}" == "1.0.0" ]]; then
    echo "[build] Tag v1.0.0 not found, using known commit ae8270b713446948246f16fadf4e2a32e35d0f62"
    git -C "${TRTLLM_REPO_DIR}" checkout ae8270b713446948246f16fadf4e2a32e35d0f62
  else
    echo "[build] ERROR: Could not checkout version ${TRTLLM_VER}" >&2
    exit 1
  fi
fi

echo "[build] ============================================"
echo "[build] Step 1/2: Convert HF checkpoint to TRT-LLM format (FP16)"
echo "[build] ============================================"

# Install conversion dependencies
LLAMA_REQUIREMENTS="${TRTLLM_REPO_DIR}/examples/models/core/llama/requirements.txt"
if [ -f "${LLAMA_REQUIREMENTS}" ]; then
  echo "[build] Installing Llama conversion requirements..."
  pip install -r "${LLAMA_REQUIREMENTS}"
else
  echo "[build] WARNING: requirements.txt not found at ${LLAMA_REQUIREMENTS}, skipping..."
fi

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Skip conversion if directory exists and not forcing
if [[ -d "${CHECKPOINT_DIR}" && "${FORCE}" != true ]]; then
  echo "[build] Reusing existing TRT-LLM checkpoint at ${CHECKPOINT_DIR}"
else
  echo "[build] Converting HF checkpoint to TRT-LLM format..."
  rm -rf "${CHECKPOINT_DIR}"
  mkdir -p "${CHECKPOINT_DIR}"
  
  # Download model from HF if it's a model ID (not a local path)
  if [[ ! -d "${MODEL_ID}" ]]; then
    echo "[build] Downloading model from HuggingFace: ${MODEL_ID}"
    LOCAL_MODEL_DIR="${PWD}/models/$(basename ${MODEL_ID})-hf"
    
    if [[ ! -d "${LOCAL_MODEL_DIR}" ]]; then
      mkdir -p "${LOCAL_MODEL_DIR}"
      "${PYTHON_EXEC}" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${MODEL_ID}',
    local_dir='${LOCAL_MODEL_DIR}',
    local_dir_use_symlinks=False
)
"
    else
      echo "[build] Using cached HF model at ${LOCAL_MODEL_DIR}"
    fi
    MODEL_DIR_FOR_CONVERT="${LOCAL_MODEL_DIR}"
  else
    echo "[build] Using local model directory: ${MODEL_ID}"
    MODEL_DIR_FOR_CONVERT="${MODEL_ID}"
  fi
  
  # Find the appropriate convert_checkpoint.py script
  # Orpheus is based on Llama architecture
  CONVERT_SCRIPT="${TRTLLM_REPO_DIR}/examples/models/core/llama/convert_checkpoint.py"
  
  if [ ! -f "${CONVERT_SCRIPT}" ]; then
    echo "[build] ERROR: convert_checkpoint.py not found at ${CONVERT_SCRIPT}" >&2
    exit 1
  fi
  
  CONVERT_CMD=(
    "${PYTHON_EXEC}" "${CONVERT_SCRIPT}"
    --model_dir "${MODEL_DIR_FOR_CONVERT}"
    --output_dir "${CHECKPOINT_DIR}"
    --dtype "${TRTLLM_DTYPE}"
  )
  echo "[build] Running: ${CONVERT_CMD[*]}"
  "${CONVERT_CMD[@]}"
fi

echo ""
echo "[build] ============================================"
echo "[build] Step 2/2: Build TensorRT-LLM engine"
echo "[build] ============================================"

# Skip engine build if directory exists and not forcing
if [[ -d "${TRTLLM_ENGINE_DIR}" && "${FORCE}" != true ]]; then
  echo "[build] Reusing existing engine at ${TRTLLM_ENGINE_DIR}"
else
  echo "[build] Building TensorRT FP16 engine..."
  
  BUILD_CMD=(
    trtllm-build
    --checkpoint_dir "${CHECKPOINT_DIR}"
    --output_dir "${TRTLLM_ENGINE_DIR}"
    --gemm_plugin float16
    --gpt_attention_plugin float16
    --context_fmha enable
    --paged_kv_cache enable
    --remove_input_padding enable
    --max_input_len "${TRTLLM_MAX_INPUT_LEN}"
    --max_seq_len "$((TRTLLM_MAX_INPUT_LEN + TRTLLM_MAX_OUTPUT_LEN))"
    --max_batch_size "${TRTLLM_MAX_BATCH_SIZE}"
    --log_level info
    --workers "$(nproc --all)"
  )
  echo "[build] Running: ${BUILD_CMD[*]}"
  "${BUILD_CMD[@]}"
fi

echo ""
echo "[build] ============================================"
echo "[build] Done. Engine: ${TRTLLM_ENGINE_DIR}"
echo "[build] To run server:"
echo "  export TRTLLM_ENGINE_DIR=\"${TRTLLM_ENGINE_DIR}\""
echo "  bash scripts/04-run-server.sh"
echo "[build] ============================================"

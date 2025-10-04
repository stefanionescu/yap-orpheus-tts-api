#!/usr/bin/env bash
set -euo pipefail

source "scripts/lib/common.sh"
load_env_if_present

: "${VENV_DIR:=$PWD/.venv}"
: "${MODEL_ID:=canopylabs/orpheus-3b-0.1-ft}"
: "${CHECKPOINT_DIR:=$PWD/models/orpheus-trtllm-ckpt-int4-awq}"
: "${TRTLLM_ENGINE_DIR:=$PWD/models/orpheus-trt-int4-awq}"
: "${TRTLLM_DTYPE:=float16}"
: "${TRTLLM_MAX_INPUT_LEN:=48}" # Optimized for sentence-by-sentence TTS
: "${TRTLLM_MAX_OUTPUT_LEN:=1024}"
: "${TRTLLM_MAX_BATCH_SIZE:=24}"
: "${PYTHON_EXEC:=python}"
: "${TRTLLM_REPO_DIR:=$PWD/.trtllm-repo}"
: "${AWQ_BLOCK_SIZE:=128}" # 128 is optimal for quality
: "${CALIB_SIZE:=256}" # Sufficient for AWQ

usage() {
  cat <<USAGE
Usage: $0 [--model ID_OR_PATH] [--checkpoint-dir DIR] [--engine-dir DIR] [--dtype float16|bfloat16] [--max-input-len N] [--max-output-len N] [--max-batch-size N] [--force]

End-to-end INT4-AWQ build (weight-only quantization):
  HF checkpoint → Quantized checkpoint → TRT engine

Defaults:
  --model              ${MODEL_ID}
  --checkpoint-dir     ${CHECKPOINT_DIR}
  --engine-dir         ${TRTLLM_ENGINE_DIR}
  --dtype              ${TRTLLM_DTYPE}
  --max-input-len      ${TRTLLM_MAX_INPUT_LEN}
  --max-output-len     ${TRTLLM_MAX_OUTPUT_LEN}
  --max-batch-size     ${TRTLLM_MAX_BATCH_SIZE}
  --awq-block-size     ${AWQ_BLOCK_SIZE}
  --calib-size         ${CALIB_SIZE}
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
    --awq-block-size) AWQ_BLOCK_SIZE="$2"; shift 2 ;;
    --calib-size) CALIB_SIZE="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]:-}"

[ -d "${VENV_DIR}" ] || { echo "[build-awq] venv missing. Run scripts/01-install-trt.sh first." >&2; exit 1; }
source "${VENV_DIR}/bin/activate"

if [ "$(uname -s)" != "Linux" ]; then
  echo "[build-awq] ERROR: TensorRT-LLM builds require Linux with NVIDIA GPUs." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[build-awq] ERROR: nvidia-smi not detected. Ensure the GPU is visible." >&2
  exit 1
fi

source_env_dir "scripts/env"
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
  export HF_HUB_TOKEN="${HF_HUB_TOKEN:-$HF_TOKEN}"
fi

# Ensure TensorRT-LLM repo is available for quantization scripts
if [ ! -d "${TRTLLM_REPO_DIR}" ]; then
  echo "[build-awq] Cloning TensorRT-LLM repo for quantization scripts..."
  git clone https://github.com/NVIDIA/TensorRT-LLM.git "${TRTLLM_REPO_DIR}"
fi

# Pin repo tag to installed wheel version
echo "[build-awq] Syncing repo version with installed wheel..."
TRTLLM_VER="$(${PYTHON_EXEC} -c 'import tensorrt_llm as t; print(t.__version__)' 2>/dev/null | tail -1 | tr -d '[:space:]')"
echo "[build-awq] Detected TensorRT-LLM version: ${TRTLLM_VER}"
git -C "${TRTLLM_REPO_DIR}" fetch --tags

if ! git -C "${TRTLLM_REPO_DIR}" checkout "v${TRTLLM_VER}" 2>/dev/null; then
  if [[ "${TRTLLM_VER}" == "1.0.0" ]]; then
    echo "[build-awq] Tag v1.0.0 not found, using known commit ae8270b713446948246f16fadf4e2a32e35d0f62"
    git -C "${TRTLLM_REPO_DIR}" checkout ae8270b713446948246f16fadf4e2a32e35d0f62
  else
    echo "[build-awq] ERROR: Could not checkout version ${TRTLLM_VER}" >&2
    exit 1
  fi
fi

echo "[build-awq] ============================================"
echo "[build-awq] Step 1/2: Quantize to INT4-AWQ (weight-only)"
echo "[build-awq] ============================================"

# Install quantization dependencies
QUANT_REQUIREMENTS="${TRTLLM_REPO_DIR}/examples/quantization/requirements.txt"
if [ -f "${QUANT_REQUIREMENTS}" ]; then
  echo "[build-awq] Installing quantization requirements..."
  pip install -r "${QUANT_REQUIREMENTS}"
else
  echo "[build-awq] WARNING: quantization requirements.txt not found, skipping..."
fi

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Skip quantization if directory exists and not forcing
if [[ -d "${CHECKPOINT_DIR}" && "${FORCE}" != true ]]; then
  echo "[build-awq] Reusing existing quantized checkpoint at ${CHECKPOINT_DIR}"
else
  echo "[build-awq] Quantizing to INT4-AWQ..."
  rm -rf "${CHECKPOINT_DIR}"
  mkdir -p "${CHECKPOINT_DIR}"
  
  # Download model from HF if it's a model ID (not a local path)
  if [[ ! -d "${MODEL_ID}" ]]; then
    echo "[build-awq] Downloading model from HuggingFace: ${MODEL_ID}"
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
      echo "[build-awq] Using cached HF model at ${LOCAL_MODEL_DIR}"
    fi
    MODEL_DIR_FOR_QUANT="${LOCAL_MODEL_DIR}"
  else
    echo "[build-awq] Using local model directory: ${MODEL_ID}"
    MODEL_DIR_FOR_QUANT="${MODEL_ID}"
  fi
  
  # Quantize using ModelOpt
  QUANT_SCRIPT="${TRTLLM_REPO_DIR}/examples/quantization/quantize.py"
  
  if [ ! -f "${QUANT_SCRIPT}" ]; then
    echo "[build-awq] ERROR: quantize.py not found at ${QUANT_SCRIPT}" >&2
    exit 1
  fi
  
  QUANT_CMD=(
    "${PYTHON_EXEC}" "${QUANT_SCRIPT}"
    --model_dir "${MODEL_DIR_FOR_QUANT}"
    --output_dir "${CHECKPOINT_DIR}"
    --dtype "${TRTLLM_DTYPE}"
    --qformat int4_awq
    --awq_block_size "${AWQ_BLOCK_SIZE}"
    --calib_size "${CALIB_SIZE}"
  )
  echo "[build-awq] Running: ${QUANT_CMD[*]}"
  "${QUANT_CMD[@]}"
fi

# Sanity check the quantized checkpoint before building
echo "[build-awq] Validating quantized checkpoint..."
test -f "${CHECKPOINT_DIR}/config.json" || { echo "[build-awq] ERROR: No config.json found in ${CHECKPOINT_DIR}"; exit 1; }
ls "${CHECKPOINT_DIR}"/rank*.safetensors >/dev/null 2>&1 || { echo "[build-awq] ERROR: No rank*.safetensors found in ${CHECKPOINT_DIR}"; exit 1; }
echo "[build-awq] Quantized checkpoint validation passed."

echo ""
echo "[build-awq] ============================================"
echo "[build-awq] Step 2/2: Build TensorRT-LLM engine"
echo "[build-awq] ============================================"

# Skip engine build if directory exists and not forcing
if [[ -d "${TRTLLM_ENGINE_DIR}" && "${FORCE}" != true ]]; then
  echo "[build-awq] Reusing existing engine at ${TRTLLM_ENGINE_DIR}"
else
  echo "[build-awq] Building TensorRT INT4-AWQ engine..."
  
  BUILD_CMD=(
    trtllm-build
    --checkpoint_dir "${CHECKPOINT_DIR}"
    --output_dir "${TRTLLM_ENGINE_DIR}"
    --gemm_plugin auto
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
  echo "[build-awq] Running: ${BUILD_CMD[*]}"
  "${BUILD_CMD[@]}"
fi

echo ""
echo "[build-awq] ============================================"
echo "[build-awq] Done. Engine: ${TRTLLM_ENGINE_DIR}"
echo "[build-awq] Configuration: INT4-AWQ weight-only"
echo "[build-awq] Model weights: 6GB → ~1.5GB (≈4x smaller)"
echo "[build-awq] To run server:"
echo "  export TRTLLM_ENGINE_DIR=\"${TRTLLM_ENGINE_DIR}\""
echo "  bash scripts/04-run-server.sh"
echo "[build-awq] ============================================"


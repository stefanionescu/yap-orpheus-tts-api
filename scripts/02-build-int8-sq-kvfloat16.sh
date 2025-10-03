#!/usr/bin/env bash
set -euo pipefail

source "scripts/lib/common.sh"
load_env_if_present

: "${VENV_DIR:=$PWD/.venv}"
: "${MODEL_ID:=canopylabs/orpheus-3b-0.1-ft}"
: "${QUANTIZED_DIR:=$PWD/models/orpheus-int8sq-kvint8}"
: "${TRTLLM_ENGINE_DIR:=$PWD/models/orpheus-trt-int8sq}"
: "${TRTLLM_DTYPE:=float16}"
: "${TRTLLM_MAX_INPUT_LEN:=128}"
: "${TRTLLM_MAX_OUTPUT_LEN:=2048}"
: "${TRTLLM_MAX_BATCH_SIZE:=1}"
: "${QUANTIZE_DTYPE:=float16}"
: "${CALIB_SIZE:=512}"
: "${CALIB_BATCH_SIZE:=8}"
: "${CALIB_MAX_SEQ_LENGTH:=128}"
: "${PYTHON_EXEC:=python}"
: "${TRTLLM_REPO_DIR:=$PWD/.trtllm-repo}"

usage() {
  cat <<USAGE
Usage: $0 [--model ID_OR_PATH] [--quantized-dir DIR] [--engine-dir DIR] [--dtype float16|bfloat16] [--max-input-len N] [--max-output-len N] [--max-batch-size N] [--calib-size N] [--force]

End-to-end INT8 SmoothQuant + INT8 KV build:
  HF checkpoint → W8A8 + INT8 KV quantized checkpoint → TRT engine

Defaults:
  --model              ${MODEL_ID}
  --quantized-dir      ${QUANTIZED_DIR}
  --engine-dir         ${TRTLLM_ENGINE_DIR}
  --dtype              ${QUANTIZE_DTYPE}
  --max-input-len      ${TRTLLM_MAX_INPUT_LEN}
  --max-output-len     ${TRTLLM_MAX_OUTPUT_LEN}
  --max-batch-size     ${TRTLLM_MAX_BATCH_SIZE}
  --calib-size         ${CALIB_SIZE}
USAGE
}

ARGS=()
FORCE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ID="$2"; shift 2 ;;
    --quantized-dir) QUANTIZED_DIR="$2"; shift 2 ;;
    --engine-dir) TRTLLM_ENGINE_DIR="$2"; shift 2 ;;
    --dtype) QUANTIZE_DTYPE="$2"; TRTLLM_DTYPE="$2"; shift 2 ;;
    --max-input-len) TRTLLM_MAX_INPUT_LEN="$2"; shift 2 ;;
    --max-output-len) TRTLLM_MAX_OUTPUT_LEN="$2"; shift 2 ;;
    --max-batch-size) TRTLLM_MAX_BATCH_SIZE="$2"; shift 2 ;;
    --calib-size) CALIB_SIZE="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]:-}"

[ -d "${VENV_DIR}" ] || { echo "[build-int8sq] venv missing. Run scripts/01-install.sh first." >&2; exit 1; }
source "${VENV_DIR}/bin/activate"

if [ "$(uname -s)" != "Linux" ]; then
  echo "[build-int8sq] ERROR: TensorRT-LLM builds require Linux with NVIDIA GPUs." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[build-int8sq] ERROR: nvidia-smi not detected. Ensure the GPU is visible." >&2
  exit 1
fi

source_env_dir "scripts/env"
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
  export HF_HUB_TOKEN="${HF_HUB_TOKEN:-$HF_TOKEN}"
fi

# Ensure TensorRT-LLM repo is available for quantization scripts
if [ ! -d "${TRTLLM_REPO_DIR}" ]; then
  echo "[build-int8sq] Cloning TensorRT-LLM repo for quantization scripts..."
  git clone https://github.com/NVIDIA/TensorRT-LLM.git "${TRTLLM_REPO_DIR}"
fi

# Pin repo tag to installed wheel version (critical for compatibility)
echo "[build-int8sq] Syncing repo version with installed wheel..."
# Extract version number, filtering out TRT-LLM log messages
TRTLLM_VER="$(${PYTHON_EXEC} -c 'import tensorrt_llm as t; print(t.__version__)' 2>/dev/null | tail -1 | tr -d '[:space:]')"
echo "[build-int8sq] Detected TensorRT-LLM version: ${TRTLLM_VER}"
git -C "${TRTLLM_REPO_DIR}" fetch --tags

# Try tag first, fallback to known commit for v1.0.0
if ! git -C "${TRTLLM_REPO_DIR}" checkout "v${TRTLLM_VER}" 2>/dev/null; then
  if [[ "${TRTLLM_VER}" == "1.0.0" ]]; then
    echo "[build-int8sq] Tag v1.0.0 not found, using known commit ae8270b713446948246f16fadf4e2a32e35d0f62"
    git -C "${TRTLLM_REPO_DIR}" checkout ae8270b713446948246f16fadf4e2a32e35d0f62
  else
    echo "[build-int8sq] ERROR: Could not checkout version ${TRTLLM_VER}" >&2
    exit 1
  fi
fi

# Install quantization dependencies
echo "[build-int8sq] Installing quantization requirements..."
pip install -r "${TRTLLM_REPO_DIR}/examples/quantization/requirements.txt"

if [ ! -f "${TRTLLM_REPO_DIR}/examples/quantization/quantize.py" ]; then
  echo "[build-int8sq] ERROR: quantize.py not found in TensorRT-LLM repo at ${TRTLLM_REPO_DIR}" >&2
  exit 1
fi

echo "[build-int8sq] ============================================"
echo "[build-int8sq] Step 1/2: Quantize to INT8 SQ + FP16 KV"
echo "[build-int8sq] ============================================"

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Skip quantization if directory exists and not forcing
if [[ -d "${QUANTIZED_DIR}" && "${FORCE}" != true ]]; then
  echo "[build-int8sq] Reusing existing quantized checkpoint at ${QUANTIZED_DIR}"
else
  echo "[build-int8sq] Creating quantized checkpoint..."
  mkdir -p "${QUANTIZED_DIR}"
  
  QUANT_CMD=(
    "${PYTHON_EXEC}" "${TRTLLM_REPO_DIR}/examples/quantization/quantize.py"
    --model_dir "${MODEL_ID}"
    --output_dir "${QUANTIZED_DIR}"
    --dtype "${QUANTIZE_DTYPE}"
    --qformat int8_sq
    --kv_cache_dtype float16
    --calib_size "${CALIB_SIZE}"
    --batch_size "${CALIB_BATCH_SIZE}"
    --calib_max_seq_length "${CALIB_MAX_SEQ_LENGTH}"
  )
  echo "[build-int8sq] Running: ${QUANT_CMD[*]}"
  "${QUANT_CMD[@]}"
fi

# Sanity check the quantized checkpoint before building
echo "[build-int8sq] Validating quantized checkpoint..."
test -f "${QUANTIZED_DIR}/config.json" || { echo "[build-int8sq] ERROR: No config.json found in ${QUANTIZED_DIR}"; exit 1; }
ls "${QUANTIZED_DIR}"/rank*.safetensors >/dev/null 2>&1 || { echo "[build-int8sq] ERROR: No rank*.safetensors found in ${QUANTIZED_DIR}"; exit 1; }
echo "[build-int8sq] Quantized checkpoint validation passed."

echo ""
echo "[build-int8sq] ============================================"
echo "[build-int8sq] Step 2/2: Build TensorRT-LLM engine"
echo "[build-int8sq] ============================================"

# Skip engine build if directory exists and not forcing
if [[ -d "${TRTLLM_ENGINE_DIR}" && "${FORCE}" != true ]]; then
  echo "[build-int8sq] Reusing existing engine at ${TRTLLM_ENGINE_DIR}"
else
  echo "[build-int8sq] Building TensorRT engine..."
  
  BUILD_CMD=(
    trtllm-build
    --checkpoint_dir "${QUANTIZED_DIR}"
    --output_dir "${TRTLLM_ENGINE_DIR}"
    --max_input_len "${TRTLLM_MAX_INPUT_LEN}"
    --max_seq_len "$((TRTLLM_MAX_INPUT_LEN + TRTLLM_MAX_OUTPUT_LEN))"
    --max_batch_size "${TRTLLM_MAX_BATCH_SIZE}"
    --remove_input_padding enable
    --gemm_plugin auto
    --log_level info
    --workers "$(nproc --all)"
  )
  echo "[build-int8sq] Running: ${BUILD_CMD[*]}"
  "${BUILD_CMD[@]}"
fi

echo ""
echo "[build-int8sq] ============================================"
echo "[build-int8sq] Done. Engine: ${TRTLLM_ENGINE_DIR}"
echo "[build-int8sq] To run server:"
echo "  export BACKEND=trtllm"
echo "  export TRTLLM_ENGINE_DIR=\"${TRTLLM_ENGINE_DIR}\""
echo "  bash scripts/04-run-server.sh"
echo "[build-int8sq] ============================================"

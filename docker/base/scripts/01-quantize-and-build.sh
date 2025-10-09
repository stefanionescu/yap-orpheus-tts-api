#!/usr/bin/env bash
set -euo pipefail

# Quantize (INT4-AWQ) and build a TensorRT-LLM engine using preinstalled deps.

# Defaults (override via env or flags)
MODEL_ID=${MODEL_ID:-canopylabs/orpheus-3b-0.1-ft}
TRTLLM_REPO_DIR=${TRTLLM_REPO_DIR:-/opt/TensorRT-LLM}
MODELS_DIR=${MODELS_DIR:-/opt/models}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/opt/checkpoints/orpheus-trtllm-ckpt-int4-awq}
ENGINE_OUTPUT_DIR=${TRTLLM_ENGINE_DIR:-/opt/engines/orpheus-trt-int4-awq}

# Build parameters (match project defaults)
TRTLLM_DTYPE=${TRTLLM_DTYPE:-float16}
TRTLLM_MAX_INPUT_LEN=${TRTLLM_MAX_INPUT_LEN:-48}
TRTLLM_MAX_OUTPUT_LEN=${TRTLLM_MAX_OUTPUT_LEN:-1024}
TRTLLM_MAX_BATCH_SIZE=${TRTLLM_MAX_BATCH_SIZE:-16}
AWQ_BLOCK_SIZE=${AWQ_BLOCK_SIZE:-128}
CALIB_SIZE=${CALIB_SIZE:-256}

PYTHON_EXEC=${PYTHON_EXEC:-python}

usage() {
  echo "Usage: $0 [--model ID_OR_PATH] [--checkpoint-dir DIR] [--engine-dir DIR] [--dtype float16|bfloat16] [--max-input-len N] [--max-output-len N] [--max-batch-size N] [--awq-block-size N] [--calib-size N] [--force]"
}

ARGS=()
FORCE_REBUILD=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ID="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --engine-dir) ENGINE_OUTPUT_DIR="$2"; shift 2 ;;
    --dtype) TRTLLM_DTYPE="$2"; shift 2 ;;
    --max-input-len) TRTLLM_MAX_INPUT_LEN="$2"; shift 2 ;;
    --max-output-len) TRTLLM_MAX_OUTPUT_LEN="$2"; shift 2 ;;
    --max-batch-size) TRTLLM_MAX_BATCH_SIZE="$2"; shift 2 ;;
    --awq-block-size) AWQ_BLOCK_SIZE="$2"; shift 2 ;;
    --calib-size) CALIB_SIZE="$2"; shift 2 ;;
    --force) FORCE_REBUILD=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]:-}"

echo "=== Quantize and Build TRT-LLM Engine ==="

# Env validation
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not detected. GPU required for engine build." >&2
  exit 1
fi

if [[ ! -d "$TRTLLM_REPO_DIR" ]]; then
  echo "ERROR: TRTLLM_REPO_DIR not found at $TRTLLM_REPO_DIR" >&2
  exit 1
fi

# Install quantization requirements now (match custom/build/build.sh)
quant_requirements="$TRTLLM_REPO_DIR/examples/quantization/requirements.txt"
if [[ -f "$quant_requirements" ]]; then
  echo "[build] Installing quantization requirements..."
  pip install -r "$quant_requirements"
else
  echo "[build] WARNING: quantization requirements.txt not found, continuing"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export HF_HUB_TOKEN="${HF_HUB_TOKEN:-$HF_TOKEN}"

# Resolve model directory (use pre-downloaded model inside image)
local_model_dir="$MODEL_ID"
if [[ ! -d "$MODEL_ID" ]]; then
  basename="${MODEL_ID##*/}"
  local_model_dir="${MODELS_DIR}/${basename}-hf"
  if [[ ! -d "$local_model_dir" ]]; then
    echo "ERROR: Expected pre-downloaded model at ${local_model_dir}. Rebuild image with HF_TOKEN secret."
    exit 1
  else
    echo "[build] Using cached HF model at ${local_model_dir}"
  fi
else
  echo "[build] Using local model directory: ${MODEL_ID}"
fi

# Skip if already built
if [[ -f "$ENGINE_OUTPUT_DIR/rank0.engine" && -f "$ENGINE_OUTPUT_DIR/config.json" && "$FORCE_REBUILD" != true ]]; then
  echo "[build] Engine already exists at: $ENGINE_OUTPUT_DIR"
  echo "[build] Use --force to rebuild"
  exit 0
fi

echo "[build] Configuration:"
echo "  Model: ${MODEL_ID}"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo "  Engine dir: ${ENGINE_OUTPUT_DIR}"
echo "  DType: ${TRTLLM_DTYPE}"
echo "  Max input/output: ${TRTLLM_MAX_INPUT_LEN}/${TRTLLM_MAX_OUTPUT_LEN}"
echo "  Max batch size: ${TRTLLM_MAX_BATCH_SIZE}"
echo "  AWQ block size: ${AWQ_BLOCK_SIZE}  Calib size: ${CALIB_SIZE}"

echo "[build] Step 1/2: Quantize to INT4-AWQ"
rm -rf "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

quant_script="$TRTLLM_REPO_DIR/examples/quantization/quantize.py"
if [[ ! -f "$quant_script" ]]; then
  echo "ERROR: quantize.py not found at $quant_script" >&2
  exit 1
fi

"$PYTHON_EXEC" "$quant_script" \
  --model_dir "$local_model_dir" \
  --output_dir "$CHECKPOINT_DIR" \
  --dtype "$TRTLLM_DTYPE" \
  --qformat int4_awq \
  --awq_block_size "$AWQ_BLOCK_SIZE" \
  --calib_size "$CALIB_SIZE" \
  --kv_cache_dtype int8

echo "[build] Step 2/2: Build TensorRT-LLM engine"
mkdir -p "$ENGINE_OUTPUT_DIR"

trtllm-build \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --output_dir "$ENGINE_OUTPUT_DIR" \
  --gemm_plugin auto \
  --gpt_attention_plugin float16 \
  --context_fmha enable \
  --paged_kv_cache enable \
  --remove_input_padding enable \
  --max_input_len "$TRTLLM_MAX_INPUT_LEN" \
  --max_seq_len "$((TRTLLM_MAX_INPUT_LEN + TRTLLM_MAX_OUTPUT_LEN))" \
  --max_batch_size "$TRTLLM_MAX_BATCH_SIZE" \
  --log_level info \
  --workers "$(nproc --all)"

echo "[build] ✓ Done. Engine at: $ENGINE_OUTPUT_DIR"


#!/usr/bin/env bash
set -euo pipefail

# Common helpers and env
source "scripts/lib/common.sh"
load_env_if_present

# Defaults
: "${VENV_DIR:=$PWD/.venv}"
: "${MODEL_ID:=canopylabs/orpheus-3b-0.1-ft}"
: "${TRTLLM_ENGINE_DIR:=$PWD/models/orpheus-trt}"
: "${FP16_MODEL_DIR:=$PWD/models/orpheus-fp16}"
: "${TRTLLM_DTYPE:=float16}"
: "${TRTLLM_MAX_INPUT_LEN:=128}"
: "${TRTLLM_MAX_OUTPUT_LEN:=2048}"
: "${TRTLLM_MAX_BATCH_SIZE:=16}"
: "${TRTLLM_KV_CACHE_DTYPE:=int8}"
: "${PYTHON_EXEC:=python}"

usage() {
  cat <<USAGE
Usage: $0 [--model ID] [--output DIR] [--dtype float16|bfloat16] [--max-input-len N] [--max-output-len N] [--max-batch-size N] [--kv-cache-dtype fp8|int8] [--quantized-dir DIR] [--minimal] [--cli]

Build a TensorRT-LLM engine directory for Orpheus.

Options:
  --minimal        Build tiny config (128/128/1) to validate toolchain
  --cli            Use trtllm-build CLI instead of Python builder

Defaults:
  --model           ${MODEL_ID}
  --output          ${TRTLLM_ENGINE_DIR}
  --dtype           ${TRTLLM_DTYPE}
  --max-input-len   ${TRTLLM_MAX_INPUT_LEN}
  --max-output-len  ${TRTLLM_MAX_OUTPUT_LEN}
  --max-batch-size  ${TRTLLM_MAX_BATCH_SIZE}
  --kv-cache-dtype  ${TRTLLM_KV_CACHE_DTYPE:-<unset>}
  --quantized-dir   ${TRTLLM_QUANTIZED_DIR:-<unset>}
USAGE
}

ARGS=()
MINIMAL=false
USE_CLI=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ID="$2"; shift 2 ;;
    --output) TRTLLM_ENGINE_DIR="$2"; shift 2 ;;
    --dtype) TRTLLM_DTYPE="$2"; shift 2 ;;
    --max-input-len) TRTLLM_MAX_INPUT_LEN="$2"; shift 2 ;;
    --max-output-len) TRTLLM_MAX_OUTPUT_LEN="$2"; shift 2 ;;
    --max-batch-size) TRTLLM_MAX_BATCH_SIZE="$2"; shift 2 ;;
    --kv-cache-dtype) TRTLLM_KV_CACHE_DTYPE="$2"; shift 2 ;;
    --quantized-dir) TRTLLM_QUANTIZED_DIR="$2"; shift 2 ;;
    --minimal) MINIMAL=true; shift ;;
    --cli) USE_CLI=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]:-}"

[ -d "${VENV_DIR}" ] || { echo "[build-trt] venv missing. Run scripts/01-install.sh first." >&2; exit 1; }
source "${VENV_DIR}/bin/activate"

if [ "$(uname -s)" != "Linux" ]; then
  echo "[build-trt] ERROR: TensorRT-LLM builds require Linux with NVIDIA GPUs." >&2
  exit 1
fi

if [ -z "${MODEL_FOR_BUILD}" ]; then
  echo "[build-trt] ERROR: Failed to resolve checkpoint directory." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[build-trt] ERROR: nvidia-smi not detected. Ensure the GPU is visible to this runtime." >&2
  exit 1
fi

# Source modular env snippets (may override defaults)
source_env_dir "scripts/env"

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
  export HF_HUB_TOKEN="${HF_HUB_TOKEN:-$HF_TOKEN}"
fi

MODE="python"
if [ "$USE_CLI" = true ]; then
  MODE="cli"
fi

MODEL_FOR_BUILD=""
if [ -d "${MODEL_ID}" ]; then
  SENTINEL_PATH="${MODEL_ID%/}/.fp16-export.json"
  if [ -f "${SENTINEL_PATH}" ]; then
    MODEL_FOR_BUILD="$(cd "${MODEL_ID}" && pwd)"
  else
    echo "[build-trt] Source directory lacks FP16 sentinel; exporting to ${FP16_MODEL_DIR}"
    bash scripts/01b-export-fp16-checkpoint.sh --model "${MODEL_ID}" --output "${FP16_MODEL_DIR}"
    MODEL_FOR_BUILD="$(cd "${FP16_MODEL_DIR}" && pwd)"
  fi
else
  echo "[build-trt] Exporting FP16 checkpoint for ${MODEL_ID}"
  bash scripts/01b-export-fp16-checkpoint.sh --model "${MODEL_ID}" --output "${FP16_MODEL_DIR}"
  MODEL_FOR_BUILD="$(cd "${FP16_MODEL_DIR}" && pwd)"
fi

echo "[build-trt] Model: ${MODEL_ID}"
echo "[build-trt] Using checkpoint directory: ${MODEL_FOR_BUILD}"
echo "[build-trt] Output: ${TRTLLM_ENGINE_DIR}"
echo "[build-trt] Mode: ${MODE}${MINIMAL:+ (minimal)}"

mkdir -p "${TRTLLM_ENGINE_DIR}"

if [ "$USE_CLI" = true ]; then
  CONTEXT_FMHA_ARGS=""
  if [ "${TRTLLM_KV_CACHE_DTYPE}" = "fp8" ]; then
    CONTEXT_FMHA_ARGS=" --use_fp8_context_fmha enable"
  fi

  if [ "$MINIMAL" = true ]; then
    echo "[build-trt] Running minimal CLI build..."
    trtllm-build \
      --checkpoint_dir "$MODEL_FOR_BUILD" \
      --output_dir "$TRTLLM_ENGINE_DIR" \
      --max_input_len 128 \
      --max_seq_len 256 \
      --max_batch_size 1 \
      --remove_input_padding enable \
      --log_level info${CONTEXT_FMHA_ARGS}
  else
    echo "[build-trt] Running full CLI build..."
    trtllm-build \
      --checkpoint_dir "$MODEL_FOR_BUILD" \
      --output_dir "$TRTLLM_ENGINE_DIR" \
      --max_input_len "${TRTLLM_MAX_INPUT_LEN}" \
      --max_seq_len "$((TRTLLM_MAX_INPUT_LEN + TRTLLM_MAX_OUTPUT_LEN))" \
      --max_batch_size "${TRTLLM_MAX_BATCH_SIZE}" \
      --remove_input_padding enable \
      --log_level info${CONTEXT_FMHA_ARGS}
  fi
else
  CMD=(
    "${PYTHON_EXEC}" server/build/build-trt-engine.py
    --model "${MODEL_FOR_BUILD}"
    --output "${TRTLLM_ENGINE_DIR}"
    --dtype "${TRTLLM_DTYPE}"
    --max_input_len "${TRTLLM_MAX_INPUT_LEN}"
    --max_output_len "${TRTLLM_MAX_OUTPUT_LEN}"
    --max_batch_size "${TRTLLM_MAX_BATCH_SIZE}"
    --context_fmha "${TRTLLM_CONTEXT_FMHA:-disable}"
  )
  if [ -n "${TRTLLM_KV_CACHE_DTYPE}" ]; then CMD+=(--kv_cache_dtype "${TRTLLM_KV_CACHE_DTYPE}"); fi
  # No offline quantization flags; KV cache quant is runtime-only via KvCacheConfig
  if [ "$MINIMAL" = true ]; then CMD+=(--minimal); fi
  echo "[build-trt] Running: ${CMD[*]}"
  "${CMD[@]}"
fi

echo "[build-trt] Engine directory ready: ${TRTLLM_ENGINE_DIR}"
echo "[build-trt] To use it: export BACKEND=trtllm; export TRTLLM_ENGINE_DIR=\"${TRTLLM_ENGINE_DIR}\"; bash scripts/03-run-server.sh"

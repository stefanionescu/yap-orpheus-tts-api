#!/usr/bin/env bash
set -euo pipefail

# Common helpers and env
source "scripts/lib/common.sh"
load_env_if_present

# Defaults
: "${VENV_DIR:=$PWD/.venv}"
: "${MODEL_ID:=canopylabs/orpheus-3b-0.1-ft}"
: "${TRTLLM_ENGINE_DIR:=$PWD/models/orpheus-trt}"
: "${TRTLLM_DTYPE:=bfloat16}"  # float16|bfloat16
: "${TRTLLM_MAX_INPUT_LEN:=2048}"
: "${TRTLLM_MAX_OUTPUT_LEN:=2048}"
: "${TRTLLM_MAX_BATCH_SIZE:=16}"

usage() {
  cat <<USAGE
Usage: $0 [--model ID] [--output DIR] [--dtype float16|bfloat16] [--max-input-len N] [--max-output-len N] [--max-batch-size N]

Build a TensorRT-LLM engine directory for Orpheus.

Defaults:
  --model           ${MODEL_ID}
  --output          ${TRTLLM_ENGINE_DIR}
  --dtype           ${TRTLLM_DTYPE}
  --max-input-len   ${TRTLLM_MAX_INPUT_LEN}
  --max-output-len  ${TRTLLM_MAX_OUTPUT_LEN}
  --max-batch-size  ${TRTLLM_MAX_BATCH_SIZE}
USAGE
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ID="$2"; shift 2 ;;
    --output) TRTLLM_ENGINE_DIR="$2"; shift 2 ;;
    --dtype) TRTLLM_DTYPE="$2"; shift 2 ;;
    --max-input-len) TRTLLM_MAX_INPUT_LEN="$2"; shift 2 ;;
    --max-output-len) TRTLLM_MAX_OUTPUT_LEN="$2"; shift 2 ;;
    --max-batch-size) TRTLLM_MAX_BATCH_SIZE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]:-}"

[ -d "${VENV_DIR}" ] || { echo "[build-trt] venv missing. Run scripts/01-install.sh first." >&2; exit 1; }
source "${VENV_DIR}/bin/activate"

# Source modular env snippets (may override defaults)
source_env_dir "scripts/env"

echo "[build-trt] Model: ${MODEL_ID}"
echo "[build-trt] Output: ${TRTLLM_ENGINE_DIR}"

# Ensure TRT-LLM is installed; if not, install it
python - <<'PY'
try:
    import tensorrt_llm  # noqa: F401
    print("[build-trt] TensorRT-LLM present")
except Exception:
    import sys
    sys.exit(42)
PY
if [ $? -ne 0 ]; then
  echo "[build-trt] Installing TensorRT-LLM backend..."
  bash scripts/01-install-trt.sh
fi

mkdir -p "${TRTLLM_ENGINE_DIR}"

CMD=(
  python server/build/build-trt-engine.py
  --model "${MODEL_ID}"
  --output "${TRTLLM_ENGINE_DIR}"
  --dtype "${TRTLLM_DTYPE}"
  --max_input_len "${TRTLLM_MAX_INPUT_LEN}"
  --max_output_len "${TRTLLM_MAX_OUTPUT_LEN}"
  --max_batch_size "${TRTLLM_MAX_BATCH_SIZE}"
)

echo "[build-trt] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[build-trt] Engine directory ready: ${TRTLLM_ENGINE_DIR}"
echo "[build-trt] To use it: export BACKEND=trtllm; export TRTLLM_ENGINE_DIR=\"${TRTLLM_ENGINE_DIR}\"; bash scripts/03-run-server.sh"



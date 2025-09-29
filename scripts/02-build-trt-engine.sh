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
Usage: $0 [--model ID] [--output DIR] [--dtype float16|bfloat16] [--max-input-len N] [--max-output-len N] [--max-batch-size N] [--minimal] [--cli]

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
    --minimal) MINIMAL=true; shift ;;
    --cli) USE_CLI=true; shift ;;
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
echo "[build-trt] Mode: ${USE_CLI:+cli}${USE_CLI:-python}${MINIMAL:+ (minimal)}"

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

if $USE_CLI; then
  # Pre-download HF snapshot to avoid symlink surprises and mid-build network I/O
  PYTHON_EXEC=${PYTHON_EXEC:-python}
  echo "[build-trt] Pre-downloading weights for ${MODEL_ID}..."
  "$PYTHON_EXEC" - <<'PY'
from huggingface_hub import snapshot_download
import os
repo_id = os.environ.get("MODEL_ID")
local_dir = os.path.join(os.getcwd(), ".hf", repo_id.replace("/", "-"))
path = snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
print(f"[build-trt] Weights cached at: {path}")
PY

  CKPT_DIR="${PWD}/.hf/${MODEL_ID//\//-}"
  OUT_DIR="${TRTLLM_ENGINE_DIR}"

  if $MINIMAL; then
    echo "[build-trt] Running minimal CLI build..."
    trtllm-build \
      --checkpoint_dir "$CKPT_DIR" \
      --output_dir "$OUT_DIR" \
      --max_input_len 128 \
      --max_seq_len 256 \
      --max_batch_size 1 \
      --remove_input_padding enable \
      --log_level info
  else
    echo "[build-trt] Running full CLI build..."
    trtllm-build \
      --checkpoint_dir "$CKPT_DIR" \
      --output_dir "$OUT_DIR" \
      --max_input_len "${TRTLLM_MAX_INPUT_LEN}" \
      --max_seq_len "$((TRTLLM_MAX_INPUT_LEN + TRTLLM_MAX_OUTPUT_LEN))" \
      --max_batch_size "${TRTLLM_MAX_BATCH_SIZE}" \
      --remove_input_padding enable \
      --log_level info
  fi
else
  CMD=(
    python server/build/build-trt-engine.py
    --model "${MODEL_ID}"
    --output "${TRTLLM_ENGINE_DIR}"
    --dtype "${TRTLLM_DTYPE}"
    --max_input_len "${TRTLLM_MAX_INPUT_LEN}"
    --max_output_len "${TRTLLM_MAX_OUTPUT_LEN}"
    --max_batch_size "${TRTLLM_MAX_BATCH_SIZE}"
  )
  if $MINIMAL; then CMD+=(--minimal); fi
  echo "[build-trt] Running: ${CMD[*]}"
  "${CMD[@]}"
fi

echo "[build-trt] Engine directory ready: ${TRTLLM_ENGINE_DIR}"
echo "[build-trt] To use it: export BACKEND=trtllm; export TRTLLM_ENGINE_DIR=\"${TRTLLM_ENGINE_DIR}\"; bash scripts/03-run-server.sh"



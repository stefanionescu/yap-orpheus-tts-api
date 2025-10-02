#!/usr/bin/env bash
set -euo pipefail

source "scripts/lib/common.sh"
load_env_if_present

: "${VENV_DIR:=$PWD/.venv}"
: "${MODEL_ID:=canopylabs/orpheus-3b-0.1-ft}"
: "${FP16_MODEL_DIR:=$PWD/models/orpheus-fp16}"
: "${PYTHON_EXEC:=python}"

usage() {
  cat <<USAGE
Usage: $0 [--model ID_OR_PATH] [--output DIR] [--force]

Convert a Hugging Face checkpoints to FP16 for TensorRT builds. If the
output already exists with a valid sentinel, the conversion is skipped.
USAGE
}

ARGS=()
FORCE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ID="$2"; shift 2 ;;
    --output) FP16_MODEL_DIR="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]:-}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "[export-fp16] ERROR: venv missing. Run scripts/01-install.sh first." >&2
  exit 1
fi
source "${VENV_DIR}/bin/activate"

# Load project env overrides (HF token, cache paths, etc.)
source_env_dir "scripts/env"

CMD=(
  "${PYTHON_EXEC}" server/build/export_fp16_checkpoint.py
  --model "${MODEL_ID}"
  --output "${FP16_MODEL_DIR}"
)

if [ "${FORCE}" = true ]; then
  CMD+=(--force)
fi

echo "[export-fp16] Running: ${CMD[*]}"
"${CMD[@]}"

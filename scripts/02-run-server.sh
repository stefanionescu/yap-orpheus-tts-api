#!/usr/bin/env bash
set -euo pipefail
source ".env"

[ -d "${VENV_DIR}" ] || { echo "venv missing. Run scripts/01-install.sh"; exit 1; }
source "${VENV_DIR}/bin/activate"

# perf environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9
export OMP_NUM_THREADS=$(nproc)
export NVIDIA_TF32_OVERRIDE=1

echo "[run] Starting FastAPI on ${HOST}:${PORT}"
exec uvicorn app.server:app --host "${HOST}" --port "${PORT}" --timeout-keep-alive 75 --log-level info



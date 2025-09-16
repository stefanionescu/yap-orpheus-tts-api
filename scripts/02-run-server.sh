#!/usr/bin/env bash
set -euo pipefail
# Load env if present (optional)
if [ -f ".env" ]; then source ".env"; fi

[ -d "${VENV_DIR}" ] || { echo "venv missing. Run scripts/01-install.sh"; exit 1; }
source "${VENV_DIR}/bin/activate"

# perf environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9
export OMP_NUM_THREADS=$(nproc)
export NVIDIA_TF32_OVERRIDE=1

echo "[run] Starting FastAPI on ${HOST:-0.0.0.0}:${PORT:-8000}"

# Detached mode if DETACH=1 env or --detach arg
DETACH_FLAG=${DETACH:-0}
if [ "${1:-}" = "--detach" ]; then DETACH_FLAG=1; fi

CMD="uvicorn server.server:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8000}\" --timeout-keep-alive 75 --log-level info"

if [ "$DETACH_FLAG" = "1" ]; then
  mkdir -p logs .run
  # run in background detached from TTY, redirect to log
  nohup bash -lc "$CMD" > logs/server.log 2>&1 &
  PID=$!
  echo $PID > .run/server.pid
  echo "[run] Server started in background (PID $PID). Logs: logs/server.log"
  exit 0
fi

exec uvicorn server.server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --timeout-keep-alive 75 --log-level info


#!/usr/bin/env bash
set -euo pipefail
# Load env if present (optional)
if [ -f ".env" ]; then source ".env"; fi
# Defaults
: "${VENV_DIR:=$PWD/.venv}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"

[ -d "${VENV_DIR}" ] || { echo "venv missing. Run scripts/01-install.sh"; exit 1; }
source "${VENV_DIR}/bin/activate"

# perf environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9
export OMP_NUM_THREADS=$(nproc)
export NVIDIA_TF32_OVERRIDE=1
# Ensure vLLM uses spawn (decoder.py initializes CUDA early)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Disable torch.compile/inductor to avoid building Triton/C extensions at runtime
export VLLM_TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TRITON_DISABLE_COMPILATION=1

echo "[run] Starting FastAPI on ${HOST:-0.0.0.0}:${PORT:-8000}"
CMD="uvicorn server.server:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8000}\" --timeout-keep-alive 75 --log-level info"

# Always start detached and write PID + logs
mkdir -p logs .run
nohup bash -lc "$CMD" > logs/server.log 2>&1 &
PID=$!
echo $PID > .run/server.pid
echo "[run] Server started in background (PID $PID)."
echo "[run] Following logs (Ctrl-C detaches, server keeps running)"
touch logs/server.log || true
exec tail -n +1 -F logs/server.log


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
# vLLM tuning defaults (override via env if needed)
export VLLM_DTYPE=${VLLM_DTYPE:-half}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
export VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.92}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-24}
# SNAC cadence: 5 ≈ ~100ms
export SNAC_DECODE_FRAMES=${SNAC_DECODE_FRAMES:-5}
# Ensure vLLM uses spawn (safe with CUDA init in background thread)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Disable torch.compile/inductor to avoid building Triton/C extensions at runtime
export VLLM_TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TRITON_DISABLE_COMPILATION=1

echo "[run] Starting FastAPI on ${HOST:-0.0.0.0}:${PORT:-8000}"
CMD="uvicorn server.server:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8000}\" --timeout-keep-alive 75 --log-level info"

# Always start detached and write PID + logs
mkdir -p logs .run
# Fully detach from TTY so Ctrl-C on this shell won't signal the server
setsid bash -lc "$CMD" </dev/null > logs/server.log 2>&1 &
PID=$!
echo $PID > .run/server.pid
echo "[run] Server started in background (PID $PID)."
echo "[run] Following logs (Ctrl-C detaches, server keeps running)"
touch logs/server.log || true
exec tail -n +1 -F logs/server.log


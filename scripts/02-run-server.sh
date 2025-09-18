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
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export NVIDIA_TF32_OVERRIDE=1
# vLLM tuning defaults (override via env if needed)
export VLLM_DTYPE=${VLLM_DTYPE:-float16}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
export VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.95}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-24}
# Disable torch.compile for SNAC by default to match Baseten testing
export SNAC_TORCH_COMPILE=${SNAC_TORCH_COMPILE:-0}
# Enable vLLM prefix cache to cut prefill on repeated voice/prompt preambles
export VLLM_PREFIX_CACHE=${VLLM_PREFIX_CACHE:-1}
# Server-side chunking size for Baseten Mode A (~280 chars recommended)
export MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-280}
# SNAC dynamic batching
export SNAC_MAX_BATCH=${SNAC_MAX_BATCH:-64}
export SNAC_BATCH_TIMEOUT_MS=${SNAC_BATCH_TIMEOUT_MS:-10}
# Ensure vLLM uses spawn (safe with CUDA init in background thread)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Disable torch.compile/inductor to avoid building Triton/C extensions at runtime (vLLM path)
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


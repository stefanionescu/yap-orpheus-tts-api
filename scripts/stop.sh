#!/usr/bin/env bash
set -euo pipefail

# Flags:
#  --clean-install  Remove venv and Python/HF/torch caches created by 01-install-trt.sh
#  --clean-system   Clean apt caches created by 00-bootstrap.sh (best-effort)
#  --clean-trt      Remove TensorRT-LLM build artefacts (engines, quantized checkpoints, repo clone, caches)

# Default paths - must match build scripts
: "${TRTLLM_REPO_DIR:=$PWD/.trtllm-repo}"
: "${MODELS_DIR:=$PWD/models}"

CLEAN_INSTALL=0
CLEAN_SYSTEM=0
CLEAN_TRT=0
for arg in "$@"; do
  case "$arg" in
    --clean-install) CLEAN_INSTALL=1 ;;
    --clean-system) CLEAN_SYSTEM=1 ;;
    --clean-trt) CLEAN_TRT=1 ;;
    *) ;;
  esac
done

echo "[stop] Stopping server if running"
if [ -f .run/server.pid ]; then
  PID=$(cat .run/server.pid || true)
  if [ -n "$PID" ]; then
    kill "$PID" || true
    sleep 1
  fi
  rm -f .run/server.pid || true
fi
pkill -f "uvicorn server.server:app" || true
sleep 1

if [ -f .run/run-all.pid ]; then
  RA_PID=$(cat .run/run-all.pid || true)
  if [ -n "$RA_PID" ]; then
    kill "$RA_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$RA_PID" 2>/dev/null || true
  fi
  rm -f .run/run-all.pid || true
fi
pkill -f "scripts/run-all.sh" 2>/dev/null || true

# Kill any lingering GPU compute processes (free VRAM)
if command -v nvidia-smi >/dev/null 2>&1; then
  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
  if [ -n "$PIDS" ]; then
    echo "$PIDS" | xargs -r -n1 kill 2>/dev/null || true
    sleep 1
    # hard kill stragglers
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    if [ -n "$PIDS" ]; then
      echo "$PIDS" | xargs -r -n1 kill -9 2>/dev/null || true
    fi
  fi
  
  pkill -9 -f "python.*tensorrt" 2>/dev/null || true
  pkill -9 -f "python.*trtllm" 2>/dev/null || true
  pkill -9 -f "mpirun" 2>/dev/null || true
  pkill -9 -f "mpi4py" 2>/dev/null || true
  sleep 1
  
  if command -v fuser >/dev/null 2>&1; then
    for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia0 /dev/nvidia1; do
      [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
    done
  fi
  
  rm -rf /dev/shm/nvidia* /dev/shm/cuda* /tmp/cuda* /tmp/.X11-unix/* 2>/dev/null || true
  
  if command -v ipcs >/dev/null 2>&1; then
    CURRENT_USER=$(whoami)
    ipcs -s | grep "$CURRENT_USER" | awk '{print $2}' | xargs -r -n1 ipcrm -s 2>/dev/null || true
    ipcs -q | grep "$CURRENT_USER" | awk '{print $2}' | xargs -r -n1 ipcrm -q 2>/dev/null || true
    ipcs -m | grep "$CURRENT_USER" | awk '{print $2}' | xargs -r -n1 ipcrm -m 2>/dev/null || true
  fi
  
  sleep 2
  
  LEFT=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
  if [ -z "$LEFT" ]; then
    GPU_INDEX=${GPU_INDEX:-0}
    nvidia-smi --gpu-reset -i "$GPU_INDEX" 2>/dev/null || true
    nvidia-smi -i "$GPU_INDEX" -pm 0 2>/dev/null || true
    nvidia-smi -i "$GPU_INDEX" -pm 1 2>/dev/null || true
  fi
fi

rm -rf .run logs 2>/dev/null || true

if [ "$CLEAN_INSTALL" = "1" ]; then
  echo "[stop] Cleaning install artifacts"
  rm -rf .venv || true
  rm -rf ~/.cache/huggingface ~/.local/share/huggingface || true
  [ -n "${HF_HOME:-}" ] && rm -rf "${HF_HOME}" || true
  [ -n "${HF_HUB_CACHE:-}" ] && rm -rf "${HF_HUB_CACHE}" || true
  rm -rf /tmp/huggingface* ~/.cache/torch ~/.cache/torch_extensions ~/.torch || true
  rm -rf /tmp/torch* /tmp/.torch* /dev/shm/torch* || true
  rm -rf ~/.triton ~/.cache/triton /tmp/triton* || true
  rm -rf ~/.cache/pip ~/.cache/hf_transfer ~/.cache/clip ~/.cache/transformers || true
  rm -rf /tmp/.hf* 2>/dev/null || true
fi

if [ "$CLEAN_TRT" = "1" ]; then
  echo "[stop] Cleaning TRT artifacts"
  rm -rf "${TRTLLM_REPO_DIR}" "${MODELS_DIR}" || true
  find "$PWD" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  rm -rf ~/.cache/tensorrt* ~/.cache/nv_tensorrt ~/.cache/nvidia ~/.nv || true
  rm -rf ~/.cache/modelopt ~/.cache/model_optimizer ~/.cache/nvidia_modelopt || true
  rm -rf /tmp/trt* /tmp/tensorrt* /tmp/modelopt* /tmp/quantiz* /tmp/calib* 2>/dev/null || true
  rm -rf /dev/shm/trt* /dev/shm/quantiz* 2>/dev/null || true
fi

if [ "$CLEAN_SYSTEM" = "1" ]; then
  echo "[stop] Cleaning system caches"
  command -v apt-get >/dev/null 2>&1 && apt-get clean || true
  rm -rf /var/lib/apt/lists/* /tmp/pip-* /tmp/tmp* /var/tmp/* 2>/dev/null || true
fi

# Always clean workspace temp files
rm -rf audio/*.wav .pytest_cache .mypy_cache .ruff_cache *.egg-info 2>/dev/null || true

echo "[stop] Done"


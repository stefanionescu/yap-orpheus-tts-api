#!/usr/bin/env bash
set -euo pipefail

# Flags:
#  --clean-install  Remove venv and Python/HF/torch caches created by 01-install.sh
#  --clean-system   Clean apt caches created by 00-bootstrap.sh (best-effort)
#  --clean-trt      Remove TensorRT-LLM build artefacts (engines, local HF/FP16 caches)

: "${TRTLLM_ENGINE_DIR:=$PWD/models/orpheus-trt}"
: "${FP16_MODEL_DIR:=$PWD/models/orpheus-fp16}"
: "${HF_SNAPSHOT_DIR:=$PWD/.hf}"

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

echo "[stop] Stopping server and cleaning run artifacts"

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
  echo "[stop] Stopping background pipeline"
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
  echo "[stop] Killing GPU compute processes (if any)"
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
  
  # Kill any Python processes that might have CUDA contexts
  echo "[stop] Killing Python/TensorRT processes"
  pkill -9 -f "python.*tensorrt" 2>/dev/null || true
  pkill -9 -f "python.*trtllm" 2>/dev/null || true
  pkill -9 -f "mpirun" 2>/dev/null || true
  pkill -9 -f "mpi4py" 2>/dev/null || true
  sleep 1
  
  # As a fallback, kill any process holding /dev/nvidia* (container-local)
  if command -v fuser >/dev/null 2>&1; then
    echo "[stop] Killing processes using NVIDIA devices"
    for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia0 /dev/nvidia1; do
      if [ -e "$dev" ]; then
        fuser -k -9 "$dev" 2>/dev/null || true
      fi
    done
  fi
  
  # Clear CUDA IPC handles and shared memory
  echo "[stop] Clearing CUDA shared memory"
  rm -rf /dev/shm/nvidia* 2>/dev/null || true
  rm -rf /dev/shm/cuda* 2>/dev/null || true
  rm -rf /tmp/cuda* 2>/dev/null || true
  rm -rf /tmp/.X11-unix/* 2>/dev/null || true
  
  # Clear any lingering IPC semaphores/message queues
  if command -v ipcs >/dev/null 2>&1; then
    echo "[stop] Clearing IPC resources"
    # Get current user
    CURRENT_USER=$(whoami)
    # Clear semaphores
    ipcs -s | grep "$CURRENT_USER" | awk '{print $2}' | xargs -r -n1 ipcrm -s 2>/dev/null || true
    # Clear message queues
    ipcs -q | grep "$CURRENT_USER" | awk '{print $2}' | xargs -r -n1 ipcrm -q 2>/dev/null || true
    # Clear shared memory segments
    ipcs -m | grep "$CURRENT_USER" | awk '{print $2}' | xargs -r -n1 ipcrm -m 2>/dev/null || true
  fi
  
  sleep 2
  
  # Attempt GPU reset if no compute procs remain (may require privileges)
  LEFT=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
  if [ -z "$LEFT" ]; then
    echo "[stop] Resetting GPU"
    GPU_INDEX=${GPU_INDEX:-0}
    # Try multiple reset methods
    nvidia-smi --gpu-reset -i "$GPU_INDEX" 2>/dev/null || true
    # Alternative: reset compute mode
    nvidia-smi -i "$GPU_INDEX" -pm 0 2>/dev/null || true
    nvidia-smi -i "$GPU_INDEX" -pm 1 2>/dev/null || true
    # Clear persistence mode and re-enable
    nvidia-smi -i "$GPU_INDEX" -pm 0 2>/dev/null || true
    sleep 1
    nvidia-smi -i "$GPU_INDEX" -pm 1 2>/dev/null || true
  else
    echo "[stop] WARNING: GPU processes still running: $LEFT"
  fi
  
  # Final check
  echo "[stop] Final GPU memory status:"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
else
  echo "[stop] nvidia-smi not available; skipping GPU process kill"
fi

echo "[stop] Removing run-time artifacts (.run/, logs/)"
rm -rf .run || true
rm -rf logs || true

if [ "$CLEAN_INSTALL" = "1" ]; then
  echo "[stop] Removing venv and caches from install step"
  rm -rf .venv || true
  # Hugging Face caches (default and alternative locations)
  rm -rf ~/.cache/huggingface || true
  rm -rf ~/.local/share/huggingface || true
  if [ -n "${HF_HOME:-}" ]; then rm -rf "${HF_HOME}" || true; fi
  if [ -n "${HF_HUB_CACHE:-}" ]; then rm -rf "${HF_HUB_CACHE}" || true; fi

  # Torch/pip caches
  rm -rf ~/.cache/torch || true
  rm -rf ~/.cache/torch_extensions || true
  rm -rf ~/.cache/pip || true
  rm -rf ~/.cache/hf_transfer || true
  rm -rf ~/.cache/clip || true
  rm -rf ~/.cache/vllm || true
  rm -rf ~/.cache/triton || true
  # TensorRT-LLM and NVIDIA caches
  rm -rf ~/.cache/tensorrt_llm || true
  rm -rf ~/.cache/tensorrt-llm || true
  rm -rf ~/.cache/nv_tensorrt || true
  rm -rf ~/.cache/nvidia || true
  rm -rf ~/.nv || true

  # Temp files that may accumulate
  rm -rf /tmp/vllm* 2>/dev/null || true
  rm -rf /tmp/huggingface* 2>/dev/null || true
  rm -rf /tmp/torch* 2>/dev/null || true
  rm -rf /dev/shm/vllm* 2>/dev/null || true
  rm -rf /tmp/trt* 2>/dev/null || true
  rm -rf /tmp/tensorrt* 2>/dev/null || true
  rm -rf /dev/shm/trt* 2>/dev/null || true
fi

# Remove TensorRT-LLM artefacts generated by build scripts
if [ "$CLEAN_TRT" = "1" ]; then
  echo "[stop] Removing TensorRT-LLM artefacts"
  # Engine output directory (env override or default used by scripts/03-build-trt-engine.sh)
  rm -rf "${TRTLLM_ENGINE_DIR}" || true
  # TensorRT-LLM checkpoint directories (converted from HF format)
  rm -rf "${TRTLLM_ENGINE_DIR}-ckpt" || true
  rm -rf "$PWD/models/orpheus-trt-ckpt" || true
  # FP16 checkpoints generated ahead of TRT builds
  rm -rf "${FP16_MODEL_DIR}" || true
  # Common streaming engine directory used by experiments
  rm -rf "$PWD/models/orpheus-streaming" || true
  rm -rf "$PWD/models/orpheus-streaming-ckpt" || true
  # Local HF snapshot cache used by build flow
  if [ -n "${TRTLLM_CACHE_DIR:-}" ]; then
    rm -rf "${TRTLLM_CACHE_DIR}" || true
  fi
  rm -rf "${HF_SNAPSHOT_DIR}" || true
  # As a safety net, remove stray engine files and checkpoint dirs under models/
  if [ -d "$PWD/models" ]; then
    find "$PWD/models" -type f -name "*.engine" -delete 2>/dev/null || true
    find "$PWD/models" -type f -name "model.plan" -delete 2>/dev/null || true
    find "$PWD/models" -type f -name "rank*.safetensors" -delete 2>/dev/null || true
    find "$PWD/models" -type d -name "*-ckpt" -exec rm -rf {} + 2>/dev/null || true
  fi
  # Clear Python bytecode that might hold references
  find "$PWD/server" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
fi

if [ "$CLEAN_SYSTEM" = "1" ]; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "[stop] Cleaning apt caches (system step)"
    apt-get clean || true
    rm -rf /var/lib/apt/lists/* || true
  else
    echo "[stop] apt-get not available; skipping system cache clean"
  fi
fi

echo "[stop] Wipe complete."

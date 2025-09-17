#!/usr/bin/env bash
set -euo pipefail

# Flags:
#  --clean-install  Remove venv and Python/HF/torch caches
#  --clean-system   Clean apt caches

CLEAN_INSTALL=0
CLEAN_SYSTEM=0

for arg in "$@"; do
  case "$arg" in
    --clean-install) CLEAN_INSTALL=1 ;;
    --clean-system) CLEAN_SYSTEM=1 ;;
    *) ;;
  esac
done

echo "[stop] Stopping server and cleaning run artifacts"

# 1) Kill uvicorn main proc (pidfile + pattern)
echo "[stop] Stopping uvicorn if running"
if [ -f .run/server.pid ]; then
  PID=$(cat .run/server.pid || true)
  if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" || true
    sleep 1
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f .run/server.pid || true
fi
# Fallback: pattern kill
pkill -f "uvicorn server.server:app" 2>/dev/null || true
sleep 1

# 2) Kill vLLM core engine & related workers (best-effort)
echo "[stop] Killing vLLM/ray workers (best-effort)"
# Common vLLM patterns
pkill -f "python.*vllm" 2>/dev/null || true
pkill -f "vllm.v1" 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "openai.api_server" 2>/dev/null || true
# Ray (if any were spawned by other configs)
pkill -f "ray::" 2>/dev/null || true
pkill -f "raylet" 2>/dev/null || true
pkill -f "plasma_store" 2>/dev/null || true
pkill -f "gcs_server" 2>/dev/null || true
sleep 1

# 3) Last resort: kill any process with open CUDA handles, then reset GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[stop] Checking for GPU compute processes"
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
  # Kill any process holding /dev/nvidia* (container-local)
  if command -v fuser >/dev/null 2>&1; then
    for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia0 /dev/nvidia1; do
      [ -e "$dev" ] && fuser -k "$dev" 2>/dev/null || true
    done
  fi
  # If nothing left, attempt GPU reset (may require privileges)
  LEFT=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
  if [ -z "$LEFT" ]; then
    GPU_INDEX=${GPU_INDEX:-0}
    echo "[stop] Attempting nvidia-smi --gpu-reset -i ${GPU_INDEX}"
    nvidia-smi --gpu-reset -i "$GPU_INDEX" 2>/dev/null || true
  else
    echo "[stop] GPU processes still present; skipping reset"
  fi
else
  echo "[stop] nvidia-smi not available; skipping GPU process kill"
fi

# 4) Remove run-time artifacts and SHM leftovers
echo "[stop] Removing .run/, logs/, and shared-memory artifacts"
rm -rf .run || true
rm -rf logs || true
rm -rf /dev/shm/vllm* 2>/dev/null || true
rm -rf /tmp/vllm* 2>/dev/null || true
rm -rf /tmp/huggingface* 2>/dev/null || true
rm -rf /tmp/torch* 2>/dev/null || true

# 5) Optional deep clean of install caches
if [ "$CLEAN_INSTALL" = "1" ]; then
  echo "[stop] Removing venv and caches from install step"
  rm -rf .venv || true
  rm -rf ~/.cache/huggingface ~/.local/share/huggingface || true
  [ -n "${HF_HOME:-}" ] && rm -rf "${HF_HOME}" || true
  [ -n "${HF_HUB_CACHE:-}" ] && rm -rf "${HF_HUB_CACHE}" || true
  rm -rf ~/.cache/torch ~/.cache/torch_extensions ~/.cache/pip ~/.cache/hf_transfer ~/.cache/vllm ~/.cache/triton ~/.nv || true
fi

# 6) Optional apt cache clean
if [ "$CLEAN_SYSTEM" = "1" ]; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "[stop] Cleaning apt caches"
    apt-get clean || true
    rm -rf /var/lib/apt/lists/* || true
  else
    echo "[stop] apt-get not available; skipping system cache clean"
  fi
fi

echo "[stop] Wipe complete."

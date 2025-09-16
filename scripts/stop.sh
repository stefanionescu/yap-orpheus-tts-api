#!/usr/bin/env bash
set -euo pipefail

# Flags:
#  --clean-install  Remove venv and Python/HF/torch caches created by 01-install.sh
#  --clean-system   Clean apt caches created by 00-bootstrap.sh (best-effort)

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
  rm -rf ~/.nv || true

  # Temp files that may accumulate
  rm -rf /tmp/vllm* 2>/dev/null || true
  rm -rf /tmp/huggingface* 2>/dev/null || true
  rm -rf /tmp/torch* 2>/dev/null || true
  rm -rf /dev/shm/vllm* 2>/dev/null || true
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



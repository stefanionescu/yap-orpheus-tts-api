#!/usr/bin/env bash
set -euo pipefail
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

read -p "This will remove venv and caches (~/.cache/huggingface, ~/.cache/torch, pip). Continue? [y/N] " yn
case "$yn" in
  [Yy]* ) ;;
  * ) echo "Aborted"; exit 0;;
esac

rm -rf .venv
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch
rm -rf ~/.cache/pip
echo "[stop] Wiped."



#!/usr/bin/env bash
set -euo pipefail
pkill -f "uvicorn app.server" || true
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



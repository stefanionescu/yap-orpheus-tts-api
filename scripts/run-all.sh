#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .env ]; then
  echo "[run-all] Missing .env. Copy .env.example to .env and set HF_TOKEN." >&2
  exit 1
fi

echo "[run-all] 1/3 bootstrap"
bash scripts/00-bootstrap.sh

echo "[run-all] 2/3 install"
bash scripts/01-install.sh

echo "[run-all] 3/3 start server"
if [ "${TRTLLM_ENABLE:-0}" = "1" ]; then
  exec bash scripts/02-run-server-trtllm.sh
else
  exec bash scripts/02-run-server.sh
fi



#!/usr/bin/env bash
set -euo pipefail

# HF_TOKEN must be set by the deployment environment (no .env required)
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[run-all] ERROR: HF_TOKEN not set. Export HF_TOKEN in the shell." >&2
  echo "           Example: export HF_TOKEN=\"hf_xxx\"" >&2
  exit 1
fi

echo "[run-all] 1/3 bootstrap"
bash scripts/00-bootstrap.sh

echo "[run-all] 2/3 install"
bash scripts/01-install.sh

echo "[run-all] 3/3 start server"
exec bash scripts/02-run-server.sh

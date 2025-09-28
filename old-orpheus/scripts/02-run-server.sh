#!/usr/bin/env bash
set -euo pipefail
# Common helpers and env
source "scripts/lib/common.sh"
load_env_if_present
# Defaults
: "${VENV_DIR:=$PWD/.venv}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"

[ -d "${VENV_DIR}" ] || { echo "venv missing. Run scripts/01-install.sh"; exit 1; }
source "${VENV_DIR}/bin/activate"

# Source modular env snippets
source_env_dir "scripts/env"

echo "[run] Starting FastAPI on ${HOST:-0.0.0.0}:${PORT:-8000}"
CMD=$(build_uvicorn_cmd)
start_background "$CMD" ".run/server.pid" "logs/server.log"


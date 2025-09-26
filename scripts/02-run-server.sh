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

# Print selected backend
echo "[run] Backend: ${ORPHEUS_BACKEND:-trtllm}"

# Optional: help dynamic loader find TRT/LLM libs if needed
if [ "${ORPHEUS_BACKEND:-trtllm}" != "vllm" ]; then
  # Match TF32 override used during engine build; the engine logs will warn if mismatched
  export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-1}
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(python - <<'PY'
import site,glob,os
paths = []
for sp in site.getsitepackages():
    paths += glob.glob(os.path.join(sp, 'tensorrt*'))
    paths += glob.glob(os.path.join(sp, 'nvidia/cuda_runtime*'))
if paths:
    print(os.path.dirname(paths[0]))
PY
)"
  # Common system lib dirs (libpython, openmpi)
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/lib/x86_64-linux-gnu/openmpi/lib:${LD_LIBRARY_PATH}"
fi

echo "[run] Starting FastAPI on ${HOST:-0.0.0.0}:${PORT:-8000}"
CMD=$(build_uvicorn_cmd)
start_server "$CMD" ".run/server.pid" "logs/server.log"


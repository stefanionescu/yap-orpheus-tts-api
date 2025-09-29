#!/usr/bin/env bash
set -euo pipefail

: "${VENV_DIR:=$PWD/.venv}"
: "${TRTLLM_WHEEL_URL:=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.20.0-cp310-cp310-linux_x86_64.whl}"

if [ "$(uname -s)" != "Linux" ]; then
  echo "[install-trt] TensorRT-LLM requires Linux with NVIDIA GPUs. Skipping."
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[install-trt] NVIDIA driver / nvidia-smi not detected. Ensure GPU drivers are installed."
fi

[ -d "${VENV_DIR}" ] || { echo "[install-trt] venv missing. Run scripts/01-install.sh first."; exit 1; }
source "${VENV_DIR}/bin/activate"

echo "[install-trt] Installing mpi4py (MPI Python bindings)"
pip install "mpi4py>=3.1"

echo "[install-trt] Checking Python shared library (libpython)"
if ! python - <<'PY'
import ctypes
import ctypes.util
import sys

version = f"{sys.version_info.major}.{sys.version_info.minor}"
lib_name = ctypes.util.find_library(f"python{version}")
if not lib_name:
    raise SystemExit(
        "Unable to locate libpython shared library. Install python3-dev (or python3.10-dev) "
        "and ensure LD_LIBRARY_PATH includes its directory."
    )

try:
    ctypes.CDLL(lib_name)
except OSError as exc:
    raise SystemExit(
        f"Found {lib_name} but failed to load it (LD_LIBRARY_PATH?): {exc}"
    )
PY
then
  echo "[install-trt] ERROR: libpython shared library missing or unloadable." >&2
  echo "[install-trt] Hint: apt-get install python3-dev python3.10-dev and re-run bootstrap." >&2
  exit 1
fi

echo "[install-trt] Verifying mpi4py can access MPI runtime"
if ! python - <<'PY'
import sys
try:
    from mpi4py import MPI  # noqa: WPS433
    MPI.Get_version()
except ImportError as exc:  # pragma: no cover - runtime dependency only
    sys.exit(f"mpi4py is not installed in the current virtualenv: {exc}")
except AttributeError as exc:  # pragma: no cover - runtime dependency only
    sys.exit(
        "mpi4py is installed but its MPI extension failed to load. "
        f"Ensure libmpi is available and reinstall mpi4py. Details: {exc}"
    )
except RuntimeError as exc:  # pragma: no cover - runtime dependency only
    sys.exit(
        "mpi4py could not bind to the MPI runtime. "
        f"Install OpenMPI (libopenmpi-dev openmpi-bin) and rerun this script. Details: {exc}"
    )
except Exception as exc:  # pragma: no cover - runtime dependency only
    sys.exit(f"Unexpected mpi4py/MPI error: {exc}")
else:
    sys.exit(0)
PY
then
  echo "[install-trt] ERROR: mpi4py is unavailable or MPI runtime missing." >&2
  echo "[install-trt] Hint: run scripts/00-bootstrap.sh or install OpenMPI (libopenmpi-dev openmpi-bin)." >&2
  exit 1
fi

echo "[install-trt] Installing base deps (ensures FastAPI==0.115.4)"
pip install -r requirements-base.txt

echo "[install-trt] Installing TensorRT-LLM wheel"
pip install --extra-index-url https://pypi.nvidia.com "${TRTLLM_WHEEL_URL}"

echo "[install-trt] Installing TRT extras"
pip install -r requirements-trt.txt

echo "[install-trt] Checking CUDA Python bindings (cuda-python)"
if ! python - <<'PY'
try:
    from cuda import cuda, cudart  # noqa: WPS433
except ImportError as exc:
    raise SystemExit(
        "cuda-python not installed or failed to import. Install cuda-python>=12.4 "
        "matching your driver."
    ) from exc
else:
    err, _ = cudart.cudaDriverGetVersion()
    if err not in (0,):
        raise SystemExit(
            "cuda-python imported but cudart driver query failed. Ensure the runtime "
            "can access CUDA libraries (LD_LIBRARY_PATH to libcuda.so/libcudart.so)."
        )
PY
then
  echo "[install-trt] ERROR: cuda-python bindings missing or unusable." >&2
  echo "[install-trt] Hint: pip install cuda-python>=12.4 and ensure CUDA libs are on LD_LIBRARY_PATH." >&2
  exit 1
fi

echo "[install-trt] Verifying env"
pip check || { echo "[install-trt] Dependency conflict detected"; exit 1; }

echo "[install-trt] Done."

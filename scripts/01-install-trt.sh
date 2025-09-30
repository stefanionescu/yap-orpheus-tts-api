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

NEED_MPI="${NEED_MPI:-0}"   # set NEED_MPI=1 only for multi-GPU builds
if [ "$NEED_MPI" = "1" ]; then
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
else
  echo "[install-trt] Skipping MPI check (NEED_MPI=0)"
fi

echo "[install-trt] Installing base deps (ensures FastAPI==0.115.4)"
pip install -r requirements-base.txt

echo "[install-trt] Installing TensorRT-LLM + libs"
pip install --upgrade --extra-index-url https://pypi.nvidia.com \
  "${TRTLLM_WHEEL_URL:-tensorrt-llm==0.20.0}" \
  "tensorrt-cu12-bindings" \
  "tensorrt-cu12-libs"

echo "[install-trt] Installing TRT extras"
pip install -r requirements-trt.txt

echo "[install-trt] Ensuring cuda-python is installed"
pip install --upgrade "cuda-python>=12.6,<13"

if command -v ldconfig >/dev/null 2>&1; then
  CUDA_LIB_DIR=$(ldconfig -p 2>/dev/null | awk '/libcuda\\.so/{print $NF; exit}' | xargs dirname 2>/dev/null || true)
  if [ -n "${CUDA_LIB_DIR:-}" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${CUDA_LIB_DIR}:"*) : ;;
      *) export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" ;;
    esac
  fi
fi

echo "[install-trt] Checking CUDA Python bindings (cuda-python)"
CUDA_CHECK_OUTPUT="$(
python - <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version


def ok(msg):
    print(msg)
    sys.exit(0)


def fail(msg):
    print(msg)
    sys.exit(1)


try:
    ver = version("cuda-python")
except PackageNotFoundError:
    fail("MISSING: cuda-python not installed")

major = int(ver.split(".", 1)[0])
try:
    if major >= 13:
        from cuda.bindings import runtime as cudart  # new path in 13.x
    else:
        from cuda import cudart  # legacy path in 12.x
except Exception as exc:  # pragma: no cover - runtime dependency only
    fail(f"IMPORT_ERROR: {type(exc).__name__}: {exc}")

err, _ = cudart.cudaDriverGetVersion()
if err != 0:
    fail(f"CUDART_ERROR: cudaDriverGetVersion -> {err}")
ok("OK")
PY
)" || true

if ! printf '%s' "$CUDA_CHECK_OUTPUT" | grep -q '^OK$'; then
  echo "[install-trt] ERROR: cuda-python not usable." >&2
  echo "[install-trt] Details:" >&2
  printf '%s\n' "$CUDA_CHECK_OUTPUT" >&2
  echo "[install-trt] Hint: ensure cuda-python>=12.6,<13 and that libcuda/libcudart are on LD_LIBRARY_PATH." >&2
  exit 1
fi

echo "[install-trt] Verifying env"
pip check || { echo "[install-trt] Dependency conflict detected"; exit 1; }

echo "[install-trt] Done."

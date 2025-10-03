# Yap Orpheus TTS API

Run Orpheus 3B TTS behind a FastAPI server using TensorRT-LLM backend only. Optimized for A100 GPU instances (Runpod/AWS) using a plain Python virtualenv.

- **Server**: `server/`
- **Scripts**: `scripts/`
- **Tests**: `tests/`

### Prerequisites

- NVIDIA GPU with CUDA 12.x drivers (A100 recommended)
- Ubuntu-based image with `nvidia-smi`
- OpenMPI runtime (`libopenmpi-dev` + `openmpi-bin`). Run `bash scripts/00-bootstrap.sh` on apt-based images to install.
- Python 3.10 with shared libraries (`apt-get install python3-dev python3.10-dev` on Ubuntu)
- CUDA Python bindings (`pip install cuda-python>=12.4`, matched to system CUDA)
- Hugging Face token (`HF_TOKEN`) with access to the model

### Quickstart

```bash
# 1) Set required token (deployment step)
export HF_TOKEN="hf_xxx"

# 2) Bootstrap → install → run (tails logs)
bash scripts/run-all.sh

# 3) Health check
curl -s http://127.0.0.1:8000/healthz
```

### Environment

- Required: `HF_TOKEN`
- Optional knobs (already surfaced via `scripts/env/`):
  - `FIRST_CHUNK_WORDS` (default 40), `NEXT_CHUNK_WORDS` (140), `MIN_TAIL_WORDS` (12)
  - `SNAC_TORCH_COMPILE` (0), `SNAC_MAX_BATCH` (64), `SNAC_BATCH_TIMEOUT_MS` (10)
- TRT-LLM: set `TRTLLM_ENGINE_DIR=/path/to/engine_dir`; see `scripts/env/trt.sh`
  - TRT-LLM cache (optional): `TRTLLM_CACHE_DIR` to control where HF snapshots are stored (defaults to `./.hf`)

### Use the TensorRT-LLM backend (Linux + NVIDIA)

Trivial path (one command runs install → engine build → server):

```bash
export HF_TOKEN="hf_xxx"
 # Optional: customize engine output directory (default: $PWD/models/orpheus-trt-int8sq-kvfp16)
 # export TRTLLM_ENGINE_DIR=/models/orpheus-trt

bash scripts/run-all.sh
```

Manual path (if you prefer explicit steps):

```bash
export HF_TOKEN="hf_xxx"

# 1) Base deps (creates .venv)
bash scripts/01-install.sh

# 2) Install TensorRT-LLM backend (requires working OpenMPI + mpi4py + cuda-python + libpython3.10.so)
bash scripts/01-install-trt.sh

# 3) Build engine directory (installs TRT-LLM if missing)
bash scripts/02-build-int8-sq-kvfloat16.sh

# 4) Run server
export TRTLLM_ENGINE_DIR=/models/orpheus-trt-int8sq-kvfp16
bash scripts/04-run-server.sh
```

### Recommended container base for TRT-LLM

- Linux with NVIDIA drivers and CUDA 12.x available (must have `nvidia-smi`).
- Known-good base images:
  - `nvidia/cuda:12.2.0-devel-ubuntu22.04` (builder + runtime)
  - `nvidia/cuda:12.4.1-runtime-ubuntu22.04` (runtime; install build tools as needed)
- Runpod: choose an Ubuntu 22.04 image exposing `nvidia-smi` with CUDA 12.x (A100 recommended).

### Concurrency up to 16

```bash
export HF_TOKEN="hf_xxx"
bash scripts/run-all.sh
```

Validate with 16 concurrent streams:

```bash
python tests/bench.py --n 16 --concurrency 16
```

### Stop and cleanup

```bash
# Stop server and remove .run/ and logs/
bash scripts/stop.sh

# Also remove venv and caches from install step
bash scripts/stop.sh --clean-install

# Also remove TensorRT-LLM artefacts (engine outputs and local HF cache from TRT builds)
bash scripts/stop.sh --clean-trt

# Also clean system apt caches from bootstrap step
bash scripts/stop.sh --clean-system

# Delete everything
bash scripts/stop.sh --clean-install --clean-trt --clean-system
```

### Tests (optional)

```bash
# Enter the virtual environment
source .venv/bin/activate

# Warmup (single WS stream)
python tests/warmup.py

# Benchmark (concurrent WS sessions)
python tests/bench.py
```

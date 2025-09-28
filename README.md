## Yap Orpheus TTS API

Run Orpheus 3B TTS behind a FastAPI server with either vLLM or TensorRT-LLM backends. Optimized for A100 GPU instances (Runpod/AWS) using a plain Python virtualenv.

- **Server**: `server/`
- **Scripts**: `scripts/`
- **Tests**: `tests/`

### Prerequisites

- NVIDIA GPU with CUDA 12.x drivers (A100 recommended)
- Ubuntu-based image with `nvidia-smi`
- Python 3.10
- Hugging Face token (`HF_TOKEN`) with access to the model

### Quickstart (vLLM backend, default)

```bash
# 1) Set required token (deployment step)
export HF_TOKEN="hf_xxx"

# 2) Bootstrap → install → run (tails logs)
bash scripts/run-all.sh

# 3) Health check
curl -s http://127.0.0.1:8000/healthz

# 4) One-off WS synthesis to WAV (24 kHz)
python - <<'PY'
import asyncio, json, websockets, wave
async def main():
    uri = "ws://127.0.0.1:8000/ws/tts"
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"text": "hello from orpheus", "voice": "female"}))
        with wave.open("out.wav", "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    break
                if isinstance(msg, (bytes, bytearray)):
                    wf.writeframes(msg)
asyncio.run(main())
PY
```

### Environment

- Required: `HF_TOKEN`
- Optional knobs (already surfaced via `scripts/env/`):
  - `FIRST_CHUNK_WORDS` (default 40), `NEXT_CHUNK_WORDS` (140), `MIN_TAIL_WORDS` (12)
  - `SNAC_TORCH_COMPILE` (0), `SNAC_MAX_BATCH` (64), `SNAC_BATCH_TIMEOUT_MS` (10)
  - vLLM: see `server/vllm_config.py` and `scripts/env/tts.sh`
  - TRT-LLM: set `BACKEND=trtllm` and `TRTLLM_ENGINE_DIR=/path/to/engine_dir`; see `scripts/env/trt.sh`

### Use the TensorRT-LLM backend (Linux + NVIDIA)

```bash
export HF_TOKEN="hf_xxx"
export BACKEND=trtllm

# 1) Install base deps
bash scripts/01-install.sh

# 2) Install TRT-LLM wheel and extras
bash scripts/01-install-trt.sh

# 3) Build or provide a prebuilt TRT engine directory
#    Option A (in-repo builder):
#      python server/build/build-trt-engine.py --model canopylabs/orpheus-3b-0.1-ft \
#             --output /models/orpheus-trt --dtype float16 --max_input_len 512 --max_output_len 1024
#    Option B (experiments): inference-experiments/simple_build.py
export TRTLLM_ENGINE_DIR=/models/orpheus-trt

# 4) Start server (tails logs)
bash scripts/02-run-server.sh
```

### Recommended container base for TRT-LLM

- Linux with NVIDIA drivers and CUDA 12.x available (must have `nvidia-smi`).
- Known-good base images:
  - `nvidia/cuda:12.2.0-devel-ubuntu22.04` (builder + runtime)
  - `nvidia/cuda:12.4.1-runtime-ubuntu22.04` (runtime; install build tools as needed)
- Runpod: choose an Ubuntu 22.04 image exposing `nvidia-smi` with CUDA 12.x (A100 recommended).

- Inspect current values:
```bash
bash scripts/print-env.sh
```

### Stop and cleanup

```bash
# Stop server and remove .run/ and logs/
bash scripts/stop.sh

# Also remove venv and caches from install step
bash scripts/stop.sh --clean-install

# Also clean system apt caches from bootstrap step
bash scripts/stop.sh --clean-system

# Delete everything
bash scripts/stop.sh --clean-install --clean-system
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

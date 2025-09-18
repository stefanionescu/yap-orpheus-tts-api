### Yap Orpheus TTS API

Run Orpheus 3B TTS behind a FastAPI server with vLLM continuous batching. Optimized for A100 GPU instances (Runpod/AWS) using a plain Python virtualenv.

- **Server**: `server/`
- **Scripts**: `scripts/`
- **Tests**: `tests/`

### Prerequisites

- NVIDIA GPU with CUDA 12.x drivers (A100 recommended)
- Ubuntu-based image with `nvidia-smi`
- Python 3.10
- Hugging Face token (`HF_TOKEN`) with access to the model

### Quickstart

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
  - vLLM: see `server/vllm_config.py` and `scripts/env/vllm.sh`
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
```

### Tests (optional)

```bash
# Warmup (single WS stream)
python tests/warmup.py

# Benchmark (concurrent WS sessions)
python tests/bench.py
```

### Project layout

```
server/
  server.py
  engine_vllm.py
  prompts.py
  vllm_config.py
  streaming.py
  core/
    __init__.py
    utils.py
    chunking.py
    custom_tokens.py
    snac_batcher.py
scripts/
  00-bootstrap.sh
  01-install.sh
  02-run-server.sh
  run-all.sh
  print-env.sh
  stop.sh
  env/
    perf.sh
    vllm.sh
    tts.sh
  lib/
    common.sh
tests/
  warmup.py
  bench.py
  client.py
```

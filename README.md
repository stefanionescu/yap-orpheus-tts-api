# Yap Orpheus TTS API

This repo provides a docker-free setup to run Orpheus 3B TTS behind a FastAPI server with vLLM continuous batching. It’s designed for Runpod/AWS-style GPU instances and a plain Python virtualenv.

- Server module lives in `server/`
- Scripts live in `scripts/`
- Load/latency tests live in `tests/`
- All scripts must be run with `bash` (not sh)

## Requirements

- NVIDIA GPU with CUDA 12.x drivers (A100 recommended)
- Ubuntu-based image with `nvidia-smi`
- A Hugging Face access token with access to the model
- Python 3.10 on the instance

## Quickstart (vLLM path)

1) Clone and enter the project directory
```bash
cd /Users/dr_stone/Documents/work/yap-orpheus-tts-api
```

2) Set required environment variable (deployment step)
```bash
export HF_TOKEN="hf_xxx"  # must be set in the shell/session
```

3) Bootstrap system packages and checks
```bash
bash scripts/00-bootstrap.sh
```

4) Create virtualenv and install Python deps (Torch, vLLM, deps)
```bash
bash scripts/01-install.sh
```

5) Start the server on port 8000 (vLLM backend) — auto-detached with live logs
```bash
bash scripts/02-run-server.sh   # starts in background and tails logs
# Press Ctrl-C to stop following logs; server keeps running (PID in .run/server.pid)
# Re-attach later:
tail -f logs/server.log

# Show last 200 lines of the server log (one-time)
tail -n 200 logs/server.log
```

Alternatively, you can run all three steps with one command:
```bash
bash scripts/run-all.sh
```

## Health check and basic call

- Health
```bash
curl -s http://127.0.0.1:8000/healthz
```

- WebSocket synthesis (server streams raw PCM16, 24 kHz). Mode A (Baseten best-performance): send a single JSON with the full text. The server will split into ~280-char chunks internally and stream audio sequentially.

  Example (Mode A):
```bash
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

## Voice and parameters

- Voices: `tara` (female), `zac` (male). Aliases supported: `female`→`tara`, `male`→`zac`.
- WebSocket protocol (Mode A): send a single JSON `{ "text": "...", "voice": "...", "max_tokens": 4096? }`. The server chunks internally.
-- Tuning envs (server-side):
  - `MAX_CHUNK_SIZE` (default 280): chunk size in characters before prompt formatting
  - `SNAC_TORCH_COMPILE` (default 0): compile SNAC modules (0 recommended)
  - vLLM knobs in `server/vllm_config.py` or env

## Running tests (warmup and benchmark)

Enter the virtualenv first:
```bash
source .venv/bin/activate
```

Warmup (defaults, WebSocket):
```bash
python tests/warmup.py
```

Warmup (with params):
```bash
python tests/warmup.py --server 127.0.0.1:8000 --voice female
python tests/warmup.py --server 127.0.0.1:8000 --voice male --seed 42
```

Benchmark (defaults, WebSocket):
```bash
python tests/bench.py
```

Benchmark (with params, concurrent WS streaming sessions):
```bash
# 32 requests, 16 concurrent, male (Zac)
python tests/bench.py --server 127.0.0.1:8000 --n 32 --concurrency 16 --voice male
# with extra generation params
python tests/bench.py --server 127.0.0.1:8000 --n 32 --concurrency 16 --voice female --seed 123
```

Defaults:
- Server: `127.0.0.1:8000`
- Text: "Oh my god Danny, you're so smart and handsome! You're gonna love talking to me once Stefan is done with the app. Can't wait to see you there sweetie!"
- Voice: `female` (Tara)

Notes:
- Tests do not write WAV files; they count streamed bytes to infer audio seconds and compute metrics (TTFB, RTF, xRT, throughput).
- `tests/bench.py` writes per-session metrics JSONL to `tests/results/bench_metrics.jsonl`.

## Client for remote testing

The `tests/client.py` script connects to a remote Orpheus server and saves audio as WAV files locally. Perfect for testing your RunPod deployment from your laptop.

**Setup local environment (one-time):**

1) Create a local Python virtualenv:
```bash
# On your laptop/local machine (not the server)
cd /path/to/yap-orpheus-tts-api
python3 -m venv .venv
source .venv/bin/activate
```

2) Install minimal client dependencies:
```bash
pip install --upgrade pip
pip install websockets python-dotenv
```

3) Create `.env` file with your RunPod details:
```bash
# .env (in project root)
RUNPOD_TCP_HOST=your-pod-id-12345.proxy.runpod.net
RUNPOD_TCP_PORT=8000
RUNPOD_API_KEY=your-runpod-api-key-if-needed
```

**Usage:**

Local testing (server running on same machine):
```bash
source .venv/bin/activate
python tests/client.py --server 127.0.0.1:8000 --voice female --text "Hello from my laptop"
```

Remote RunPod testing (reads from .env):
```bash
source .venv/bin/activate
python tests/client.py --voice tara --text "Testing from my laptop to RunPod"
```

Custom server:
```bash
python tests/client.py --server wss://your-domain.com:8000 --voice zac --max-tokens 4096
```

**Output:**
- WAV files saved to `audio/tts_<timestamp>.wav`
- Metrics printed: TTFB, wall time, audio duration, RTF, xRT
- Play the WAV files with any audio player to hear the results

## Stopping and cleanup

Stop the server and clean artifacts:
```bash
bash scripts/stop.sh                    # stop server, remove .run/ and logs/
# also remove venv and caches from install step:
bash scripts/stop.sh --clean-install
# also clean system apt caches from bootstrap step:
bash scripts/stop.sh --clean-system
# combine:
bash scripts/stop.sh --clean-install --clean-system
```
The script runs non-interactively.

## Performance tuning

- Concurrency: `VLLM_MAX_SEQS` controls continuous batching parallelism (e.g., 16–24 on A100 40GB)
- GPU util: `VLLM_GPU_UTIL` (e.g., 0.95). If OOM, drop to 0.92
- Context: `VLLM_MAX_MODEL_LEN` (default 8192; increase for longer texts if memory allows)
- Dtype: `VLLM_DTYPE=float16|bfloat16` (Ampere runs FP16/BF16 well)
- SNAC batching: `SNAC_MAX_BATCH` (default 64), `SNAC_BATCH_TIMEOUT_MS` (default 10). Greatly improves decoder throughput at load
- Prefix cache: enabled by default (`VLLM_PREFIX_CACHE=1`)
- Threads: set to 1 to avoid CPU thrash: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
- FlashAttention 2: If compatible wheels exist for your Torch/CUDA, install:
  ```bash
  pip install --no-build-isolation "flash-attn>=2.5.7"
  ```
  vLLM detects it automatically on restart.
- Eager mode: we disable torch.compile/triton JIT to avoid in-container builds and speed startup.

Install speed knobs:
- Skip apt packages: `SKIP_APT=1 bash scripts/00-bootstrap.sh`
- Skip model prefetch: `PREFETCH=0 bash scripts/01-install.sh`

## Troubleshooting

- Ensure `HF_TOKEN` env is set
- If Torch install fails, check that `nvidia-smi` reports a CUDA 12.x driver and re-run `bash scripts/01-install.sh`
- If vLLM issues arise, try setting `VLLM_VERSION_PIN=0.7.3` in `.env`

## Project layout

```
server/                # FastAPI + Orpheus (vLLM)
  server.py            # FastAPI app (port 8000). WS: /ws/tts (PCM16 frames)
  engine_vllm.py       # vLLM async engine holder
  prompts.py           # Prompt helpers and audio control wrappers
  vllm_config.py       # vLLM tuning knobs
  utils.py
scripts/               # All runnable scripts (bash)
  00-bootstrap.sh
  01-install.sh
  02-run-server.sh     # vLLM path (default)
  run-all.sh           # bootstrap → install → start
  print-env.sh
  stop.sh
tests/
  warmup.py            # single request warmup test
  bench.py             # concurrent load testing
  client.py            # remote client that saves WAV files
```

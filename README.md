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

5) Start the server on port 8000 (vLLM backend)
```bash
bash scripts/02-run-server.sh
# or start detached so Ctrl-C in console won't stop it:
DETACH=1 bash scripts/02-run-server.sh
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

- Synthesize (streaming; PCM16, 24 kHz)
```bash
curl -s -X POST http://127.0.0.1:8000/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello from orpheus","voice":"female","stream":true}' \
  -o /dev/null
```

The HTTP response streams raw PCM16 bytes. For human listening, use the Python tests below.

## Voice presets and parameters

- Voices: `female` → Tara, `male` → Zac (aliases: `tara`, `zac`)
- Tara defaults: `temperature=0.80`, `top_p=0.80`, `repetition_penalty=1.90`, `seed=42`
- Zac defaults:  `temperature=0.40`, `top_p=0.80`, `repetition_penalty=1.85`, `seed=42`
- Extra request params supported by `POST /tts`: `seed`, `temperature`, `top_p`, `repetition_penalty`, `chunk_chars`, `stream`
- Policy: `num_predict` is fixed at 49152 and cannot be overridden. `chunk_chars` default is 500.

## Running tests (warmup and benchmark)

Enter the virtualenv first:
```bash
source .venv/bin/activate
```

Warmup (defaults, no params):
```bash
python tests/warmup.py
```

Warmup (with params):
```bash
python tests/warmup.py --server 127.0.0.1:8000 --voice female
python tests/warmup.py --server 127.0.0.1:8000 --voice male --seed 42
```

Benchmark (defaults, no params):
```bash
python tests/bench.py
```

Benchmark (with params, concurrent streaming requests):
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
- chunking: `chunk_chars=500`

Notes:
- Tests do not write WAV files; they count streamed bytes to infer audio seconds and compute metrics (TTFB, RTF, xRT, throughput).
- `tests/bench.py` writes per-session metrics JSONL to `tests/results/bench_metrics.jsonl`.

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
- Memory packing: `VLLM_GPU_UTIL` (e.g., 0.92)
- Context: `VLLM_MAX_MODEL_LEN` (default 8192; increase for longer texts if memory allows)
- Dtype: `VLLM_DTYPE=float16|bfloat16` (Ampere runs FP16/BF16 well)
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
server/                 # FastAPI + Orpheus (vLLM)
  server.py            # FastAPI app (port 8000)
  tts_engine.py        # Orpheus wrapper + voice presets
  vllm_config.py       # vLLM tuning knobs
  text_chunker.py
  utils.py
scripts/                # All runnable scripts (bash)
  00-bootstrap.sh
  01-install.sh
  02-run-server.sh          # vLLM path (default)
  run-all.sh                # bootstrap → install → start
  print-env.sh
  stop.sh
tests/
  warmup.py
  bench.py
```

# Yap Orpheus TTS API

Run Orpheus 3B TTS behind a FastAPI server using TensorRT-LLM backend with INT4-AWQ quantization. Optimized for A100 GPUs to support **17-20 concurrent users** with minimal quality loss.

- **Server**: `server/`
- **Scripts**: `scripts/`
- **Tests**: `tests/`

## Key Features

- **INT4-AWQ weight quantization** + **INT8 KV cache** for 3x memory efficiency vs FP16
- **Optimized for streaming TTS**: 48-token input, 1024-token output
- **Low TTFB**: Sentence-by-sentence chunking with dynamic SNAC batching
- **High throughput**: 16-18 concurrent real-time users on single A100

## Prerequisites

- NVIDIA GPU with CUDA 12.x drivers (A100 recommended)
- Ubuntu-based image with `nvidia-smi`
- OpenMPI runtime (installed automatically by `00-bootstrap.sh`)
- Python 3.10 with shared libraries (installed automatically by `00-bootstrap.sh`)
- Hugging Face token (`HF_TOKEN`) with access to `canopylabs/orpheus-3b-0.1-ft`

### Quickstart

```bash
# 1) Set required token (deployment step)
export HF_TOKEN="hf_xxx"

# 2) Bootstrap → install → run (tails logs)
bash scripts/run-all.sh

# 3) Health check
curl -s http://127.0.0.1:8000/healthz
```

## Configuration

### Required
- `HF_TOKEN`: Hugging Face token for model access

### TRT-LLM Settings (`scripts/env/trt.sh`)
- `TRTLLM_ENGINE_DIR`: Path to built engine (default: `models/orpheus-trt-int4-awq`)
- `TRTLLM_MAX_INPUT_LEN`: 48 tokens (optimized for sentences)
- `TRTLLM_MAX_OUTPUT_LEN`: 1024 tokens
- `TRTLLM_MAX_BATCH_SIZE`: 20 concurrent users
- `KV_FREE_GPU_FRAC`: 0.92 (use 92% of free GPU memory for KV cache)

### TTS Settings (`scripts/env/tts.sh`)
- `SNAC_MAX_BATCH`: 64 (SNAC decoder batching)
- `SNAC_BATCH_TIMEOUT_MS`: 5ms

## Installation & Deployment

### One-Command Deploy (Recommended)

Runs bootstrap → install → build INT4-AWQ engine → start server:

```bash
export HF_TOKEN="hf_xxx"
bash scripts/run-all.sh
```

### Manual Steps

```bash
export HF_TOKEN="hf_xxx"

# 1) Bootstrap system dependencies (OpenMPI, Python dev libs)
bash scripts/00-bootstrap.sh

# 2) Install TensorRT-LLM + dependencies
bash scripts/01-install-trt.sh

# 3) Build INT4-AWQ + INT8 KV cache engine
bash scripts/02-build.sh

# 4) Run server
bash scripts/03-run-server.sh
```

### Start/Stop Server Manually (no rebuild)

Use this when you already have a built TensorRT-LLM engine and just want to restart the API server.

1) Stop any running server (no flags):

```bash
bash scripts/stop.sh
```

2) Ensure your Hugging Face token is exported for runtime:

```bash
export HF_TOKEN="hf_xxx"
```

3) Set the TensorRT-LLM engine directory (must contain rank0.engine):

```bash
# If you used the provided build scripts, this is typically:
export TRTLLM_ENGINE_DIR="$(pwd)/models/orpheus-trt-int4-awq"

# Optional: verify the engine exists
[ -f "$TRTLLM_ENGINE_DIR/rank0.engine" ] && echo "Engine OK" || echo "Missing rank0.engine"
```

4) Start the server:

```bash
bash scripts/03-run-server.sh
```

5) Health check:

```bash
curl -s http://127.0.0.1:8000/healthz
```

Note on HF token precedence:
- If you see a warning like:
  "Note: Environment variable `HF_TOKEN` is set and is the current active token independently from the token you've just configured."
  it just means the `HF_TOKEN` environment variable overrides any saved login. To switch tokens, update `HF_TOKEN` accordingly.

### Rebuild Engine Only

```bash
# Force rebuild with new settings
bash scripts/02-build.sh --force

# Custom settings
bash scripts/02-build.sh --max-batch-size 32 --force
```

## Benchmarking

```bash
# Activate venv
source .venv/bin/activate

# Warmup (single request)
python tests/warmup.py

# Benchmark concurrent streams
python tests/bench.py --n 8 --concurrency 8
```

## Performance Tuning

### High Concurrency RTF Optimization

If you experience RTF degradation at high concurrency (16-20+ users):

**Rebuild with INT8 KV Cache** (primary fix):
```bash
# INT8 KV cache is now enabled by default in the build script
bash scripts/02-build.sh --force
```

This reduces KV cache memory usage by 50%, allowing more concurrent requests.

**Optional Tuning**:
```bash
# Adjust CUDA concurrency (default is 2)
export CUDA_DEVICE_MAX_CONNECTIONS=4  # Allow more concurrent kernel launches

# Monitor KV cache utilization
export TLLM_LOG_LEVEL=DEBUG
# Look for "waiting for free blocks" in logs → increase KV_FREE_GPU_FRAC
```

## Recommended Container Images

- `nvidia/cuda:12.2.0-devel-ubuntu22.04` (build + runtime)
- `nvidia/cuda:12.4.1-runtime-ubuntu22.04` (runtime only)
- **Runpod**: Ubuntu 22.04 template with A100 + CUDA 12.x

## Cleanup

```bash
# Stop server only
bash scripts/stop.sh

# Stop + remove build artifacts (~10-50GB)
bash scripts/stop.sh --clean-trt

# Stop + remove venv and caches (~20-100GB)
bash scripts/stop.sh --clean-install

# Nuclear option: remove everything
bash scripts/stop.sh --clean-install --clean-trt --clean-system
```

## Architecture

### Quantization Strategy
- **Weights**: INT4-AWQ (4x compression, ~2% quality loss)
- **KV Cache**: INT8 (2x compression, ~0.5% quality loss, **critical for high concurrency**)
- **Activations**: FP16 (no quantization - preserves quality)

### Why This Works for TTS
TTS models generate discrete audio codes where activation precision is critical. Weight-only quantization (INT4-AWQ) compresses the model without degrading the forward pass quality, while **INT8 KV cache is essential for high-concurrency performance** - it doubles the number of concurrent users the GPU can handle by reducing KV cache memory from 16-bit to 8-bit.

**Avoid**: 
- Full quantization (W8A8 SmoothQuant) destroys TTS quality by quantizing activations
- FP8 quantization (A100 doesn't support FP8 instructions)

## Client Usage

WebSocket endpoint: `ws://host:8000/ws/tts`

```python
import websockets
import json

async with websockets.connect("ws://localhost:8000/ws/tts") as ws:
    await ws.send(json.dumps({"text": "Hello world", "voice": "tara"}))
    while True:
        msg = await ws.recv()
        if msg == b"__END__":
            break
        # msg is raw PCM audio bytes (24kHz, int16)
```

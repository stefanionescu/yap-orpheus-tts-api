# Yap Orpheus TTS API

Run Orpheus 3B TTS behind a FastAPI server using TensorRT-LLM backend with INT4-AWQ quantization. Optimized for A100 GPUs to support **16 concurrent users** with minimal to no quality loss.

- **Server**: `server/` - Clean, modular FastAPI application
- **Scripts**: `scripts/` - Organized setup, build, and runtime scripts  
- **Tests**: `tests/` - Benchmarking and validation tools

## Key Features

- **INT4-AWQ weight quantization** + **INT8 KV cache** for 3x memory efficiency vs FP16
- **Optimized for streaming TTS**: 48-token input, 1024-token output
- **Low TTFB**: Sentence-by-sentence chunking with dynamic SNAC batching
- **High throughput**: 16 concurrent real-time users on single A100

## Prerequisites

- NVIDIA GPU with CUDA 12.x drivers (A100 recommended)
- Ubuntu-based image with `nvidia-smi`
- OpenMPI runtime (installed automatically by bootstrap script)
- Python 3.10 with shared libraries (installed automatically by bootstrap script)
- Hugging Face token (`HF_TOKEN`) with access to `canopylabs/orpheus-3b-0.1-ft`

### Quickstart

#### Option 1: Docker (2-5 minutes)
```bash
# Pull and run pre-built image
docker pull your_username/orpheus-3b-tts:latest
docker run --gpus all -p 8000:8000 --name orpheus-tts your_username/orpheus-3b-tts:latest

# Health check
curl -s http://127.0.0.1:8000/healthz
```

#### Option 2: Scripts (45 minutes)
```bash
# 1) Set required token
export HF_TOKEN="hf_xxx"

# 2) Bootstrap → install → run (tails logs)
bash scripts/main.sh

# 3) Health check
curl -s http://127.0.0.1:8000/healthz
```

## Configuration

All configuration is centralized in `scripts/environment.sh` with comprehensive documentation.

### Required Environment Variables
- `HF_TOKEN`: Hugging Face token for model access
- `TRTLLM_ENGINE_DIR`: Path to built engine (set automatically by build scripts)

### Key Configuration Settings
- **Engine**: `TRTLLM_MAX_BATCH_SIZE=16` (concurrent users), `KV_FREE_GPU_FRAC=0.92` (GPU memory usage)
- **TTS**: `SNAC_MAX_BATCH=64` (audio decoder batching), `ORPHEUS_MAX_TOKENS=1024` (output length)
- **Server**: `HOST=0.0.0.0`, `PORT=8000`, `DEFAULT_VOICE=tara`
- **Performance**: CUDA, PyTorch, and threading optimizations

See `scripts/environment.sh` for all available options and detailed documentation.

## Installation & Deployment

### Docker Deployment (Recommended - 2-5 minutes)

For rapid deployment using pre-built image:

```bash
# Simple deployment
docker pull your_username/orpheus-3b-tts:latest
docker run --gpus all -p 8000:8000 --name orpheus-tts your_username/orpheus-3b-tts:latest
```

### Scripts Deployment (45 minutes)

Runs bootstrap → install → build INT4-AWQ engine → start server:

```bash
export HF_TOKEN="hf_xxx"
bash scripts/main.sh
```

### Building Docker Image

To build your own Docker image with pre-built TensorRT engine:

```bash
# Set credentials
export HF_TOKEN="hf_your_token"
export DOCKER_USERNAME="your_dockerhub_username"
export DOCKER_PASSWORD="your_dockerhub_password"

# Login to Docker Hub
echo $DOCKER_PASSWORD | docker login --username $DOCKER_USERNAME --password-stdin

# Build and push (takes ~45 minutes)
docker build \
  --build-arg HF_TOKEN="$HF_TOKEN" \
  --build-arg DOCKER_USERNAME="$DOCKER_USERNAME" \
  --build-arg DOCKER_PASSWORD="$DOCKER_PASSWORD" \
  -t $DOCKER_USERNAME/orpheus-3b-tts:latest \
  -f docker/Dockerfile .

docker push $DOCKER_USERNAME/orpheus-3b-tts:latest
```

### Manual Steps

```bash
export HF_TOKEN="hf_xxx"

# 1) Bootstrap system dependencies (OpenMPI, Python dev libs)
bash scripts/setup/bootstrap.sh
# or: bash scripts/00-bootstrap.sh  (compatibility wrapper)

# 2) Install Python environment and TensorRT-LLM
bash scripts/setup/install-dependencies.sh
# or: bash scripts/01-install-trt.sh  (compatibility wrapper)

# 3) Build INT4-AWQ + INT8 KV cache engine
bash scripts/build/build-engine.sh
# or: bash scripts/02-build.sh  (compatibility wrapper)

# 4) Start TTS server
bash scripts/runtime/start-server.sh
# or: bash scripts/03-run-server.sh  (compatibility wrapper)
```

### Script Organization

The scripts are now organized into logical directories:
- **`scripts/setup/`** - System bootstrap and dependency installation
- **`scripts/build/`** - TensorRT-LLM engine building
- **`scripts/runtime/`** - Server startup and management
- **`scripts/utils/`** - Cleanup and maintenance utilities
- **`scripts/lib/`** - Shared helper functions

Old numbered script names (`00-bootstrap.sh`, etc.) are maintained as compatibility wrappers.

### Start/Stop Server Manually (no rebuild)

Use this when you already have a built TensorRT-LLM engine and just want to restart the API server.

1) Stop any running server:

```bash
bash scripts/utils/cleanup.sh
# or: bash scripts/stop.sh  (compatibility wrapper)
```

2) Ensure your Hugging Face token is exported:

```bash
export HF_TOKEN="hf_xxx"
```

3) Set the TensorRT-LLM engine directory (if not using default):

```bash
# Default location (set automatically by build scripts):
export TRTLLM_ENGINE_DIR="$(pwd)/models/orpheus-trt-int4-awq"

# Verify the engine exists:
[ -f "$TRTLLM_ENGINE_DIR/rank0.engine" ] && echo "Engine OK" || echo "Missing rank0.engine"
```

4) Start the server:

```bash
bash scripts/runtime/start-server.sh
# or: bash scripts/03-run-server.sh  (compatibility wrapper)
```

5) Health check:

```bash
curl -s http://127.0.0.1:8000/healthz
```

### Checking Server Logs

After starting the server, logs are automatically tailed. If you need to check logs later:

```bash
# View current server logs (follows new output)
tail -f logs/server.log

# View all server logs from the beginning
cat logs/server.log

# View last 50 lines of server logs
tail -n 50 logs/server.log

# Search for errors in logs
grep -i error logs/server.log

# Check if server is running
ps aux | grep "uvicorn server.server:app"
```

**Note**: If you run `cleanup.sh` and then restart the server with `03-run-server.sh`, the logs directory is recreated and new logs start fresh. Previous logs are removed during cleanup.

Note on HF token precedence:
- If you see a warning like:
  "Note: Environment variable `HF_TOKEN` is set and is the current active token independently from the token you've just configured."
  it just means the `HF_TOKEN` environment variable overrides any saved login. To switch tokens, update `HF_TOKEN` accordingly.

### Rebuild Engine Only

```bash
# Force rebuild with new settings
bash scripts/build/build-engine.sh --force
# or: bash scripts/02-build.sh --force  (compatibility wrapper)

# Custom batch size
bash scripts/build/build-engine.sh --max-batch-size 32 --force
# or: bash scripts/02-build.sh --max-batch-size 32 --force
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
bash scripts/build/build-engine.sh --force
# or: bash scripts/02-build.sh --force  (compatibility wrapper)
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
bash scripts/utils/cleanup.sh
# or: bash scripts/stop.sh  (compatibility wrapper)

# Stop + remove build artifacts (~10-50GB)
bash scripts/utils/cleanup.sh --clean-trt
# or: bash scripts/stop.sh --clean-trt

# Stop + remove venv and caches (~20-100GB)  
bash scripts/utils/cleanup.sh --clean-install
# or: bash scripts/stop.sh --clean-install

# Nuclear option: remove everything
bash scripts/utils/cleanup.sh --clean-install --clean-trt --clean-system
# or: bash scripts/stop.sh --clean-install --clean-trt --clean-system

# Get help on cleanup options
bash scripts/utils/cleanup.sh --help
```

### Cleanup Options Explained

- **No flags**: Stop processes, clean runtime files only
- **`--clean-install`**: Remove Python venv, pip/torch/HF caches (~20-100GB)
- **`--clean-trt`**: Remove TensorRT engines, model files, build artifacts (~10-50GB)  
- **`--clean-system`**: Remove system package caches (apt, etc.)

**Warning**: `--clean-trt` removes the built TensorRT engine. You'll need to rebuild it with `scripts/build/build-engine.sh` before running the server again.

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

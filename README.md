# Yap Orpheus TTS API

Run Orpheus 3B TTS behind a FastAPI server using TensorRT-LLM backend with INT4-AWQ quantization. Optimized for A100 GPUs to support **16 concurrent users** with minimal to no quality loss.

- **Server**: `server/` - Clean, modular FastAPI application
- **Scripts**: `custom/` - Organized setup, build, and runtime scripts  
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
- 60–70 GB free disk space to run `custom/` (optimized model download, engine build, caches)

### Quickstart
```bash
# 1) Set required token
export HF_TOKEN="hf_xxx"

# 2) Bootstrap → install → run (tails logs)
bash custom/main.sh

# 3) Health check
curl -s http://127.0.0.1:8000/healthz
```

**Note:** `GPU_SM_ARCH` is only needed if you plan to push to HuggingFace (set `HF_PUSH_AFTER_BUILD=1`).

## Configuration

All configuration is centralized in `custom/environment.sh` with comprehensive documentation.

### Required Environment Variables
- `HF_TOKEN`: Hugging Face token for model access
- `TRTLLM_ENGINE_DIR`: Path to built engine (set automatically by build scripts)

### Optional Environment Variables
- `GPU_SM_ARCH`: GPU architecture - **only required for HuggingFace push** (A100: `sm80`, RTX 4090: `sm89`, H100: `sm90`)

### Key Configuration Settings
- **Engine**: `TRTLLM_MAX_BATCH_SIZE=16` (concurrent users), `KV_FREE_GPU_FRAC=0.92` (GPU memory usage)
- **TTS**: `SNAC_MAX_BATCH=64` (audio decoder batching), `ORPHEUS_MAX_TOKENS=1024` (output length)
- **Server**: `HOST=0.0.0.0`, `PORT=8000`, `DEFAULT_VOICE=female`
- **Performance**: CUDA, PyTorch, and threading optimizations
- **GPU**: `GPU_SM_ARCH=sm80` (only required for HuggingFace push)

See `custom/environment.sh` for all available options and detailed documentation.

## Installation & Deployment

### Scripts Deployment

Runs bootstrap → install → build INT4-AWQ engine → start server:

```bash
export HF_TOKEN="hf_xxx"
bash custom/main.sh
```

### Optional: Push artifacts to Hugging Face after build

You can optionally publish the converted/quantized TRT-LLM checkpoint and/or the built engine(s) to a Hugging Face model repo. **Requires GPU_SM_ARCH to be set** - the pipeline will abort if GPU architecture is not explicitly configured. Engines are not portable across GPU architectures and TRT/CUDA versions, so prefer pushing TRT-LLM checkpoints for broad reuse.

1) Set publishing variables (only if you want to push):

```bash
export HF_TOKEN="hf_xxx"                           # required for access and upload
export GPU_SM_ARCH="sm80"                          # required: A100: sm80, RTX4090: sm89, H100: sm90
export HF_PUSH_AFTER_BUILD=1                       # enable push step in pipeline
export HF_PUSH_REPO_ID="your-org/my-model-trtllm"  # target HF repo
export HF_PUSH_PRIVATE=0                           # 1=private, 0=public
export HF_PUSH_WHAT=both                           # engines | checkpoints | both

# Optional: label for engine subtree (auto-detected if omitted)
# e.g., sm80_trt-llm-1.0.0_cuda12.4
export HF_PUSH_ENGINE_LABEL=""
```

2) Run the normal pipeline; a push occurs right after build:

```bash
bash custom/main.sh
```

Artifacts layout pushed to HF:

```
tokenizer.json
tokenizer.model
trt-llm/
  checkpoints/
    awq_config.json
    rank0.safetensors
    rank1.safetensors
    ...
  engines/
    <engine_label>/
      rank0.engine
      rank1.engine
      build_command.sh
      build_metadata.json
```

LFS rules are included automatically for large files (`.engine`, `.plan`, `.safetensors`, `.bin`).

Practical guidance:
- Engines are great for your own homogeneous fleet; risky for general reuse.
- Prefer publishing TRT-LLM checkpoints (post-convert, pre-engine) for portability.
- If you publish engines, include the metadata we generate next to them to avoid “invalid engine” surprises.

### Manual Steps

```bash
export HF_TOKEN="hf_xxx"

# 1) Bootstrap system dependencies (OpenMPI, Python dev libs)
bash custom/00-bootstrap.sh

# 2) Install Python environment and TensorRT-LLM
bash custom/01-install-trt.sh

# 3) Build INT4-AWQ + INT8 KV cache engine
bash custom/02-build.sh

# 4) Start TTS server
bash custom/03-run-server.sh
```

### Script Organization

The scripts are now organized into logical directories:
- **`custom/setup/`** - System bootstrap and dependency installation
- **`custom/build/`** - TensorRT-LLM engine building
- **`custom/runtime/`** - Server startup and management
- **`custom/utils/`** - Cleanup and maintenance utilities
- **`custom/lib/`** - Shared helper functions

Old numbered script names (`00-bootstrap.sh`, etc.) are maintained as compatibility wrappers.

### Start/Stop Server Manually (no rebuild)

Use this when you already have a built TensorRT-LLM engine and just want to restart the API server.

1) Stop any running server:

```bash
bash custom/utils/cleanup.sh
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
bash custom/03-run-server.sh
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
bash custom/02-build.sh --force

# Custom batch size
bash custom/02-build.sh --max-batch-size 12 --force
```

## Testing and Benchmarking

### Testing

```bash
# Activate venv
source .venv/bin/activate

# Warmup (single request)
python tests/warmup.py

# Benchmark concurrent streams
python tests/bench.py --n 8 --concurrency 8

# Custom text and voice
python tests/warmup.py --voice female --text "Your custom text here"
```

### External Client Testing (from your laptop)

Run the streaming client against a remote/local server using a clean virtual environment:

```bash
# 1) Clone this repo (or ensure you're in the repo root)
git clone https://github.com/your_org/yap-orpheus-tts-api.git
cd yap-orpheus-tts-api

# 2) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install lightweight client dependencies only
python -m pip install --upgrade pip
pip install websockets python-dotenv

# 4) Run the client against your server (replace with your URL/host)
python tests/client.py --voice female

# Example for local machine
# python tests/client.py --server 127.0.0.1:8000 --voice male

# 5) When done
# deactivate
```

## Performance Tuning

### High Concurrency RTF Optimization

If you experience RTF degradation at high concurrency (16-20+ users):

**Rebuild with INT8 KV Cache** (primary fix):
```bash
# INT8 KV cache is now enabled by default in the build script
bash custom/02-build.sh --force
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

## Recommended GPU Environments

- **Local/Cloud VM**: Ubuntu 20.04/22.04/24.04 with NVIDIA drivers
- **Runpod**: Ubuntu 22.04 template with A100 + CUDA 12.x
- **Vast.ai**: Any Ubuntu template with A100/A6000 + CUDA 12.x

## Cleanup

```bash
# Stop server only
bash custom/utils/cleanup.sh

# Stop + remove build artifacts
bash custom/utils/cleanup.sh --clean-trt

# Stop + remove venv and caches  
bash custom/utils/cleanup.sh --clean-install

# Nuclear option: remove everything
bash custom/utils/cleanup.sh --clean-install --clean-trt --clean-system

# Get help on cleanup options
bash custom/utils/cleanup.sh --help
```

### Cleanup Options Explained

- **No flags**: Stop processes, clean runtime files only
- **`--clean-install`**: AGGRESSIVELY remove Python venv and ALL caches
- **`--clean-trt`**: Remove TensorRT engines, CUDA artifacts, force uninstall packages
- **`--clean-system`**: NUCLEAR system cleanup - removes ALL caches/logs/temp files
- **`--clean-models`**: Remove downloaded models, checkpoints, and engines
- **`--clean-all`**: NUCLEAR OPTION - removes EVERYTHING (all of the above)

**Warning**: `--clean-trt` removes the built TensorRT engine. `--clean-all` is extremely aggressive and will remove gigabytes of cached data. You'll need to rebuild everything from scratch.

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
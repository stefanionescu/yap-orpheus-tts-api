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

### Authentication
- `YAP_API_KEY` (optional): If set, the server requires `Authorization: Bearer <YAP_API_KEY>` on incoming requests. For Docker, pass with `-e YAP_API_KEY=...`.

### Key Configuration Settings
- **Engine**: `TRTLLM_MAX_BATCH_SIZE=16` (concurrent users), `KV_FREE_GPU_FRAC=0.92` (GPU memory usage)
- **TTS**: `SNAC_MAX_BATCH=64` (audio decoder batching), `ORPHEUS_MAX_TOKENS=1024` (output length)
- **Server**: `HOST=0.0.0.0`, `PORT=8000`, `DEFAULT_VOICE=female`
- **Performance**: CUDA, PyTorch, and threading optimizations
- **GPU**: `GPU_SM_ARCH=sm80` (only required for HuggingFace push)
- **Build metadata**: `DEFAULT_TRTLLM_VERSION=1.0.0` (fallback TensorRT-LLM version when the wheel metadata is unavailable during engine build)

See `custom/environment.sh` for all available options and detailed documentation.

## Installation & Deployment

### Docker (containerized) path

If you want to build and run everything inside a container, see `docker/README.md` for image build (with BuildKit secret for `HF_TOKEN`) and instructions to quantize/build the engine and start the server inside the image.

Quickstart (container):
```bash
# Build base image (CUDA 12.1, Python 3.10 venv, torch==2.4.1)
export HF_TOKEN="hf_xxx"   # optional at build time
DOCKER_BUILDKIT=1 bash docker/build.sh

# Push to Docker Hub (optional)
DOCKER_BUILDKIT=1 bash docker/build.sh --push

# Single-shot: quantize → build engine → start server (background)
docker run --gpus all --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -e YAP_API_KEY=your_secret_key \
  -e MODEL_ID=canopylabs/orpheus-3b-0.1-ft \
  -e TRTLLM_ENGINE_DIR=/opt/engines/orpheus-trt-int4-awq \
  -v /path/for/engines:/opt/engines \
  -p 8000:8000 \
  -it IMAGE:TAG run.sh
```

### Scripts Deployment

Runs bootstrap → install → build INT4-AWQ engine → start server:

```bash
export HF_TOKEN="hf_xxx"
bash custom/main.sh
```

### Deploy from a Hugging Face checkpoint or prebuilt engines (skip local quantization)

You can bypass local quantization by pulling artifacts from a Hugging Face model repo produced with our `server/hf/push_to_hf.py` tool or compatible layout. Set the variables below and run the same pipeline; the build step will skip unnecessary work.

```bash
# Required for HF access
export HF_TOKEN="hf_xxx"

# Point to the repo that contains either:
# - trt-llm/checkpoints/**                (portable; engine will be built locally)
# - trt-llm/engines/<engine_label>/**     (non-portable; may be used as-is if compatible)
export HF_DEPLOY_REPO_ID="your-org/my-model-trtllm"

# Optional selection and behavior
# auto|engines|checkpoints (default: auto)
export HF_DEPLOY_USE=auto
# If pulling engines and multiple labels exist, pick one (e.g., sm80_trt-llm-1.0.0_cuda12.4)
export HF_DEPLOY_ENGINE_LABEL=""
# If engines are downloaded and environment matches, skip local build (default: 1)
export HF_DEPLOY_SKIP_BUILD_IF_ENGINES=1
# Enforce GPU SM match when using engines (default: 1)
export HF_DEPLOY_STRICT_ENV_MATCH=1

# Run the normal pipeline; it will:
# - Prefer engines if compatible; otherwise fall back to checkpoints
# - If only checkpoints exist, skip quantization and just build the engine
bash custom/main.sh
```

Integrity checks ensure required files exist after download. When engines are used, a basic SM-arch compatibility check is performed using `build_metadata.json` (or folder label) vs your local GPU.

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

### Voices
The server accepts `voice` values `female` or `male` (mapped internally to `tara` and `zac`).

### Generation Parameters

The TTS API supports customizable generation parameters that can be set per-connection (via metadata message) or per-text request. If not specified, the server uses voice-specific defaults optimized for each voice.

#### Supported Parameters

- **`temperature`** (float, 0.3-0.9): Controls randomness in generation. Lower values produce more consistent output, higher values add more variation.
- **`top_p`** (float, 0.7-1.0): Nucleus sampling parameter. Controls the cumulative probability cutoff for token selection.
- **`repetition_penalty`** (float, 1.1-1.9): Penalizes repeated tokens. Higher values reduce repetition more aggressively.

#### Voice-Specific Defaults

If generation parameters are not specified, the server uses optimized defaults based on the selected voice:

- **Female (Tara)**: `temperature=0.45`, `top_p=0.95`, `repetition_penalty=1.25`  
- **Male (Zac)**: `temperature=0.45`, `top_p=0.95`, `repetition_penalty=1.15`

#### Usage Examples

**Set parameters for entire connection via metadata message:**
```json
{"voice": "female", "temperature": 0.4, "top_p": 0.9, "repetition_penalty": 1.3}
```

**Override parameters for specific text:**
```json
{"text": "Hello world!", "voice": "female", "temperature": 0.6}
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

**Note**: Running `cleanup.sh` without flags leaves `logs/` intact. Use `cleanup.sh --clean-all` if you want to wipe cached logs along with the rest of the workspace.

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

# Control trimming of leading silence (default: true)
# Disable trimming for warmup
python tests/warmup.py --trim-silence false
# Disable trimming for benchmark
python tests/bench.py --trim-silence false --n 8 --concurrency 8

# Custom generation parameters
python tests/warmup.py --voice female --temperature 0.4 --top-p 0.9 --repetition-penalty 1.3
python tests/bench.py --voice male --temperature 0.6 --n 8 --concurrency 4
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

# Client: disable trimming example
python tests/client.py --trim-silence false --voice male

# Custom generation parameters
python tests/client.py --voice female --temperature 0.4 --top-p 0.9 --repetition-penalty 1.3

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
# Stop server and leave dependencies/models untouched (default)
bash custom/utils/cleanup.sh

# Full reset: remove models, TensorRT artifacts, venv, caches, temp files
bash custom/utils/cleanup.sh --clean-all

# Show help
bash custom/utils/cleanup.sh --help
```

### Cleanup Options Explained

- **No flags**: Stop server processes, release GPU resources, clear `.run/`
- **`--clean-all`**: Remove models, TensorRT repo clones (`.trtllm-repo`, `TensorRT-LLM`), Python venv, `~/.local/lib/python*`, Hugging Face caches, and temp build files

**Warning**: `--clean-all` nukes every locally installed dependency (TensorRT engines, TRT repo, Hugging Face snapshots, pip site-packages, caches). Expect to re-run the full setup afterward.

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

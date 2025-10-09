### Orpheus TRT-LLM Base Image

⚠️ **IMPORTANT NOTICE**: This is a **large (~100GB), resource-intensive base image** intended for **research, experimentation, and quantization workflows**. It is **NOT optimized for production use** due to its size and comprehensive tooling stack.

**Use Cases:**
- Experimenting with different Orpheus quantization methods using TensorRT-LLM
- Research and development workflows requiring full TensorRT-LLM toolchain
- Testing various quantization parameters and engine configurations
- Prototyping new TTS quantization approaches

**Not Recommended For:**
- Production deployments (use lightweight runtime images instead)
- CI/CD pipelines requiring fast image pulls
- Resource-constrained environments

This image pre-installs everything done by `scripts/00-bootstrap.sh` and `scripts/01-install-trt.sh` so cloud jobs can skip slow dependency setup, but comes with the trade-off of significant storage and transfer overhead.

#### Contents

This comprehensive base image includes **all the logic and code needed for quantizing Orpheus using TensorRT-LLM**, making it suitable for experimentation with different quantization approaches:

- **Base**: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- **System deps**: git, wget, curl, jq, OpenMPI runtime/dev, Python 3.10 + dev/venv
- **Python env**: venv at `/opt/venv` and set as default `PATH`
- **PyTorch**: installed from CU121 index (torch==2.4.1)
- **App deps**: Complete `requirements.txt` dependencies for full functionality
- **TensorRT-LLM**: Full TensorRT-LLM installation via NVIDIA PyPI (wheel URL is build-arg)
- **TRT-LLM repo**: Complete repository cloned to `/opt/TensorRT-LLM` and pinned to installed wheel version
- **Complete codebase**: All server code (`/app/server/`) and test utilities (`/app/tests/`) for comprehensive testing and validation
- **HF model cache (optional)**: If `HF_TOKEN` build-arg is provided, the Orpheus model snapshot is pre-downloaded to `/opt/models/<basename>-hf` during build (token is not persisted).
- **Runtime scripts** (inside image at `/usr/local/bin`):
  - `01-quantize-and-build.sh`: performs INT4-AWQ quantization and builds TRT engine
  - `02-start-server.sh`: starts the FastAPI server
  - `run.sh`: orchestrates build → server (background)
- **Environment defaults**: `docker/base/scripts/environment.sh` is baked in and auto-sourced for every Bash shell via `BASH_ENV` and `/etc/profile.d`, so defaults like `YAP_API_KEY`, sampling params, and performance knobs are always present. Override with `-e VAR=...` at runtime.

**Image Characteristics:**
- **Size**: ~100GB (includes full CUDA runtime, PyTorch, TensorRT-LLM, complete codebase, and dependencies)
- **Performance**: Slower to pull/start compared to production-optimized images
- **Completeness**: Contains everything needed for Orpheus quantization experimentation

#### Build
```bash
cd /path/to/yap-orpheus-tts-api
export MODEL_ID=canopylabs/orpheus-3b-0.1-ft           # optional
# Optionally provide HF token as a BuildKit secret (not persisted):
export HF_TOKEN=hf_xxx
DOCKER_BUILDKIT=1 bash docker/base/build.sh            # uses --secret id=HF_TOKEN,env=HF_TOKEN when set

# or with overrides:
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
TRTLLM_WHEEL_URL=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl \
IMAGE_NAME=sionescu/orpheus-trtllm-base IMAGE_TAG=cu126-py310-trt1.0 \
bash docker/base/build.sh

# Push existing image (skip build script)
docker login                              # only needed once per registry
docker push ${IMAGE_NAME:-sionescu/orpheus-trtllm}:${IMAGE_TAG:-cu121-py310}

# Optional: build + push in one go  
DOCKER_BUILDKIT=1 bash docker/base/build.sh --push
```

Notes:
- `HF_TOKEN` is optional at build time. When set, it is passed as a BuildKit secret, used only during the relevant RUN steps, and never persisted in layers.
- Do not bake secrets into images pushed to registries; provide them at build via `--secret` or at runtime via `-e`.

#### Use in Cloud
Use this image as a base for your runtime image where you add model engines and start the server.
```Dockerfile
FROM sionescu/orpheus-trtllm-base:cu126-py310-trtllm1.0.0
WORKDIR /app
COPY . /app
# Supply runtime env like HF_TOKEN, TRTLLM_ENGINE_DIR, etc.
ENV HOST=0.0.0.0 PORT=8000
CMD ["bash", "-lc", "uvicorn server.server:app --host $HOST --port $PORT --timeout-keep-alive 75 --log-level info"]
```

#### Notes
- This base image intentionally mirrors bootstrap steps; additionally it pre-clones the TRT-LLM repo and caches the Orpheus model. Engine build (`scripts/02-build.sh`) and server run are left to downstream images or runtime.
 - Secrets like `HF_TOKEN` and `YAP_API_KEY` must be provided at runtime (e.g., `docker run -e ...`, Compose/Orchestrator secrets). They are not persisted by the image.

#### Runtime
Examples after pulling the image:
```bash
# 1) Single-shot pipeline: quantize → build → start server (background)
docker run --gpus all --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -e YAP_API_KEY=your_secret_key \
  -e MODEL_ID=canopylabs/orpheus-3b-0.1-ft \
  -e TRTLLM_ENGINE_DIR=/opt/engines/orpheus-trt-int4-awq \
  -v /path/for/engines:/opt/engines \
  -p 8000:8000 \
  -it IMAGE:TAG run.sh

# 2) Manual steps (optional):
# Quantize and build engine only
docker run --gpus all --rm \
  -e HF_TOKEN=$HF_TOKEN -e MODEL_ID=canopylabs/orpheus-3b-0.1-ft \
  -v /path/for/engines:/opt/engines \
  -it IMAGE:TAG 01-quantize-and-build.sh --engine-dir /opt/engines/orpheus-trt-int4-awq

# Start server only (assumes engine exists at /opt/engines/...)
docker run --gpus all --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -e YAP_API_KEY=your_secret_key \
  -e TRTLLM_ENGINE_DIR=/opt/engines/orpheus-trt-int4-awq \
  -p 8000:8000 -it IMAGE:TAG 02-start-server.sh
```

RunPod:
```bash
# Use IMAGE:TAG in RunPod template with GPU, and set Env Vars:
# - HF_TOKEN (required)
# - MODEL_ID (optional)
# - TRTLLM_ENGINE_DIR (e.g. /opt/engines/orpheus-trt-int4-awq)
# Add container command: run.sh
```
- Ensure the host runtime provides NVIDIA GPU access (`--gpus all` in Docker, or equivalent in your cloud).
#### Tests
Inside the container (after server is running):
```bash
# Warmup
python /app/tests/warmup.py --server 127.0.0.1:8000 --voice female

# Benchmark (adjust concurrency as needed)
python /app/tests/bench.py --n 8 --concurrency 8
```

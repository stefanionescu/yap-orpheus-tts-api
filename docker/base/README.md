### Orpheus TRT-LLM Base Image

⚠️ Large (~100GB) image for research/experimentation and quantization workflows. Not for production use. Use the fast image for runtime serving.

---

#### Purpose

- Explore/iterate on Orpheus quantization with TensorRT-LLM
- Build engines with different parameters
- Start the API server for validation after a build

---

#### What's Included

- CUDA 12.1 runtime, Python 3.10 venv
- PyTorch (CU121), complete requirements
- TensorRT-LLM wheel; TRT-LLM repo cloned and pinned in `/opt/TensorRT-LLM`
- App code `/app/server/` and `/app/tests/`
- Runtime scripts in `/usr/local/bin`:
  - `01-quantize-and-build.sh`: INT4-AWQ + engine build
  - `02-start-server.sh`: start FastAPI server
  - `run.sh`: orchestrate build → server
- Environment defaults auto-sourced from `docker/base/scripts/environment.sh`

---

#### Build

```bash
cd /path/to/yap-orpheus-tts-api
export HF_TOKEN=hf_xxx                   # optional at build time (as BuildKit secret)
DOCKER_BUILDKIT=1 bash docker/base/build.sh

# Overrides (optional)
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
TRTLLM_WHEEL_URL=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl \
IMAGE_NAME=sionescu/orpheus-trtllm-base IMAGE_TAG=cu126-py310-trt1.0 \
bash docker/base/build.sh
```

Notes:
- `HF_TOKEN` (when provided) is passed as a BuildKit secret and not persisted.
- Provide secrets at build via `--secret` or at runtime via `-e`.

---

#### Runtime Examples

Quantize → build → start server (background):
```bash
docker run --gpus all --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -e YAP_API_KEY=your_secret_key \
  -e MODEL_ID=canopylabs/orpheus-3b-0.1-ft \
  -e TRTLLM_ENGINE_DIR=/opt/engines/orpheus-trt-int4-awq \
  -v /path/for/engines:/opt/engines \
  -p 8000:8000 \
  -it IMAGE:TAG run.sh
```

Build engine only:
```bash
docker run --gpus all --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_ID=canopylabs/orpheus-3b-0.1-ft \
  -v /path/for/engines:/opt/engines \
  -it IMAGE:TAG 01-quantize-and-build.sh --engine-dir /opt/engines/orpheus-trt-int4-awq
```

Start server (engine already present):
```bash
docker run --gpus all --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -e YAP_API_KEY=your_secret_key \
  -e TRTLLM_ENGINE_DIR=/opt/engines/orpheus-trt-int4-awq \
  -p 8000:8000 -it IMAGE:TAG 02-start-server.sh
```

Ensure the host runtime provides NVIDIA GPU access (`--gpus all`).

---

#### Testing

Inside the container (after server is running):
```bash
python /app/tests/warmup.py --server 127.0.0.1:8000 --voice female
python /app/tests/bench.py --n 8 --concurrency 8
```

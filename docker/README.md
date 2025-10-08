### Orpheus TRT-LLM Base Image

This image pre-installs everything done by `scripts/00-bootstrap.sh` and `scripts/01-install-trt.sh` so cloud jobs can skip slow dependency setup.

#### Contents
- **Base**: `nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04`
- **System deps**: git, wget, curl, jq, OpenMPI runtime/dev, Python 3.10 + dev/venv
- **Python env**: venv at `/opt/venv` and set as default `PATH`
- **PyTorch**: installed from CU126 index (torch==2.4.1)
- **App deps**: `requirements.txt`
- **TensorRT-LLM**: via NVIDIA PyPI (wheel URL is build-arg)
- **TRT-LLM repo**: cloned to `/opt/TensorRT-LLM` and pinned to installed wheel version
- **HF model cache**: Orpheus model snapshot downloaded to `/opt/models/<basename>-hf`

#### Build
```bash
cd /path/to/yap-orpheus-tts-api
export HF_TOKEN=hf_xxx   # REQUIRED at build time
export MODEL_ID=canopylabs/orpheus-3b-0.1-ft  # optional override
bash docker/build-base.sh
# or with overrides:
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
TRTLLM_WHEEL_URL=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl \
IMAGE_NAME=yapai/orpheus-trtllm-base IMAGE_TAG=cu126-py310-trt1.0 \
HF_TOKEN=$HF_TOKEN \
bash docker/build-base.sh
```

`HF_TOKEN` is mandatory during build to validate access to models and mirror the install script behavior.

#### Use in Cloud
Use this image as a base for your runtime image where you add model engines and start the server.
```Dockerfile
FROM yapai/orpheus-trtllm-base:cu126-py310-trtllm1.0.0
WORKDIR /app
COPY . /app
# Supply runtime env like HF_TOKEN, TRTLLM_ENGINE_DIR, etc.
ENV HOST=0.0.0.0 PORT=8000
CMD ["bash", "-lc", "uvicorn server.server:app --host $HOST --port $PORT --timeout-keep-alive 75 --log-level info"]
```

#### Notes
- This base image intentionally mirrors bootstrap steps; additionally it pre-clones the TRT-LLM repo and caches the Orpheus model. Engine build (`scripts/02-build.sh`) and server run are left to downstream images or runtime.
- Ensure the host runtime provides NVIDIA GPU access (`--gpus all` in Docker, or equivalent in your cloud).



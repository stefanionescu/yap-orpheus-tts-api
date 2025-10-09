### Orpheus TTS Fast Image

This is a **lean, fast Docker image** optimized for **production deployments**. It contains only the essential dependencies and runtime environment - models and engines are loaded at runtime for maximum flexibility and deployment speed.

**Use Cases:**
- Production deployments with pre-built TensorRT engines
- CI/CD pipelines requiring fast image pulls and deployment
- Kubernetes/cloud deployments with dynamic model/engine mounting
- Scaling scenarios where you want to pull models/engines separately
- Multi-model serving where different containers use different models

**Key Advantages:**
- Fast deployment
- Flexible model management: Pull/mount models at runtime
- Cost effective: Faster pulls = lower cloud egress costs
- Production optimized: Only essential runtime dependencies included

---

#### Contents

This lean production image includes **only the essential dependencies** needed to run Orpheus TTS inference:

- **Base**: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- **System deps**: Essential tools (git, wget, curl, Python 3.10)
- **Python env**: Optimized venv at `/opt/venv` 
- **PyTorch**: CUDA 12.1 optimized (torch==2.4.1)
- **Core dependencies**: Complete `requirements.txt` including Hugging Face Hub
- **TensorRT-LLM**: Runtime wheel only (no repo, no build tools)
- **Server code**: Complete `/app/server/` and test utilities (`/app/tests/`)
- **Runtime scripts**:
  - `start-server.sh`: Production server startup with model auto-download
  - `environment.sh`: Production environment defaults

**What's NOT Included (loaded at runtime):**
- Pre-downloaded models (pulled at runtime via HF Hub)
- TensorRT-LLM repository (not needed for inference)
- Pre-built engines (mounted or built separately)
- Development/build tools (keeps image lean)

---

#### Build

```bash
cd /path/to/yap-orpheus-tts-api

# Basic build
bash docker/fast/build.sh

# Build with custom tags/registries
IMAGE_NAME=myregistry/orpheus-tts IMAGE_TAG=prod \
bash docker/fast/build.sh

# Build and push to registry
bash docker/fast/build.sh --push

# Custom PyTorch/TensorRT versions
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
TRTLLM_WHEEL_URL=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.1.0-cp310-cp310-linux_x86_64.whl \
bash docker/fast/build.sh
```

---

#### Production Usage

##### Option 1: Pre-built Engine (Recommended)
Mount a pre-built TensorRT engine for fastest startup:

```bash
# Assuming you have a pre-built engine directory
docker run --gpus all -d \
  --name orpheus-tts \
  -e HF_TOKEN=$HF_TOKEN \
  -e YAP_API_KEY=$YAP_API_KEY \
  -e TRTLLM_ENGINE_DIR=/opt/engines/orpheus-trt-int4 \
  -v /path/to/your/engine:/opt/engines/orpheus-trt-int4:ro \
  -p 8000:8000 \
  sionescu/orpheus-trtllm-fast:cu121-py310
```

##### Option 2: Runtime Model Download
Let the container download the model automatically:

```bash
docker run --gpus all -d \
  --name orpheus-tts \
  -e HF_TOKEN=$HF_TOKEN \
  -e YAP_API_KEY=$YAP_API_KEY \
  -e MODEL_ID=canopylabs/orpheus-3b-0.1-ft \
  -e TRTLLM_ENGINE_DIR=/opt/engines/orpheus-trt-int4 \
  -v /path/to/engines:/opt/engines \
  -v /path/to/models:/opt/models \
  -p 8000:8000 \
  sionescu/orpheus-trtllm-fast:cu121-py310
```

##### Option 3: Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orpheus-tts
spec:
  replicas: 3
  selector:
    matchLabels:
      app: orpheus-tts
  template:
    metadata:
      labels:
        app: orpheus-tts
    spec:
      containers:
      - name: orpheus-tts
        image: sionescu/orpheus-tts-fast:cu121-py310
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: orpheus-secrets
              key: hf-token
        - name: YAP_API_KEY
          valueFrom:
            secretKeyRef:
              name: orpheus-secrets
              key: yap-api-key
        - name: TRTLLM_ENGINE_DIR
          value: "/opt/engines/orpheus-trt-int4"
        volumeMounts:
        - name: engine-volume
          mountPath: /opt/engines
          readOnly: true
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
      volumes:
      - name: engine-volume
        persistentVolumeClaim:
          claimName: orpheus-engines-pvc
```

---

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *required* | Hugging Face token for model access |
| `YAP_API_KEY` | `"your_secret_key_here"` | API authentication key |
| `MODEL_ID` | `"canopylabs/orpheus-3b-0.1-ft"` | Hugging Face model identifier |
| `TRTLLM_ENGINE_DIR` | *required* | Path to TensorRT engine directory |
| `MODELS_DIR` | `"/opt/models"` | Directory for downloaded models |
| `ENGINES_DIR` | `"/opt/engines"` | Base directory for engines |
| `HOST` | `"0.0.0.0"` | Server bind address |
| `PORT` | `8000` | Server port |
| `MAX_OUTPUT_LEN` | `2048` | Maximum generation length |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `TOP_P` | `0.9` | Nucleus sampling parameter |

---

#### Development & Testing

Run tests inside the container:
```bash
# Health check
curl http://localhost:8000/health

# Warmup test
docker exec orpheus-tts python /app/tests/warmup.py --server 127.0.0.1:8000 --voice female

# Performance benchmark
docker exec orpheus-tts python /app/tests/bench.py --n 10 --concurrency 4
```

---

#### Docker Compose Example

```yaml
version: '3.8'
services:
  orpheus-tts:
    image: sionescu/orpheus-tts-fast:cu121-py310
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - YAP_API_KEY=${YAP_API_KEY}
      - TRTLLM_ENGINE_DIR=/opt/engines/orpheus-trt-int4
      - MODEL_ID=canopylabs/orpheus-3b-0.1-ft
    volumes:
      - ./engines:/opt/engines:ro
      - ./models:/opt/models
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

#### Comparison: Fast vs Base Images

| Aspect | Fast Image | Base Image |
|--------|------------|------------|
| **Size** | ~10-15GB | ~100GB |
| **Purpose** | Production | Research/Development |
| **Startup** | Fast (seconds) | Slow (minutes) |
| **Model Loading** | Runtime | Build-time (optional) |
| **Engine Building** | External | Included |
| **Use Case** | Production deployment | Experimentation/Quantization |
| **Cost** | Low (fast pulls) | High (large transfers) |

---

#### Building Your Own Engine

If you need to build a TensorRT engine, use the base image or a separate build container:

```bash
# Use base image to build engine, then copy to fast image
docker run --gpus all --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_ID=canopylabs/orpheus-3b-0.1-ft \
  -v $(pwd)/engines:/opt/engines \
  sionescu/orpheus-trtllm:cu121-py310 \
  01-quantize-and-build.sh --engine-dir /opt/engines/orpheus-trt-int4

# Then use the built engine with fast image
docker run --gpus all -d \
  -e HF_TOKEN=$HF_TOKEN \
  -e YAP_API_KEY=$YAP_API_KEY \
  -e TRTLLM_ENGINE_DIR=/opt/engines/orpheus-trt-int4 \
  -v $(pwd)/engines:/opt/engines:ro \
  -p 8000:8000 \
  sionescu/orpheus-trtllm-fast:cu121-py310
```

---

#### Health & Monitoring

The image includes a built-in health check endpoint:
- **Health**: `GET /health` - Returns service status
- **Logs**: Use `docker logs <container>` for debugging
- **Metrics**: Server logs include timing and performance metrics

For production monitoring, integrate with your observability stack (Prometheus, DataDog, etc.).

# Docker Deployment for Orpheus 3B TTS

This directory contains Docker configuration for building and deploying the Orpheus 3B TTS server with **pre-built INT4-AWQ quantized TensorRT engines**.

**What gets built:** Orpheus 3B model with INT4-AWQ weight-only quantization + INT8 KV cache (6GB → ~1.5GB, 4x compression)

## Quick Start (2-5 minutes)

### Build & Push on GPU Machine

Clone the repo and run the build script **directly on a GPU machine**:

```bash
# On the GPU machine (Ubuntu 20.04/22.04 or Debian 11/12)
git clone https://github.com/your_username/yap-orpheus-tts-api.git
cd yap-orpheus-tts-api

bash docker/scripts/build.sh \
  --docker-username YOUR_DOCKERHUB_USERNAME \
  --docker-password YOUR_DOCKERHUB_PASSWORD \
  --hf-token hf_xxx
```

What it does:
1. Installs Docker + NVIDIA Container Toolkit
2. Builds the image with **INT4-AWQ quantized TensorRT engine** (30-45 min)
   - Downloads Orpheus 3B model from HuggingFace
   - Quantizes weights to INT4-AWQ (4x compression)
   - Builds optimized TensorRT engine with INT8 KV cache
3. Pushes complete image to `docker.io/YOUR_DOCKERHUB_USERNAME/orpheus-3b-tts:latest`

Requirements:
- GPU machine with NVIDIA drivers installed
- Ubuntu 20.04/22.04 or Debian 11/12
- Root/sudo access

Troubleshooting:
- If `nvcr.io/nvidia/tensorrt:23.12-py3` requires login: `docker login nvcr.io -u '$oauthtoken' -p '<NGC_API_KEY>'`

### Deploy Pre-built Image

After building with `build.sh`, deploy the **pre-quantized image** anywhere with GPU:

```bash
# Set your Docker image name and API key
export DOCKER_IMAGE="your_username/orpheus-3b-tts:latest"
export YAP_API_KEY="your_api_key_here"

# Pull and run pre-built image
docker pull $DOCKER_IMAGE
docker run --gpus all -p 8000:8000 -e YAP_API_KEY="$YAP_API_KEY" --name orpheus-tts $DOCKER_IMAGE

# Check health (server starts after API key validation)
curl http://localhost:8000/healthz
```

### Runpod Deployment

#### Option 1: Automatic Startup (Recommended)
When using Runpod with automatic server startup:

1. **Create pod** using Docker image: `your_username/orpheus-3b-tts:latest`
2. **Server starts automatically** when pod boots
3. **Access immediately** via Runpod's public URL (check pod interface)
4. **Test**: `curl http://localhost:8000/healthz` (from pod terminal)

#### Option 2: Manual Control
For manual server control inside the pod:

1. **Create pod** with Docker image but override command:
   - Use custom Docker command: `bash` (prevents auto-start)
2. **Connect to pod** (SSH/Jupyter/Web terminal)
3. **Start server manually:**

```bash
# Easy startup (foreground)
bash /app/start-server.sh

# Or run in background
bash /app/start-server.sh --background

# Check if running
curl http://localhost:8000/healthz

# View logs
docker logs -f orpheus-tts

# Stop/restart container (server auto-restarts)
docker restart orpheus-tts
```

4. **Access via Runpod's public URL** (check pod interface for the URL)

## Testing and Benchmarking

The Docker image includes test scripts for validating and benchmarking the TTS server:

### Available Tests

```bash
# Navigate to app directory
cd /app

# Activate virtual environment
source .venv/bin/activate

# Quick warmup test (single request)
python tests/warmup.py --host localhost --port 8000

# Benchmark concurrent requests
python tests/bench.py --host localhost --port 8000 --n 4 --concurrency 4

# Custom text benchmark
python tests/bench.py --host localhost --port 8000 --text "Your custom text here" --n 2
```

### Test Options

**warmup.py** - Single request validation:
```bash
python tests/warmup.py [OPTIONS]
  --host HOST          Server host (default: localhost)
  --port PORT          Server port (default: 8000)  
  --voice VOICE        Voice to use (default: female)
  --text TEXT          Text to synthesize
```

**bench.py** - Concurrent load testing:
```bash
python tests/bench.py [OPTIONS]
  --host HOST          Server host (default: localhost)
  --port PORT          Server port (default: 8000)
  --n N                Number of requests per worker (default: 5)
  --concurrency N      Number of concurrent workers (default: 2)
  --voice VOICE        Voice to use (default: female)
  --text TEXT          Text to synthesize
```

### Example Testing Workflow

```bash
# 1. Start server in background
bash /app/start-server.sh --background

# 2. Wait for server to be ready
sleep 10

# 3. Quick validation
python tests/warmup.py

# 4. Benchmark performance
python tests/bench.py --n 3 --concurrency 2

# 5. View server logs
tail -f /tmp/tts-server.log
```

## Pre-built Image Usage

If you have access to the pre-built Docker image:

```bash
# Simple deployment
docker pull your_username/orpheus-3b-tts:latest
docker run --gpus all -p 8000:8000 -e YAP_API_KEY=your_key your_username/orpheus-3b-tts:latest
```

## Building Your Own Image

To build the Docker image with pre-built TensorRT engine (45 minutes):

### Prerequisites

You can skip manual prerequisites by using the automated remote build above. If you prefer to build locally, ensure:
1. GPU-enabled machine with NVIDIA drivers
2. Docker with GPU support (nvidia-container-toolkit)

### Manual Build Steps

The `build.sh` script handles everything automatically. For manual control, you can run the individual steps from `build.sh` separately if needed.

**What the build process does:**
- Installs all system dependencies and TensorRT-LLM
- Downloads Orpheus 3B model from HuggingFace (`canopylabs/orpheus-3b-0.1-ft`)
- **Quantizes model weights to INT4-AWQ** (reduces 6GB → ~1.5GB)
- **Builds TensorRT engine with INT8 KV cache** for memory efficiency
- Packages everything into a Docker image
- Pushes complete image to Docker Hub
- **Result:** 2-5 minute deployments anywhere with pre-quantized model

## Cloud Deployment

### Standard Cloud VM/Server

For cloud VMs where you have Docker control:

```bash
# Set your image name
export DOCKER_IMAGE="your_username/orpheus-3b-tts:latest"

# Pull and run
docker pull $DOCKER_IMAGE
docker run -d \
  --name orpheus-tts \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  --shm-size=2g \
  $DOCKER_IMAGE

# Check status
docker logs -f orpheus-tts
curl http://localhost:8000/healthz
```

### Runpod/Vast.ai/Similar Services

For services where you start a pod with a Docker image:

1. **Create pod/instance** with image: `your_username/orpheus-3b-tts:latest`
2. **Connect via SSH/Jupyter/Terminal**
3. **Start server with one command:**

```bash
# Simple startup (foreground - shows logs)
bash /app/start-server.sh

# Background startup (for production)
bash /app/start-server.sh --background

# Check status
curl http://localhost:8000/healthz

# View logs (background mode)
tail -f /tmp/tts-server.log

# Stop server
bash /app/stop-server.sh
```

4. **Access via the pod's public URL/IP** (check service interface)

## Configuration

### Environment Variables

**Build-time (required for building):**
- HF_TOKEN - HuggingFace token for model access
- DOCKER_USERNAME - Docker Hub username  
- DOCKER_PASSWORD - Docker Hub password

**Runtime (optional):**
- HOST_PORT - Host port mapping (default: 8000)

### Image Configuration

The Docker image includes:
- **Base:** NVIDIA TensorRT 23.12 container (includes CUDA, cuDNN, TensorRT)
- **Python:** 3.10 with optimized virtual environment
- **PyTorch:** CUDA 12.1 support
- **TensorRT-LLM:** Version 1.0.0
- **Pre-built Engine:** Orpheus 3B with INT4-AWQ weight quantization + INT8 KV cache
- **Model Size:** ~1.5GB (down from 6GB original)
- **Performance:** Optimized for 16 concurrent users on A100

## Performance Comparison

| Method | Setup Time | Use Case |
|--------|------------|----------|
| Docker (pre-built) | 2-5 minutes | Production, cloud deployment |
| Scripts (from scratch) | 45 minutes | Development, customization |

## Troubleshooting

### GPU Support Issues
```bash
# Check GPU availability
nvidia-smi

# Test Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi
```

### Container Issues
```bash
# View container logs (server logs)
docker logs -f orpheus-tts

# Check container status
docker ps -a

# Restart container (server auto-restarts)
docker restart orpheus-tts

# Stop container
docker stop orpheus-tts

# Remove container
docker rm orpheus-tts
```

### Image Issues
```bash
# Verify image contents
docker run --rm your_username/orpheus-3b-tts:latest ls -la /app/models/

# Check image size
docker image inspect your_username/orpheus-3b-tts:latest --format='{{.Size}}' | numfmt --to=iec
```

## Security Notes

- Container runs as non-root user (tts)
- No sensitive data in image layers
- Build-time secrets via build args (not stored in image)
- Health checks enabled for monitoring

## Optimization

The Docker image is optimized for:
- **Fast startup:** Pre-built TensorRT engine (no build time on deployment)
- **Memory efficiency:** INT4-AWQ weight quantization + INT8 KV cache
- **Model compression:** 6GB → ~1.5GB (4x smaller)
- **High concurrency:** 16 concurrent users on A100
- **Small image size:** Multi-stage build with cleanup

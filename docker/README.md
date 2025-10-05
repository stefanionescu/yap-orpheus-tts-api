# Docker Deployment for Orpheus 3B TTS

This directory contains Docker configuration for building and deploying the Orpheus 3B TTS server with pre-built TensorRT engines for rapid deployment.

## Quick Start (2-5 minutes)

### Standard Docker Deployment

For instant deployment on any GPU-enabled machine:

```bash
# Set your Docker image name
export DOCKER_IMAGE="your_username/orpheus-3b-tts:latest"

# Pull and run pre-built image (server starts automatically)
docker pull $DOCKER_IMAGE
docker run --gpus all -p 8000:8000 --name orpheus-tts $DOCKER_IMAGE

# Server starts automatically - check health
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
  --max-tokens N       Max tokens to generate
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
  --max-tokens N       Max tokens to generate
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
docker run --gpus all -p 8000:8000 your_username/orpheus-3b-tts:latest
```

## Building Your Own Image

To build the Docker image with pre-built TensorRT engine (45 minutes):

### Prerequisites

1. GPU-enabled machine with NVIDIA drivers
2. Docker with GPU support (nvidia-container-toolkit)

### Build Steps

```bash
# 1. Set required environment variables
export HF_TOKEN="hf_your_token_here"
export DOCKER_USERNAME="your_dockerhub_username"
export DOCKER_PASSWORD="your_dockerhub_password"

# 2. Login to Docker Hub
echo $DOCKER_PASSWORD | docker login --username $DOCKER_USERNAME --password-stdin

# 3. Build image
docker build \
  --build-arg HF_TOKEN="$HF_TOKEN" \
  --build-arg DOCKER_USERNAME="$DOCKER_USERNAME" \
  --build-arg DOCKER_PASSWORD="$DOCKER_PASSWORD" \
  -t $DOCKER_USERNAME/orpheus-3b-tts:latest \
  -f docker/Dockerfile .

# 4. Push to Docker Hub
docker push $DOCKER_USERNAME/orpheus-3b-tts:latest
```

This will:
- Build Docker image with all dependencies
- Install TensorRT-LLM and PyTorch
- Build optimized INT4-AWQ + INT8 KV cache engine
- Push complete image to Docker Hub
- Enable 2-5 minute deployments anywhere

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
- Base: NVIDIA TensorRT 23.12 container (includes CUDA, cuDNN, TensorRT)
- Python: 3.10 with optimized virtual environment
- PyTorch: CUDA 12.1 support
- TensorRT-LLM: Version 1.0.0 with INT4-AWQ + INT8 KV cache
- Pre-built Engine: Optimized for 16 concurrent users on A100

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
- Fast startup: Pre-built TensorRT engine
- Memory efficiency: INT4-AWQ weights + INT8 KV cache
- High concurrency: 16 concurrent users on A100
- Small size: Multi-stage build with cleanup

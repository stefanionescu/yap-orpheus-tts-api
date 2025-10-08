#!/usr/bin/env bash
# =============================================================================
# Main Docker Script - Complete GPU Setup, Build & Push
# =============================================================================
# Run this script DIRECTLY on a remote GPU machine to:
# 1. Install Docker + NVIDIA Container Toolkit
# 2. Clone the repo
# 3. Build the Docker image with TensorRT engine
# 4. Push to Docker Hub
#
# Usage (on remote GPU machine):
#   curl -fsSL https://raw.githubusercontent.com/YOUR_REPO/main/docker/scripts/build.sh | bash -s -- \
#     --docker-username YOUR_USERNAME \
#     --docker-password YOUR_PASSWORD \
#     [--hf-token HF_TOKEN] \
#     [--repo-url https://github.com/YOUR_REPO.git] \
#     [--image-name orpheus-3b-tts] \
#     [--image-tag latest]
#
# Or download and run:
#   wget https://raw.githubusercontent.com/YOUR_REPO/main/docker/scripts/build.sh
#   bash build.sh --docker-username USER --docker-password PASS
# =============================================================================

set -euo pipefail

echo "=== Orpheus 3B TTS - Complete Docker Setup ==="

# =============================================================================
# Configuration and Arguments
# =============================================================================

DOCKER_USERNAME=""
DOCKER_PASSWORD=""
HF_TOKEN=""
REPO_URL="https://github.com/Yap-With-AI/yap-orpheus-tts-api.git"
IMAGE_NAME="orpheus-3b-tts"
IMAGE_TAG="latest"
WORK_DIR="/tmp/yap-orpheus-build"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker-username)
      DOCKER_USERNAME="$2"; shift 2 ;;
    --docker-password)
      DOCKER_PASSWORD="$2"; shift 2 ;;
    --hf-token)
      HF_TOKEN="$2"; shift 2 ;;
    --repo-url)
      REPO_URL="$2"; shift 2 ;;
    --image-name)
      IMAGE_NAME="$2"; shift 2 ;;
    --image-tag)
      IMAGE_TAG="$2"; shift 2 ;;
    --work-dir)
      WORK_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --docker-username USER --docker-password PASS [--hf-token TOKEN] [--repo-url URL] [--image-name NAME] [--image-tag TAG]"
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$DOCKER_USERNAME" ] || [ -z "$DOCKER_PASSWORD" ]; then
  echo "ERROR: --docker-username and --docker-password are required" >&2
  exit 1
fi

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "[main] Docker Hub: $DOCKER_USERNAME"
echo "[main] Image: $FULL_IMAGE_NAME"
echo "[main] Repository: $REPO_URL"
echo "[main] Work directory: $WORK_DIR"

# =============================================================================
# Step 1: System Prerequisites
# =============================================================================

echo "[main] === Step 1/4: Installing system prerequisites ==="

# OS validation
if [ -f /etc/os-release ]; then
    . /etc/os-release
else
    echo "ERROR: /etc/os-release not found; unsupported OS" >&2
    exit 1
fi

case "${ID}-${VERSION_ID}" in
    ubuntu-20.04|ubuntu-22.04|debian-11|debian-12)
        echo "[main] Detected supported OS: ${PRETTY_NAME}"
        ;;
    *)
        echo "ERROR: Unsupported OS: ${ID} ${VERSION_ID}. Supported: Ubuntu 20.04/22.04, Debian 11/12." >&2
        exit 1
        ;;
esac

# Install base tools
echo "[main] Installing base prerequisites..."
apt-get update -y
apt-get install -y ca-certificates curl gnupg lsb-release software-properties-common git

# =============================================================================
# Step 2: Docker + NVIDIA Container Toolkit
# =============================================================================

echo "[main] === Step 2/4: Setting up Docker + GPU support ==="

# Install Docker if needed
if command -v docker >/dev/null 2>&1; then
    echo "[main] Docker already installed"
else
    echo "[main] Installing Docker Engine..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
fi

# Add user to docker group
if groups "$USER" | grep -q docker; then
    echo "[main] User already in docker group"
else
    echo "[main] Adding user $USER to docker group"
    usermod -aG docker "$USER"
fi

# Install NVIDIA Container Toolkit if needed
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "[main] NVIDIA Container Toolkit already installed"
else
    echo "[main] Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo ${ID}${VERSION_ID})
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    apt-get update -y
    apt-get install -y nvidia-container-toolkit
fi

# Configure Docker for NVIDIA
echo "[main] Configuring Docker runtime for NVIDIA..."
nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 || true
systemctl restart docker

# Validate GPU support
echo "[main] Validating GPU support..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. GPU drivers required." >&2
    exit 1
fi

echo "[main] GPU check:"
nvidia-smi

if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: Docker GPU runtime test failed." >&2
    exit 1
fi

echo "[main] ✓ Docker + GPU support ready"

# =============================================================================
# Step 3: Clone Repository and Build Image
# =============================================================================

echo "[main] === Step 3/4: Building Docker image ==="

# Clean and create work directory
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository
echo "[main] Cloning repository..."
git clone "$REPO_URL" .

# Docker Hub authentication
echo "[main] Authenticating with Docker Hub..."
echo "$DOCKER_PASSWORD" | docker login --username "$DOCKER_USERNAME" --password-stdin

if [ $? -ne 0 ]; then
    echo "ERROR: Docker Hub authentication failed" >&2
    exit 1
fi

# Build Docker image
echo "[main] Building Docker image (this takes 30-45 minutes)..."
docker build \
    --file docker/Dockerfile \
    --tag "$FULL_IMAGE_NAME" \
    --build-arg HF_TOKEN="$HF_TOKEN" \
    --build-arg DOCKER_USERNAME="$DOCKER_USERNAME" \
    --build-arg DOCKER_PASSWORD="$DOCKER_PASSWORD" \
    --build-arg IMAGE_NAME="$IMAGE_NAME" \
    --progress=plain \
    .

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed" >&2
    exit 1
fi

# Validate built image
echo "[main] Validating built image..."
if ! docker image inspect "$FULL_IMAGE_NAME" >/dev/null 2>&1; then
    echo "ERROR: Built image not found" >&2
    exit 1
fi

# Check TensorRT engine exists
docker run --rm "$FULL_IMAGE_NAME" ls -la /app/models/orpheus-trt-int4-awq/rank0.engine

if [ $? -ne 0 ]; then
    echo "ERROR: TensorRT engine not found in built image" >&2
    exit 1
fi

echo "[main] ✓ Image built and validated"

# =============================================================================
# Step 4: Push to Docker Hub
# =============================================================================

echo "[main] === Step 4/4: Pushing to Docker Hub ==="

docker push "$FULL_IMAGE_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: Docker push failed" >&2
    exit 1
fi

# Get image size for summary
IMAGE_SIZE=$(docker image inspect "$FULL_IMAGE_NAME" --format='{{.Size}}' | numfmt --to=iec)

echo "[main] ✓ Image pushed successfully"

# =============================================================================
# Success Summary
# =============================================================================

echo ""
echo "🚀 === Build Complete! ==="
echo "Image: $FULL_IMAGE_NAME"
echo "Size: $IMAGE_SIZE"
echo "Registry: Docker Hub"
echo "Status: ✓ Ready for deployment"
echo ""
echo "🔥 Deploy anywhere with GPU:"
echo "  export YAP_API_KEY=your_key"
echo "  docker pull $FULL_IMAGE_NAME"
echo "  docker run --gpus all -p 8000:8000 -e YAP_API_KEY=\$YAP_API_KEY $FULL_IMAGE_NAME"
echo ""
echo "🎉 Success! Your TTS image is ready for global deployment."

# Cleanup
echo "[main] Cleaning up work directory..."
cd /
rm -rf "$WORK_DIR"

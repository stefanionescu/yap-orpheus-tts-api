#!/usr/bin/env bash
# =============================================================================
# Example Usage Script for Docker Deployment
# =============================================================================
# This script demonstrates the complete workflow from building to deployment.
# Use this as a reference for your own deployment pipeline.
# =============================================================================

set -euo pipefail

echo "=== Orpheus 3B TTS Docker Example ==="

# =============================================================================
# Step 1: Set Required Environment Variables
# =============================================================================

echo "Step 1: Setting up environment variables..."

# These need to be set before building
export HF_TOKEN="hf_your_token_here"                    # Replace with your HF token
export DOCKER_USERNAME="your_dockerhub_username"        # Replace with your Docker Hub username  
export DOCKER_PASSWORD="your_dockerhub_password"        # Replace with your Docker Hub password

# Optional customization
export IMAGE_NAME="orpheus-3b-tts"                      # Custom image name
export IMAGE_TAG="latest"                               # Custom tag

echo "✓ Environment configured"

# =============================================================================
# Step 2: Build and Push Image (One-time, 45 minutes)
# =============================================================================

echo ""
echo "Step 2: Building Docker image with pre-built TensorRT engine..."
echo "This step takes a while but only needs to be done once."

read -p "Do you want to build and push the image? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash docker/scripts/build-and-push.sh
    echo "✓ Image built and pushed to Docker Hub"
else
    echo "Skipping build step..."
fi

# =============================================================================
# Step 3: Deploy on Any GPU Machine (2-5 minutes)
# =============================================================================

echo ""
echo "Step 3: Deploying pre-built image..."
echo "This step takes ~2-5 minutes and can be done on any GPU machine."

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

read -p "Do you want to deploy the image locally? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash docker/scripts/deploy.sh "$FULL_IMAGE_NAME"
    echo "✓ Server deployed and running"
else
    echo "Skipping deployment..."
fi

# =============================================================================
# Step 4: Test the Deployment
# =============================================================================

echo ""
echo "Step 4: Testing the deployment..."

if curl -f -s http://localhost:8000/healthz >/dev/null 2>&1; then
    echo "✓ Server is healthy and responding"
    echo ""
    echo "🎉 Success! Your TTS server is ready."
    echo ""
    echo "Endpoints:"
    echo "  Health: http://localhost:8000/healthz"
    echo "  WebSocket: ws://localhost:8000/ws/tts"
    echo ""
    echo "Quick test:"
    echo "  curl http://localhost:8000/healthz"
else
    echo "❌ Server health check failed"
    echo "Check logs with: docker logs -f orpheus-tts-server"
fi

# =============================================================================
# Usage Summary
# =============================================================================

echo ""
echo "=== Usage Summary ==="
echo ""
echo "🔨 Build once (45 min):"
echo "  bash docker/scripts/build-and-push.sh"
echo ""
echo "🚀 Deploy anywhere (2-5 min):"
echo "  bash docker/scripts/deploy.sh $FULL_IMAGE_NAME"
echo ""
echo "⚡ Ultra-fast cloud deployment:"
echo "  bash <(curl -s https://raw.githubusercontent.com/.../docker/scripts/quick-start.sh)"
echo ""
echo "📋 Management:"
echo "  docker logs -f orpheus-tts-server    # View logs"
echo "  docker stop orpheus-tts-server       # Stop server"
echo "  docker restart orpheus-tts-server    # Restart server"

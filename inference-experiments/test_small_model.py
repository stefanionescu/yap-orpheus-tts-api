#!/usr/bin/env python3
"""
Test script with a smaller model to verify the build process works
"""

import modal
import os
import subprocess
import sys

# Define persistent volume
volume = modal.Volume.from_name("tensorrt-workspace", create_if_missing=True)

# Build the Modal image with required system and Python deps
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "wget", "curl", "build-essential", "cmake", "ninja-build", "pkg-config",
        "libopenmpi-dev", "openmpi-bin", "openmpi-common", "libopenmpi3",
        "software-properties-common", "ca-certificates"
    )
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "transformers", "datasets", "accelerate",
        "numpy", "packaging", "wheel", "setuptools", "ninja",
        "mpi4py", "huggingface_hub", "tqdm"
    )
    .run_commands(
        "pip install https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.20.0-cp310-cp310-linux_x86_64.whl"
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_VISIBLE_DEVICES": "0",
        "TENSORRT_LLM_LOG_LEVEL": "INFO",
        "HF_TOKEN": "",
        "HF_TRANSFER": "1"
    })
    .add_local_python_source("simple_build", copy=True)
)

# Define the Modal app
app = modal.App("test-small-model", image=image)

@app.function(
    gpu="A10G",
    memory=16384,
    timeout=3600,
    volumes={"/workspace": volume},
)
def test_small_model():
    """Test with a smaller model to verify the process works"""
    
    # Use a much smaller model for testing
    model_dir = "microsoft/DialoGPT-small"  # ~200MB model
    output_dir = "/workspace/models/test-small-model"
    
    print(f"🧪 Testing with smaller model: {model_dir}")
    
    cmd = f"python -m simple_build --model_dir {model_dir} --output_dir {output_dir} --dtype float16 --max_batch_size 1 --max_input_len 512 --max_output_len 512"
    print(f"🔨 Executing: {cmd}")
    
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f"Build failed with code {ret}")
        
    print(f"✅ Test build completed successfully")
    
    # List what was created
    if os.path.exists(output_dir):
        print(f"📁 Contents of {output_dir}:")
        for item in os.listdir(output_dir):
            print(f"  - {item}")
    
    return output_dir

@app.local_entrypoint()
def main():
    """Test with small model"""
    print("🚀 Testing with small model...")
    result = test_small_model.remote()
    print(f"✅ Test completed: {result}")
import modal
import os

def download_tensorrt_deps():
    """Download and verify TensorRT-LLM dependencies"""
    import subprocess
    import sys
    
    # Verify MPI installation
    try:
        import mpi4py
        print(f"MPI4PY version: {mpi4py.__version__}")
    except ImportError:
        print("MPI4PY not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mpi4py"])
    
    # Test TensorRT-LLM import
    try:
        import tensorrt_llm
        print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")
    except ImportError:
        print("TensorRT-LLM import failed - will be installed via wheel")

def download_test_model():
    """Download a small test model for TensorRT-LLM testing"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "microsoft/DialoGPT-small"  # Small model for testing
    print(f"Downloading test model: {model_name}")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)
    print("Test model downloaded successfully")

# Create Modal image with TensorRT-LLM
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10"  # Use 3.10 for stability
    )
    .apt_install(
        "git",
        "wget", 
        "curl",
        "build-essential",
        "cmake",
        "ninja-build",
        "pkg-config",
        # MPI dependencies for TensorRT-LLM
        "libopenmpi-dev",
        "openmpi-bin", 
        "openmpi-common",
        "libopenmpi3",
        # Additional system deps
        "software-properties-common",
        "ca-certificates"
    )
    .pip_install(
        "torch",
        "torchvision", 
        "torchaudio",
        "transformers==4.44.2",
        "datasets",
        "accelerate",
        "numpy",
        "packaging",
        "wheel",
        "setuptools",
        "ninja",
        "mpi4py",  # Install MPI before TensorRT-LLM
        "jupyter",
        "jupyterlab",
        "ipywidgets",
        "matplotlib",
        "seaborn",
        "pandas",
        "plotly",
        "tqdm"
    )
    .run_commands(
        # Install TensorRT-LLM stable version
        "pip install https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.20.0-cp310-cp310-linux_x86_64.whl"
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_VISIBLE_DEVICES": "0",
        "TENSORRT_LLM_LOG_LEVEL": "INFO"
    })
    .run_function(download_tensorrt_deps)
    .run_function(download_test_model)
)

app = modal.App("tensorrt-llm-jupyter", image=image)
volume = modal.Volume.from_name("tensorrt-workspace", create_if_missing=True)

JUPYTER_TOKEN = "tensorrt_secure_token_123"  # Change this to a secure token

@app.function(
    gpu="A10G",  # A100 for TensorRT-LLM performance
    timeout=86400,  # 24 hour timeout
    memory=32768,  # 32GB RAM
    volumes={"/workspace": volume},
    concurrency_limit=1,
    allow_concurrent_inputs=100
)
def run_jupyter(timeout: int = 28800):  # 8 hour default timeout
    import subprocess
    import time
    import json
    import os
    
    jupyter_port = 8888
    
    # Create sample TensorRT-LLM notebook
    os.makedirs("/workspace", exist_ok=True)
    
    sample_notebook = "/workspace/tensorrt_llm_example.ipynb"
    if not os.path.exists(sample_notebook):
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# TensorRT-LLM Example\n",
                        "\n",
                        "This notebook demonstrates TensorRT-LLM usage with Modal"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import tensorrt_llm\n",
                        "import torch\n",
                        "print(f\"TensorRT-LLM version: {tensorrt_llm.__version__}\")\n",
                        "print(f\"PyTorch version: {torch.__version__}\")\n",
                        "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f\"GPU: {torch.cuda.get_device_name()}\")\n",
                        "    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")"
                    ]
                },
                {
                    "cell_type": "code", 
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Test MPI installation\n",
                        "import mpi4py\n",
                        "from mpi4py import MPI\n",
                        "print(f\"MPI4PY version: {mpi4py.__version__}\")\n",
                        "print(f\"MPI rank: {MPI.COMM_WORLD.Get_rank()}\")\n",
                        "print(f\"MPI size: {MPI.COMM_WORLD.Get_size()}\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None, 
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Your TensorRT-LLM model code here\n",
                        "# Example: Load and optimize a model with TensorRT-LLM\n",
                        "print(\"Ready for TensorRT-LLM model development!\")"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python", 
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open(sample_notebook, 'w') as f:
            json.dump(notebook_content, f, indent=2)
    
    with modal.forward(jupyter_port) as tunnel:
        process = subprocess.Popen([
            "jupyter", "lab",
            "--no-browser",
            "--allow-root", 
            "--ip=0.0.0.0",
            f"--port={jupyter_port}",
            "--NotebookApp.allow_origin='*'",
            "--NotebookApp.allow_remote_access=1",
            "--notebook-dir=/workspace",
            f"--NotebookApp.token={JUPYTER_TOKEN}"
        ], env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN})
        
        print(f"\n🚀 TensorRT-LLM Jupyter Lab is available at: {tunnel.url}")
        print(f"🔑 Token: {JUPYTER_TOKEN}")
        print(f"📁 Workspace: /workspace")
        print(f"⏰ Timeout: {timeout/3600:.1f} hours")
        print(f"🖥️  GPU: A100-40GB")
        print(f"💾 Memory: 32GB")
        
        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                if process.poll() is not None:
                    break
                time.sleep(1)
        finally:
            process.terminate()
            process.wait()

@app.function(gpu="A10G", memory=32768)
def get_gpu_info():
    """Get detailed GPU and system information"""
    import subprocess
    import torch
    
    print("=" * 50)
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 50)
    
    # Run nvidia-smi
    subprocess.run(["nvidia-smi"])
    
    print("\n" + "=" * 50)
    print("🔥 PYTORCH GPU INFORMATION")
    print("=" * 50)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA version: {torch.version.cuda}")
    
    print("\n" + "=" * 50)
    print("🚀 TENSORRT-LLM INFORMATION") 
    print("=" * 50)
    try:
        import tensorrt_llm
        print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")
        print("✅ TensorRT-LLM imported successfully")
    except Exception as e:
        print(f"❌ TensorRT-LLM import failed: {e}")
    
    print("\n" + "=" * 50)
    print("🔗 MPI INFORMATION")
    print("=" * 50)
    try:
        import mpi4py
        from mpi4py import MPI
        print(f"MPI4PY version: {mpi4py.__version__}")
        print(f"MPI rank: {MPI.COMM_WORLD.Get_rank()}")
        print(f"MPI size: {MPI.COMM_WORLD.Get_size()}")
        print("✅ MPI configured successfully")
    except Exception as e:
        print(f"❌ MPI test failed: {e}")

@app.function(gpu="A10G", memory=32768)
def test_tensorrt_llm():
    """Test TensorRT-LLM basic functionality"""
    import tensorrt_llm
    print(f"🎯 Testing TensorRT-LLM {tensorrt_llm.__version__}")
    
    # Add your TensorRT-LLM test code here
    # Example: Basic model compilation test
    
    return "TensorRT-LLM test completed successfully!"

@app.function(gpu="A100-40GB", memory=32768)
def run_llm_inference(prompt: str = "whaaaarr"):
    """Run inference with TensorRT-LLM optimized model"""
    import tensorrt_llm
    print(f"🔥 Running inference with prompt: '{prompt}'")
    
    # Your TensorRT-LLM inference code here
    # This is a placeholder - replace with actual model loading and inference
    
    result = f"TensorRT-LLM processed: {prompt}"
    return result

# CLI entrypoints
@app.local_entrypoint()
def main():
    """Test the TensorRT-LLM setup"""
    print("🚀 Testing TensorRT-LLM setup...")
    get_gpu_info.remote()
    test_result = test_tensorrt_llm.remote()
    print(test_result)

@app.local_entrypoint() 
def jupyter():
    """Start Jupyter Lab with TensorRT-LLM"""
    print("🚀 Starting TensorRT-LLM Jupyter Lab...")
    print("📝 This will start a Jupyter Lab instance with TensorRT-LLM pre-installed")
    print("🔗 Access URL will be displayed once the server starts")
    print("💾 Files will be saved in persistent volume")
    run_jupyter.remote()

@app.local_entrypoint()
def info():
    """Get system and GPU information"""
    get_gpu_info.remote()

if __name__ == "__main__":
    # Default: show system info and test TensorRT-LLM
    info()
    main()

import modal
import os
import subprocess
import sys

# Define persistent volume
volume = modal.Volume.from_name("tensorrt-workspace", create_if_missing=True)

# Function to download and verify dependencies
def download_tensorrt_deps():
    """Download and verify TensorRT-LLM dependencies"""
    try:
        import mpi4py
        print(f"MPI4PY version: {mpi4py.__version__}")
    except ImportError:
        print("MPI4PY not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mpi4py"])

    try:
        import tensorrt_llm
        print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")
    except ImportError:
        print("TensorRT-LLM import failed - will be installed via wheel")

# Build the Modal image with required system and Python deps
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        # core tools
        "git", "wget", "curl", "build-essential", "cmake", "ninja-build", "pkg-config",
        # MPI
        "libopenmpi-dev", "openmpi-bin", "openmpi-common", "libopenmpi3",
        # extras
        "software-properties-common", "ca-certificates"
    )
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "transformers", "datasets", "accelerate",
        "numpy", "packaging", "wheel", "setuptools", "ninja",
        "mpi4py", "jupyter", "jupyterlab", "ipywidgets",
        "matplotlib", "seaborn", "pandas", "plotly", "tqdm",
        "snac"  # For decoder.py
    )
    .run_commands(
        # Install TensorRT-LLM
        "pip install https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.20.0-cp310-cp310-linux_x86_64.whl"
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_VISIBLE_DEVICES": "0",
        "TENSORRT_LLM_LOG_LEVEL": "INFO",
        "HF_TOKEN": ""
    })
    .run_function(download_tensorrt_deps)
    .add_local_python_source("decoder", copy=True) 
    .add_local_python_source("build_engine", copy=True) # Add decoder.py
)

# Define the Modal app
app = modal.App("tensorrt-llm-jupyter", image=image)

@app.function(
    gpu="A10G",
    memory=32768,
    timeout=86400,  # 24h
    volumes={"/workspace": volume},
    max_containers=1,
)
def build_engine(
    model_dir: str = "canopylabs/orpheus-tts-0.1-finetune-prod",
    output_dir: str = "/workspace/models/orpheus-tts-0.1-finetune-prod",
    dtype: str = "float16",  # Changed from bfloat16 to float16 for better A10 compatibility
    max_batch_size: int = 1,
    max_input_len: int = 1024,
    max_output_len: int = 1024,
    max_beam_width: int = 1,
    use_gpt_attention_plugin: bool = True,
    use_gemm_plugin: bool = True,
   use_layernorm_plugin: bool = False,
    use_rmsnorm_plugin: bool = False,
    enable_context_fmha: bool = False,
    enable_remove_input_padding: bool = False,
    enable_paged_kv_cache: bool = False,
    tokens_per_block: int = 64,
    max_num_tokens: int = None,
    int8_kv_cache: bool = False,
    fp8_kv_cache: bool = False,
    use_weight_only: bool = False,
    weight_only_precision: str = "bfloat16",  # Changed from int8 to bfloat16 for better A10 compatibility
    world_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
):
    """Build and save the TensorRT-LLM engine into persistent volume"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command with all parameters
    cmd_parts = [
        f"python -m build_engine",
        f"--model_dir {model_dir}",
        f"--output_dir {output_dir}",
        f"--dtype {dtype}",
        f"--max_batch_size {max_batch_size}",
        f"--max_input_len {max_input_len}",
        f"--max_output_len {max_output_len}",
        f"--max_beam_width {max_beam_width}",
        f"--tokens_per_block {tokens_per_block}",
        f"--world_size {world_size}",
        f"--tp_size {tp_size}",
        f"--pp_size {pp_size}",
        f"--weight_only_precision {weight_only_precision}",
    ]
    
    # Add optional max_num_tokens if specified
    if max_num_tokens is not None:
        cmd_parts.append(f"--max_num_tokens {max_num_tokens}")
    
    # Add boolean flags based on their values
    if use_gpt_attention_plugin:
        cmd_parts.append("--use_gpt_attention_plugin")
    else:
        cmd_parts.append("--no_gpt_attention_plugin")
        
    if use_gemm_plugin:
        cmd_parts.append("--use_gemm_plugin")
    else:
        cmd_parts.append("--no_gemm_plugin")
        
    if use_layernorm_plugin:
        cmd_parts.append("--use_layernorm_plugin")
    else:
        cmd_parts.append("--no_layernorm_plugin")
        
    if use_rmsnorm_plugin:
        cmd_parts.append("--use_rmsnorm_plugin")
    else:
        cmd_parts.append("--no_rmsnorm_plugin")
        
    if enable_context_fmha:
        cmd_parts.append("--enable_context_fmha")
    else:
        cmd_parts.append("--no_context_fmha")
        
    if enable_remove_input_padding:
        cmd_parts.append("--enable_remove_input_padding")
    else:
        cmd_parts.append("--no_remove_input_padding")
        
    if enable_paged_kv_cache:
        cmd_parts.append("--enable_paged_kv_cache")
    else:
        cmd_parts.append("--no_paged_kv_cache")
        
    if use_weight_only:
        cmd_parts.append("--use_weight_only")
        
    if int8_kv_cache:
        cmd_parts.append("--int8_kv_cache")
        
    if fp8_kv_cache:
        cmd_parts.append("--fp8_kv_cache")
    
    cmd = " ".join(cmd_parts)
    print(f"🔨 Executing: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f"Engine build failed with code {ret}")
    engine_path = os.path.join(output_dir, "model.plan")
    if not os.path.isfile(engine_path):
        raise FileNotFoundError(f"Expected engine file not found: {engine_path}")
    print(f"✅ Engine saved to: {engine_path}")
    return engine_path

@app.function(
    gpu="A10G",
    memory=32768,
    volumes={"/workspace": volume},
    timeout=3600,  # 1 hour
)
def run_llm_inference(
    engine_path: str = "/workspace/models/orpheus-tts-0.1-finetune-prod/model.plan",
    prompt: str = "Hello, TensorRT!",
    output_dir: str = "/workspace/models/orpheus-tts-0.1-finetune-prod/output"
):
    """Load the TensorRT engine, run inference, and decode tokens to audio."""
    import tensorrt_llm
    from decoder import tokens_decoder_sync  # Import from decoder.py

    os.makedirs(output_dir, exist_ok=True)

    print(f"🔋 Loading engine from: {engine_path}")
    trt_llm = tensorrt_llm.TRTLMEngine(engine_path)

    print(f"💬 Inference prompt: {prompt}")
    token_gen = trt_llm.generate([prompt], max_tokens=128)

    audio_chunks = []
    for chunk in tokens_decoder_sync(token_gen):
        audio_chunks.append(chunk)
    print(f"🎵 Generated {len(audio_chunks)} audio chunks")

    audio_file = os.path.join(output_dir, "generated_audio.wav")
    with open(audio_file, "wb") as f:
        for chunk in audio_chunks:
            f.write(chunk)
    print(f"💾 Audio saved to: {audio_file}")

    return audio_file

# Local entrypoints
@app.local_entrypoint()
def main():
        """Test build and inference locally"""
        print("🚀 Start engine build...")
        engine_path = build_engine.remote()
        print(f"🗂 Built engine at: {engine_path}")

        print("🚀 Start inference test...")
        audio_file = run_llm_inference.remote(engine_path, "Test prompt for TRT-LLM")
        print(f"📝 Audio output saved to: {audio_file}")

        # To copy the audio file to local machine, use:
        # modal volume get tensorrt-workspace /workspace/models/orpheus-tts-0.1-finetune-prod/output/generated_audio.wav ./local_audio.wav

@app.local_entrypoint()
def jupyter():
    """Start Jupyter Lab"""
    print("🚀 Starting Jupyter Lab...")
    run_jupyter = modal.Function.lookup(app, "run_jupyter")
    run_jupyter.remote()
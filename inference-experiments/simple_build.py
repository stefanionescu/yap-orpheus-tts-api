"""
Simple TensorRT-LLM Engine Builder for Orpheus Model
Minimal configuration approach for TensorRT-LLM 0.20.0
"""

import os
import argparse
from pathlib import Path
import tensorrt_llm
from tensorrt_llm.llmapi import LLM
from tensorrt_llm.llmapi import BuildConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import time

def download_model_with_progress(model_dir: str, cache_dir: str = None):
    """Download HuggingFace model with progress tracking"""
    print(f"[1/3] Downloading HuggingFace model: {model_dir}")
    
    try:
        # Enable hf_transfer for faster downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        # Try to import hf_transfer to verify it's available
        try:
            import hf_transfer
            print(f"  - hf_transfer available: {hf_transfer.__version__}")
        except ImportError:
            print("  - hf_transfer not available, using standard download")
        
        # First try to load tokenizer to verify model exists
        print("  - Checking model accessibility...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
        print(f"  - Model accessible, vocab size: {tokenizer.vocab_size}")
        
        # Download model files with progress updates
        print("  - Downloading model files (this may take 5-15 minutes for 3B model)...")
        start_time = time.time()
        
        # Use a thread to show periodic progress updates
        import threading
        
        def progress_monitor():
            while not download_complete:
                elapsed = time.time() - start_time
                print(f"    ... still downloading ({elapsed:.0f}s elapsed)")
                time.sleep(30)  # Update every 30 seconds
        
        download_complete = False
        progress_thread = threading.Thread(target=progress_monitor)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # Enable hf_transfer for faster downloads
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            
            local_model_path = snapshot_download(
                repo_id=model_dir,
                cache_dir=cache_dir,
                local_files_only=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Keep safetensors but skip unnecessary files
                resume_download=True,  # Resume if interrupted
                max_workers=4,  # Parallel downloads
            )
        finally:
            download_complete = True
            progress_thread.join(timeout=1)
        
        elapsed = time.time() - start_time
        print(f"  - Download completed in {elapsed:.1f}s")
        print(f"  - Model cached at: {local_model_path}")
        
        return local_model_path
        
    except Exception as e:
        print(f"  - Download failed: {e}")
        raise

def build_simple_engine(
    model_dir: str,
    output_dir: str,
    dtype: str = 'float16',
    max_batch_size: int = 1,
    max_input_len: int = 1024,
    max_output_len: int = 1024,
):
    """
    Build TensorRT-LLM engine using the simplified LLM API
    This approach uses TensorRT-LLM's high-level API which handles most configuration automatically
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {model_dir}")
    print(f"Building engine with dtype={dtype}, max_batch_size={max_batch_size}")
    
    # Create build configuration with minimal settings
    build_config = BuildConfig()
    build_config.max_batch_size = max_batch_size
    build_config.max_input_len = max_input_len
    build_config.max_seq_len = max_input_len + max_output_len
    
    # Set precision
    if dtype == 'float16':
        build_config.precision = 'float16'
    elif dtype == 'bfloat16':
        build_config.precision = 'bfloat16' 
    else:
        build_config.precision = 'bfloat16'  # Default to bfloat16 for better precision
    
    # Disable complex optimizations that may cause issues
    build_config.use_paged_kv_cache = False  # Disable for simplicity
    build_config.remove_input_padding = False  # Disable for simplicity
    
    try:
        # Use persistent volume for caching models
        cache_dir = "/workspace/hf_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if model is already downloaded
        model_cache_path = os.path.join(cache_dir, "models--" + model_dir.replace("/", "--"))
        if os.path.exists(model_cache_path):
            print(f"✅ Found cached model at {model_cache_path}")
            # Find the actual model directory
            snapshot_dirs = [d for d in os.listdir(model_cache_path) if d.startswith("snapshots")]
            if snapshot_dirs:
                snapshot_dir = os.path.join(model_cache_path, snapshot_dirs[0])
                refs_dir = os.path.join(snapshot_dir, os.listdir(snapshot_dir)[0])
                local_model_path = refs_dir
                print(f"  - Using cached model from {local_model_path}")
            else:
                local_model_path = download_model_with_progress(model_dir, cache_dir)
        else:
            local_model_path = download_model_with_progress(model_dir, cache_dir)
        
        print("[2/3] Creating TensorRT-LLM engine...")
        print("  - Initializing LLM with build config...")
        
        # Use the high-level LLM API which handles engine building automatically
        llm = LLM(
            model=local_model_path,
            build_config=build_config,
        )
        print("  - Model loaded successfully")
        
        print("[3/3] Saving engine...")
        # Save the engine to the output directory
        llm.save(output_dir)
        
        print(f"✅ Engine successfully built and saved to {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"❌ Engine build failed: {e}")
        print("\nTrying even simpler configuration...")
        
        # Fallback: Ultra-minimal configuration
        try:
            print("[1/3] Attempting fallback with minimal configuration...")
            llm = LLM(model=model_dir)  # Use all defaults
            llm.save(output_dir)
            print(f"✅ Engine built with defaults and saved to {output_dir}")
            return output_dir
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            raise e2


def build_legacy_engine(
    model_dir: str,
    output_dir: str,
    dtype: str = 'float16',
    max_batch_size: int = 1,
    max_input_len: int = 1024,
    max_output_len: int = 1024,
):
    """
    Fallback: Build engine using the older convert_checkpoint approach
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using legacy checkpoint conversion for {model_dir}")
    
    # Import required modules for legacy approach
    from tensorrt_llm.models.llama.model import LLaMAForCausalLM
    from tensorrt_llm.quantization import QuantMode
    from tensorrt_llm.builder import Builder
    
    # Load HuggingFace config
    from transformers import AutoConfig, AutoModelForCausalLM
    hf_config = AutoConfig.from_pretrained(model_dir)
    
    # Create minimal TensorRT-LLM config
    config = {
        'architecture': 'LLaMAForCausalLM',
        'dtype': dtype,
        'num_hidden_layers': getattr(hf_config, 'num_hidden_layers', 32),
        'num_attention_heads': getattr(hf_config, 'num_attention_heads', 32),
        'hidden_size': getattr(hf_config, 'hidden_size', 4096),
        'vocab_size': getattr(hf_config, 'vocab_size', 32000),
        'max_position_embeddings': getattr(hf_config, 'max_position_embeddings', 2048),
        'max_batch_size': max_batch_size,
        'max_input_len': max_input_len,
        'max_output_len': max_output_len,
        'quantization': {
            'quant_mode': QuantMode(0),  # No quantization
        },
        'mapping': {
            'world_size': 1,
            'tp_size': 1,
            'pp_size': 1,
        }
    }
    
    print("Building TensorRT engine with legacy method...")
    
    # This is a simplified version - you would need to implement the full weight conversion
    # For now, we'll just create the directory structure
    checkpoint_dir = output_dir / 'checkpoint'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save config
    import json
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Checkpoint structure created at {checkpoint_dir}")
    print("⚠️  Note: Full weight conversion not implemented in this simplified version")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple TensorRT-LLM engine builder')
    
    # Required arguments
    parser.add_argument('--model_dir', type=str, required=True, help='Path to HuggingFace model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for TRT engine')
    
    # Simple configuration
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], 
                       help='Model precision (float16 recommended for A10)')
    parser.add_argument('--max_batch_size', type=int, default=1, help='Maximum batch size')
    parser.add_argument('--max_input_len', type=int, default=1024, help='Maximum input sequence length')
    parser.add_argument('--max_output_len', type=int, default=1024, help='Maximum output sequence length')
    parser.add_argument('--legacy', action='store_true', help='Use legacy checkpoint conversion method')
    
    args = parser.parse_args()
    
    try:
        if args.legacy:
            build_legacy_engine(
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                dtype=args.dtype,
                max_batch_size=args.max_batch_size,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
            )
        else:
            build_simple_engine(
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                dtype=args.dtype,
                max_batch_size=args.max_batch_size,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
            )
    except Exception as e:
        print(f"❌ Build failed: {e}")
        print("\n🔄 Trying legacy method as fallback...")
        try:
            build_legacy_engine(
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                dtype=args.dtype,
                max_batch_size=args.max_batch_size,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
            )
        except Exception as e2:
            print(f"❌ Legacy method also failed: {e2}")
            exit(1)
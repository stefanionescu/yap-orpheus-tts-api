"""
TensorRT-LLM Engine Builder for Orpheus Model
This script converts a Hugging Face model to TensorRT-LLM format with optimizations
"""

import os
import torch
import argparse
from pathlib import Path
import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin import PluginConfig
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.models import PretrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def build_orpheus_engine(
    model_dir: str,
    output_dir: str,
    dtype: str = 'float16',
    max_batch_size: int = 8,
    max_input_len: int = 512,
    max_output_len: int = 1200,
    max_beam_width: int = 1,
    use_gpt_attention_plugin: bool = True,
    use_gemm_plugin: bool = True,
    use_layernorm_plugin: bool = False,
    use_rmsnorm_plugin: bool = False,
    enable_context_fmha: bool = False,
    enable_remove_input_padding: bool = False,
    enable_paged_kv_cache: bool = False,
    tokens_per_block: int = 128,
    max_num_tokens: int = None,
    int8_kv_cache: bool = False,
    fp8_kv_cache: bool = False,
    use_weight_only: bool = False,
    weight_only_precision: str = 'bfloat16',
    world_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1
):
    """
    Build TensorRT-LLM engine for Orpheus model
    
    Args:
        model_dir: Path to HuggingFace model
        output_dir: Directory to save TRT engine
        dtype: Data type (float16, float32, bfloat16)
        max_batch_size: Maximum batch size
        max_input_len: Maximum input sequence length
        max_output_len: Maximum output sequence length
        ... (other optimization parameters)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load HuggingFace model and config
    print(f"Loading model from {model_dir}")
    hf_config = AutoConfig.from_pretrained(model_dir)
    
    # Map dtype
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    
    # Configure quantization
    quant_mode = QuantMode(0)
    if use_weight_only:
        if weight_only_precision == 'int8':
            quant_mode = QuantMode.use_weight_only(use_int8_weight_only=True)
        elif weight_only_precision == 'int4':
            quant_mode = QuantMode.use_weight_only(use_int4_weight_only=True)
    
    if int8_kv_cache:
        quant_mode = quant_mode.set_int8_kv_cache()
    elif fp8_kv_cache:
        quant_mode = quant_mode.set_fp8_kv_cache()
    
    # Plugin configuration - TensorRT-LLM expects string values for plugins
    plugin_config = PluginConfig()
    
    # Set plugin values based on dtype and boolean flags
    plugin_dtype = dtype if dtype != 'float32' else 'float16'  # Use float16 if float32 is specified
    
    if use_gpt_attention_plugin:
        plugin_config.gpt_attention_plugin = plugin_dtype
    else:
        plugin_config.gpt_attention_plugin = None
        
    if use_gemm_plugin:
        plugin_config.gemm_plugin = plugin_dtype
    else:
        plugin_config.gemm_plugin = None
        
    if use_layernorm_plugin:
        plugin_config.layernorm_plugin = plugin_dtype
    else:
        plugin_config.layernorm_plugin = None
        
    if use_rmsnorm_plugin:
        plugin_config.rmsnorm_plugin = plugin_dtype
    else:
        plugin_config.rmsnorm_plugin = None
    
    # Boolean configurations
    plugin_config.context_fmha = enable_context_fmha
    plugin_config.remove_input_padding = enable_remove_input_padding
    plugin_config.paged_kv_cache = enable_paged_kv_cache
    plugin_config.tokens_per_block = tokens_per_block
    
    # Calculate max_num_tokens if not provided
    if max_num_tokens is None:
        max_num_tokens = max_batch_size * max_input_len
    
    # Build configuration
    build_config = {
        'max_batch_size': max_batch_size,
        'max_input_len': max_input_len,
        'max_output_len': max_output_len,
        'max_beam_width': max_beam_width,
        'max_num_tokens': max_num_tokens,
        'strongly_typed': True,
        'builder_opt': None,
    }
    
    # Create TensorRT-LLM model configuration
    print("Creating TensorRT-LLM model configuration")
    
    # This is a simplified example - you'll need to adapt based on your specific model architecture
    # For a GPT-like model (which Orpheus likely is), you can use:
    
    tensorrt_llm_config = {
        'architecture': 'GPTForCausalLM',  # Adjust based on actual architecture
        'dtype': str(torch_dtype).split('.')[-1],
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_attention_heads,
        'num_key_value_heads': getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads),
        'hidden_size': hf_config.hidden_size,
        'intermediate_size': getattr(hf_config, 'intermediate_size', 4 * hf_config.hidden_size),
        'vocab_size': hf_config.vocab_size,
        'max_position_embeddings': hf_config.max_position_embeddings,
        'hidden_act': getattr(hf_config, 'hidden_act', 'gelu'),
        'norm_epsilon': getattr(hf_config, 'layer_norm_epsilon', 1e-5),
        'position_embedding_type': getattr(hf_config, 'position_embedding_type', 'learned'),
        'use_parallel_embedding': False,
        'embedding_sharding_dim': 0,
        'share_embedding_table': False,
        'quantization': {
            'quant_mode': quant_mode,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': tp_size,
            'pp_size': pp_size,
        },
        'head_size': hf_config.hidden_size // hf_config.num_attention_heads,
    }
    
    # Convert weights from HuggingFace to TensorRT-LLM format
    print("Converting model weights...")
    
    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map='cpu'  # Load on CPU first
    )
    
    # Save converted checkpoint
    checkpoint_dir = output_dir / 'checkpoint'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Convert and save weights (this is model-specific)
    # You'll need to implement the actual weight conversion based on your model architecture
    # This is a placeholder for the conversion process
    print("Saving converted checkpoint...")
    
    # Example of how to save config
    import json
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(tensorrt_llm_config, f, indent=2)
    
    # Build TensorRT engine
    print("Building TensorRT engine...")
    
    # Initialize builder
    builder = Builder()
    
    # Build engine with configuration
    engine = builder.build_engine(
        model_config=tensorrt_llm_config,
        weights_path=checkpoint_dir,
        output_dir=output_dir,
        build_config=build_config,
        plugin_config=plugin_config,
    )
    
    # Save additional metadata
    metadata = {
        'model_name': 'orpheus-tts',
        'precision': dtype,
        'max_batch_size': max_batch_size,
        'max_input_len': max_input_len,
        'max_output_len': max_output_len,
        'quantization': str(quant_mode),
        'plugin_config': {
            'gpt_attention_plugin': plugin_config.gpt_attention_plugin,
            'gemm_plugin': plugin_config.gemm_plugin,
            'layernorm_plugin': plugin_config.layernorm_plugin,
            'rmsnorm_plugin': plugin_config.rmsnorm_plugin,
            'context_fmha': enable_context_fmha,
            'remove_input_padding': enable_remove_input_padding,
            'paged_kv_cache': enable_paged_kv_cache,
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Engine successfully built and saved to {output_dir}")
    return output_dir


def benchmark_engine_build(model_dir: str, output_dir: str, configs: list):
    """
    Benchmark different engine configurations
    """
    import time
    
    results = []
    
    for config_name, config in configs:
        print(f"\n=== Building engine with config: {config_name} ===")
        start_time = time.time()
        
        try:
            engine_dir = build_orpheus_engine(
                model_dir=model_dir,
                output_dir=f"{output_dir}_{config_name}",
                **config
            )
            build_time = time.time() - start_time
            
            # Get engine size
            engine_size = sum(
                f.stat().st_size for f in Path(engine_dir).rglob('*') if f.is_file()
            ) / (1024 * 1024)  # MB
            
            results.append({
                'config': config_name,
                'build_time': build_time,
                'engine_size_mb': engine_size,
                'status': 'success',
                'settings': config
            })
            
            print(f"Build time: {build_time:.2f}s")
            print(f"Engine size: {engine_size:.2f} MB")
            
        except Exception as e:
            results.append({
                'config': config_name,
                'status': 'failed',
                'error': str(e)
            })
            print(f"Build failed: {e}")
    
    # Print summary
    print("\n=== Build Benchmark Summary ===")
    for result in results:
        if result['status'] == 'success':
            print(f"{result['config']}:")
            print(f"  Build time: {result['build_time']:.2f}s")
            print(f"  Engine size: {result['engine_size_mb']:.2f} MB")
        else:
            print(f"{result['config']}: FAILED - {result['error']}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build TensorRT-LLM engine for Orpheus model')
    
    # Required arguments
    parser.add_argument('--model_dir', type=str, required=True, help='Path to HuggingFace model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for TRT engine')
    
    # Basic configuration (optimized for single A10 GPU)
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32', 'bfloat16'], help='Model precision (float16 recommended for A10)')
    parser.add_argument('--max_batch_size', type=int, default=4, help='Maximum batch size (reduced for A10 24GB VRAM)')
    parser.add_argument('--max_input_len', type=int, default=512, help='Maximum input sequence length')
    parser.add_argument('--max_output_len', type=int, default=1200, help='Maximum output sequence length')
    parser.add_argument('--max_beam_width', type=int, default=1, help='Maximum beam width for beam search')
    parser.add_argument('--max_num_tokens', type=int, default=None, help='Maximum number of tokens (auto-calculated if None)')
    parser.add_argument('--tokens_per_block', type=int, default=64, help='Tokens per block for paged attention (reduced for A10)')
    
    # Plugin configuration (with enable/disable options that map to dtype strings)
    parser.add_argument('--use_gpt_attention_plugin', action='store_true', default=True, help='Use GPT attention plugin with dtype precision')
    parser.add_argument('--no_gpt_attention_plugin', dest='use_gpt_attention_plugin', action='store_false', help='Disable GPT attention plugin')
    parser.add_argument('--use_gemm_plugin', action='store_true', default=True, help='Use GEMM plugin with dtype precision')
    parser.add_argument('--no_gemm_plugin', dest='use_gemm_plugin', action='store_false', help='Disable GEMM plugin')
    parser.add_argument('--use_layernorm_plugin', action='store_true', default=False, help='Use LayerNorm plugin with dtype precision')
    parser.add_argument('--no_layernorm_plugin', dest='use_layernorm_plugin', action='store_false', help='Disable LayerNorm plugin')
    parser.add_argument('--use_rmsnorm_plugin', action='store_true', default=False, help='Use RMSNorm plugin with dtype precision')
    parser.add_argument('--no_rmsnorm_plugin', dest='use_rmsnorm_plugin', action='store_false', help='Disable RMSNorm plugin')
    
    # Optimization features
    parser.add_argument('--enable_context_fmha', action='store_true', default=True, help='Enable context FMHA')
    parser.add_argument('--no_context_fmha', dest='enable_context_fmha', action='store_false', help='Disable context FMHA')
    parser.add_argument('--enable_remove_input_padding', action='store_true', default=True, help='Enable remove input padding')
    parser.add_argument('--no_remove_input_padding', dest='enable_remove_input_padding', action='store_false', help='Disable remove input padding')
    parser.add_argument('--enable_paged_kv_cache', action='store_true', default=True, help='Enable paged KV cache')
    parser.add_argument('--no_paged_kv_cache', dest='enable_paged_kv_cache', action='store_false', help='Disable paged KV cache')
    
    # Quantization options
    parser.add_argument('--use_weight_only', action='store_true', help='Use weight-only quantization')
    parser.add_argument('--weight_only_precision', type=str, default='bfloat16', choices=['int8', 'int4', 'bfloat16'], help='Weight-only quantization precision')
    parser.add_argument('--int8_kv_cache', action='store_true', help='Enable int8 KV cache')
    parser.add_argument('--fp8_kv_cache', action='store_true', help='Enable fp8 KV cache')
    
    # Parallel processing (single A10 GPU defaults)
    parser.add_argument('--world_size', type=int, default=1, help='Total number of GPUs (single A10)')
    parser.add_argument('--tp_size', type=int, default=1, help='Tensor parallelism size (single A10)')
    parser.add_argument('--pp_size', type=int, default=1, help='Pipeline parallelism size (single A10)')
    
    # Benchmark mode
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark with different configs')
    
    args = parser.parse_args()
    
    # Validate parallelism settings (should all be 1 for single A10)
    if args.world_size != 1 or args.tp_size != 1 or args.pp_size != 1:
        print("Warning: For single A10 GPU, parallelism settings should all be 1")
        print(f"Current settings: world_size={args.world_size}, tp_size={args.tp_size}, pp_size={args.pp_size}")
    
    # Validate KV cache options (mutually exclusive)
    if args.int8_kv_cache and args.fp8_kv_cache:
        raise ValueError("Cannot enable both int8_kv_cache and fp8_kv_cache simultaneously")
    
    # A10 GPU memory optimization recommendations
    if args.max_batch_size > 4:
        print(f"Warning: Batch size {args.max_batch_size} may be too large for A10 24GB VRAM. Consider reducing to 4 or lower.")
    
    if args.dtype == 'float32':
        print("Warning: float32 precision may consume too much VRAM on A10. Consider using float16 or bfloat16.")
    
    if args.benchmark:
        # Define configurations to benchmark (optimized for single A10 GPU)
        configs = [
            ('fp16_conservative', {
                'dtype': 'float16',
                'max_batch_size': 2,
                'use_gpt_attention_plugin': True,
                'use_gemm_plugin': True,
                'enable_context_fmha': True,
                'tokens_per_block': 64,
            }),
            ('fp16_optimized', {
                'dtype': 'float16',
                'max_batch_size': 4,
                'use_gpt_attention_plugin': True,
                'use_gemm_plugin': True,
                'enable_context_fmha': True,
                'enable_remove_input_padding': True,
                'enable_paged_kv_cache': True,
                'tokens_per_block': 64,
            }),
            ('int8_weight_only', {
                'dtype': 'float16',
                'max_batch_size': 6,
                'use_weight_only': True,
                'weight_only_precision': 'int8',
                'use_gpt_attention_plugin': True,
                'use_gemm_plugin': True,
                'enable_paged_kv_cache': True,
                'tokens_per_block': 64,
            }),
            ('int4_weight_only', {
                'dtype': 'float16',
                'max_batch_size': 8,
                'use_weight_only': True,
                'weight_only_precision': 'int4',
                'use_gpt_attention_plugin': True,
                'use_gemm_plugin': True,
                'enable_paged_kv_cache': True,
                'tokens_per_block': 64,
            }),
        ]
        
        benchmark_engine_build(args.model_dir, args.output_dir, configs)
    else:
        # Single build with all provided arguments
        build_orpheus_engine(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            dtype=args.dtype,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            max_beam_width=args.max_beam_width,
            use_gpt_attention_plugin=args.use_gpt_attention_plugin,
            use_gemm_plugin=args.use_gemm_plugin,
     #       use_layernorm_plugin=args.use_layernorm_plugin,
            use_rmsnorm_plugin=args.use_rmsnorm_plugin,
            enable_context_fmha=args.enable_context_fmha,
            enable_remove_input_padding=args.enable_remove_input_padding,
            enable_paged_kv_cache=args.enable_paged_kv_cache,
            tokens_per_block=args.tokens_per_block,
            max_num_tokens=args.max_num_tokens,
            int8_kv_cache=args.int8_kv_cache,
            fp8_kv_cache=args.fp8_kv_cache,
            use_weight_only=args.use_weight_only,
            weight_only_precision=args.weight_only_precision,
            world_size=args.world_size,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
        )
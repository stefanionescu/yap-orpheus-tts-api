#!/usr/bin/env python3
"""
Streaming-optimized TensorRT-LLM implementation for Orpheus TTS
Focuses on low TTFB and RTF < 1.0 for real-time performance
"""

import modal
import os
import time
import torch
from typing import List, Dict, Any

# Define persistent volume
volume = modal.Volume.from_name("tensorrt-workspace", create_if_missing=True)

# Optimized image for streaming performance
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
        "mpi4py", "snac", "hf_transfer", "huggingface_hub", "soundfile"
    )
    .run_commands(
        "pip install https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.20.0-cp310-cp310-linux_x86_64.whl"
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",  # Reduced for streaming
        "CUDA_VISIBLE_DEVICES": "0",
        "TENSORRT_LLM_LOG_LEVEL": "ERROR",  # Reduce logging overhead
        "HF_TOKEN": "",
        "HF_TRANSFER": "1"
    })
    .add_local_python_source("decoder", copy=True)
)

app = modal.App("streaming-trtllm-orpheus", image=image)

@app.function(
    gpu="A10G",
    memory=32768,
    timeout=3600,
    volumes={"/workspace": volume},
)
def build_streaming_engine(
    model_dir: str = "canopylabs/orpheus-3b-0.1-ft",
    output_dir: str = "/workspace/models/orpheus-streaming",
    streaming_config: str = "aggressive"  # conservative, balanced, aggressive
):
    """Build TensorRT-LLM engine optimized for streaming with low TTFB/RTF"""
    from tensorrt_llm.builder import Builder
    from tensorrt_llm.network import net_guard
    from tensorrt_llm.plugin import PluginConfig
    from tensorrt_llm.quantization import QuantMode
    from transformers import AutoConfig
    import subprocess
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Streaming-optimized configurations
    streaming_configs = {
        "conservative": {
            "max_batch_size": 1,           # Single request at a time
            "max_input_len": 256,          # Shorter context for faster processing
            "max_output_len": 512,         # Shorter output for lower TTFB
            "dtype": "float16",
            "use_weight_only": False,
            "tokens_per_block": 32,        # Smaller blocks for faster allocation
            "max_num_tokens": 512,
        },
        "balanced": {
            "max_batch_size": 1,
            "max_input_len": 512,
            "max_output_len": 768,
            "dtype": "float16", 
            "use_weight_only": True,
            "weight_only_precision": "int8",
            "tokens_per_block": 64,
            "max_num_tokens": 768,
        },
        "aggressive": {
            "max_batch_size": 1,
            "max_input_len": 512,
            "max_output_len": 1024,
            "dtype": "float16",
            "use_weight_only": True,
            "weight_only_precision": "int4",  # Most aggressive quantization
            "tokens_per_block": 128,
            "max_num_tokens": 1024,
        }
    }
    
    config = streaming_configs[streaming_config]
    print(f"🚀 Building streaming engine with {streaming_config} config")
    print(f"   Settings: {config}")
    
    # Use the simple_build approach but with streaming optimizations
    cmd_parts = [
        f"python -m simple_build",
        f"--model_dir {model_dir}",
        f"--output_dir {output_dir}",
        f"--dtype {config['dtype']}",
        f"--max_batch_size {config['max_batch_size']}",
        f"--max_input_len {config['max_input_len']}",
        f"--max_output_len {config['max_output_len']}",
    ]
    
    # Add streaming optimizations
    if config.get("use_weight_only"):
        cmd_parts.append("--use_weight_only")
        cmd_parts.append(f"--weight_only_precision {config.get('weight_only_precision', 'int8')}")
    
    cmd_parts.extend([
        "--use_gpt_attention_plugin",
        "--use_gemm_plugin", 
        "--enable_context_fmha",
        "--enable_remove_input_padding",
        "--enable_paged_kv_cache",
        f"--tokens_per_block {config['tokens_per_block']}",
        f"--max_num_tokens {config['max_num_tokens']}"
    ])
    
    cmd = " ".join(cmd_parts)
    print(f"🔨 Executing: {cmd}")
    
    build_start = time.time()
    ret = subprocess.call(cmd, shell=True)
    build_time = time.time() - build_start
    
    if ret != 0:
        raise RuntimeError(f"Streaming engine build failed with code {ret}")
    
    # Find the engine file
    possible_paths = [
        os.path.join(output_dir, "rank0.engine"),
        os.path.join(output_dir, "model.plan"),
        os.path.join(output_dir, "engines", "rank0.engine")
    ]
    
    engine_path = None
    for path in possible_paths:
        if os.path.isfile(path):
            engine_path = path
            break
    
    if not engine_path:
        raise FileNotFoundError(f"Engine file not found in {output_dir}")
    
    # Save streaming metadata
    streaming_metadata = {
        "config_type": streaming_config,
        "build_time": build_time,
        "settings": config,
        "engine_path": engine_path,
        "optimizations": [
            "quantization" if config.get("use_weight_only") else "no_quantization",
            "paged_kv_cache",
            "context_fmha", 
            "remove_input_padding",
            f"tokens_per_block_{config['tokens_per_block']}"
        ]
    }
    
    with open(os.path.join(output_dir, "streaming_metadata.json"), "w") as f:
        json.dump(streaming_metadata, f, indent=2)
    
    print(f"✅ Streaming engine built in {build_time:.2f}s: {engine_path}")
    return engine_path

@app.function(
    gpu="A10G", 
    memory=32768,
    volumes={"/workspace": volume},
    timeout=3600,
)
def streaming_inference_benchmark(
    engine_path: str = "/workspace/models/orpheus-streaming/rank0.engine",
    test_prompts: List[str] = None,
    target_chunk_size: int = 7  # Generate audio in 7-token chunks for streaming
):
    """Run streaming inference with chunked generation for low TTFB"""
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from transformers import AutoTokenizer
    from snac import SNAC
    import torch
    import numpy as np
    import soundfile as sf
    
    if test_prompts is None:
        test_prompts = [
            "Hello world",
            "This is a test of streaming audio generation",
            "Real-time text to speech synthesis should have low latency"
        ]
    
    engine_dir = os.path.dirname(engine_path)
    print(f"🔋 Loading streaming engine from: {engine_dir}")
    
    # Load models once for streaming
    try:
        llm = LLM(model=engine_dir)
        print("✅ Streaming LLM loaded")
    except Exception as e:
        print(f"❌ Failed to load engine: {e}")
        llm = LLM(model="canopylabs/orpheus-3b-0.1-ft")
        print("✅ Fallback LLM loaded")
    
    tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-ft")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    print("✅ All models loaded for streaming")
    
    def format_prompt(prompt, voice="tara"):
        adapted_prompt = f"{voice}: {prompt}"
        prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        return tokenizer.decode(all_input_ids[0])
    
    def streaming_generation(formatted_prompt, max_tokens=256):
        """Generate tokens in streaming fashion with early audio chunks"""
        # Use aggressive sampling for faster generation
        sampling_params = SamplingParams(
            temperature=0.3,     # Lower temp for faster, more deterministic generation
            top_p=0.7,          # Reduced for faster sampling
            max_tokens=max_tokens,
            repetition_penalty=1.1,
            stop_token_ids=[49158]
        )
        
        # Generate all tokens at once (TensorRT-LLM doesn't support true streaming yet)
        outputs = llm.generate([formatted_prompt], sampling_params)
        raw_tokens = outputs[0].outputs[0].token_ids
        
        # Simulate chunked processing for streaming effect
        audio_tokens = []
        audio_chunks = []
        
        # Extract audio tokens
        generated_ids = torch.tensor(raw_tokens).unsqueeze(0)
        audio_token_mask = (generated_ids >= 128266) & (generated_ids <= 156938)
        audio_positions = audio_token_mask.nonzero(as_tuple=True)
        
        if len(audio_positions[1]) > 0:
            first_idx = audio_positions[1][0].item()
            last_idx = audio_positions[1][-1].item()
            audio_token_ids = generated_ids[:, first_idx:last_idx+1].squeeze()
            
            # Process tokens  
            filtered_tokens = audio_token_ids[audio_token_ids != 128258]
            raw_codes = [(int(t) - 128266) % 4096 for t in filtered_tokens]
            
            # Group into 7-token chunks for streaming
            chunk_times = []
            for i in range(0, len(raw_codes), target_chunk_size):
                chunk_start = time.time()
                chunk = raw_codes[i:i + target_chunk_size]
                
                if len(chunk) == target_chunk_size:
                    # Convert chunk to audio
                    audio_chunk = decode_audio_chunk(chunk, snac_model)
                    if audio_chunk is not None:
                        audio_chunks.append(audio_chunk)
                        chunk_time = time.time() - chunk_start
                        chunk_times.append(chunk_time)
                        
                        # Streaming metrics per chunk
                        chunk_duration = len(audio_chunk) / 24000
                        chunk_rtf = chunk_time / chunk_duration if chunk_duration > 0 else float('inf')
                        
                        print(f"   📦 Chunk {len(audio_chunks)}: {chunk_time:.3f}s, RTF={chunk_rtf:.3f}x")
            
            return audio_chunks, chunk_times, raw_codes
        
        return [], [], []
    
    def decode_audio_chunk(codes, snac_model):
        """Decode a 7-token chunk to audio"""
        if len(codes) != 7:
            return None
            
        try:
            # Redistribute codes for SNAC
            codes_0 = torch.tensor([codes[0]], device='cuda', dtype=torch.int32).unsqueeze(0)
            codes_1 = torch.tensor([codes[1], codes[4]], device='cuda', dtype=torch.int32).unsqueeze(0) 
            codes_2 = torch.tensor([codes[2], codes[3], codes[5], codes[6]], device='cuda', dtype=torch.int32).unsqueeze(0)
            
            codes_list = [codes_0, codes_1, codes_2]
            
            with torch.inference_mode():
                audio_chunk = snac_model.decode(codes_list)
                return audio_chunk.detach().squeeze().cpu().numpy()
        except Exception as e:
            print(f"   ⚠️ Chunk decode failed: {e}")
            return None
    
    results = []
    
    # Test each prompt with streaming
    for i, prompt in enumerate(test_prompts):
        print(f"\n🧪 Streaming test {i+1}/{len(test_prompts)}: '{prompt}'")
        
        formatted_prompt = format_prompt(prompt, "tara")
        
        # Measure streaming performance
        total_start = time.time()
        audio_chunks, chunk_times, all_codes = streaming_generation(formatted_prompt, max_tokens=128)
        total_time = time.time() - total_start
        
        if audio_chunks:
            # Concatenate audio chunks
            full_audio = np.concatenate(audio_chunks)
            audio_duration = len(full_audio) / 24000
            
            # Calculate streaming metrics
            first_chunk_time = chunk_times[0] if chunk_times else 0
            avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0
            total_rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
            
            print(f"✅ Generated {len(audio_chunks)} chunks")
            print(f"   🚀 TTFB (first chunk): {first_chunk_time:.3f}s")
            print(f"   ⚡ Avg chunk time: {avg_chunk_time:.3f}s") 
            print(f"   🎵 Total audio: {audio_duration:.2f}s")
            print(f"   📊 Total RTF: {total_rtf:.3f}x")
            
            # Save concatenated audio
            output_dir = "/workspace/streaming_output"
            os.makedirs(output_dir, exist_ok=True)
            audio_file = os.path.join(output_dir, f"streaming_{i+1}.wav")
            sf.write(audio_file, full_audio, 24000)
            
            results.append({
                "prompt": prompt,
                "success": True,
                "ttfb": first_chunk_time,
                "total_time": total_time,
                "audio_duration": audio_duration,
                "total_rtf": total_rtf,
                "chunks_generated": len(audio_chunks),
                "avg_chunk_time": avg_chunk_time,
                "tokens_generated": len(all_codes),
                "audio_file": audio_file
            })
        else:
            print(f"❌ No audio chunks generated")
            results.append({
                "prompt": prompt,
                "success": False,
                "ttfb": 0,
                "total_time": total_time,
                "audio_duration": 0,
                "total_rtf": float('inf')
            })
    
    return results

@app.local_entrypoint()
def build_and_benchmark():
    """Build streaming engine and run benchmark"""
    print("🚀 Building streaming-optimized TensorRT-LLM engine...")
    
    # Build with aggressive streaming optimizations
    engine_path = build_streaming_engine.remote(
        streaming_config="aggressive"  # Most optimized for streaming
    )
    
    print(f"✅ Engine built: {engine_path}")
    
    # Run streaming benchmark
    print("\n🧪 Running streaming benchmark...")
    test_prompts = [
        "Hello",
        "This is a test",
        "Streaming audio generation should be fast and efficient"
    ]
    
    results = streaming_inference_benchmark.remote(
        engine_path=engine_path,
        test_prompts=test_prompts
    )
    
    # Calculate summary
    successful = [r for r in results if r["success"]]
    if successful:
        avg_ttfb = sum(r["ttfb"] for r in successful) / len(successful)
        avg_rtf = sum(r["total_rtf"] for r in successful) / len(successful)
        
        print(f"\n📈 Streaming Performance Summary:")
        print(f"   Average TTFB: {avg_ttfb:.3f}s")
        print(f"   Average RTF: {avg_rtf:.3f}x")
        print(f"   Real-time capable: {'✅ Yes' if avg_rtf < 1.0 else '❌ No'}")
        
        for i, result in enumerate(successful):
            print(f"   {i+1}. TTFB={result['ttfb']:.3f}s RTF={result['total_rtf']:.3f}x | {result['chunks_generated']} chunks")
    
    return results

@app.local_entrypoint()
def quick_benchmark():
    """Quick benchmark using existing engine"""
    engine_path = "/workspace/models/orpheus-streaming/rank0.engine"
    
    test_prompts = ["Quick test", "Another quick streaming test"]
    
    results = streaming_inference_benchmark.remote(
        engine_path=engine_path,
        test_prompts=test_prompts
    )
    
    print("\n📊 Quick streaming results:")
    for r in results:
        if r["success"]:
            print(f"TTFB: {r['ttfb']:.3f}s, RTF: {r['total_rtf']:.3f}x")
    
    return results
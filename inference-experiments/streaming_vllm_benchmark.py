#!/usr/bin/env python3
"""
Streaming vLLM benchmark using OrpheusModel from engine_class.py
Measures TTFB and RTF for real-time streaming performance
"""

import modal
import os
import time
import torch
from typing import List, Dict, Any

# Use the same volume and image as vLLM test
volume = modal.Volume.from_name("tensorrt-workspace", create_if_missing=True)

# Same image as test_vllm_inference.py but with engine_class
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git", "wget", "curl", "build-essential", "pkg-config"
    )
    .pip_install(
        # Install compatible PyTorch first - match vLLM 0.7.3 requirements
        "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "transformers", "datasets", "accelerate",
        "numpy", "packaging", "wheel", "setuptools",
        "snac", "huggingface_hub", "soundfile", "asyncio"
    )
    .pip_install(
        # Install newer vLLM that supports streaming
        "vllm==0.7.3"
    )
    # .pip_install(
    #     "wave"
    # )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_TOKEN": "",
        "HF_TRANSFER": "1"
    })
    .add_local_python_source("decoder", copy=True)
    .add_local_python_source("engine_class", copy=True)
)

app = modal.App("streaming-vllm-orpheus", image=image)

@app.function(
    gpu="L40S",
    memory=32768,
    timeout=3600,
    volumes={"/workspace": volume},
)
def streaming_vllm_benchmark(
    test_prompts=None,
    streaming_config=None
):
    """Run streaming vLLM benchmark using OrpheusModel with real-time streaming"""
    from engine_class import OrpheusModel
    import time
    import asyncio
    from collections import deque
    
    if test_prompts is None:
        test_prompts = [
            "Hello world",
            "This is a streaming test",
            "Real-time text to speech should have very low latency for good user experience"
        ]
    
    if streaming_config is None:
        # Optimized config for streaming performance
        streaming_config = {
            "max_model_len": 1024,          # Reduced context for faster processing
            "gpu_memory_utilization": 0.90, # Leave memory for audio processing
            # "swap_space": 4,                # Reduced swap
            # "enforce_eager": True,          # Disable graph optimization for lower latency
            "max_num_seqs": 1
            ,             # Single sequence for streaming
            "max_num_batched_tokens": 1024,  # Smaller batches for lower TTFB
            "swap_space": 4,                # Reduced swap
        }
    
    print(f"🔋 Loading OrpheusModel with streaming config...")
    print(f"   Config: {streaming_config}")
    
    # Initialize OrpheusModel with streaming optimizations
    model = OrpheusModel(
        model_name="maya-research/Veena",
        dtype=torch.torch.bfloat16,
        **streaming_config
    )
    print("✅ OrpheusModel loaded for streaming")
    
    def measure_streaming_performance(prompt, voice="kavya"):
        """Measure streaming performance with chunked audio generation"""
        print(f"\n🧪 Streaming: '{prompt}'")
        
        # Streaming parameters optimized for low latency
        generation_params = {
            "voice": voice,
            "temperature": 0.3,        # Lower for faster, more deterministic generation
            "top_p": 0.7,             # Reduced for faster sampling
            "max_tokens": 512,        # Shorter for lower TTFB
            "repetition_penalty": 1.1,
            "request_id": f"stream-{int(time.time())}"
        }
        
        # Track streaming metrics
        tokens_received = []
        token_times = []
        first_token_time = None
        start_time = time.time()
        
        print(f"🚀 Starting token streaming...")
        
        # Stream tokens using OrpheusModel's streaming generator
        try:
            for i, token_text in enumerate(model.generate_tokens_sync(prompt, **generation_params)):
                token_time = time.time()
                tokens_received.append(token_text)
                token_times.append(token_time)
                
                # Record TTFB (first token)
                if first_token_time is None:
                    first_token_time = token_time - start_time
                    print(f"   ⚡ TTFB (first token): {first_token_time:.3f}s")
                
                # Show streaming progress every 10 tokens
                if (i + 1) % 85 == 0:
                    elapsed = token_time - start_time
                    tokens_per_sec = (i + 1) / elapsed
                    print(f"   📊 {i+1} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
        except Exception as e:
            print(f"❌ Streaming failed: {e}")
            return None
        
        total_time = time.time() - start_time
        
        # Generate final audio from all streamed tokens
        print(f"🎵 Converting {len(tokens_received)} tokens to audio...")
        
        try:
            audio_processing_start = time.time()
            
            # Use the generate_speech method which handles token decoding
            print(f"   Processing audio generation...")
            audio_result = model.generate_speech(
                prompt=prompt,
                voice=voice,
                temperature=generation_params["temperature"],
                top_p=generation_params["top_p"],
                max_tokens=generation_params["max_tokens"],
                repetition_penalty=generation_params["repetition_penalty"]
            )
            print(f"   Audio generation completed")
            
            # Process audio chunks directly like the working example
            import wave
            
            # Save streaming audio to both locations
            output_dir = "/workspace/streaming_vllm_output"
            os.makedirs(output_dir, exist_ok=True)
            audio_file = os.path.join(output_dir, f"streaming_{int(time.time())}.wav")
            
            # Process audio chunks as they stream (like your working example)
            with wave.open(audio_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                
                total_frames = 0
                chunk_counter = 0
                
                for audio_chunk in audio_result:
                    chunk_counter += 1
                    frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_chunk)
                
                audio_duration = total_frames / wf.getframerate()
                total_rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
            
            audio_processing_time = time.time() - audio_processing_start
            print(f"   Audio processing took {audio_processing_time:.3f}s")
            print(f"   Processed {chunk_counter} chunks, duration: {audio_duration:.2f}s")
            print(f"   💾 Saved audio to: {audio_file}")
            
            # Also save a local copy
            with wave.open("output.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                
                # Re-generate for local copy (audio_result is consumed)
                audio_result_copy = model.generate_speech(
                    prompt=prompt,
                    voice=voice,
                    temperature=generation_params["temperature"],
                    top_p=generation_params["top_p"],
                    max_tokens=generation_params["max_tokens"],
                    repetition_penalty=generation_params["repetition_penalty"]
                )
                
                for audio_chunk in audio_result_copy:
                    wf.writeframes(audio_chunk)
            
            print(f"   💾 Local copy saved to: output.wav")
            
            print(f"✅ Streaming completed:")
            print(f"   🚀 TTFB: {first_token_time:.3f}s")
            print(f"   ⏱️  Total time: {total_time:.3f}s")
            print(f"   🎵 Audio duration: {audio_duration:.2f}s")
            print(f"   📊 RTF: {total_rtf:.3f}x")
            print(f"   🔤 Tokens: {len(tokens_received)}")
            print(f"   💾 Saved: {audio_file}")
            
            return {
                "prompt": prompt,
                "success": True,
                "ttfb": first_token_time,
                "total_time": total_time,
                "audio_duration": audio_duration,
                "total_rtf": total_rtf,
                "tokens_streamed": len(tokens_received),
                "tokens_per_second": len(tokens_received) / total_time,
                "audio_processing_time": audio_processing_time,
                "audio_file": audio_file,
                "realtime_capable": total_rtf < 1.0
            }
                
        except Exception as e:
            print(f"❌ Audio processing failed: {e}")
            return {
                "prompt": prompt,
                "success": False,
                "ttfb": first_token_time or 0,
                "total_time": total_time,
                "audio_duration": 0,
                "total_rtf": float('inf'),
                "tokens_streamed": len(tokens_received),
                "error": str(e)
            }
    
    # Run streaming benchmark on all test prompts
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"🧪 Streaming Test {i+1}/{len(test_prompts)}")
        print(f"{'='*60}")
        
        result = measure_streaming_performance(prompt)
        if result:
            results.append(result)
    
    return results

@app.local_entrypoint()
def benchmark():
    """Run vLLM streaming benchmark"""
    print("🚀 Running vLLM streaming benchmark with OrpheusModel...")
    
    test_prompts = [
        "Hello",
        "This is a streaming test",
        "आज मैंने एक नई तकनीक के बारे में सीखा जो कृत्रिम बुद्धिमत्ता का उपयोग करके मानव जैसी आवाज़ उत्पन्न कर सकती है",
        "Testing streaming performance with longer text to see how it handles extended content",
        "Final streaming test to evaluate overall performance"
    ]
    
    # Optimized streaming configuration
    streaming_config = {
        "max_model_len": 128,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,          # Critical for low latency
        "max_num_seqs": 1,             # Single sequence processing
        "max_num_batched_tokens": 128, # Smaller batches
        "swap_space": 2,               # Minimal swap for speed
    }
    
    results = streaming_vllm_benchmark.remote(
        test_prompts=test_prompts,
        streaming_config=streaming_config
    )
    
    # Calculate streaming performance summary
    successful_results = [r for r in results if r["success"]]
    
    if successful_results:
        avg_ttfb = sum(r["ttfb"] for r in successful_results) / len(successful_results)
        avg_rtf = sum(r["total_rtf"] for r in successful_results) / len(successful_results)
        avg_tokens_per_sec = sum(r["tokens_per_second"] for r in successful_results) / len(successful_results)
        realtime_count = sum(1 for r in successful_results if r["realtime_capable"])
        
        print(f"\n📈 vLLM Streaming Performance Summary:")
        print(f"   Success rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
        print(f"   Average TTFB: {avg_ttfb:.3f}s")
        print(f"   Average RTF: {avg_rtf:.3f}x")
        print(f"   Average token speed: {avg_tokens_per_sec:.1f} tok/s")
        print(f"   Real-time capable: {realtime_count}/{len(successful_results)} tests")
        print(f"   🎯 Streaming target: {'✅ ACHIEVED' if avg_rtf < 1.0 and avg_ttfb < 0.5 else '❌ NOT MET'}")
        
        print(f"\n📊 Per-prompt breakdown:")
        for i, result in enumerate(successful_results):
            status = "🚀" if result["realtime_capable"] else "⚠️"
            print(f"   {i+1:2d}. {status} TTFB={result['ttfb']:.3f}s RTF={result['total_rtf']:.3f}x | {result['tokens_streamed']} tokens | {len(result['prompt'])} chars")
    else:
        print(f"\n❌ No successful streaming tests")
    
    return results

@app.local_entrypoint()
def quick_stream():
    """Quick streaming test"""
    print("🚀 Quick vLLM streaming test...")
    
    results = streaming_vllm_benchmark.remote(
        test_prompts=["Quick streaming test"],
        streaming_config={
            "max_model_len": 128,
            "gpu_memory_utilization": 0.8,
            "enforce_eager": True,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 128,
        }
    )
    
    if results and results[0]["success"]:
        r = results[0]
        print(f"✅ TTFB: {r['ttfb']:.3f}s, RTF: {r['total_rtf']:.3f}x")
        print(f"   Real-time: {'✅ Yes' if r['realtime_capable'] else '❌ No'}")
    
    return results
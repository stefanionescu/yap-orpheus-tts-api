#!/usr/bin/env python3
"""
Test vLLM inference with Orpheus model to compare with TensorRT-LLM
"""

import modal
import os
import asyncio
import torch
from typing import AsyncGenerator

# Define persistent volume
volume = modal.Volume.from_name("tensorrt-workspace", create_if_missing=True)

# Build the Modal image with vLLM and dependencies
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
        "snac", "huggingface_hub", "soundfile"
    )
    .pip_install(
        # Install newer vLLM that supports more models
        "vllm==0.7.3"
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_TOKEN": "",
        "HF_TRANSFER": "1"
    })
    .add_local_python_source("decoder", copy=True)
)

app = modal.App("test-vllm-orpheus", image=image)

@app.cls(
    gpu="A10G",
    memory=32768,
    timeout=3600,
    volumes={"/workspace": volume},
)
class VLLMInferenceEngine:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.snac_model = None
        
    @modal.enter()
    def load_models(self):
        """Load models on container startup"""
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        from snac import SNAC
        
        model_name = "canopylabs/orpheus-3b-0.1-ft"
        print(f"🔋 Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"📝 Tokenizer loaded, vocab size: {self.tokenizer.vocab_size}")
        
        # Load vLLM model with explicit configuration
        try:
            self.llm = LLM(
                model=model_name,
                dtype=torch.bfloat16,
                max_model_len=256,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"❌ Failed to load with default settings: {e}")
            print("🔄 Trying with reduced settings...")
            self.llm = LLM(
                model=model_name,
                dtype=torch.bfloat16,
                max_model_len=256,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                enforce_eager=True,
                disable_custom_all_reduce=True
            )
        print(f"✅ vLLM model loaded successfully")
        
        # Load SNAC model
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        snac_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.snac_model = self.snac_model.to(snac_device)
        print("✅ SNAC model loaded")
        
    @modal.method()
    def inference_with_profiling(self, 
                                prompt: str = "Hello, this is a test of text to speech synthesis.", 
                                voice: str = "tara") -> dict:
        """Run inference with detailed profiling"""
        import time
        
        start_time = time.time()
        print(f"🚀 Starting inference with profiling...")
        
        # Format prompt
        format_start = time.time()
        formatted_prompt, token_ids = self.format_prompt(prompt, voice)
        format_time = time.time() - format_start
        
        print(f"💬 Original prompt: {prompt}")
        print(f"🔧 Formatted prompt: {formatted_prompt}")
        print(f"⏱️ Prompt formatting: {format_time:.3f}s")
        
        # Generate with vLLM
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.8,
            max_tokens=128,
            stop_token_ids=[49158],
            repetition_penalty=1.1
        )
        
        generation_start = time.time()
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        generation_time = time.time() - generation_start
        
        # Extract tokens
        raw_generated_ids = outputs[0].outputs[0].token_ids
        generated_text = outputs[0].outputs[0].text
        
        ttfb = generation_time  # Simplified TTFB measurement
        
        print(f"📝 Generated text: {generated_text[:100]}...")
        print(f"🔤 Generated {len(raw_generated_ids)} tokens")
        print(f"⚡ TTFB: {ttfb:.3f}s")
        
        # Process tokens to audio
        audio_start = time.time()
        audio_np = self.convert_tokens_to_audio(raw_generated_ids)
        audio_time = time.time() - audio_start
        
        total_time = time.time() - start_time
        
        if audio_np is not None:
            audio_duration = len(audio_np) / 24000
            rtf = total_time / audio_duration
            
            print(f"🎵 Audio duration: {audio_duration:.2f}s")
            print(f"📊 RTF: {rtf:.3f}x")
            print(f"⏱️ Total time: {total_time:.3f}s")
            
            # Save audio
            output_dir = "/workspace/vllm_test_output"
            os.makedirs(output_dir, exist_ok=True)
            audio_file = os.path.join(output_dir, "profiled_audio.wav")
            
            import soundfile as sf
            sf.write(audio_file, audio_np, 24000)
            print(f"💾 Audio saved to: {audio_file}")
            
            return {
                "success": True,
                "metrics": {
                    "ttfb": ttfb,
                    "total_time": total_time,
                    "audio_duration": audio_duration,
                    "rtf": rtf,
                    "tokens_generated": len(raw_generated_ids),
                    "text_length": len(prompt),
                    "format_time": format_time,
                    "generation_time": generation_time,
                    "audio_processing_time": audio_time
                },
                "audio_file": audio_file
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate audio"
            }
            
    def format_prompt(self, prompt, voice="tara"):
        """Format prompt for Orpheus model"""
        import torch
        adapted_prompt = f"{voice}: {prompt}"
        prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        prompt_string = self.tokenizer.decode(all_input_ids[0])
        return prompt_string, all_input_ids[0]
        
    def convert_tokens_to_audio(self, raw_generated_ids):
        """Convert tokens to audio using proper modulo operation"""
        if not raw_generated_ids:
            return None
            
        import torch
        generated_ids = torch.tensor(raw_generated_ids).unsqueeze(0)
        
        # Look for audio tokens
        audio_token_mask = (generated_ids >= 128266) & (generated_ids <= 156938)
        audio_positions = audio_token_mask.nonzero(as_tuple=True)
        
        if len(audio_positions[1]) > 0:
            first_audio_idx = audio_positions[1][0].item()
            last_audio_idx = audio_positions[1][-1].item()
            cropped_tensor = generated_ids[:, first_audio_idx:last_audio_idx+1]
        else:
            print("  ⚠️ No audio tokens found")
            return None
        
        # Remove EOS tokens
        token_to_remove = 128258
        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)
        
        # Process audio tokens
        audio_tokens = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            
            # Convert using proper modulo operation
            raw_codes = [int(t) - 128266 for t in trimmed_row]
            final_codes = [code % 4096 for code in raw_codes]
            audio_tokens.extend(final_codes)
            
        if len(audio_tokens) < 7:
            print("  ⚠️ Insufficient audio tokens")
            return None
            
        return self.redistribute_and_decode(audio_tokens)
        
    def redistribute_and_decode(self, audio_tokens):
        """Redistribute codes and decode to audio"""
        import torch
        snac_device = next(self.snac_model.parameters()).device
        
        codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)
        
        num_frames = len(audio_tokens) // 7
        frame = audio_tokens[:num_frames*7]
        
        for j in range(num_frames):
            i = 7*j
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
            else:
                codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])
            
            if codes_1.shape[0] == 0:
                codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
            else:
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
            
            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
            else:
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
        
        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        
        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)
            
        return audio_hat.detach().squeeze().cpu().numpy()


@app.local_entrypoint()
def main():
    """Test vLLM inference with profiling"""
    print("🚀 Testing vLLM inference with profiling...")
    
    engine = VLLMInferenceEngine()
    
    result = engine.inference_with_profiling.remote(
        prompt="Hello, this is a test of text to speech synthesis with detailed profiling.",
        voice="tara"
    )
    
    if result["success"]:
        print(f"✅ Success! Audio generated: {result['audio_file']}")
        print(f"📊 Metrics: {result['metrics']}")
        print("To download: modal volume get tensorrt-workspace /workspace/vllm_test_output/profiled_audio.wav ./vllm_profiled_audio.wav")
    else:
        print(f"❌ Failed: {result['error']}")

@app.function(
    gpu="A10G",
    memory=32768,
    timeout=3600,
    volumes={"/workspace": volume},
)
def benchmark_vllm_sequential(test_prompts = None):
    """Run sequential vLLM inference with single model load for benchmarking"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from snac import SNAC
    import time
    import torch
    import os
    import soundfile as sf
    
    if test_prompts is None:
        test_prompts = [
            "Short text.",
            "This is a medium length text for benchmarking vLLM performance.",
            "This is a much longer text to test vLLM capabilities with extended content generation.",
            "Another prompt to test the system's response.",    
            "Testing with a different voice: Tara, can you read this text for me?",
            "Can you generate audio for this prompt using the Orpheus model?",
            "Let's see how well vLLM handles this prompt with a longer context and multiple sentences.",
            "This is a test of the vLLM system to see how it performs with various prompts and audio generation tasks.",
            "Can you synthesize this text into audio? Let's check the quality and performance of the generated audio.",
            "Final test prompt to evaluate the overall performance and capabilities of vLLM in generating audio from text prompts."
        ]
    
    model_name = "canopylabs/orpheus-3b-0.1-ft"
    print(f"🔋 Loading vLLM model once: {model_name}")
    
    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"📝 Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
    
    # Load vLLM model once
    try:
        llm = LLM(
            model=model_name,
            dtype=torch.bfloat16,
            max_model_len=2048,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        print("✅ vLLM model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load with default settings: {e}")
        llm = LLM(
            model=model_name,
            dtype=torch.bfloat16,
            max_model_len=1024,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=True
        )
        print("✅ vLLM model loaded with fallback settings")
    
    # Load SNAC model once
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    snac_device = "cuda" if torch.cuda.is_available() else "cpu"
    snac_model = snac_model.to(snac_device)
    print("✅ SNAC model loaded")
    
    def format_prompt(prompt, voice="tara"):
        """Format prompt for Orpheus model"""
        adapted_prompt = f"{voice}: {prompt}"
        prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        prompt_string = tokenizer.decode(all_input_ids[0])
        return prompt_string
    
    def convert_tokens_to_audio(raw_generated_ids):
        """Convert tokens to audio using proper modulo operation"""
        if not raw_generated_ids:
            return None
            
        generated_ids = torch.tensor(raw_generated_ids).unsqueeze(0)
        
        # Look for audio tokens
        audio_token_mask = (generated_ids >= 128266) & (generated_ids <= 156938)
        audio_positions = audio_token_mask.nonzero(as_tuple=True)
        
        if len(audio_positions[1]) > 0:
            first_audio_idx = audio_positions[1][0].item()
            last_audio_idx = audio_positions[1][-1].item()
            cropped_tensor = generated_ids[:, first_audio_idx:last_audio_idx+1]
        else:
            print("  ⚠️ No audio tokens found")
            return None
        
        # Remove EOS tokens
        token_to_remove = 128258
        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)
        
        # Process audio tokens
        audio_tokens = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            
            # Convert using proper modulo operation
            raw_codes = [int(t) - 128266 for t in trimmed_row]
            final_codes = [code % 4096 for code in raw_codes]
            audio_tokens.extend(final_codes)
            
        if len(audio_tokens) < 7:
            print("  ⚠️ Insufficient audio tokens")
            return None
            
        return redistribute_and_decode(audio_tokens)
    
    def redistribute_and_decode(audio_tokens):
        """Redistribute codes and decode to audio"""
        codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)
        
        num_frames = len(audio_tokens) // 7
        frame = audio_tokens[:num_frames*7]
        
        for j in range(num_frames):
            i = 7*j
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
            else:
                codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])
            
            if codes_1.shape[0] == 0:
                codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
            else:
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
            
            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
            else:
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
        
        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        
        with torch.inference_mode():
            audio_hat = snac_model.decode(codes)
            
        return audio_hat.detach().squeeze().cpu().numpy()
    
    results = []
    
    # Process each prompt sequentially
    for i, prompt in enumerate(test_prompts):
        print(f"\n🧪 Testing prompt {i+1}/{len(test_prompts)}: '{prompt[:50]}...'")
        
        # Format prompt
        formatted_prompt = format_prompt(prompt, "tara")
        print(f"🔧 Formatted prompt: {formatted_prompt}")
        
        # Time the generation
        start_time = time.time()
        
        # Generate with vLLM
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.8,
            max_tokens=256,
            stop_token_ids=[49158],
            repetition_penalty=1.3
        )
        
        generation_start = time.time()
        outputs = llm.generate([formatted_prompt], sampling_params)
        generation_end = time.time()
        
        ttfb = generation_end - generation_start
        
        # Extract tokens
        raw_generated_ids = outputs[0].outputs[0].token_ids
        print(f"⚡ TTFB: {ttfb:.3f}s, Generated {len(raw_generated_ids)} tokens")
        
        # Convert to audio
        audio_processing_start = time.time()
        audio_np = convert_tokens_to_audio(raw_generated_ids)
        audio_processing_time = time.time() - audio_processing_start
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if audio_np is not None:
            audio_duration = len(audio_np) / 24000
            rtf = total_time / audio_duration
            
            print(f"✅ Completed: TTFB={ttfb:.3f}s, Total={total_time:.3f}s, Audio={audio_duration:.2f}s, RTF={rtf:.3f}x")
            
            results.append({
                "prompt": prompt,
                "success": True,
                "ttfb": ttfb,
                "total_time": total_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "tokens_generated": len(raw_generated_ids),
                "prompt_length": len(prompt),
                "audio_processing_time": audio_processing_time
            })
        else:
            print(f"❌ Failed to generate audio")
            results.append({
                "prompt": prompt,
                "success": False,
                "ttfb": ttfb,
                "total_time": total_time,
                "audio_duration": 0.0,
                "rtf": 0.0,
                "tokens_generated": len(raw_generated_ids),
                "prompt_length": len(prompt),
                "audio_processing_time": audio_processing_time
            })
    
    return results

@app.local_entrypoint()
def benchmark():
    """Benchmark vLLM with different text lengths"""
    print("🏃‍♂️ Running vLLM benchmark...")
    
    test_prompts = [
        "Short text.",
        "This is a medium length text for benchmarking vLLM performance.",
        "This is a much longer text to test vLLM capabilities with extended content generation.",
        "Another prompt to test the system's response.",
        "Testing with a different voice: Tara, can you read this text for me?",
        "Can you generate audio for this prompt using the Orpheus model?",
        "Let's see how well vLLM handles this prompt with a longer context and multiple sentences.",
        "This is a test of the vLLM system to see how it performs with various prompts and audio generation tasks.",
        "Can you synthesize this text into audio? Let's check the quality and performance of the generated audio.",
        "Final test prompt to evaluate the overall performance and capabilities of vLLM in generating audio from text prompts."
    ]
    
    # Run sequential inference with single model load
    results = benchmark_vllm_sequential.remote(test_prompts=test_prompts)
    
    # Calculate summary statistics
    successful_results = [r for r in results if r["success"]]
    success_rate = len(successful_results) / len(results)
    
    if successful_results:
        avg_ttfb = sum(r["ttfb"] for r in successful_results) / len(successful_results)
        avg_total = sum(r["total_time"] for r in successful_results) / len(successful_results)
        avg_rtf = sum(r["rtf"] for r in successful_results) / len(successful_results)
        total_audio = sum(r["audio_duration"] for r in successful_results)
        
        print(f"\n📈 vLLM Benchmark Summary:")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Total tests: {len(results)}")
        print(f"   Average TTFB: {avg_ttfb:.3f}s")
        print(f"   Average total time: {avg_total:.3f}s")
        print(f"   Average RTF: {avg_rtf:.3f}x")
        print(f"   Total audio generated: {total_audio:.2f}s")
        
        # Show per-prompt breakdown
        print(f"\n📊 Per-prompt breakdown:")
        for i, result in enumerate(results):
            status = "✅" if result["success"] else "❌"
            print(f"   {i+1:2d}. {status} TTFB={result['ttfb']:.3f}s RTF={result['rtf']:.3f}x | {result['prompt_length']:3d} chars | {result['prompt'][:40]}...")
    else:
        print(f"\n📈 vLLM Benchmark Summary:")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Total tests: {len(results)}")
        print(f"   ❌ No successful requests to analyze")
    
    return results
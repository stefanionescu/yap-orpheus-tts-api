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
        "snac", "hf_transfer", "huggingface_hub", "soundfile" # For decoder.py and download progress
    )
    .run_commands(
        # Install TensorRT-LLM
        "pip install https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.20.0-cp310-cp310-linux_x86_64.whl"
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_VISIBLE_DEVICES": "0",
        "TENSORRT_LLM_LOG_LEVEL": "INFO",
        "HF_TOKEN": "",
        "HF_TRANSFER": "1"  # Enable HF transfer for large models
    })
    .run_function(download_tensorrt_deps)
    .add_local_python_source("decoder", copy=True) 
    .add_local_python_source("simple_build", copy=True) # Add decoder.py
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
    model_dir: str = "canopylabs/orpheus-3b-0.1-ft",
    output_dir: str = "/workspace/models/orpheus-tts-0.1-finetune-prod",
    dtype: str = "bfloat16",
    max_batch_size: int = 1,
    max_input_len: int = 1024,
    max_output_len: int = 1024,
    legacy: bool = False,
):
    """Build and save the TensorRT-LLM engine into persistent volume using simplified approach"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command with minimal parameters for simplicity
    cmd_parts = [
        f"python -m simple_build",
        f"--model_dir {model_dir}",
        f"--output_dir {output_dir}",
        f"--dtype {dtype}",
        f"--max_batch_size {max_batch_size}",
        f"--max_input_len {max_input_len}",
        f"--max_output_len {max_output_len}",
    ]
    
    # Add legacy flag if requested
    if legacy:
        cmd_parts.append("--legacy")
    
    cmd = " ".join(cmd_parts)
    print(f"🔨 Executing: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f"Engine build failed with code {ret}")
        
    # Check for multiple possible engine file locations
    possible_engine_paths = [
        os.path.join(output_dir, "model.plan"),
        os.path.join(output_dir, "rank0.engine"),
        os.path.join(output_dir, "engines", "rank0.engine"),
        os.path.join(output_dir, "checkpoint", "config.json"),  # For legacy method
    ]
    
    engine_path = None
    for path in possible_engine_paths:
        if os.path.isfile(path):
            engine_path = path
            break
    
    if not engine_path:
        # List directory contents for debugging
        print(f"❌ Engine file not found. Contents of {output_dir}:")
        for item in os.listdir(output_dir):
            print(f"  {item}")
        raise FileNotFoundError(f"Expected engine file not found in {output_dir}")
        
    print(f"✅ Engine saved to: {engine_path}")
    return engine_path

@app.function(
    gpu="A10G",
    memory=32768,
    volumes={"/workspace": volume},
    timeout=3600,  # 1 hour
)
def run_llm_inference(
    engine_path: str = "/workspace/models/orpheus-tts-0.1-finetune-prod/rank0.engine",
    prompt: str = "Hello, this is a test of text to speech synthesis.",
    output_dir: str = "/workspace/models/orpheus-tts-0.1-finetune-prod/output"
):
    """Load the TensorRT engine, run inference, and decode tokens to audio."""
    from tensorrt_llm.llmapi import LLM
    from decoder import tokens_decoder_sync  # Import from decoder.py
    import time
    # start = time.time()

    os.makedirs(output_dir, exist_ok=True)

    print(f"🔋 Loading engine from: {engine_path}")
    # The LLM was saved as a complete directory, so we need to load it from there
    engine_dir = os.path.dirname(engine_path)
    print(f"📁 Loading LLM from directory: {engine_dir}")
    
    # List contents to see what's there
    if os.path.exists(engine_dir):
        print(f"📋 Contents of {engine_dir}:")
        for item in os.listdir(engine_dir):
            print(f"  - {item}")
    
    # Try to load the LLM using the saved directory
    try:
        # The LLM should be loadable from the saved directory
        llm = LLM(model=engine_dir)
        print("✅ LLM loaded successfully from saved directory")
    except Exception as e:
        print(f"❌ Failed to load from directory: {e}")
        # Fallback: try to load from the original model
        print("🔄 Trying to load from original model...")
        llm = LLM(model="canopylabs/orpheus-3b-0.1-ft")

    print(f"💬 Original prompt: {prompt}")
    
    # Load tokenizer to format prompt correctly
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-ft")
        print(f"📝 Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        # Fallback to basic prompt
        formatted_prompt = f"tara: {prompt}"
        token_ids = None
    else:
        # Format prompt like in the OrpheusModel class
        voice = "tara"
        adapted_prompt = f"{voice}: {prompt}"
        
        import torch
        start = time.time()

        prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        formatted_prompt = tokenizer.decode(all_input_ids[0])
        token_ids = all_input_ids[0]
        
        print(f"🔧 Formatted prompt: {formatted_prompt}")
        print(f"🔢 Token IDs: {token_ids.tolist()}")
    
    # Generate with TensorRT-LLM using the correct format
    try:
        # Try with sampling parameters matching vLLM approach
        from tensorrt_llm.llmapi import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.8,
            max_tokens=256,  # Increased from 1200 to generate more audio tokens
            repetition_penalty=1.1,
            stop_token_ids=[49158]  # Same as vLLM
        )
        outputs = llm.generate([formatted_prompt], sampling_params=sampling_params)
        print("✅ Generated with Orpheus-style sampling parameters")
        
    except Exception as e:
        print(f"❌ Sampling generation failed: {e}")
        # Try with basic parameters
        try:
            outputs = llm.generate([formatted_prompt])
            print("✅ Generated with basic parameters")
        except Exception as e2:
            print(f"❌ Basic generation failed: {e2}")
            raise e2
    
    # Extract tokens from the output
    raw_generated_ids = []
    generated_text = []
    
    print(f"🔍 Examining output structure...")
    for i, output in enumerate(outputs):
        print(f"  Output {i}: {type(output)}")
        
        # Get the generated token IDs from the output
        if hasattr(output, 'token_ids'):
            print(f"  Found token_ids: {len(output.token_ids)} tokens")
            raw_generated_ids.extend(output.token_ids)
        elif hasattr(output, 'outputs'):
            print(f"  Found outputs: {len(output.outputs)}")
            for j, out in enumerate(output.outputs):
                if hasattr(out, 'token_ids'):
                    print(f"    Sub-output {j} token_ids: {len(out.token_ids)} tokens")
                    raw_generated_ids.extend(out.token_ids)
                if hasattr(out, 'text'):
                    generated_text.append(out.text)
        elif hasattr(output, 'text'):
            generated_text.append(output.text)
    
    print(f"🔤 Raw generated tokens: {len(raw_generated_ids)}")
    print(f"📝 Generated text: {generated_text}")
    
    # Process tokens according to HuggingFace Orpheus implementation
    print("🔧 Processing tokens with correct Orpheus format...")
    
    if raw_generated_ids:
        # Convert to tensor for processing
        import torch
        generated_ids = torch.tensor(raw_generated_ids).unsqueeze(0)
        
        # Look for audio tokens in the range [128266, 156938] (audio token range)
        # These are the tokens that represent actual audio codes
        audio_token_mask = (generated_ids >= 128266) & (generated_ids <= 156938)
        
        # Find the first and last audio tokens to extract the audio sequence
        audio_positions = audio_token_mask.nonzero(as_tuple=True)
        
        if len(audio_positions[1]) > 0:
            first_audio_idx = audio_positions[1][0].item()
            last_audio_idx = audio_positions[1][-1].item()
            cropped_tensor = generated_ids[:, first_audio_idx:last_audio_idx+1]
            print(f"  Found {len(audio_positions[1])} audio tokens from position {first_audio_idx} to {last_audio_idx}")
            print(f"  Cropped to {cropped_tensor.shape[1]} audio tokens")
        else:
            print("  ⚠️ No audio tokens found in range [128266, 156938]")
            cropped_tensor = generated_ids
        
        # Remove EOS tokens (128258)
        token_to_remove = 128258
        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)
        
        # Process each row (batch)
        audio_code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7  # Must be multiple of 7
            trimmed_row = row[:new_length]
            
            # CRITICAL: Convert audio tokens to SNAC codes using proper modulo operation
            # Based on Orpheus implementation: token_id = int(number_str) - 10 - ((index % 7) * 4096)
            # Audio tokens are in range [128266, 156938] and need to be converted properly
            
            # First convert from token IDs to raw codes
            raw_codes = [int(t) - 128266 for t in trimmed_row]
            
            # Apply the proper modulo operation based on position in 7-token groups
            # This handles the offset cycling that Orpheus uses
            trimmed_row = []
            for i, code in enumerate(raw_codes):
                # Apply the modulo operation to get the actual SNAC code
                # The 10 offset and 4096 cycling is built into the token generation
                actual_code = code % 4096
                trimmed_row.append(actual_code)
            
            # Debug: Check token conversion
            print(f"  Original token range: {min([int(t) for t in row[:new_length]])} to {max([int(t) for t in row[:new_length]])}")
            print(f"  Raw code range: {min(raw_codes)} to {max(raw_codes)}")
            print(f"  Final SNAC code range: {min(trimmed_row)} to {max(trimmed_row)}")
            audio_code_lists.append(trimmed_row)
            
            print(f"  Processed {len(trimmed_row)} audio tokens (trimmed to multiple of 7)")
            print(f"  First 10 audio codes: {trimmed_row[:10]}")
        
        generated_tokens = audio_code_lists[0] if audio_code_lists else []
    else:
        print("  ⚠️ No raw token IDs found")
        generated_tokens = []

    # Decode tokens to audio using HuggingFace redistribution method
    audio_chunks = []
    if generated_tokens:
        print(f"🎵 Decoding {len(generated_tokens)} audio tokens using correct redistribution...")
        
        try:
            # Redistribute codes exactly like decoder.py implementation
            def redistribute_codes(multiframe):
                # Exact copy of decoder.py convert_to_audio function logic
                if len(multiframe) < 7:
                    print("  ❌ Insufficient tokens for redistribution (need at least 7)")
                    return None
                
                # Import SNAC model for decoding
                from snac import SNAC
                import torch
                import os
                
                snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
                snac_device = os.environ.get("SNAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
                snac_model = snac_model.to(snac_device)
                
                codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
                codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
                codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)
                
                num_frames = len(multiframe) // 7
                frame = multiframe[:num_frames*7]
                
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
                
                print(f"  Layer 0: {codes_0.shape[0]} codes")
                print(f"  Layer 1: {codes_1.shape[0]} codes")
                print(f"  Layer 2: {codes_2.shape[0]} codes")
                
                codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
                
                # Check that all tokens are between 0 and 4096 (from decoder.py)
                if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
                    print("  ❌ Invalid token values detected (outside 0-4096 range)")
                    return None
                
                with torch.inference_mode():
                    audio_hat = snac_model.decode(codes)
                
                return audio_hat
            
            # Generate audio from redistributed codes
            audio_hat = redistribute_codes(generated_tokens)
            
            # Convert to audio file
            import soundfile as sf
            import numpy as np
            
            audio_np = audio_hat.detach().squeeze().cpu().numpy()
            audio_file = os.path.join(output_dir, "generated_audio.wav")
            difference = start - time.time()
            # Save as WAV file
            sf.write(audio_file, audio_np, 24000)
            print(f"💾 Audio saved to: {audio_file}")
            print(f"🎵 Audio duration: {len(audio_np)/24000:.2f} seconds")
            
            print(f"⏱ Time taken: {difference:.2f} seconds")
            print(f"✅ rtf = {len(audio_np)/24000/(len(generated_tokens)/7):.2f}")
            print(f"🎵 Audio generated successfully.")
            
        except Exception as e:
            print(f"❌ Audio decoding failed: {e}")
            import traceback
            traceback.print_exc()
            audio_file = None
    else:
        print("⚠️  No tokens generated, skipping audio generation")
        audio_file = None

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

@app.function(
    gpu="A10G",
    memory=32768,
    volumes={"/workspace": volume},
    timeout=3600,
)
def benchmark_sequential_inference(
    engine_path: str = "/workspace/models/orpheus-tts-0.1-finetune-prod/rank0.engine",
    test_prompts: list = None
):
    """Run sequential inference with single model load for benchmarking"""
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from decoder import tokens_decoder_sync
    import time
    import os
    
    if test_prompts is None:
        test_prompts = [
            "Short text.",
            "This is a medium length text for benchmarking TensorRT-LLM performance.",
            "This is a much longer text to test TensorRT-LLM capabilities with extended content generation."
        ]
    
    print(f"🔋 Loading engine once from: {engine_path}")
    engine_dir = os.path.dirname(engine_path)
    
    # Load LLM once
    try:
        llm = LLM(model=engine_dir)
        print("✅ LLM loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load from directory: {e}")
        llm = LLM(model="canopylabs/orpheus-3b-0.1-ft")
        print("✅ LLM loaded from fallback")
    
    # Load tokenizer once
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-ft")
        print(f"📝 Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        tokenizer = None
    
    # Load SNAC once
    from snac import SNAC
    import torch
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    snac_device = "cuda" if torch.cuda.is_available() else "cpu"
    snac_model = snac_model.to(snac_device)
    print("✅ SNAC model loaded")
    
    results = []
    
    # Process each prompt
    for i, prompt in enumerate(test_prompts):
        print(f"\n🧪 Testing prompt {i+1}/{len(test_prompts)}: '{prompt[:50]}...'")
        
        # Format prompt
        voice = "tara"
        adapted_prompt = f"{voice}: {prompt}"
        
        if tokenizer:
            import torch
            prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
            all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
            formatted_prompt = tokenizer.decode(all_input_ids[0])
        else:
            formatted_prompt = f"tara: {prompt}"
        
        print(f"🔧 Formatted prompt: {formatted_prompt}")
        
        # Time the generation
        start_time = time.time()
        
        # Generate with profiling
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.8,
            max_tokens=256,
            repetition_penalty=1.1,
            stop_token_ids=[49158]
        )
        
        generation_start = time.time()
        outputs = llm.generate([formatted_prompt], sampling_params=sampling_params)
        generation_end = time.time()
        
        ttfb = generation_end - generation_start
        
        # Extract tokens
        raw_generated_ids = []
        for output in outputs:
            if hasattr(output, 'outputs'):
                for out in output.outputs:
                    if hasattr(out, 'token_ids'):
                        raw_generated_ids.extend(out.token_ids)
        
        print(f"⚡ TTFB: {ttfb:.3f}s, Generated {len(raw_generated_ids)} tokens")
        
        # Convert to audio (simplified version)
        audio_processing_start = time.time()
        audio_duration = 0.0
        rtf = 0.0
        
        if raw_generated_ids:
            # Process tokens like in run_llm_inference
            generated_ids = torch.tensor(raw_generated_ids).unsqueeze(0)
            audio_token_mask = (generated_ids >= 128266) & (generated_ids <= 156938)
            audio_positions = audio_token_mask.nonzero(as_tuple=True)
            
            if len(audio_positions[1]) > 0:
                first_audio_idx = audio_positions[1][0].item()
                last_audio_idx = audio_positions[1][-1].item()
                cropped_tensor = generated_ids[:, first_audio_idx:last_audio_idx+1]
                
                # Process audio tokens
                processed_rows = []
                for row in cropped_tensor:
                    masked_row = row[row != 128258]  # Remove EOS
                    processed_rows.append(masked_row)
                
                # Get audio codes
                audio_codes = []
                for row in processed_rows:
                    row_length = row.size(0)
                    new_length = (row_length // 7) * 7
                    trimmed_row = row[:new_length]
                    raw_codes = [int(t) - 128266 for t in trimmed_row]
                    final_codes = [code % 4096 for code in raw_codes]
                    audio_codes.extend(final_codes)
                
                # Estimate audio duration (126 tokens ≈ 1.54s based on your output)
                if len(audio_codes) >= 7:
                    audio_duration = (len(audio_codes) / 7) * 0.086  # ~0.086s per frame
                
        end_time = time.time()
        total_time = end_time - start_time
        
        if audio_duration > 0:
            rtf = total_time / audio_duration
        
        print(f"✅ Completed: TTFB={ttfb:.3f}s, Total={total_time:.3f}s, Audio={audio_duration:.2f}s, RTF={rtf:.3f}x")
        
        results.append({
            "prompt": prompt,
            "success": len(raw_generated_ids) > 0,
            "ttfb": ttfb,
            "total_time": total_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
            "tokens_generated": len(raw_generated_ids),
            "prompt_length": len(prompt)
        })
    
    return results

@app.local_entrypoint()
def benchmark():
    """Benchmark TensorRT-LLM performance"""
    print("🏃‍♂️ Running TensorRT-LLM benchmark...")
    
    test_prompts = [
        "Short text.",
        "This is a medium length text for benchmarking TensorRT-LLM performance.",
        "This is a much longer text to test TensorRT-LLM capabilities with extended content generation.",
        "Another prompt to test the system's response.",
        "Testing with a different voice: Tara, can you read this text for me?",
        "Can you generate audio for this prompt using the Orpheus model?",
        "Let's see how well TensorRT-LLM handles this prompt with a longer context and multiple sentences.",
        "This is a test of the TensorRT-LLM system to see how it performs with various prompts and audio generation tasks.",
        "Can you synthesize this text into audio? Let's check the quality and performance of the generated audio.",
        "Final test prompt to evaluate the overall performance and capabilities of TensorRT-LLM in generating audio from text prompts."
    ]
    
    # Run sequential inference with single model load
    results = benchmark_sequential_inference.remote(
        engine_path="/workspace/models/orpheus-tts-0.1-finetune-prod/rank0.engine",
        test_prompts=test_prompts
    )
    
    # Calculate summary statistics
    successful_results = [r for r in results if r["success"]]
    success_rate = len(successful_results) / len(results)
    
    if successful_results:
        avg_ttfb = sum(r["ttfb"] for r in successful_results) / len(successful_results)
        avg_total = sum(r["total_time"] for r in successful_results) / len(successful_results)
        avg_rtf = sum(r["rtf"] for r in successful_results) / len(successful_results)
        total_audio = sum(r["audio_duration"] for r in successful_results)
        
        print(f"\n📈 TensorRT-LLM Benchmark Summary:")
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
        print(f"\n📈 TensorRT-LLM Benchmark Summary:")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Total tests: {len(results)}")
        print(f"   ❌ No successful requests to analyze")
    
    return results

@app.local_entrypoint()
def jupyter():
    """Start Jupyter Lab"""
    print("🚀 Starting Jupyter Lab...")
    run_jupyter = modal.Function.lookup(app, "run_jupyter")
    run_jupyter.remote()
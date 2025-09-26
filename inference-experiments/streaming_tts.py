#!/usr/bin/env python3
"""
Streaming TTS with profiling, queuing, and audio stitching
"""

import modal
import os
import time
import asyncio
import torch
import numpy as np
import soundfile as sf
from typing import List, Dict, AsyncGenerator, Optional
from dataclasses import dataclass
from queue import Queue
import threading
import json

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

app = modal.App("streaming-orpheus-tts", image=image)

@dataclass
class TTSRequest:
    id: str
    text: str
    voice: str = "tara"
    timestamp: float = 0.0

@dataclass
class TTSMetrics:
    request_id: str
    ttfb: float  # Time to first byte (first audio chunk)
    total_time: float  # Total processing time
    audio_duration: float  # Duration of generated audio
    rtf: float  # Real-time factor (processing_time / audio_duration)
    tokens_generated: int
    text_length: int

class AudioStitcher:
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.audio_buffer = []
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio chunk to buffer"""
        self.audio_buffer.append(audio_data.flatten())
        
    def get_stitched_audio(self) -> np.ndarray:
        """Get the complete stitched audio"""
        if not self.audio_buffer:
            return np.array([])
        return np.concatenate(self.audio_buffer)
        
    def clear(self):
        """Clear the buffer"""
        self.audio_buffer = []

@app.cls(
    gpu="A100",
    memory=32768,
    timeout=3600,
    volumes={"/workspace": volume},
    max_containers=1,
)
class StreamingTTSEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.snac_model = None
        self.request_queue = Queue()
        self.metrics = []
        
    @modal.enter()
    def load_models(self):
        """Load models on container startup"""
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        from snac import SNAC
        
        print("🔋 Loading Orpheus TTS model...")
        model_name = "canopylabs/orpheus-3b-0.1-ft"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"📝 Tokenizer loaded, vocab size: {self.tokenizer.vocab_size}")
        
        # Load vLLM model
        self.model = LLM(
            model=model_name,
            dtype=torch.bfloat16,
            max_model_len=2048,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        print("✅ vLLM model loaded successfully")
        
        # Load SNAC model for decoding
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        snac_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.snac_model = self.snac_model.to(snac_device)
        print("✅ SNAC model loaded successfully")
        
    def format_prompt(self, text: str, voice: str = "tara") -> str:
        """Format prompt for Orpheus model"""
        adapted_prompt = f"{voice}: {text}"
        prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        prompt_string = self.tokenizer.decode(all_input_ids[0])
        return prompt_string
        
    def convert_tokens_to_audio(self, raw_token_ids: List[int]) -> Optional[np.ndarray]:
        """Convert tokens to audio using proper modulo operation"""
        if not raw_token_ids:
            return None
            
        # Convert to tensor for processing
        generated_ids = torch.tensor(raw_token_ids).unsqueeze(0)
        
        # Look for audio tokens in the range [128266, 156938]
        audio_token_mask = (generated_ids >= 128266) & (generated_ids <= 156938)
        audio_positions = audio_token_mask.nonzero(as_tuple=True)
        
        if len(audio_positions[1]) == 0:
            print("  ⚠️ No audio tokens found")
            return None
            
        first_audio_idx = audio_positions[1][0].item()
        last_audio_idx = audio_positions[1][-1].item()
        cropped_tensor = generated_ids[:, first_audio_idx:last_audio_idx+1]
        
        # Remove EOS tokens
        token_to_remove = 128258
        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)
        
        # Process tokens
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
            
        # Redistribute codes like decoder.py
        return self.redistribute_and_decode(audio_tokens)
        
    def redistribute_and_decode(self, audio_tokens: List[int]) -> np.ndarray:
        """Redistribute codes and decode to audio"""
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
        
    @modal.method()
    def process_single_request(self, request: TTSRequest) -> Dict:
        """Process a single TTS request with profiling"""
        start_time = time.time()
        ttfb = None
        
        print(f"🚀 Processing request {request.id}: '{request.text[:50]}...'")
        
        # Format prompt
        formatted_prompt = self.format_prompt(request.text, request.voice)
        
        # Generate with profiling
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.8,
            max_tokens=2000,
            stop_token_ids=[49158],
            repetition_penalty=1.3
        )
        
        generation_start = time.time()
        outputs = self.model.generate([formatted_prompt], sampling_params)
        
        # Extract tokens
        raw_token_ids = outputs[0].outputs[0].token_ids
        ttfb = time.time() - generation_start  # First token time
        
        print(f"⚡ TTFB: {ttfb:.3f}s, Generated {len(raw_token_ids)} tokens")
        
        # Convert to audio
        audio_conversion_start = time.time()
        audio_np = self.convert_tokens_to_audio(raw_token_ids)
        
        if audio_np is None:
            return {
                "success": False,
                "error": "Failed to generate audio",
                "request_id": request.id
            }
            
        total_time = time.time() - start_time
        audio_duration = len(audio_np) / 24000  # 24kHz sample rate
        rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
        
        # Create metrics
        metrics = TTSMetrics(
            request_id=request.id,
            ttfb=ttfb,
            total_time=total_time,
            audio_duration=audio_duration,
            rtf=rtf,
            tokens_generated=len(raw_token_ids),
            text_length=len(request.text)
        )
        
        self.metrics.append(metrics)
        
        print(f"✅ Request {request.id} completed:")
        print(f"   📊 RTF: {rtf:.3f}x, Audio: {audio_duration:.2f}s, Total: {total_time:.3f}s")
        
        return {
            "success": True,
            "request_id": request.id,
            "audio_data": audio_np.tolist(),
            "sample_rate": 24000,
            "metrics": {
                "ttfb": ttfb,
                "total_time": total_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "tokens_generated": len(raw_token_ids),
                "text_length": len(request.text)
            }
        }
        
    @modal.method()
    def process_batch_requests(self, requests: List[Dict]) -> Dict:
        """Process multiple TTS requests with queuing and audio stitching"""
        batch_start_time = time.time()
        stitcher = AudioStitcher()
        results = []
        
        print(f"🎯 Processing batch of {len(requests)} requests")
        
        for i, req_data in enumerate(requests):
            request = TTSRequest(
                id=req_data.get("id", f"req_{i}"),
                text=req_data["text"],
                voice=req_data.get("voice", "tara"),
                timestamp=time.time()
            )
            
            # Process request inline instead of calling modal method
            start_time = time.time()
            print(f"🚀 Processing request {request.id}: '{request.text[:50]}...'")
            
            # Format prompt
            formatted_prompt = self.format_prompt(request.text, request.voice)
            
            # Generate with profiling
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.8,
                max_tokens=2000,
                stop_token_ids=[49158],
                repetition_penalty=1.3
            )
            
            generation_start = time.time()
            outputs = self.model.generate([formatted_prompt], sampling_params)
            
            # Extract tokens
            raw_token_ids = outputs[0].outputs[0].token_ids
            ttfb = time.time() - generation_start
            
            print(f"⚡ TTFB: {ttfb:.3f}s, Generated {len(raw_token_ids)} tokens")
            
            # Convert to audio
            audio_np = self.convert_tokens_to_audio(raw_token_ids)
            
            if audio_np is None:
                result = {
                    "success": False,
                    "error": "Failed to generate audio",
                    "request_id": request.id
                }
            else:
                total_time = time.time() - start_time
                audio_duration = len(audio_np) / 24000
                rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
                
                result = {
                    "success": True,
                    "request_id": request.id,
                    "audio_data": audio_np.tolist(),
                    "sample_rate": 24000,
                    "metrics": {
                        "ttfb": ttfb,
                        "total_time": total_time,
                        "audio_duration": audio_duration,
                        "rtf": rtf,
                        "tokens_generated": len(raw_token_ids),
                        "text_length": len(request.text)
                    }
                }
                
                print(f"✅ Request {request.id} completed:")
                print(f"   📊 RTF: {rtf:.3f}x, Audio: {audio_duration:.2f}s, Total: {total_time:.3f}s")
            
            results.append(result)
            
            # Add to stitcher if successful
            if result["success"]:
                audio_data = np.array(result["audio_data"])
                stitcher.add_audio(audio_data)
                
        # Get stitched audio
        stitched_audio = stitcher.get_stitched_audio()
        batch_total_time = time.time() - batch_start_time
        
        # Calculate batch metrics
        successful_requests = [r for r in results if r["success"]]
        total_audio_duration = sum(r["metrics"]["audio_duration"] for r in successful_requests)
        average_rtf = sum(r["metrics"]["rtf"] for r in successful_requests) / len(successful_requests) if successful_requests else 0
        
        print(f"🏁 Batch completed in {batch_total_time:.3f}s")
        print(f"   📊 Average RTF: {average_rtf:.3f}x")
        print(f"   🎵 Total audio: {total_audio_duration:.2f}s")
        
        return {
            "success": True,
            "batch_metrics": {
                "total_requests": len(requests),
                "successful_requests": len(successful_requests),
                "total_processing_time": batch_total_time,
                "total_audio_duration": total_audio_duration,
                "average_rtf": average_rtf,
                "throughput": len(successful_requests) / batch_total_time if batch_total_time > 0 else 0
            },
            "stitched_audio": stitched_audio.tolist(),
            "sample_rate": 24000,
            "individual_results": results
        }
        
    @modal.method()
    def get_metrics(self) -> List[Dict]:
        """Get collected metrics"""
        return [
            {
                "request_id": m.request_id,
                "ttfb": m.ttfb,
                "total_time": m.total_time,
                "audio_duration": m.audio_duration,
                "rtf": m.rtf,
                "tokens_generated": m.tokens_generated,
                "text_length": m.text_length
            }
            for m in self.metrics
        ]

@app.local_entrypoint()
def test_streaming():
    """Test streaming TTS with multiple requests"""
    engine = StreamingTTSEngine()
    
    # Test single request
    print("🧪 Testing single request...")
    single_request = TTSRequest(
        id="test_001",
        text="Hello, this is a test of our streaming text to speech system.",
        voice="tara"
    )
    
    result = engine.process_single_request.remote(single_request)
    print(f"Single request result: {result['success']}")
    
    # Test batch requests
    print("\n🧪 Testing batch requests...")
    batch_requests = [
        {
            "id": "batch_001",
            "text": "Welcome to our advanced text to speech system.",
            "voice": "tara"
        },
        {
            "id": "batch_002", 
            "text": "This system can process multiple requests efficiently.",
            "voice": "tara"
        },
        {
            "id": "batch_003",
            "text": "Real-time factors and audio stitching work perfectly together.",
            "voice": "tara"
        }
    ]
    
    batch_result = engine.process_batch_requests.remote(batch_requests)
    print(f"Batch processing successful: {batch_result['success']}")
    print(f"Batch metrics: {batch_result['batch_metrics']}")
    
    # Save stitched audio
    if batch_result["success"]:
        stitched_audio = np.array(batch_result["stitched_audio"])
        sf.write("stitched_output.wav", stitched_audio, 24000)
        print(f"💾 Stitched audio saved: {len(stitched_audio)/24000:.2f}s duration")
    
    # Get metrics
    metrics = engine.get_metrics.remote()
    print(f"\n📊 Collected {len(metrics)} metric records")
    
    return batch_result

@app.local_entrypoint()
def benchmark():
    """Benchmark performance with various text lengths"""
    engine = StreamingTTSEngine()
    
    test_texts = [
        "Short text.",
        "This is a medium length text that should take a bit more time to process and generate audio for testing purposes.",
        "This is a much longer text that will really test the capabilities of our text to speech system. It includes multiple sentences with various punctuation marks, complex words, and should generate a substantial amount of audio content. The goal is to measure how well the system performs with longer inputs and whether the real-time factor remains acceptable even for extended content generation tasks.",
    ]
    
    print("🏃‍♂️ Running performance benchmark...")
    
    requests = []
    for i, text in enumerate(test_texts):
        requests.append({
            "id": f"bench_{i}",
            "text": text,
            "voice": "tara"
        })
    
    result = engine.process_batch_requests.remote(requests)
    
    if result["success"]:
        print("\n📈 Benchmark Results:")
        print(f"Average RTF: {result['batch_metrics']['average_rtf']:.3f}x")
        print(f"Throughput: {result['batch_metrics']['throughput']:.2f} req/s")
        print(f"Total audio: {result['batch_metrics']['total_audio_duration']:.2f}s")
        
        # Individual results
        for i, res in enumerate(result["individual_results"]):
            if res["success"]:
                m = res["metrics"]
                print(f"Text {i+1}: RTF={m['rtf']:.3f}x, TTFB={m['ttfb']:.3f}s, Len={m['text_length']} chars")
    
    return result
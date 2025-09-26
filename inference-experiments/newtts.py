"""
Orpheus-TTS Modal Server with Streaming Audio Support
Enables real-time audio streaming for faster response times
"""

import modal
import io
import wave
import time
import os
import struct
from typing import Optional, List, Dict, Any, Generator
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import json
import logging
from datetime import datetime
import uuid
import asyncio

# Modal app configuration
app = modal.App("orpheus-tts-streaming")

# FIXED: Set environment variables to prevent CUDA initialization during import
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Main image with GPU support
orpheus_image_gpu = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install([
        "git",
        "build-essential",
        "ffmpeg",
        "libsndfile1",
        "wget",
        "curl",
        "ninja-build"
    ])
    .pip_install([
        "ninja",
        "packaging",
        "wheel",
        "torch",
        "torchaudio",
        "transformers",
        "vllm==0.7.3",
        "fastapi",
        "uvicorn[standard]",
        "websockets",
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "requests",
        "huggingface_hub",
        "python-multipart",
    ])
    # Install flash-attn with error handling
    .run_commands([
        "pip install flash-attn --no-build-isolation || echo 'Flash-attn install failed, continuing'"
    ])
    # Install WORKING FlashInfer version (v0.1.2 from GitHub issue)
    .run_commands([
        "pip uninstall flashinfer -y || echo 'No flashinfer to uninstall'",
        "python3 -m pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.2/flashinfer-0.1.2+cu121torch2.4-cp311-cp311-linux_x86_64.whl || echo 'FlashInfer install failed, continuing'"
    ])
    # Install orpheus-speech LAST to avoid import issues
    .pip_install(["orpheus-speech"])
    .env({
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_HOME": "/usr/local/cuda",
        # CRITICAL: Prevent CUDA initialization during import
        "CUDA_VISIBLE_DEVICES": "",  # Will be overridden in GPU functions
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "HF_TOKEN": ""
    })
)

# CPU-only image for testing (won't try to use CUDA)
orpheus_image_cpu = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["ffmpeg", "libsndfile1", "git"])
    .pip_install([
        "torch", 
        "transformers",
        "fastapi",
        "uvicorn[standard]",
        "websockets",
        "numpy",
        "scipy",
        "librosa",
        "soundfile", 
        "requests",
        "huggingface_hub",
    ])
    .env({
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "-1",  # Force CPU mode
        "TORCH_CUDA_INIT": "0",  # Prevent CUDA initialization
        "HF_TOKEN": ""
        
    })
)

# Request models
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "tara"
    streaming: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.1
    max_tokens: Optional[int] = 2000
    temperature: Optional[float] = 0.4
    top_p: Optional[float] = 0.9

class TTSStreamRequest(BaseModel):
    text: str
    voice: Optional[str] = "tara"
    repetition_penalty: Optional[float] = 1.1
    max_tokens: Optional[int] = 2000
    temperature: Optional[float] = 0.4
    top_p: Optional[float] = 0.9

# Volume for model caching
volume = modal.Volume.from_name("orpheus-models", create_if_missing=True)

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create WAV header for streaming audio"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0  # Unknown size for streaming
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,       
        b'WAVE',
        b'fmt ',
        16,                  
        1,                   # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

@app.cls(
    image=orpheus_image_gpu,
    gpu="A100",
    scaledown_window=300,
    timeout=600,
    volumes={"/models": volume},
    memory=24576,
    cpu=6,
)
class OrpheusStreamingServer:
    """Orpheus TTS Server with streaming support"""
    
    def __init__(self):
        self.model = None
        self.model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
        self.sample_rate = 24000
        
        # Available voices
        self.available_voices = [
            "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"
        ]
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_processing_time": 0.0,
            "streaming_requests": 0,
        }
    
    @modal.enter()
    def load_model(self):
        """Load Orpheus model with proper GPU initialization"""
        print("🚀 Loading Orpheus TTS model on GPU...")
        
        # CRITICAL: Enable CUDA for this GPU container
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Use FlashInfer v0.1.2 for MAXIMUM SPEED on A10G
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        
        try:
            import torch
            print(f"🔥 PyTorch version: {torch.__version__}")
            print(f"🎯 CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
                print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                raise RuntimeError("GPU not available in GPU container!")
            
            # Import orpheus AFTER setting up CUDA
            print("📦 Importing Orpheus TTS...")
            from orpheus_tts import OrpheusModel
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            
            # Monkey patch OrpheusModel to work with A10G GPU
            def custom_setup_engine(self):
                print("🔧 Using custom engine setup for A10G GPU compatibility...")
                engine_args = AsyncEngineArgs(
                    model=self.model_name,
                    dtype=self.dtype,
                    max_model_len=8192,  # Reduced from default 131072
                    gpu_memory_utilization=0.8,  # Conservative memory usage
                    # Keep CUDA graphs enabled for speed
                )
                return AsyncLLMEngine.from_engine_args(engine_args)
            
            # Apply the monkey patch
            OrpheusModel._setup_engine = custom_setup_engine
            
            # Conservative GPU configuration for A10G
            model_config = {
                "model_name": self.model_name,
                "dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            }
            
            print(f"🔧 Model config: {model_config}")
            
            # Load model
            self.model = OrpheusModel(**model_config)
            
            print("✅ Model loaded successfully on GPU!")
            print(f"📢 Available voices: {', '.join(self.available_voices)}")
            
            # Warm up
            self._warmup_model()
            
        except Exception as e:
            print(f"❌ GPU model loading failed: {e}")
            raise RuntimeError(f"Could not load Orpheus model on GPU: {e}")
    
    def _warmup_model(self):
        """Warm up the model"""
        try:
            print("🔥 Warming up model...")
            warmup_text = "This is a warmup generation."
            # Generate a short warmup sample
            warmup_result = self.model.generate_speech(
                prompt=warmup_text, 
                voice="tara",
                max_tokens=100
            )
            # Consume the generator if it's a generator
            if hasattr(warmup_result, '__iter__') and not isinstance(warmup_result, (bytes, str, np.ndarray)):
                list(warmup_result)
            print("✅ Model warmup completed")
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")
    
    @modal.method()
    def generate_speech_stream(
        self, 
        text: str, 
        voice: str = "tara",
        repetition_penalty: float = 1.1,
        max_tokens: int = 2000,
        temperature: float = 0.4,
        top_p: float = 0.9
    ):
        """Generate streaming speech from text"""
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.metrics["total_requests"] += 1
        self.metrics["streaming_requests"] += 1
        
        print(f"🎯 Streaming speech generation (ID: {request_id[:8]}): {text[:100]}...")
        
        try:
            # First yield the WAV header
            header_time = time.time()
            yield create_wav_header(sample_rate=self.sample_rate)
            print(f"⏱️ WAV header created in {time.time() - header_time:.4f}s")
            
            # Generate speech with streaming parameters
            model_start_time = time.time()
            print(f"🚀 Starting Orpheus model inference...")
            audio_stream = self.model.generate_speech(
                prompt=text,
                voice=voice,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[128258],  # Stop token for Orpheus
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            model_init_time = time.time() - model_start_time
            print(f"⚡ Orpheus model initialized streaming in {model_init_time:.4f}s")
            
            chunk_count = 0
            total_audio_bytes = 0
            first_chunk_time = None
            chunk_processing_times = []
            
            # Stream audio chunks as they're generated
            for chunk in audio_stream:
                chunk_start_time = time.time()
                chunk_count += 1
                
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    print(f"🎵 First audio chunk received in {first_chunk_time - start_time:.4f}s")
                
                # Convert chunk to bytes if needed
                conversion_start = time.time()
                if isinstance(chunk, np.ndarray):
                    # Convert numpy array to bytes (assuming int16)
                    if chunk.dtype != np.int16:
                        chunk = (chunk * 32767).astype(np.int16)
                    chunk_bytes = chunk.tobytes()
                elif isinstance(chunk, bytes):
                    chunk_bytes = chunk
                else:
                    # Try to convert to bytes
                    try:
                        chunk_bytes = bytes(chunk)
                    except:
                        print(f"⚠️ Skipping chunk of type {type(chunk)}")
                        continue
                
                conversion_time = time.time() - conversion_start
                total_audio_bytes += len(chunk_bytes)
                
                # Yield the audio chunk
                yield chunk_bytes
                
                chunk_processing_time = time.time() - chunk_start_time
                chunk_processing_times.append(chunk_processing_time)
                
                if chunk_count <= 5 or chunk_count % 20 == 0:  # Log first 5 and every 20th chunk
                    print(f"📦 Chunk {chunk_count}: {len(chunk_bytes)} bytes, processed in {chunk_processing_time:.4f}s (conversion: {conversion_time:.4f}s)")
                
                # Optional: Add small delay to prevent overwhelming the client
                # time.sleep(0.001)  # 1ms delay - adjust as needed
            
            processing_time = time.time() - start_time
            duration = total_audio_bytes / (2 * self.sample_rate)  # Assuming 16-bit audio
            
            # Calculate timing statistics
            avg_chunk_time = sum(chunk_processing_times) / len(chunk_processing_times) if chunk_processing_times else 0
            max_chunk_time = max(chunk_processing_times) if chunk_processing_times else 0
            orpheus_generation_time = processing_time - model_init_time if first_chunk_time else processing_time
            
            self.metrics["successful_generations"] += 1
            
            print(f"✅ Streaming completed - TIMING BREAKDOWN:")
            print(f"   📊 Total chunks: {chunk_count}")
            print(f"   🎵 Audio duration: {duration:.2f}s")
            print(f"   ⏱️ Total processing time: {processing_time:.4f}s")
            print(f"   🚀 Model initialization: {model_init_time:.4f}s")
            print(f"   🎯 Time to first chunk: {(first_chunk_time - start_time):.4f}s")
            print(f"   🔄 Orpheus generation time: {orpheus_generation_time:.4f}s")
            print(f"   📦 Avg chunk processing: {avg_chunk_time:.4f}s")
            print(f"   📈 Max chunk processing: {max_chunk_time:.4f}s")
            print(f"   🚀 Real-time factor: {duration/processing_time:.2f}x")
            
        except Exception as e:
            self.metrics["failed_generations"] += 1
            processing_time = time.time() - start_time
            
            print(f"❌ Streaming generation failed (ID: {request_id[:8]}): {e}")
            # In case of error, we can't really do much since we're already streaming
            # The client will need to handle the incomplete stream
            raise e
    
    @modal.method()
    def generate_speech(
        self, 
        text: str, 
        voice: str = "tara",
        repetition_penalty: float = 1.1,
        max_tokens: int = 2000,
        temperature: float = 0.4,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Generate complete speech from text (non-streaming)"""
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.metrics["total_requests"] += 1
        
        try:
            print(f"🎯 Generating complete speech (ID: {request_id[:8]}): {text[:100]}...")
            
            # Generate speech
            model_start_time = time.time()
            print(f"🚀 Starting Orpheus model inference (non-streaming)...")
            audio_data = self.model.generate_speech(
                prompt=text,
                voice=voice,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[128258],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            model_generation_time = time.time() - model_start_time
            print(f"⚡ Orpheus model generation completed in {model_generation_time:.4f}s")
            
            audio_chunks = []
            
            # Collect all chunks
            processing_start_time = time.time()
            print(f"🔄 Processing audio data...")
            chunk_count = 0
            
            if hasattr(audio_data, '__iter__') and not isinstance(audio_data, (bytes, str, np.ndarray)):
                for chunk in audio_data:
                    chunk_count += 1
                    if isinstance(chunk, bytes):
                        audio_chunks.append(chunk)
                    elif isinstance(chunk, np.ndarray):
                        if chunk.dtype != np.int16:
                            chunk = (chunk * 32767).astype(np.int16)
                        audio_chunks.append(chunk.tobytes())
                    else:
                        try:
                            audio_chunks.append(bytes(chunk))
                        except:
                            continue
            else:
                # Direct output
                chunk_count = 1
                if isinstance(audio_data, bytes):
                    audio_chunks.append(audio_data)
                elif isinstance(audio_data, np.ndarray):
                    if audio_data.dtype != np.int16:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    audio_chunks.append(audio_data.tobytes())
            
            audio_processing_time = time.time() - processing_start_time
            print(f"🎵 Processed {chunk_count} audio chunks in {audio_processing_time:.4f}s")
            
            # Combine all chunks and create WAV
            wav_creation_start = time.time()
            if audio_chunks:
                combined_audio = b''.join(audio_chunks)
                
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(combined_audio)
                
                audio_bytes = buffer.getvalue()
                duration = len(combined_audio) / (2 * self.sample_rate)
            else:
                audio_bytes = b''
                duration = 0.0
            
            wav_creation_time = time.time() - wav_creation_start
            print(f"🎶 WAV file created in {wav_creation_time:.4f}s")
            
            processing_time = time.time() - start_time
            self.metrics["successful_generations"] += 1
            
            print(f"✅ Non-streaming completed - TIMING BREAKDOWN:")
            print(f"   📊 Total chunks processed: {chunk_count}")
            print(f"   🎵 Audio duration: {duration:.2f}s")
            print(f"   ⏱️ Total processing time: {processing_time:.4f}s")
            print(f"   🚀 Orpheus model generation: {model_generation_time:.4f}s ({model_generation_time/processing_time*100:.1f}%)")
            print(f"   🔄 Audio processing: {audio_processing_time:.4f}s ({audio_processing_time/processing_time*100:.1f}%)")
            print(f"   🎶 WAV creation: {wav_creation_time:.4f}s ({wav_creation_time/processing_time*100:.1f}%)")
            print(f"   🚀 Real-time factor: {duration/processing_time:.2f}x")
            
            return {
                "audio_data": audio_bytes,
                "sample_rate": self.sample_rate,
                "duration": duration,
                "processing_time": processing_time,
                "voice_used": voice,
                "request_id": request_id,
                "success": True
            }
            
        except Exception as e:
            self.metrics["failed_generations"] += 1
            processing_time = time.time() - start_time
            
            print(f"❌ Generation failed (ID: {request_id[:8]}): {e}")
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "request_id": request_id,
                "voice_used": voice
            }
    
    @modal.method()
    def health_check(self) -> Dict[str, Any]:
        """Health check for GPU server"""
        
        import torch
        import sys
        
        # System info
        system_info = {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(0),
            })
        
        return {
            "status": "healthy" if self.model is not None else "model_not_loaded",
            "model": self.model_name,
            "available_voices": self.available_voices,
            "sample_rate": self.sample_rate,
            "streaming_support": True,
            "metrics": self.metrics,
            "system_info": system_info,
            "timestamp": datetime.now().isoformat()
        }

# FastAPI application
web_app = FastAPI(
    title="Orpheus TTS Streaming API",
    version="2.0.0",
    description="Orpheus-3B TTS service with real-time streaming support"
)

from fastapi.middleware.cors import CORSMiddleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.get("/")
async def root():
    return {
        "service": "Orpheus TTS Streaming API",
        "version": "2.0.0",
        "model": "canopylabs/orpheus-tts-0.1-finetune-prod",
        "status": "running",
        "features": ["streaming", "real-time", "low-latency"],
        "endpoints": {
            "stream": "POST /stream (real-time streaming)",
            "generate": "POST /generate (complete audio)",
            "tts": "GET /tts?prompt=text&voice=tara (streaming via GET)",
            "websocket": "WS /ws/stream (WebSocket real-time streaming)",
            "health": "GET /health",
            "voices": "GET /voices",
            "demo": "GET /demo (WebSocket demo page)"
        },
        "docs": "/docs"
    }

@web_app.post("/stream")
async def stream_speech_endpoint(request: TTSStreamRequest):
    """Stream speech generation in real-time"""
    server = OrpheusStreamingServer()
    
    try:
        def audio_stream():
            # Use remote_gen for generator functions
            stream_generator = server.generate_speech_stream.remote_gen(
                text=request.text,
                voice=request.voice or "tara",
                repetition_penalty=request.repetition_penalty,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            for chunk in stream_generator:
                yield chunk
        
        return StreamingResponse(
            audio_stream(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=orpheus_stream.wav",
                "X-Voice": request.voice or "tara",
                "X-Streaming": "true",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@web_app.get("/tts")
async def tts_get_endpoint(
    prompt: str = "Hey there, looks like you forgot to provide a prompt!",
    voice: str = "tara",
    repetition_penalty: float = 1.1,
    max_tokens: int = 2000,
    temperature: float = 0.4,
    top_p: float = 0.9
):
    """GET endpoint for streaming TTS (similar to your Flask example)"""
    server = OrpheusStreamingServer()
    
    try:
        def audio_stream():
            stream_generator = server.generate_speech_stream.remote_gen(
                text=prompt,
                voice=voice,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            for chunk in stream_generator:
                yield chunk
        
        return StreamingResponse(
            audio_stream(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_{voice}.wav",
                "X-Voice": voice,
                "X-Streaming": "true"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS streaming failed: {str(e)}")

@web_app.post("/generate")
async def generate_speech_endpoint(request: TTSRequest) -> Response:
    """Generate complete speech from text (non-streaming)"""
    server = OrpheusStreamingServer()
    
    try:
        if request.streaming:
            # Redirect to streaming endpoint
            def audio_stream():
                stream_generator = server.generate_speech_stream.remote_gen(
                    text=request.text,
                    voice=request.voice or "tara",
                    repetition_penalty=request.repetition_penalty or 1.1,
                    max_tokens=request.max_tokens or 2000,
                    temperature=request.temperature or 0.4,
                    top_p=request.top_p or 0.9
                )
                
                for chunk in stream_generator:
                    yield chunk
            
            return StreamingResponse(
                audio_stream(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=orpheus_speech.wav",
                    "X-Voice": request.voice or "tara",
                    "X-Streaming": "true"
                }
            )
        else:
            # Non-streaming generation
            result = server.generate_speech.remote(
                text=request.text,
                voice=request.voice or "tara",
                repetition_penalty=request.repetition_penalty or 1.1,
                max_tokens=request.max_tokens or 2000,
                temperature=request.temperature or 0.4,
                top_p=request.top_p or 0.9
            )
            
            if not result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Generation failed: {result.get('error', 'Unknown error')}"
                )
            
            return Response(
                content=result["audio_data"],
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=orpheus_speech.wav",
                    "X-Duration": str(result["duration"]),
                    "X-Processing-Time": str(result["processing_time"]),
                    "X-Voice": result["voice_used"],
                    "X-Request-ID": result["request_id"],
                    "X-Streaming": "false"
                }
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/health")
async def health_endpoint():
    """Health check"""
    server = OrpheusStreamingServer()
    return server.health_check.remote()

@web_app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Serve WebSocket demo page"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Orpheus TTS WebSocket Streaming Demo</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
        }
        
        input, select, textarea {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            font-size: 16px;
            box-sizing: border-box;
        }
        
        textarea {
            height: 100px;
            resize: vertical;
        }
        
        button {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        button:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }
        
        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            font-weight: bold;
            text-align: center;
        }
        
        .status.connected {
            background: rgba(46, 204, 113, 0.3);
            border: 2px solid #2ecc71;
        }
        
        .status.disconnected {
            background: rgba(231, 76, 60, 0.3);
            border: 2px solid #e74c3c;
        }
        
        .status.generating {
            background: rgba(241, 196, 15, 0.3);
            border: 2px solid #f1c40f;
        }
        
        .audio-controls {
            text-align: center;
            margin: 20px 0;
        }
        
        .progress {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            height: 20px;
            margin: 20px 0;
            overflow: hidden;
        }
        
        .progress-bar {
            background: linear-gradient(45deg, #3498db, #2980b9);
            height: 100%;
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        
        .log {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 15px;
            height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            margin-top: 20px;
        }
        
        .log-entry {
            margin-bottom: 5px;
            padding: 2px 0;
        }
        
        .log-entry.info { color: #3498db; }
        .log-entry.success { color: #2ecc71; }
        .log-entry.error { color: #e74c3c; }
        .log-entry.warning { color: #f39c12; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Orpheus TTS WebSocket Streaming</h1>
        
        <div class="input-group">
            <label for="wsUrl">WebSocket URL:</label>
            <input type="text" id="wsUrl" value="wss://user-oawolrw--orpheus-tts-streaming-fastapi-app.modal.run/ws/stream" placeholder="Enter your Modal WebSocket URL">
        </div>
        
        <div class="input-group">
            <label for="textInput">Text to Synthesize:</label>
            <textarea id="textInput" placeholder="Enter the text you want to convert to speech...">Hello there! This is a test of real-time WebSocket audio streaming with Orpheus TTS. The audio should start playing as soon as the first chunks are generated, giving you a much faster and more responsive experience.</textarea>
        </div>
        
        <div class="input-group">
            <label for="voiceSelect">Voice:</label>
            <select id="voiceSelect">
                <option value="tara">Tara</option>
                <option value="leah">Leah</option>
                <option value="jess">Jess</option>
                <option value="leo">Leo</option>
                <option value="dan">Dan</option>
                <option value="mia">Mia</option>
                <option value="zac">Zac</option>
                <option value="zoe">Zoe</option>
            </select>
        </div>
        
        <div class="audio-controls">
            <button id="connectBtn" onclick="connectWebSocket()">Connect</button>
            <button id="disconnectBtn" onclick="disconnectWebSocket()" disabled>Disconnect</button>
            <button id="generateBtn" onclick="generateSpeech()" disabled>Generate Speech</button>
            <button id="stopBtn" onclick="stopAudio()" disabled>Stop Audio</button>
        </div>
        
        <div id="status" class="status disconnected">Disconnected</div>
        
        <div class="progress">
            <div id="progressBar" class="progress-bar"></div>
        </div>
        
        <div class="audio-controls">
            <audio id="audioPlayer" controls style="width: 100%; margin: 20px 0;"></audio>
        </div>
        
        <div class="log" id="log">
            <div class="log-entry info">[INFO] Ready to connect...</div>
        </div>
    </div>

    <script>
        let ws = null;
        let audioContext = null;
        let audioSource = null;
        let audioBuffer = [];
        let isPlaying = false;
        let mediaStream = null;
        
        // Audio streaming setup
        async function initAudioContext() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 24000
                });
                log('Audio context initialized', 'success');
            }
            
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
        }
        
        function connectWebSocket() {
            const url = document.getElementById('wsUrl').value;
            if (!url) {
                log('Please enter a WebSocket URL', 'error');
                return;
            }
            
            try {
                ws = new WebSocket(url);
                
                ws.onopen = function(event) {
                    log('WebSocket connected successfully', 'success');
                    updateStatus('connected', 'Connected');
                    document.getElementById('connectBtn').disabled = true;
                    document.getElementById('disconnectBtn').disabled = false;
                    document.getElementById('generateBtn').disabled = false;
                };
                
                ws.onmessage = function(event) {
                    if (typeof event.data === 'string') {
                        // JSON message
                        const message = JSON.parse(event.data);
                        handleStatusMessage(message);
                    } else {
                        // Binary audio data
                        handleAudioChunk(event.data);
                    }
                };
                
                ws.onclose = function(event) {
                    log('WebSocket disconnected', 'warning');
                    updateStatus('disconnected', 'Disconnected');
                    document.getElementById('connectBtn').disabled = false;
                    document.getElementById('disconnectBtn').disabled = true;
                    document.getElementById('generateBtn').disabled = true;
                    ws = null;
                };
                
                ws.onerror = function(error) {
                    log('WebSocket error: ' + error, 'error');
                    updateStatus('disconnected', 'Connection Error');
                };
                
            } catch (error) {
                log('Failed to connect: ' + error.message, 'error');
            }
        }
        
        function disconnectWebSocket() {
            if (ws) {
                ws.close();
                stopAudio();
            }
        }
        
        async function generateSpeech() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                log('WebSocket not connected', 'error');
                return;
            }
            
            const text = document.getElementById('textInput').value.trim();
            if (!text) {
                log('Please enter some text', 'error');
                return;
            }
            
            const voice = document.getElementById('voiceSelect').value;
            
            // Clear previous audio
            stopAudio();
            audioBuffer = [];
            
            // Initialize audio context
            await initAudioContext();
            
            // Send generation request
            const request = {
                text: text,
                voice: voice,
                repetition_penalty: 1.1,
                max_tokens: 2000,
                temperature: 0.4,
                top_p: 0.9
            };
            
            ws.send(JSON.stringify(request));
            log('Requesting speech generation: "' + text.substring(0, 50) + '..." (voice: ' + voice + ')', 'info');
            
            document.getElementById('generateBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
        }
        
        function handleStatusMessage(message) {
            switch (message.status) {
                case 'generating':
                    log('Generating speech for: "' + message.text.substring(0, 50) + '..."', 'info');
                    updateStatus('generating', 'Generating Speech...');
                    break;
                    
                case 'streaming':
                    log('Streaming audio chunks: ' + message.chunks_sent, 'info');
                    updateProgress(message.chunks_sent);
                    break;
                    
                case 'complete':
                    log('Speech generation complete! Total chunks: ' + message.total_chunks, 'success');
                    updateStatus('connected', 'Generation Complete');
                    document.getElementById('generateBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    updateProgress(100);
                    break;
                    
                case 'error':
                    log('Generation error: ' + message.error, 'error');
                    updateStatus('connected', 'Error - Ready');
                    document.getElementById('generateBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    break;
            }
        }
        
        async function handleAudioChunk(audioData) {
            try {
                // Convert ArrayBuffer to AudioBuffer and play immediately
                const arrayBuffer = await audioData.arrayBuffer();
                
                // Skip WAV header for subsequent chunks (first 44 bytes for first chunk)
                const dataStart = audioBuffer.length === 0 ? 44 : 0;
                const pcmData = arrayBuffer.slice(dataStart);
                
                if (pcmData.byteLength > 0) {
                    // Convert PCM data to Float32Array
                    const int16Array = new Int16Array(pcmData);
                    const float32Array = new Float32Array(int16Array.length);
                    
                    for (let i = 0; i < int16Array.length; i++) {
                        float32Array[i] = int16Array[i] / 32768.0;
                    }
                    
                    audioBuffer.push(float32Array);
                    
                    // Start playing if this is the first chunk
                    if (audioBuffer.length === 1 && !isPlaying) {
                        await startAudioPlayback();
                    }
                }
                
            } catch (error) {
                log('Error processing audio chunk: ' + error.message, 'error');
            }
        }
        
        async function startAudioPlayback() {
            if (!audioContext || audioBuffer.length === 0) return;
            
            try {
                isPlaying = true;
                log('Starting real-time audio playback', 'success');
                
                // Create a continuous audio stream
                await playAudioBuffer();
                
            } catch (error) {
                log('Error starting audio playback: ' + error.message, 'error');
                isPlaying = false;
            }
        }
        
        async function playAudioBuffer() {
            let bufferIndex = 0;
            let nextStartTime = audioContext.currentTime;
            
            const playNextChunk = () => {
                if (bufferIndex < audioBuffer.length && isPlaying) {
                    const chunkData = audioBuffer[bufferIndex];
                    
                    // Create AudioBuffer
                    const audioBufferSource = audioContext.createBuffer(1, chunkData.length, 24000);
                    audioBufferSource.getChannelData(0).set(chunkData);
                    
                    // Create source node
                    const sourceNode = audioContext.createBufferSource();
                    sourceNode.buffer = audioBufferSource;
                    sourceNode.connect(audioContext.destination);
                    
                    // Schedule playback
                    sourceNode.start(nextStartTime);
                    nextStartTime += audioBufferSource.duration;
                    
                    bufferIndex++;
                    
                    // Schedule next chunk
                    setTimeout(playNextChunk, audioBufferSource.duration * 1000 * 0.8); // Slight overlap
                } else if (bufferIndex >= audioBuffer.length) {
                    // Check for more chunks periodically
                    setTimeout(() => {
                        if (bufferIndex < audioBuffer.length && isPlaying) {
                            playNextChunk();
                        } else if (isPlaying) {
                            // Keep checking for more chunks
                            setTimeout(playNextChunk, 100);
                        }
                    }, 100);
                }
            };
            
            playNextChunk();
        }
        
        function stopAudio() {
            isPlaying = false;
            if (audioSource) {
                audioSource.stop();
                audioSource = null;
            }
            audioBuffer = [];
            log('Audio playback stopped', 'warning');
        }
        
        function updateStatus(type, message) {
            const statusDiv = document.getElementById('status');
            statusDiv.className = 'status ' + type;
            statusDiv.textContent = message;
        }
        
        function updateProgress(chunks) {
            const progressBar = document.getElementById('progressBar');
            const progress = Math.min(chunks, 100);
            progressBar.style.width = progress + '%';
        }
        
        function log(message, type) {
            type = type || 'info';
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry ' + type;
            logEntry.textContent = '[' + timestamp + '] ' + message;
            logDiv.appendChild(logEntry);
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            log('WebSocket TTS Demo initialized', 'success');
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@web_app.get("/voices")
async def voices_endpoint():
    """Get available voices"""
    return {
        "voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
        "default_voice": "tara",
        "streaming_support": True,
        "websocket_support": True,
        "parameters": {
            "repetition_penalty": {"default": 1.1, "range": [1.0, 2.0]},
            "max_tokens": {"default": 2000, "range": [100, 4000]},
            "temperature": {"default": 0.4, "range": [0.1, 1.0]},
            "top_p": {"default": 0.9, "range": [0.1, 1.0]}
        }
    }

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🔌 WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"❌ WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_audio_chunk(self, websocket: WebSocket, data: bytes):
        try:
            await websocket.send_bytes(data)
        except Exception as e:
            print(f"Error sending audio chunk: {e}")
            self.disconnect(websocket)

    async def send_message(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending message: {e}")
            self.disconnect(websocket)

manager = ConnectionManager()

@web_app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming"""
    await manager.connect(websocket)
    server = OrpheusStreamingServer()
    
    try:
        while True:
            # Wait for text message from client
            message = await websocket.receive_text()
            data = json.loads(message)
            
            text = data.get('text', '')
            voice = data.get('voice', 'tara')
            repetition_penalty = data.get('repetition_penalty', 1.1)
            max_tokens = data.get('max_tokens', 2000)
            temperature = data.get('temperature', 0.4)
            top_p = data.get('top_p', 0.9)
            
            if not text:
                await manager.send_message(websocket, {"error": "No text provided"})
                continue
            
            # Send generation started message
            await manager.send_message(websocket, {
                "status": "generating", 
                "text": text, 
                "voice": voice
            })
            
            try:
                # Generate speech stream
                stream_generator = server.generate_speech_stream.remote_gen(
                    text=text,
                    voice=voice,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                chunk_count = 0
                for chunk in stream_generator:
                    chunk_count += 1
                    await manager.send_audio_chunk(websocket, chunk)
                    
                    # Send progress update every 10 chunks
                    if chunk_count % 10 == 0:
                        await manager.send_message(websocket, {
                            "status": "streaming", 
                            "chunks_sent": chunk_count
                        })
                
                # Send completion message
                await manager.send_message(websocket, {
                    "status": "complete", 
                    "total_chunks": chunk_count
                })
                
            except Exception as e:
                await manager.send_message(websocket, {
                    "status": "error", 
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except json.JSONDecodeError:
        await manager.send_message(websocket, {"error": "Invalid JSON"})
    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.send_message(websocket, {"error": str(e)})
        manager.disconnect(websocket)

# Deploy web app with CPU image (won't crash on import)
@app.function(
    image=orpheus_image_cpu,
    allow_concurrent_inputs=10,
    scaledown_window=900,
    memory=4096,
    cpu=2
)
@modal.asgi_app()
def fastapi_app():
    return web_app

# Test streaming functionality
@app.function(image=orpheus_image_gpu, gpu="A10G")
def test_streaming():
    """Test streaming functionality"""
    print("🧪 Testing streaming functionality...")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    try:
        server = OrpheusStreamingServer()
        
        # Test streaming generation
        print("🎵 Testing stream generation...")
        chunk_count = 0
        total_bytes = 0
        
        stream_generator = server.generate_speech_stream.remote_gen(
            text="This is a test of streaming speech synthesis.",
            voice="tara",
            max_tokens=500
        )
        
        start_time = time.time()
        first_chunk_time = None
        
        for chunk in stream_generator:
            if first_chunk_time is None:
                first_chunk_time = time.time()
                print(f"⚡ First chunk received in {first_chunk_time - start_time:.2f}s")
            
            chunk_count += 1
            total_bytes += len(chunk)
            
            if chunk_count <= 5:  # Log first few chunks
                print(f"📦 Chunk {chunk_count}: {len(chunk)} bytes")
        
        total_time = time.time() - start_time
        
        print(f"✅ Streaming test completed:")
        print(f"   📊 Total chunks: {chunk_count}")
        print(f"   📈 Total bytes: {total_bytes}")
        print(f"   ⏱️ Total time: {total_time:.2f}s")
        print(f"   ⚡ Time to first chunk: {first_chunk_time - start_time:.2f}s")
        
        return {
            "test_status": "passed",
            "chunks_received": chunk_count,
            "total_bytes": total_bytes,
            "total_time": total_time,
            "time_to_first_chunk": first_chunk_time - start_time
        }
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return {"test_status": "failed", "error": str(e)}

if __name__ == "__main__":
    print("""
    🚀 Orpheus TTS Streaming Modal Server
    
    ✨ NEW FEATURES:
    🔥 Real-time streaming audio generation
    ⚡ Low-latency first chunk delivery
    🎵 Multiple streaming endpoints
    🔧 Configurable generation parameters
    📊 Streaming metrics and monitoring
    
    🧪 TESTING:
    
    modal run orpheus_modal_streaming.py::test_streaming
    
    🚀 DEPLOYMENT:
    
    modal deploy orpheus_modal_streaming.py
    
    📡 STREAMING ENDPOINTS:
    
    1. POST /stream - Full streaming with JSON body
    2. GET /tts?prompt=text&voice=tara - Simple GET streaming
    3. POST /generate - Choose streaming or non-streaming
    
    📈 Benefits:
    - Faster perceived response times
    - Better user experience for long text
    - Real-time audio playback capability
    - Reduced memory usage for large audio files
    """)

@app.local_entrypoint()
def main():
    """Test streaming functionality"""
    print("🧪 Testing streaming TTS server...")
    
    result = test_streaming.remote()
    print(f"📊 Streaming test result: {result}")
    
    return result

import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

from .utils import ensure_hf_login
from .tts_engine import OrpheusTTSEngine

load_dotenv(".env")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI(title="Orpheus 3B TTS (Runpod / vLLM)")

# Initialize on startup
engine: OrpheusTTSEngine | None = None

@app.on_event("startup")
async def _startup():
    global engine
    ensure_hf_login()
    engine = OrpheusTTSEngine()

@app.get("/healthz")
async def healthz():
    return {"ok": True}

class TTSIn(BaseModel):
    text: str
    voice: str = "tara"
    stream: bool = True
    chunk_chars: int = 500
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    model_config = {"extra": "forbid"}

def _gen_kwargs(req: TTSIn):
    d = {}
    if req.temperature is not None: d["temperature"] = req.temperature
    if req.top_p is not None: d["top_p"] = req.top_p
    if req.repetition_penalty is not None: d["repetition_penalty"] = req.repetition_penalty
    if req.seed is not None: d["seed"] = req.seed
    return d

@app.post("/tts")
async def tts(req: TTSIn):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    gen_kwargs = _gen_kwargs(req)

    if req.stream:
        async def streamer():
            loop = asyncio.get_event_loop()
            for chunk in engine.stream_pcm_chunks(req.text, req.voice, req.chunk_chars, **gen_kwargs):
                # yield raw PCM16 bytes. Client must know sr=24000, mono.
                yield chunk
                await asyncio.sleep(0)  # cooperative scheduling
        headers = {
            "X-Audio-Sample-Rate": "24000",
            "X-Audio-Format": "pcm_s16le",
            "X-Voice": req.voice,
        }
        return StreamingResponse(streamer(), media_type="application/octet-stream", headers=headers)

    else:
        wav_bytes = engine.synthesize_wav_bytes(req.text, req.voice, req.chunk_chars, **gen_kwargs)
        return Response(content=wav_bytes, media_type="audio/wav")



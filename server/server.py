import os
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from dotenv import load_dotenv

from .utils import ensure_hf_login
from .engine_vllm import OrpheusTTSEngine

load_dotenv(".env")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
SAMPLE_RATE = 24000

app = FastAPI(title="Orpheus 3B TTS (Runpod / vLLM+SNAC)")

engine: OrpheusTTSEngine | None = None

@app.on_event("startup")
async def _startup():
    global engine
    ensure_hf_login()
    engine = OrpheusTTSEngine()

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/tts")
async def tts(req: Dict):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    text = (req.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    voice = req.get("voice", "tara")
    stream = bool(req.get("stream", True))
    chunk_chars = int(req.get("chunk_chars", 500))
    temperature = req.get("temperature")
    top_p = req.get("top_p")
    repetition_penalty = req.get("repetition_penalty")

    if stream:
        def pcm_iter():
            for chunk in engine.stream_pcm_chunks(
                text=text,
                voice=voice,
                chunk_chars=chunk_chars,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ):
                if chunk:
                    yield chunk

        headers = {
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Format": "pcm_s16le",
            "X-Voice": voice,
        }
        return StreamingResponse(pcm_iter(), media_type="application/octet-stream", headers=headers)

    # Non-streaming → WAV
    wav_bytes = engine.synthesize_wav_bytes(
        text=text,
        voice=voice,
        chunk_chars=chunk_chars,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return Response(content=wav_bytes, media_type="audio/wav")



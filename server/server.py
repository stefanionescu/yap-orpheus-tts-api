import os
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
from dotenv import load_dotenv

from .utils import ensure_hf_login
from .engine_vllm import OrpheusTTSEngine
from .snac_stream import SnacDecoder
from .prompts import resolve_voice

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

@app.websocket("/ws/tts")
async def tts_ws(ws: WebSocket):
    """
    Bi-directional streaming over WebSocket:
      - Client sends JSON messages:
          {"text": "partial text"}
          {"end": true}
      - Server sends PCM16 audio chunks as binary frames.
    """
    await ws.accept()
    global engine
    if engine is None:
        await ws.close(code=1013)
        return

    # Per-connection decoder for continuity across messages
    decoder = SnacDecoder(sample_rate=SAMPLE_RATE)
    decode_frames = int(os.getenv("SNAC_DECODE_FRAMES", "5"))

    # Small queue for inbound text pieces
    q: asyncio.Queue[Optional[dict]] = asyncio.Queue()

    async def recv_loop():
        while True:
            msg = await ws.receive_text()
            try:
                obj = json.loads(msg)
            except Exception:
                obj = {"text": msg}
            if obj.get("end") is True:
                await q.put(None)
                break
            t = (obj.get("text") or "").strip()
            v = (obj.get("voice") or "").strip()
            if t:
                await q.put({"text": t, "voice": v})

    async def synth_loop():
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                # Stream frames for this text piece and decode in-session
                piece = item.get("text", "")
                voice = resolve_voice(item.get("voice", "tara")) or "tara"
                batch = []
                async for frame in engine.aiter_frames(piece, voice=voice):
                    batch.append(frame)
                    if len(batch) >= decode_frames:
                        decoder.add_frames(batch)
                        batch.clear()
                        pcm = decoder.take_new_pcm16()
                        if pcm:
                            await ws.send_bytes(pcm)
                            await asyncio.sleep(0)
                if batch:
                    decoder.add_frames(batch)
                    batch.clear()
                    pcm = decoder.take_new_pcm16()
                    if pcm:
                        await ws.send_bytes(pcm)
                        await asyncio.sleep(0)
        except Exception as e:
            try:
                await ws.close(code=1011)
            except Exception:
                pass
            raise e

    try:
        recv_task = asyncio.create_task(recv_loop())
        synth_task = asyncio.create_task(synth_loop())
        await asyncio.gather(recv_task, synth_task)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


"""Refactored FastAPI server with clean modular architecture."""

import asyncio
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv

from server.auth.utils import ensure_hf_login
from server.config import settings
from server.engine import OrpheusTRTEngine as _Engine
from server.websocket_handlers import message_receiver, ConnectionState
from server.synthesis_pipeline import SynthesisPipeline

load_dotenv(".env")

HOST = settings.host
PORT = settings.port
SAMPLE_RATE = settings.snac_sr

app = FastAPI(title="Orpheus 3B TTS API for Yap")

engine: _Engine | None = None


@app.on_event("startup")
async def _startup():
    global engine
    ensure_hf_login()
    engine = _Engine()

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.websocket("/ws/tts")
async def tts_ws(ws: WebSocket):
    """
    Sentence-by-sentence WS:
      - Client sends {"text": "...sentence or full text...", "voice": "..."}.
      - Server splits text into sentences only (no word-based chunking).
      - For each sentence, runs one generation and streams audio hops.
      - Strict in-order emission; no timers, no word buffering.
    """
    await ws.accept()
    global engine
    if engine is None:
        await ws.close(code=settings.ws_close_busy_code)
        return

    # Initialize clean handlers
    message_queue: asyncio.Queue[Optional[dict]] = asyncio.Queue()
    connection_state = ConnectionState()
    synthesis_pipeline = SynthesisPipeline(engine)

    async def synthesis_handler():
        """Handle synthesis requests from message queue."""
        try:
            while True:
                message = await message_queue.get()
                if message is None or message.get("type") == "end":
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    break
                
                if message.get("type") == "meta":
                    connection_state.update_from_meta(message.get("meta", {}))
                    continue
                
                if message.get("type") == "text":
                    text = message.get("text", "").strip()
                    voice_override = message.get("voice")
                    
                    if voice_override:
                        connection_state.voice = str(voice_override)
                    
                    if not text:
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break
                    
                    # Synthesize and stream
                    sampling_kwargs = connection_state.get_sampling_kwargs()
                    await synthesis_pipeline.synthesize_text(text, connection_state.voice, sampling_kwargs, ws)
                    
        except Exception as e:
            try:
                await ws.close(code=settings.ws_close_internal_code)
            except Exception:
                pass
            raise e

    try:
        recv_task = asyncio.create_task(message_receiver(ws, message_queue))
        synth_task = asyncio.create_task(synthesis_handler())
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


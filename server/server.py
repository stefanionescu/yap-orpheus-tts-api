"""Refactored FastAPI server with clean modular architecture."""

import asyncio
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv

from server.auth.utils import ensure_hf_login, extract_api_key_from_ws_headers, is_api_key_authorized
from server.config import settings
from server.engine import OrpheusTRTEngine as _Engine
from server.streaming.websocket_handlers import message_receiver, ConnectionState
from server.streaming.synthesis_pipeline import SynthesisPipeline

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
    # API key check via auth utility
    provided_key = extract_api_key_from_ws_headers(ws.headers.raw)
    if not is_api_key_authorized(provided_key):
        # Unauthorized
        try:
            await ws.close(code=1008)
        except Exception:
            pass
        return

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
                    try:
                        connection_state.update_from_meta(message.get("meta", {}))
                    except ValueError:
                        # Invalid voice parameter - close connection
                        try:
                            await ws.close(code=1008)  # Policy violation
                        except Exception:
                            pass
                        break
                    continue
                
                if message.get("type") == "text":
                    text = message.get("text", "").strip()
                    voice_override = message.get("voice")
                    
                    # Voice must always be provided and valid for each text message
                    try:
                        voice_str = str(voice_override)
                        from server.voices import resolve_voice
                        resolve_voice(voice_str)  # Raises ValueError if invalid/missing
                        connection_state.voice = voice_str
                    except Exception:
                        try:
                            await ws.close(code=1008)
                        except Exception:
                            pass
                        break
                    
                    if not text:
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break
                    
                    # Build sampling kwargs with per-message overrides
                    sampling_kwargs = connection_state.get_sampling_kwargs()
                    
                    # Allow per-message generation parameter overrides
                    if "temperature" in message:
                        try:
                            sampling_kwargs["temperature"] = float(message["temperature"])
                        except Exception:
                            pass
                    
                    if "top_p" in message:
                        try:
                            sampling_kwargs["top_p"] = float(message["top_p"])
                        except Exception:
                            pass
                    
                    if "repetition_penalty" in message:
                        try:
                            sampling_kwargs["repetition_penalty"] = float(message["repetition_penalty"])
                        except Exception:
                            pass
                    
                    trim_flag = connection_state.trim_silence
                    # Allow per-message trim_silence override when provided
                    if "trim_silence" in message:
                        try:
                            val = message["trim_silence"]
                            if isinstance(val, bool):
                                trim_flag = val
                            elif isinstance(val, str):
                                trim_flag = val.strip().lower() in {"1", "true", "yes", "y", "on"}
                            else:
                                trim_flag = bool(int(val))
                        except Exception:
                            pass
                    await synthesis_pipeline.synthesize_text(text, connection_state.voice, sampling_kwargs, ws, trim_silence=trim_flag)
                    
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

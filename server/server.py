import os
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
from dotenv import load_dotenv
import pysbd
import numpy as np

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

# ---- tiny per-process ramp state to avoid clicks between PCM chunks ----
_RAMP_MS = int(os.getenv("AUDIO_RAMP_MS", "5"))  # 5 ms by default

class _RampState:
    def __init__(self):
        self.tail = b""
        self.did_lead_skip = False

_ramp_state = _RampState()

def _int16_to_f32(b: bytes):
    a = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32767.0
    return a

def _f32_to_int16(a: np.ndarray) -> bytes:
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767.0).astype(np.int16).tobytes()

def apply_chunk_ramp(pcm_bytes: bytes, sr: int, first_chunk: bool = False, state: _RampState | None = None) -> bytes:
    if not pcm_bytes:
        return pcm_bytes

    a = _int16_to_f32(pcm_bytes)
    st = state or _ramp_state

    # One-time tiny startup lead skip to hide initial crackle
    if first_chunk and not st.did_lead_skip:
        lead = max(1, int(0.005 * sr))  # 5 ms
        if a.shape[0] > lead:
            a = a[lead:]
        st.did_lead_skip = True

    ramp_len = max(1, int((_RAMP_MS / 1000.0) * sr))

    # Fade-in current chunk
    if a.shape[0] >= ramp_len:
        fade_in = np.linspace(0.0, 1.0, ramp_len, endpoint=True, dtype=np.float32)
        a[:ramp_len] *= fade_in

    # Overlap-add with previous tail if present
    if st.tail:
        prev = _int16_to_f32(st.tail)
        n = min(prev.shape[0], a.shape[0])
        if n > 0:
            fade_out = np.linspace(1.0, 0.0, n, endpoint=True, dtype=np.float32)
            fade_in2 = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
            a[:n] = prev[:n] * fade_out + a[:n] * fade_in2

    # Save new tail
    tail_len = min(ramp_len, a.shape[0])
    st.tail = _f32_to_int16(a[-tail_len:]) if tail_len > 0 else b""

    return _f32_to_int16(a)

@app.on_event("startup")
async def _startup():
    global engine
    ensure_hf_login()
    engine = OrpheusTTSEngine()
    # Background warmup to reduce first-request TTFB (compile kernels, allocate KV cache)
    async def _warmup():
        try:
            # Small prompts per voice to prime model + SNAC path
            async def _prime(voice: str):
                # Let a short generation complete naturally to avoid abort logs
                async for _ in engine.aiter_frames("hello", voice=voice, num_predict=64):
                    pass
            await asyncio.gather(_prime("tara"), _prime("zac"))
        except Exception:
            # Do not fail startup if warmup errors
            pass
    asyncio.create_task(_warmup())

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.websocket("/ws/tts")
async def tts_ws(ws: WebSocket):
    """
    Bi-directional streaming over WebSocket. Supports two client patterns:
      1) Baseten-like: send one JSON metadata object (voice/max_tokens/buffer_size/..),
         then stream plain text words, finally send the string "__END__".
      2) Legacy JSON: send {"text": "...", "voice": "..."} one or more times,
         then send {"end": true}.
    Server responds with PCM16 audio chunks as binary frames.
    """
    await ws.accept()
    global engine
    if engine is None:
        await ws.close(code=1013)
        return

    # Per-connection decoder for continuity across messages
    decoder = SnacDecoder(sample_rate=SAMPLE_RATE)
    decode_frames = int(os.getenv("SNAC_DECODE_FRAMES", "2"))

    # State and queues
    q: asyncio.Queue[Optional[dict]] = asyncio.Queue()

    async def recv_loop():
        while True:
            msg = await ws.receive_text()
            # Baseten-style END sentinel
            if msg.strip() == "__END__":
                await q.put(None)
                break
            # Try JSON parse; fall back to treating as a word/phrase
            try:
                obj = json.loads(msg)
            except Exception:
                obj = None

            if isinstance(obj, dict):
                if obj.get("end") is True:
                    await q.put(None)
                    break
                # Metadata-only message (no text) → update state
                if ("text" not in obj) and (
                    any(k in obj for k in ("voice", "max_tokens", "temperature", "top_p", "repetition_penalty", "buffer_size"))
                ):
                    await q.put({"type": "meta", "meta": obj})
                    continue
                # Text message (may include voice override)
                t = (obj.get("text") or "").strip()
                v = obj.get("voice")
                if t:
                    await q.put({"type": "text", "text": t, "voice": v})
                continue

            # Plain text message (treat as a single word or phrase)
            t = msg.strip()
            if t:
                await q.put({"type": "word", "word": t})

    async def synth_loop():
        # Connection-local settings
        voice = "tara"
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        repetition_penalty: Optional[float] = None
        num_predict: Optional[int] = None  # a.k.a. max_tokens
        buf_sz = int(os.getenv("WS_WORD_BUFFER_SIZE", os.getenv("BUFFER_SIZE", "40")))  # safety valve only

        splitter = pysbd.Segmenter(language="en", clean=False)
        text_buffer: list[str] = []

        sent_any_audio = False
        ramp_state = _RampState()

        async def flush(final: bool = False):
            nonlocal sent_any_audio
            if not text_buffer:
                return
            full_text = " ".join(text_buffer)
            sentences = splitter.segment(full_text)
            prompt = None
            words_consumed = 0
            if len(sentences) > 1:
                # Flush all complete sentences except the last unfinished one
                complete_sents = sentences[:-1]
                prompt = " ".join(complete_sents)
                words_consumed = sum(len(s.split()) for s in complete_sents)
            # Fallback: if we buffered enough words, flush that many regardless of sentence boundary
            elif len(text_buffer) >= buf_sz:
                prompt = " ".join(text_buffer[:buf_sz])
                words_consumed = buf_sz
            elif final:
                prompt = " ".join(text_buffer)
                words_consumed = len(text_buffer)
            else:
                return

            # Consume from buffer
            del text_buffer[:words_consumed]

            # Resolve voice alias each flush (voice may have been updated)
            v = resolve_voice(voice) or "tara"

            # Stream frames for this prompt and decode in-session
            frames_batch = []
            PRIME_FRAMES = int(os.getenv("SNAC_PRIME_FRAMES", "2"))
            primed = False
            async for frame in engine.aiter_frames(
                prompt,
                voice=v,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_predict=num_predict,
            ):
                frames_batch.append(frame)
                # Hold first PCM until primed with a few frames
                if not primed and len(frames_batch) < PRIME_FRAMES:
                    continue
                if not primed:
                    decoder.add_frames(frames_batch)
                    frames_batch.clear()
                    pcm = decoder.take_new_pcm16()
                    if pcm:
                        pcm = apply_chunk_ramp(pcm, SAMPLE_RATE, first_chunk=(not sent_any_audio), state=ramp_state)
                        await ws.send_bytes(pcm)
                        sent_any_audio = True
                        await asyncio.sleep(0)
                    primed = True
                    continue
                # Smaller cadence for smoother flow
                local_decode_frames = int(os.getenv("SNAC_DECODE_FRAMES", "2"))
                if len(frames_batch) >= local_decode_frames:
                    decoder.add_frames(frames_batch)
                    frames_batch.clear()
                    pcm = decoder.take_new_pcm16()
                    if pcm:
                        pcm = apply_chunk_ramp(pcm, SAMPLE_RATE, state=ramp_state)
                        await ws.send_bytes(pcm)
                        await asyncio.sleep(0)
            if frames_batch:
                decoder.add_frames(frames_batch)
                frames_batch.clear()
                pcm = decoder.take_new_pcm16()
                if pcm:
                    pcm = apply_chunk_ramp(pcm, SAMPLE_RATE, state=ramp_state)
                    await ws.send_bytes(pcm)
                    await asyncio.sleep(0)

            if final:
                try:
                    await ws.close()
                except Exception:
                    pass

        try:
            while True:
                item = await q.get()
                if item is None:
                    await flush(final=True)
                    break
                typ = item.get("type")
                if typ == "meta":
                    m = item.get("meta", {})
                    if "voice" in m and m["voice"]:
                        voice = str(m["voice"])  # store raw; resolve later
                    if "buffer_size" in m:
                        try:
                            buf_sz = max(1, int(m["buffer_size"]))
                        except Exception:
                            pass
                    if "temperature" in m:
                        try:
                            temperature = float(m["temperature"])  # None allowed
                        except Exception:
                            pass
                    if "top_p" in m:
                        try:
                            top_p = float(m["top_p"])  # None allowed
                        except Exception:
                            pass
                    if "repetition_penalty" in m:
                        try:
                            repetition_penalty = float(m["repetition_penalty"])  # None allowed
                        except Exception:
                            pass
                    if "max_tokens" in m:
                        try:
                            num_predict = int(m["max_tokens"])  # None allowed
                        except Exception:
                            pass
                    continue
                if typ == "text":
                    t = item.get("text", "").strip()
                    v = item.get("voice")
                    if v:
                        voice = str(v)
                    if t:
                        text_buffer.extend(t.split())
                        await flush(final=False)
                    continue
                if typ == "word":
                    w = item.get("word", "").strip()
                    if w:
                        text_buffer.append(w)
                        await flush(final=False)
                    continue
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


import os
import re
from typing import Optional, Iterator, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
from dotenv import load_dotenv
import pysbd
import numpy as np
import torch

from .utils import ensure_hf_login
from .engine_vllm import OrpheusTTSEngine
from .prompts import build_prompt, resolve_voice
from vllm import SamplingParams
from vllm.utils import random_uuid
from snac import SNAC

load_dotenv(".env")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
SAMPLE_RATE = 24000

app = FastAPI(title="Orpheus 3B TTS (Runpod / vLLM+SNAC)")

engine: OrpheusTTSEngine | None = None

# ---- tiny first-chunk fade only (no overlap) ----
def apply_first_chunk_fade(pcm_bytes: bytes, sr: int) -> bytes:
    if not pcm_bytes:
        return pcm_bytes
    a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    ramp_len = max(1, int(0.003 * sr))  # 3 ms
    if a.shape[0] >= ramp_len:
        fade = np.linspace(0.0, 1.0, ramp_len, endpoint=True, dtype=np.float32)
        a[:ramp_len] *= fade
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767.0).astype(np.int16).tobytes()

# --- Baseten-compatible token parsing and SNAC batching ---
_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

SNAC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNAC_MAX_BATCH = 64
PREPROCESS_STREAM = torch.cuda.Stream() if torch.cuda.is_available() else None

_SNAC_BATCHEX = None

def _get_snac_batched():
    global _SNAC_BATCHEX
    if _SNAC_BATCHEX is not None:
        return _SNAC_BATCHEX

    class _SnacBatched:
        def __init__(self):
            self.dtype_decoder = torch.float32
            m = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(SNAC_DEVICE)
            m.decoder = m.decoder.to(self.dtype_decoder)
            if bool(int(os.getenv("SNAC_TORCH_COMPILE", "0"))):
                m.decoder = torch.compile(m.decoder, dynamic=True)
                m.quantizer = torch.compile(m.quantizer, dynamic=True)
            self.m = m
            self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        async def decode_codes(self, codes_triplet: list[torch.Tensor]) -> np.ndarray:
            # codes_triplet shapes: [(1, n), (1, 2n), (1, 4n)]
            with torch.inference_mode():
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        z_q = self.m.quantizer.from_codes(codes_triplet)
                        audio_hat = self.m.decoder(z_q.to(self.dtype_decoder))[:, :, 2048:4096]
                        torch.cuda.synchronize()
                else:
                    z_q = self.m.quantizer.from_codes(codes_triplet)
                    audio_hat = self.m.decoder(z_q.to(self.dtype_decoder))[:, :, 2048:4096]
            return audio_hat[0].detach().cpu().numpy()

    _SNAC_BATCHEX = _SnacBatched()
    return _SNAC_BATCHEX

def _split_custom_tokens(s: str) -> List[int]:
    return [int(x) for x in _TOKEN_RE.findall(s) if x != "0"]

def _turn_token_into_id(token_number: int, index: int) -> int:
    # Baseten’s exact rule
    return token_number - 10 - ((index % 7) * 4096)

async def aiter_pcm_from_custom_tokens(engine, prompt: str, voice: str, sp) -> Iterator[bytes]:
    """
    vLLM → detokenized pieces → <custom_token_…> → 28→PCM, Baseten-identical.
    """
    tok_count = 0
    buf_ids: list[int] = []
    snacx = _get_snac_batched()

    prev_len = 0
    async for out in engine.generate(build_prompt(prompt, resolve_voice(voice)), sp, random_uuid()):
        outs = out.outputs or []
        if not outs:
            continue

        # We MUST detokenize to strings to see <custom_token_…>
        piece = outs[0].text or ""
        if not piece:
            continue

        # Only process delta since last step to avoid double counting
        delta = piece[prev_len:]
        prev_len = len(piece)

        for n in _split_custom_tokens(delta):
            tid = _turn_token_into_id(n, tok_count)
            tok_count += 1
            buf_ids.append(tid)

            # Every 7 tokens is one frame; after 4 frames (28 ids) → decode
            if (tok_count % 7 == 0) and len(buf_ids) >= 28:
                window = buf_ids[-28:]
                arr = np.asarray(window, dtype=np.int32).reshape(-1, 7)
                codes_0 = torch.from_numpy(arr[:, 0]).unsqueeze(0).to(SNAC_DEVICE)
                codes_1 = torch.from_numpy(arr[:, [1, 4]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
                codes_2 = torch.from_numpy(arr[:, [2, 3, 5, 6]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)

                # sync staging stream like Baseten
                if PREPROCESS_STREAM is not None:
                    with torch.cuda.stream(PREPROCESS_STREAM):
                        c0, c1, c2 = codes_0, codes_1, codes_2
                    torch.cuda.synchronize()
                else:
                    c0, c1, c2 = codes_0, codes_1, codes_2

                audio = await snacx.decode_codes([c0, c1, c2])
                pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                if pcm:
                    yield pcm

@app.on_event("startup")
async def _startup():
    global engine
    ensure_hf_login()
    engine = OrpheusTTSEngine()
    # Background warmup to reduce first-request TTFB (compile kernels, allocate KV cache)
    async def _warmup():
        try:
            # Preload SNAC model to avoid first-request latency
            _ = _get_snac_batched()
            # Small prompts per voice to prime model + SNAC path using tokens→PCM
            sp = SamplingParams(
                temperature=0.6,
                top_p=0.8,
                repetition_penalty=1.1,
                max_tokens=64,
                detokenize=True,
                skip_special_tokens=False,
                ignore_eos=False,
                stop_token_ids=[128258, 128009],
            )
            async def _prime(voice: str):
                async for _ in aiter_pcm_from_custom_tokens(engine.engine, "hello", voice, sp):
                    break
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
        buf_sz = int(os.getenv("WS_WORD_BUFFER_SIZE", os.getenv("BUFFER_SIZE", "10")))  # safety valve only

        splitter = pysbd.Segmenter(language="en", clean=False)
        text_buffer: list[str] = []

        # pacing (timer-based flush to avoid idle gaps)
        FLUSH_MS = int(os.getenv("FLUSH_MS", "120"))
        flush_running = False
        stop_flag = False

        async def ticker():
            nonlocal stop_flag
            while not stop_flag:
                await asyncio.sleep(FLUSH_MS / 1000.0)
                if text_buffer and not flush_running:
                    try:
                        await flush(final=False)
                    except Exception:
                        pass

        async def flush(final: bool = False):
            nonlocal flush_running
            if flush_running:
                return
            if not text_buffer and not final:
                return
            flush_running = True
            try:
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
                    # Nothing to do (no complete sentence, below buffer, and not final)
                    return

                # Consume from buffer
                del text_buffer[:words_consumed]

                # Resolve voice alias each flush (voice may have been updated)
                v = resolve_voice(voice) or "tara"

                # Do not start a generation on empty prompt (e.g., final with no text)
                if not prompt or not prompt.strip():
                    if final:
                        try:
                            await ws.close()
                        except Exception:
                            pass
                    return

                # ---- Baseten-compatible streaming from vLLM ----
                sp = SamplingParams(
                    temperature=float(temperature if (temperature is not None) else 0.6),
                    top_p=float(top_p if (top_p is not None) else 0.8),
                    repetition_penalty=float(repetition_penalty if (repetition_penalty is not None) else 1.1),
                    max_tokens=int(num_predict if (num_predict is not None) else 6144),
                    detokenize=True,
                    skip_special_tokens=False,
                    ignore_eos=False,
                    stop_token_ids=[128258, 128009],
                )

                first_chunk = True
                async for pcm in aiter_pcm_from_custom_tokens(engine.engine, prompt, v, sp):
                    if first_chunk:
                        # tiny fade-in (3ms) only on the very first PCM of this flush
                        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
                        r = max(1, int(0.003 * SAMPLE_RATE))
                        if a.shape[0] > r:
                            a[:r] *= np.linspace(0.0, 1.0, r, dtype=np.float32)
                        pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                        first_chunk = False
                    await ws.send_bytes(pcm)
                    await asyncio.sleep(0)

                if final:
                    try:
                        await ws.close()
                    except Exception:
                        pass
            finally:
                flush_running = False

        tick_task = asyncio.create_task(ticker())
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
        finally:
            stop_flag = True
            try:
                tick_task.cancel()
            except Exception:
                pass

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


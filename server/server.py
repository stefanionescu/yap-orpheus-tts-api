import os
import re
from typing import Optional, Iterator, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
from dotenv import load_dotenv
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

# --- Baseten-compatible token parsing and SNAC batching ---
_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

SNAC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREPROCESS_STREAM = (
    torch.cuda.Stream(device=torch.device(SNAC_DEVICE)) if torch.cuda.is_available() else None
)

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

# --- Baseten chunking (pre-prompt formatting) ---
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "280"))

def chunk_text(text: str, max_len: int = MAX_CHUNK_SIZE) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        window = text[start:end]
        split_at = window.rfind("\n")
        if split_at == -1 or split_at < max_len * 0.5:
            split_at = max(window.rfind("."), window.rfind("?"), window.rfind("!"))
            if split_at != -1:
                split_at += 1
        if split_at == -1 or split_at < max_len * 0.33:
            split_at = window.rfind(",")
        if split_at == -1:
            split_at = window.rfind(" ")
        if split_at == -1:
            split_at = len(window)
        chunk = text[start : start + split_at].strip()
        if chunk:
            chunks.append(chunk)
        start += split_at
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks or [""]

def _split_custom_tokens(s: str) -> List[int]:
    return [int(x) for x in _TOKEN_RE.findall(s) if x != "0"]

def _turn_token_into_id(token_number: int, index: int) -> int:
    # Baseten’s exact rule
    return token_number - 10 - ((index % 7) * 4096)

async def aiter_pcm_from_custom_tokens(engine, prompt: str, voice: str, sp) -> Iterator[bytes]:
    """
    vLLM → detokenized pieces → <custom_token_…> → 28→PCM, Baseten-identical.
    Monotonic delta consumption (no rescans, no resets) for artifact-free audio.
    """
    tok_index = 0  # never reset within a single generation
    buf_ids: list[int] = []
    snacx = _get_snac_batched()

    prev_len = 0  # length of detokenized text we have already processed
    async for out in engine.generate(build_prompt(prompt, resolve_voice(voice)), sp, random_uuid()):
        outs = out.outputs or []
        if not outs:
            continue

        piece = outs[0].text or ""
        # Only process newly appended text to keep sequence monotonic
        if len(piece) <= prev_len:
            continue
        delta = piece[prev_len:]
        prev_len = len(piece)

        for n in _split_custom_tokens(delta):
            tid = _turn_token_into_id(n, tok_index)
            tok_index += 1
            buf_ids.append(tid)

            # Every 7 tokens is one frame; after 4 frames (28 ids) → decode
            if (tok_index % 7 == 0) and len(buf_ids) >= 28:
                window = buf_ids[-28:]
                arr = np.asarray(window, dtype=np.int32).reshape(-1, 7)
                codes_0 = torch.from_numpy(arr[:, 0]).unsqueeze(0).to(SNAC_DEVICE)
                codes_1 = torch.from_numpy(arr[:, [1, 4]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
                codes_2 = torch.from_numpy(arr[:, [2, 3, 5, 6]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)

                if PREPROCESS_STREAM is not None:
                    with torch.cuda.stream(PREPROCESS_STREAM):
                        pass
                    torch.cuda.synchronize()

                audio = await snacx.decode_codes([codes_0, codes_1, codes_2])
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
                # Keep warmup short and let it complete naturally to avoid 'Aborted request' logs
                max_tokens=56,
                detokenize=True,
                skip_special_tokens=False,
                ignore_eos=False,
                stop_token_ids=[128258, 128009],
            )
            async def _prime(voice: str):
                # Consume to completion (no early break) so vLLM finishes without aborting
                async for _ in aiter_pcm_from_custom_tokens(engine.engine, "hello", voice, sp):
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
    Baseten-parity WS (best-performance):
      - Client sends a single JSON with {"text": "...full text...", "voice": "..."}.
      - Server splits the original text into ~280-char chunks BEFORE formatting the prompt.
      - For each chunk, runs one generation and streams audio hops from last 28 custom tokens.
      - Strict in-order emission; no timers, no word buffering.
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

            # Plain text message → treat as full text for Baseten mode
            t = msg.strip()
            if t:
                await q.put({"type": "text", "text": t})

    async def synth_loop():
        # Connection-local settings
        voice = "tara"
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        repetition_penalty: Optional[float] = None
        num_predict: Optional[int] = None  # a.k.a. max_tokens

        try:
            while True:
                item = await q.get()
                if item is None:
                    # If no text was ever provided, just close
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    break
                typ = item.get("type")
                if typ == "meta":
                    m = item.get("meta", {})
                    if "voice" in m and m["voice"]:
                        voice = str(m["voice"])  # store raw; resolve later
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
                    full_text = item.get("text", "").strip()
                    v_override = item.get("voice")
                    if v_override:
                        voice = str(v_override)
                    if not full_text:
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break

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

                    v = resolve_voice(voice) or "tara"
                    chunks = chunk_text(full_text, MAX_CHUNK_SIZE)
                    if not chunks:
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        break

                    async def produce_to_queue(text: str, qout: asyncio.Queue):
                        async for pcm in aiter_pcm_from_custom_tokens(engine.engine, text, v, sp):
                            await qout.put(pcm)
                        await qout.put(None)  # sentinel

                    if len(chunks) == 1:
                        # Single chunk: stream directly
                        async for pcm in aiter_pcm_from_custom_tokens(engine.engine, chunks[0], v, sp):
                            await ws.send_bytes(pcm)
                            await asyncio.sleep(0)
                    else:
                        # Pipeline: stream chunk 0 directly, pre-generate chunk 1 into a queue,
                        # then for each subsequent chunk pre-generate the next while draining current queued
                        queued_q = asyncio.Queue(maxsize=128)
                        queued_task = asyncio.create_task(produce_to_queue(chunks[1], queued_q))

                        # Active 0 → socket
                        async for pcm in aiter_pcm_from_custom_tokens(engine.engine, chunks[0], v, sp):
                            await ws.send_bytes(pcm)
                            await asyncio.sleep(0)

                        # Drain queued chunks in order, starting from index 1
                        for idx in range(1, len(chunks)):
                            # Start next queued (idx+1) before draining current queued (idx)
                            next_q = None
                            next_task = None
                            if (idx + 1) < len(chunks):
                                next_q = asyncio.Queue(maxsize=128)
                                next_task = asyncio.create_task(produce_to_queue(chunks[idx + 1], next_q))

                            # Drain current queued_q (chunk idx)
                            while True:
                                b = await queued_q.get()
                                if b is None:
                                    break
                                await ws.send_bytes(b)
                                await asyncio.sleep(0)

                            # Move to next queued
                            queued_q = next_q if next_q is not None else asyncio.Queue()
                            queued_task = next_task

                    try:
                        await ws.close()
                    except Exception:
                        pass
                    break
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


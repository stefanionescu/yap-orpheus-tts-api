import os
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
from dotenv import load_dotenv

from .core.utils import ensure_hf_login

from .core.chunking import chunk_by_words, FIRST_CHUNK_WORDS, NEXT_CHUNK_WORDS, MIN_TAIL_WORDS
from .prompts import resolve_voice

load_dotenv(".env")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
SAMPLE_RATE = 24000

# Backend selection: "vllm" (default) or "trtllm"
BACKEND = os.getenv("BACKEND", "vllm").strip().lower()
if BACKEND == "trtllm":
    # Import TRT backend lazily to avoid requiring TRT deps when not in use
    from .engines.trt_engine import OrpheusTRTEngine as _Engine
    from .streaming.trt_streaming import aiter_pcm_from_custom_tokens
    from tensorrt_llm.llmapi import SamplingParams  # type: ignore
else:
    from .engines.vllm_engine import OrpheusVLLMEngine as _Engine
    from .streaming.vllm_streaming import aiter_pcm_from_custom_tokens
    from vllm import SamplingParams

app = FastAPI(title="Orpheus 3B TTS (Runpod / vLLM+SNAC / TRT-LLM)")

engine: _Engine | None = None


@app.on_event("startup")
async def _startup():
    global engine
    ensure_hf_login()
    engine = _Engine()
    # Background warmup to reduce first-request TTFB (compile kernels, allocate KV cache)
    async def _warmup():
        try:
            # Preload SNAC model to avoid first-request latency
            from .core.snac_batcher import get_snac_batched
            _ = get_snac_batched()
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
      - Server splits text into sentence-safe, word-based chunks (FIRST_CHUNK_WORDS then NEXT_CHUNK_WORDS)
        before formatting the prompt.
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

                    # Build sampling params; add vLLM-only flags when using vLLM backend
                    sp_kwargs = dict(
                        temperature=float(temperature if (temperature is not None) else 0.6),
                        top_p=float(top_p if (top_p is not None) else 0.8),
                        repetition_penalty=float(repetition_penalty if (repetition_penalty is not None) else 1.1),
                        max_tokens=int(num_predict if (num_predict is not None) else int(os.getenv("ORPHEUS_MAX_TOKENS", "2048"))),
                        # EOS + end-of-turn; never include audio start
                        stop_token_ids=[128009, 128260],
                    )
                    if BACKEND == "vllm":
                        sp_kwargs.update(dict(
                            detokenize=True,
                            skip_special_tokens=False,
                            ignore_eos=False,
                        ))
                    sp = SamplingParams(**sp_kwargs)

                    v = resolve_voice(voice) or "tara"
                    chunks = chunk_by_words(
                        full_text,
                        first_chunk_words=FIRST_CHUNK_WORDS,
                        next_chunk_words=NEXT_CHUNK_WORDS,
                        min_tail_words=MIN_TAIL_WORDS,
                    )
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


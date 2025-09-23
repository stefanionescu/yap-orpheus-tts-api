import os
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
from dotenv import load_dotenv

from .core.logging_config import get_logger
from .core.utils import ensure_hf_login
from .engine_selector import OrpheusTTSEngine
try:
    from vllm import SamplingParams  # type: ignore
except Exception:
    # Lightweight fallback for TRT-LLM backend to carry sampling params
    class SamplingParams:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

from .core.chunking import chunk_by_words, FIRST_CHUNK_WORDS, NEXT_CHUNK_WORDS, MIN_TAIL_WORDS
from .prompts import resolve_voice
from .streaming import aiter_pcm_from_custom_tokens

load_dotenv(".env")

# Initialize logging
logger = get_logger(__name__)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
SAMPLE_RATE = 24000

app = FastAPI(title="Orpheus 3B TTS (A100 / TRT-LLM or vLLM + SNAC)")

engine: OrpheusTTSEngine | None = None


@app.on_event("startup")
async def _startup():
    global engine
    logger.info("Starting Orpheus TTS server...")
    try:
        logger.info("Ensuring HuggingFace login...")
        ensure_hf_login()
        logger.info("Initializing TTS engine...")
        engine = OrpheusTTSEngine()
        logger.info("TTS engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}", exc_info=True)
        raise
    # Background warmup to reduce first-request TTFB (compile kernels, allocate KV cache)
    async def _warmup():
        try:
            logger.info("Starting background warmup...")
            # Preload SNAC model to avoid first-request latency
            from .core.snac_batcher import get_snac_batched
            logger.debug("Loading SNAC batched model...")
            _ = get_snac_batched()
            logger.debug("SNAC batched model loaded")
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
                # Only end-of-audio sentinel; do not include EOS during generation
                stop_token_ids=[128258],
            )
            async def _prime(voice: str):
                logger.debug(f"Priming voice: {voice}")
                # Consume to completion (no early break) so vLLM finishes without aborting
                async for _ in aiter_pcm_from_custom_tokens(engine.engine, "hello", voice, sp):
                    pass
                logger.debug(f"Voice {voice} primed successfully")
            await asyncio.gather(_prime("tara"), _prime("zac"))
            logger.info("Background warmup completed successfully")
        except Exception as e:
            # Do not fail startup if warmup errors
            logger.warning(f"Warmup failed but continuing: {e}", exc_info=True)
    asyncio.create_task(_warmup())

@app.get("/healthz")
async def healthz():
    logger.debug("Health check requested")
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
    logger.info(f"WebSocket connection accepted from {ws.client}")
    global engine
    if engine is None:
        logger.error("Engine not initialized, closing WebSocket")
        await ws.close(code=1013)
        return

    # State and queues
    q: asyncio.Queue[Optional[dict]] = asyncio.Queue()

    async def recv_loop():
        while True:
            try:
                msg = await ws.receive_text()
                logger.debug(f"Received WebSocket message: {msg[:100]}{'...' if len(msg) > 100 else ''}")
            except Exception as e:
                logger.error(f"Error receiving WebSocket message: {e}", exc_info=True)
                break
            # Baseten-style END sentinel
            if msg.strip() == "__END__":
                await q.put(None)
                break
            # Try JSON parse; fall back to treating as a word/phrase
            try:
                obj = json.loads(msg)
                logger.debug(f"Parsed JSON message: {obj}")
            except Exception as e:
                logger.debug(f"Failed to parse JSON, treating as plain text: {e}")
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
                    logger.debug(f"Queuing text message: voice={v}, text_length={len(t)}")
                    await q.put({"type": "text", "text": t, "voice": v})
                continue

            # Plain text message → treat as full text for Baseten mode
            t = msg.strip()
            if t:
                logger.debug(f"Queuing plain text message: text_length={len(t)}")
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
                        logger.debug(f"Voice override: {voice}")
                    if not full_text:
                        logger.warning("Empty text received, closing connection")
                        try:
                            await ws.close()
                        except Exception as e:
                            logger.error(f"Error closing WebSocket: {e}", exc_info=True)
                        break

                    # ---- Baseten-compatible streaming from vLLM ----
                    final_temp = float(temperature if (temperature is not None) else 0.6)
                    final_top_p = float(top_p if (top_p is not None) else 0.8)
                    final_rep_penalty = float(repetition_penalty if (repetition_penalty is not None) else 1.1)
                    final_max_tokens = int(num_predict if (num_predict is not None) else int(os.getenv("ORPHEUS_MAX_TOKENS", "2048")))
                    
                    sp = SamplingParams(
                        temperature=final_temp,
                        top_p=final_top_p,
                        repetition_penalty=final_rep_penalty,
                        max_tokens=final_max_tokens,
                        detokenize=True,
                        skip_special_tokens=False,
                        ignore_eos=False,
                        # Only end-of-audio sentinel; do not include EOS during generation
                        stop_token_ids=[128258],
                    )
                    logger.debug(f"Sampling params: temp={final_temp}, top_p={final_top_p}, rep_penalty={final_rep_penalty}, max_tokens={final_max_tokens}")

                    v = resolve_voice(voice) or "tara"
                    logger.info(f"Starting TTS generation: voice={v}, text_length={len(full_text)}")
                    chunks = chunk_by_words(
                        full_text,
                        first_chunk_words=FIRST_CHUNK_WORDS,
                        next_chunk_words=NEXT_CHUNK_WORDS,
                        min_tail_words=MIN_TAIL_WORDS,
                    )
                    logger.debug(f"Text chunked into {len(chunks)} chunks: {[len(chunk.split()) for chunk in chunks]} words each")
                    if not chunks:
                        logger.warning("No chunks generated from text, closing connection")
                        try:
                            await ws.close()
                        except Exception as e:
                            logger.error(f"Error closing WebSocket: {e}", exc_info=True)
                        break

                    async def produce_to_queue(text: str, qout: asyncio.Queue):
                        async for pcm in aiter_pcm_from_custom_tokens(engine.engine, text, v, sp):
                            await qout.put(pcm)
                        await qout.put(None)  # sentinel

                    if len(chunks) == 1:
                        # Single chunk: stream directly
                        logger.debug("Single chunk processing - streaming directly")
                        pcm_count = 0
                        async for pcm in aiter_pcm_from_custom_tokens(engine.engine, chunks[0], v, sp):
                            await ws.send_bytes(pcm)
                            pcm_count += 1
                            await asyncio.sleep(0)
                        logger.debug(f"Sent {pcm_count} PCM chunks for single chunk")
                    else:
                        # Pipeline: stream chunk 0 directly, pre-generate chunk 1 into a queue,
                        # then for each subsequent chunk pre-generate the next while draining current queued
                        logger.debug(f"Multi-chunk processing - pipeline mode with {len(chunks)} chunks")
                        queued_q = asyncio.Queue(maxsize=128)
                        queued_task = asyncio.create_task(produce_to_queue(chunks[1], queued_q))

                        # Active 0 → socket
                        logger.debug("Processing chunk 0 directly to socket")
                        pcm_count = 0
                        async for pcm in aiter_pcm_from_custom_tokens(engine.engine, chunks[0], v, sp):
                            await ws.send_bytes(pcm)
                            pcm_count += 1
                            await asyncio.sleep(0)
                        logger.debug(f"Sent {pcm_count} PCM chunks for chunk 0")

                        # Drain queued chunks in order, starting from index 1
                        for idx in range(1, len(chunks)):
                            logger.debug(f"Processing chunk {idx}/{len(chunks)-1}")
                            # Start next queued (idx+1) before draining current queued (idx)
                            next_q = None
                            next_task = None
                            if (idx + 1) < len(chunks):
                                next_q = asyncio.Queue(maxsize=128)
                                next_task = asyncio.create_task(produce_to_queue(chunks[idx + 1], next_q))

                            # Drain current queued_q (chunk idx)
                            pcm_count = 0
                            while True:
                                b = await queued_q.get()
                                if b is None:
                                    break
                                await ws.send_bytes(b)
                                pcm_count += 1
                                await asyncio.sleep(0)
                            logger.debug(f"Sent {pcm_count} PCM chunks for chunk {idx}")

                            # Move to next queued
                            queued_q = next_q if next_q is not None else asyncio.Queue()
                            queued_task = next_task

                    logger.info("TTS generation completed successfully")
                    try:
                        await ws.close()
                    except Exception as e:
                        logger.error(f"Error closing WebSocket after completion: {e}", exc_info=True)
                    break
        except Exception as e:
            logger.error(f"Error in synthesis loop: {e}", exc_info=True)
            try:
                await ws.close(code=1011)
            except Exception as close_e:
                logger.error(f"Error closing WebSocket after synthesis error: {close_e}", exc_info=True)
            raise e

    try:
        recv_task = asyncio.create_task(recv_loop())
        synth_task = asyncio.create_task(synth_loop())
        await asyncio.gather(recv_task, synth_task)
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: {e}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.debug("Cleaning up WebSocket connection")
        try:
            await ws.close()
        except Exception as e:
            logger.debug(f"Error during WebSocket cleanup: {e}")


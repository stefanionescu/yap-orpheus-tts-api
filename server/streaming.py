import asyncio
import os
import uuid
from typing import Iterator, Any

import numpy as np
import torch

from .prompts import build_prompt, build_prompt_ids, resolve_voice
from .core.custom_tokens import split_custom_tokens, turn_token_into_id
from .core.snac_batcher import get_snac_batched, SNAC_DEVICE
from .core.logging_config import get_logger

logger = get_logger(__name__)


def _random_uuid() -> str:
    return uuid.uuid4().hex


async def aiter_pcm_from_custom_tokens(engine: Any, prompt: str, voice: str, sp: Any) -> Iterator[bytes]:
    """
    vLLM → detokenized pieces → <custom_token_…> → 28→PCM, Baseten-identical.
    Monotonic delta consumption (no rescans, no resets) for artifact-free audio.
    """
    session_id = _random_uuid()[:8]
    logger.debug(f"[{session_id}] Starting PCM generation: voice={voice}, prompt_len={len(prompt)}")
    
    tok_index = 0  # never reset within a single generation
    buf_ids: list[int] = []
    
    logger.debug(f"[{session_id}] Loading SNAC model...")
    snacx = get_snac_batched()
    logger.debug(f"[{session_id}] SNAC model ready")

    prev_len = 0  # length of detokenized text we have already processed
    generation_step = 0
    pcm_count = 0
    tokens_processed = 0
    
    # Prefer pre-tokenized prompt ids when backend supports it (TRT-LLM wrapper accepts list[int])
    try:
        tok = getattr(engine, "tokenizer", None)
        use_ids = (tok is not None)
    except Exception:
        tok = None
        use_ids = False
    encoded = build_prompt(prompt, resolve_voice(voice))
    if use_ids and tok is not None:
        try:
            encoded = build_prompt_ids(prompt, resolve_voice(voice), tok)
            logger.info(f"PROMPT_TAIL={encoded[-16:] if isinstance(encoded, list) and len(encoded) > 16 else encoded}")
        except Exception:
            pass

    async for out in engine.generate(encoded, sp, _random_uuid()):
        generation_step += 1
        outs = out.outputs or []
        if not outs:
            continue

        piece = outs[0].text or ""
        # Only process newly appended text to keep sequence monotonic
        if len(piece) <= prev_len:
            continue
        delta = piece[prev_len:]
        prev_len = len(piece)
        
        if delta:
            logger.debug(f"[{session_id}] Step {generation_step}: Processing delta: '{delta[:50]}{'...' if len(delta) > 50 else ''}'")

        for n in split_custom_tokens(delta, tokenizer=tok):
            tid = turn_token_into_id(n, tok_index)
            tok_index += 1
            buf_ids.append(tid)
            tokens_processed += 1

            # Every 7 tokens is one frame; after 4 frames (28 ids) → decode
            if (tok_index % 7 == 0) and len(buf_ids) >= 28:
                window = buf_ids[-28:]
                arr = np.asarray(window, dtype=np.int32).reshape(-1, 7)
                codes_0 = torch.from_numpy(arr[:, 0]).unsqueeze(0).to(SNAC_DEVICE)
                codes_1 = torch.from_numpy(arr[:, [1, 4]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
                codes_2 = torch.from_numpy(arr[:, [2, 3, 5, 6]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)

                logger.debug(f"[{session_id}] Decoding SNAC frame {tok_index // 7}")
                audio = await snacx.decode_codes([codes_0, codes_1, codes_2])
                pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                if pcm:
                    pcm_count += 1
                    logger.debug(f"[{session_id}] Yielding PCM chunk {pcm_count} ({len(pcm)} bytes)")
                    yield pcm
    
    logger.info(f"[{session_id}] PCM generation completed: {generation_step} generation steps, "
                f"{tokens_processed} tokens processed, {pcm_count} PCM chunks yielded")



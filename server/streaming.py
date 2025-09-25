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
    
    buf_codes: list[int] = []
    # before the generate loop:
    emit_from = 0                # number of audio tokens already decoded
    first_emitted = False
    
    # Early TTFB controls (treat values as TOKENS, not frames). Fallback to older names if present.
    def _env_int(name: str, default: int) -> int:
        try:
            v = os.getenv(name)
            return int(v) if (v is not None and str(v).strip() != "") else default
        except Exception:
            return default
    # Early TTFB controls (TOKENS, not frames) - AGGRESSIVELY optimized for minimal latency
    MIN_TOKENS_FIRST  = _env_int("MIN_TOKENS_FIRST", 7)   # 1 frame only! Minimal TTFB 
    MIN_TOKENS_SUBSEQ = _env_int("MIN_TOKENS_SUBSEQ", 28) # 4 frames * 7, balanced for quality  
    TOKENS_EVERY      = _env_int("TOKENS_EVERY", 7)       # Tokens per SNAC frame (was hardcoded!)

    TOKENS_PER_FRAME = TOKENS_EVERY  # Now actually configurable!
    FIRST_FRAMES  = max(1, MIN_TOKENS_FIRST  // TOKENS_PER_FRAME)
    SUBSEQ_FRAMES = max(1, MIN_TOKENS_SUBSEQ // TOKENS_PER_FRAME)
    
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

    # Defensive early-stop detection string for END_OF_SPEECH (id 128258)
    try:
        EOSPEECH_STR = (tok.decode([128258], skip_special_tokens=False) if tok is not None else "<|end_of_speech|>")
    except Exception:
        EOSPEECH_STR = "<|end_of_speech|>"

    async for out in engine.generate(encoded, sp, _random_uuid()):
        generation_step += 1
        if isinstance(out, dict):
            piece = out.get("text", "")
            codes = out.get("audio_codes_delta", [])
        else:
            # Back-compat: your old _CompatResult path
            outs = getattr(out, "outputs", []) or []
            piece = outs[0].text if outs else ""
            codes = []  # no id path, but we don't rely on regex anymore
            
        # Only process newly appended text to keep sequence monotonic
        if len(piece) <= prev_len:
            continue
        delta = piece[prev_len:]
        prev_len = len(piece)
        
        if delta:
            logger.debug(f"[{session_id}] Step {generation_step}: Processing delta: '{delta[:50]}{'...' if len(delta) > 50 else ''}'")
            # Early tail logging to confirm custom tokens are present
            if generation_step <= 6:
                try:
                    logger.debug(f"[{session_id}] piece_tail={repr(piece[-120:])}")
                    logger.debug(f"[{session_id}] delta_tail={repr(delta[-120:])}")
                except Exception:
                    pass

        saw_eos = EOSPEECH_STR and (EOSPEECH_STR in piece)  # use full piece for safety

        # Use audio codes from IDs instead of regex extraction
        tokens_in_delta = codes
        try:
            logger.debug(f"[{session_id}] audio_tokens_in_delta={len(tokens_in_delta)}")
        except Exception:
            pass
        for n in tokens_in_delta:                 # n are audio code indices (not token ids)
            buf_codes.append(int(n))
            tokens_processed += 1

        # Extra sanity logging for first few steps
        if generation_step <= 3:
            total = len(buf_codes)
            rem = (total - emit_from) % TOKENS_PER_FRAME
            logger.debug(f"[{session_id}] buffered_codes={total}, ready_frames={(total-emit_from)//7}, leftover_tokens={rem}")

        # try to emit whole NEW frames only
        while (len(buf_codes) - emit_from) >= TOKENS_PER_FRAME:
            frames_ready = (len(buf_codes) - emit_from) // TOKENS_PER_FRAME
            if not first_emitted:
                need = FIRST_FRAMES
            else:
                need = SUBSEQ_FRAMES
            if frames_ready < need:
                break

            # Tentative slice — do NOT advance emit_from yet
            start = emit_from
            end   = emit_from + need * TOKENS_PER_FRAME
            chunk_codes = buf_codes[start:end]

            assert all(isinstance(x, (int, np.integer)) and 0 <= x < 4096 for x in chunk_codes), \
                   f"Bad codes in chunk: {chunk_codes[:14]}"

            arr = np.asarray(chunk_codes, dtype=np.int64).reshape(-1, TOKENS_PER_FRAME)  # [F, 7]
            F = arr.shape[0]

            # Split into three residual streams (flat layout expected by your SNAC)
            codes0_np = arr[:, 0]                    # [F]
            codes1_np = arr[:, [1, 4]].reshape(-1)   # [2F]
            codes2_np = arr[:, [2, 3, 5, 6]].reshape(-1)  # [4F]

            codes_0 = torch.from_numpy(codes0_np).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)  # [1, F]
            codes_1 = torch.from_numpy(codes1_np).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)  # [1, 2F]
            codes_2 = torch.from_numpy(codes2_np).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)  # [1, 4F]

            logger.debug(f"[{session_id}] Decoding SNAC frame {pcm_count + 1} (F={F})")

            def _normalize_audio(a):
                if a is None:
                    return None
                # snac may return list/tuple of per-item audios → take our item 0
                if isinstance(a, (list, tuple)):
                    a = a[0]
                if isinstance(a, torch.Tensor):
                    a = a.detach().cpu().numpy()
                a = np.asarray(a)
                if a.size == 0:
                    return None
                return a.astype(np.float32).reshape(-1)

            # Single (flat) attempt — this is the API your SNAC supports
            audio = await snacx.decode_codes([codes_0, codes_1, codes_2])
            audio = _normalize_audio(audio)

            if audio is None:
                # Do NOT advance emit_from; accumulate more frames and try again later
                logger.debug(f"[{session_id}] SNAC returned empty; keeping {F} frames buffered (need more)")
                break

            # Success → advance the pointer and emit PCM
            emit_from = end
            first_emitted = True

            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            if pcm:
                pcm_count += 1
                logger.debug(f"[{session_id}] Yielding PCM chunk {pcm_count} ({len(pcm)} bytes)")
                yield pcm
        
        if saw_eos and pcm_count > 0:
            break  # ok to stop; we streamed at least one hop
        # if saw_eos but no frames yet, keep looping until we accumulate 28 tokens or hit generator end
    # flush any remaining full frames (ignore leftovers < 7 tokens)  
    try:
        pending = len(buf_codes) - emit_from
        if pending >= TOKENS_PER_FRAME and pcm_count == 0:
            frames_to_emit = pending // TOKENS_PER_FRAME
            start = emit_from
            end = emit_from + frames_to_emit * TOKENS_PER_FRAME
            chunk_codes = buf_codes[start:end]
            
            arr = np.asarray(chunk_codes, dtype=np.int64).reshape(-1, TOKENS_PER_FRAME)  # [F, 7]
            F = arr.shape[0]

            # Split into three residual streams (flat layout only)
            codes0_np = arr[:, 0]                    # [F]
            codes1_np = arr[:, [1, 4]].reshape(-1)   # [2F]
            codes2_np = arr[:, [2, 3, 5, 6]].reshape(-1)  # [4F]

            codes_0 = torch.from_numpy(codes0_np).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)  # [1, F]
            codes_1 = torch.from_numpy(codes1_np).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)  # [1, 2F]
            codes_2 = torch.from_numpy(codes2_np).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)  # [1, 4F]

            def _normalize_audio(a):
                if a is None:
                    return None
                # snac may return list/tuple of per-item audios → take our item 0
                if isinstance(a, (list, tuple)):
                    a = a[0]
                if isinstance(a, torch.Tensor):
                    a = a.detach().cpu().numpy()
                a = np.asarray(a)
                if a.size == 0:
                    return None
                return a.astype(np.float32).reshape(-1)

            logger.debug(f"[{session_id}] Flushing final {frames_to_emit} full frames")
            
            # Single (flat) attempt for flush
            audio = await snacx.decode_codes([codes_0, codes_1, codes_2])
            audio = _normalize_audio(audio)

            # Convert to PCM only if we actually got samples
            if audio is not None:
                pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                if pcm:
                    pcm_count += 1
                    logger.debug(f"[{session_id}] Yielding final flushed PCM chunk ({len(pcm)} bytes)")
                    yield pcm
    except Exception:
        pass

    logger.info(f"[{session_id}] PCM generation completed: {generation_step} generation steps, "
                f"{tokens_processed} tokens processed, {pcm_count} PCM chunks yielded")



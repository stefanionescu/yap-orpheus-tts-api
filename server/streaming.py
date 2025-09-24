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
    MIN_TOKENS_FIRST = _env_int("MIN_TOKENS_FIRST", _env_int("MIN_FRAMES_FIRST", 7))
    MIN_TOKENS_SUBSEQ = _env_int("MIN_TOKENS_SUBSEQ", _env_int("MIN_FRAMES_SUBSEQ", 7))
    TOKENS_EVERY = _env_int("TOKENS_EVERY", _env_int("PROCESS_EVERY", 7))
    
    # choose frames, not windows
    TOKENS_PER_FRAME = 7
    FIRST_FRAMES = max(1, MIN_TOKENS_FIRST // TOKENS_PER_FRAME)      # e.g. 14 -> 2
    SUBSEQ_FRAMES = max(1, MIN_TOKENS_SUBSEQ // TOKENS_PER_FRAME)    # e.g. 7  -> 1
    
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

        # try to emit whole NEW frames only
        while (len(buf_codes) - emit_from) >= TOKENS_PER_FRAME:
            frames_ready = (len(buf_codes) - emit_from) // TOKENS_PER_FRAME
            if not first_emitted:
                need = FIRST_FRAMES
                if frames_ready < need:
                    break
                frames_to_emit = need
                first_emitted = True
            else:
                need = SUBSEQ_FRAMES
                if frames_ready < need:
                    break
                frames_to_emit = need

            start = emit_from
            end   = emit_from + frames_to_emit * TOKENS_PER_FRAME
            chunk_codes = buf_codes[start:end]                    # <-- strictly new frames
            emit_from = end

            # Optional sanity check while debugging
            assert all(isinstance(x, (int, np.integer)) and 0 <= x < 4096 for x in chunk_codes), \
                   f"Bad codes in chunk: {chunk_codes[:14]}"

            arr = np.asarray(chunk_codes, dtype=np.int64).reshape(-1, TOKENS_PER_FRAME)  # [F, 7]
            F = arr.shape[0]

            # Always build the three code streams from positions:
            codes0_np = arr[:, 0]                         # [F]
            codes1_np = arr[:, [1, 4]]                    # [F, 2]
            codes2_np = arr[:, [2, 3, 5, 6]]              # [F, 4]

            # Shapes v1 (flat): [B, F] / [B, 2F] / [B, 4F]  (this is what we used before)
            codes_0_v1 = torch.from_numpy(codes0_np).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)               # [1, F]
            codes_1_v1 = torch.from_numpy(codes1_np.reshape(-1)).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)   # [1, 2F]
            codes_2_v1 = torch.from_numpy(codes2_np.reshape(-1)).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)   # [1, 4F]

            # Shapes v2 (stacked): [B, F] / [B, 2, F] / [B, 4, F]
            codes_0_v2 = codes_0_v1                                                                                 # [1, F]
            codes_1_v2 = torch.from_numpy(codes1_np.T).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long).contiguous()  # [1, 2, F]
            codes_2_v2 = torch.from_numpy(codes2_np.T).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long).contiguous()  # [1, 4, F]

            # Try v1, then fall back to v2 if empty
            async def _decode_triplet(triplet):
                a = await snacx.decode_codes(triplet)
                try:
                    # Normalize to numpy
                    if hasattr(a, "detach"):
                        a = a.detach().cpu().numpy()
                    elif isinstance(a, torch.Tensor):
                        a = a.cpu().numpy()
                    return a
                except Exception:
                    return a

            logger.debug(f"[{session_id}] Decoding SNAC frame {pcm_count + 1}")
            
            # First attempt (flat)
            audio = await _decode_triplet([codes_0_v1, codes_1_v1, codes_2_v1])

            # If empty or None, try stacked layout
            if audio is None or (hasattr(audio, "size") and int(getattr(audio, "size", 0)) == 0):
                logger.debug(f"[{session_id}] SNAC returned empty audio with flat layout; retrying stacked layout")
                audio = await _decode_triplet([codes_0_v2, codes_1_v2, codes_2_v2])

            # Log what we got
            try:
                sz = int(getattr(audio, "size", 0)) if audio is not None else 0
                shp = tuple(getattr(audio, "shape", [])) if audio is not None else ()
                logger.debug(f"[{session_id}] SNAC audio shape={shp} size={sz}")
            except Exception:
                pass

            # Convert to PCM only if we actually got samples
            if audio is not None:
                # Expect 1D or 2D (B, T); flatten to mono stream
                audio = np.asarray(audio).astype(np.float32).reshape(-1)
                if audio.size > 0:
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

            # Always build the three code streams from positions:
            codes0_np = arr[:, 0]                         # [F]
            codes1_np = arr[:, [1, 4]]                    # [F, 2]
            codes2_np = arr[:, [2, 3, 5, 6]]              # [F, 4]

            # Shapes v1 (flat): [B, F] / [B, 2F] / [B, 4F]
            codes_0_v1 = torch.from_numpy(codes0_np).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)               # [1, F]
            codes_1_v1 = torch.from_numpy(codes1_np.reshape(-1)).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)   # [1, 2F]
            codes_2_v1 = torch.from_numpy(codes2_np.reshape(-1)).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long)   # [1, 4F]

            # Shapes v2 (stacked): [B, F] / [B, 2, F] / [B, 4, F]
            codes_0_v2 = codes_0_v1                                                                                 # [1, F]
            codes_1_v2 = torch.from_numpy(codes1_np.T).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long).contiguous()  # [1, 2, F]
            codes_2_v2 = torch.from_numpy(codes2_np.T).unsqueeze(0).to(SNAC_DEVICE, dtype=torch.long).contiguous()  # [1, 4, F]

            # Try v1, then fall back to v2 if empty
            async def _decode_triplet(triplet):
                a = await snacx.decode_codes(triplet)
                try:
                    # Normalize to numpy
                    if hasattr(a, "detach"):
                        a = a.detach().cpu().numpy()
                    elif isinstance(a, torch.Tensor):
                        a = a.cpu().numpy()
                    return a
                except Exception:
                    return a

            logger.debug(f"[{session_id}] Flushing final {frames_to_emit} full frames")
            
            # First attempt (flat)
            audio = await _decode_triplet([codes_0_v1, codes_1_v1, codes_2_v1])

            # If empty or None, try stacked layout
            if audio is None or (hasattr(audio, "size") and int(getattr(audio, "size", 0)) == 0):
                logger.debug(f"[{session_id}] SNAC returned empty audio with flat layout; retrying stacked layout for flush")
                audio = await _decode_triplet([codes_0_v2, codes_1_v2, codes_2_v2])

            # Convert to PCM only if we actually got samples
            if audio is not None:
                # Expect 1D or 2D (B, T); flatten to mono stream
                audio = np.asarray(audio).astype(np.float32).reshape(-1)
                if audio.size > 0:
                    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    if pcm:
                        pcm_count += 1
                        logger.debug(f"[{session_id}] Yielding final flushed PCM chunk ({len(pcm)} bytes)")
                        yield pcm
    except Exception:
        pass

    logger.info(f"[{session_id}] PCM generation completed: {generation_step} generation steps, "
                f"{tokens_processed} tokens processed, {pcm_count} PCM chunks yielded")



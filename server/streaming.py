import asyncio
import os
import re
import uuid
from typing import Iterator, Any

import numpy as np
import torch

from .prompts import build_prompt, resolve_voice, CODE_END, CODE_OFFSET, CODES_PER_LEVEL, TOKENS_PER_FRAME
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

    # --- Correct Orpheus audio token handling ---
    AUDIO_RE = re.compile(r"<custom_token_(\d+)>")

    def audio_codes_from_token_ids(token_ids: list[int]) -> list[int]:
        """Filter and normalize token IDs to 'global' audio codes [0 .. 7*4096).
        Stops at CODE_END if present.
        """
        out: list[int] = []
        upper = CODE_OFFSET + 7 * CODES_PER_LEVEL
        for tid in token_ids:
            if tid == CODE_END:
                break
            if CODE_OFFSET <= tid < upper:
                out.append(int(tid - CODE_OFFSET))
        return out

    def split_to_snac_lanes(global_codes: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split interleaved global codes into SNAC lanes q0, q1, q2.
        Returns lane-local code indices in [0, 4096).
        """
        if not global_codes:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
        arr = np.asarray(global_codes, dtype=np.int64).reshape(-1, TOKENS_PER_FRAME)  # [F,7]
        # Subtract lane offsets then pack with the correct interleave: 0 | 1,4 | 2,3,5,6
        q0 = (arr[:, 0] - 0 * CODES_PER_LEVEL).reshape(-1)
        q1 = np.stack([
            (arr[:, 1] - 1 * CODES_PER_LEVEL),
            (arr[:, 4] - 4 * CODES_PER_LEVEL),
        ], axis=1).reshape(-1)
        q2 = np.stack([
            (arr[:, 2] - 2 * CODES_PER_LEVEL),
            (arr[:, 3] - 3 * CODES_PER_LEVEL),
            (arr[:, 5] - 5 * CODES_PER_LEVEL),
            (arr[:, 6] - 6 * CODES_PER_LEVEL),
        ], axis=1).reshape(-1)
        return q0, q1, q2
        
    def _apply_crossfade(pcm_new, prev_tail_state, crossfade_samples=256):
        """
        Apply crossfade between consecutive PCM chunks to remove boundary clicks.
        crossfade_samples: number of samples to crossfade (~10.7 ms @ 24 kHz)
        Returns: (output_pcm, new_prev_tail_state)
        """
        if not prev_tail_state.get('tail'):
            # First chunk - just save tail for next time
            pcm_array = np.frombuffer(pcm_new, dtype=np.int16)
            if len(pcm_array) > crossfade_samples:
                prev_tail_state['nontail'] = pcm_new[:-crossfade_samples*2]  # int16 = 2 bytes
                prev_tail_state['tail'] = pcm_new[-crossfade_samples*2:]
            else:
                prev_tail_state['nontail'] = b""
                prev_tail_state['tail'] = pcm_new
            return pcm_new, prev_tail_state
            
        # Crossfade with previous tail
        prev_tail = prev_tail_state['tail']
        prev_nontail = prev_tail_state['nontail']
        
        a = np.frombuffer(prev_tail, dtype=np.int16).astype(np.float32)
        b_start = np.frombuffer(pcm_new[:len(prev_tail)], dtype=np.int16).astype(np.float32)
        
        # Ensure same length for crossfade 
        min_len = min(len(a), len(b_start))
        if min_len == 0:
            return pcm_new, prev_tail_state
            
        a = a[:min_len]
        b_start = b_start[:min_len]
        
        # Linear crossfade
        w = np.linspace(0.0, 1.0, min_len, endpoint=False)
        mix = (a*(1.0-w) + b_start*w).astype(np.int16).tobytes()
        
        # Construct output: prev_nontail + crossfade + rest_of_current
        rest_of_current = pcm_new[len(prev_tail):]
        output_pcm = prev_nontail + mix + rest_of_current
        
        # Update state for next time
        pcm_array = np.frombuffer(pcm_new, dtype=np.int16)
        if len(pcm_array) > crossfade_samples:
            prev_tail_state['nontail'] = pcm_new[:-crossfade_samples*2]
            prev_tail_state['tail'] = pcm_new[-crossfade_samples*2:]
        else:
            prev_tail_state['nontail'] = b""
            prev_tail_state['tail'] = pcm_new
            
        return output_pcm, prev_tail_state
        
    # Early TTFB controls (TOKENS, not frames) - AGGRESSIVELY optimized for minimal latency
    tokens_per_frame = _env_int("TOKENS_EVERY", TOKENS_PER_FRAME)  # default to model constant
    MIN_TOKENS_FIRST  = _env_int("MIN_TOKENS_FIRST", tokens_per_frame)   # 1 frame only! Minimal TTFB 
    MIN_TOKENS_SUBSEQ = _env_int("MIN_TOKENS_SUBSEQ", 4 * tokens_per_frame) # 4 frames, balanced for quality  

    FIRST_FRAMES  = max(1, MIN_TOKENS_FIRST  // tokens_per_frame)
    SUBSEQ_FRAMES = max(1, MIN_TOKENS_SUBSEQ // tokens_per_frame)
    
    logger.debug(f"[{session_id}] Loading SNAC model...")
    snacx = get_snac_batched()
    logger.debug(f"[{session_id}] SNAC model ready")

    prev_len = 0  # length of detokenized text we have already processed
    generation_step = 0
    pcm_count = 0
    tokens_processed = 0
    
    # Crossfade state for smooth PCM transitions
    crossfade_state = {'tail': None, 'nontail': None}
    
    # Prefer pre-tokenized prompt ids when backend supports it (TRT-LLM wrapper accepts list[int])
    try:
        tok = getattr(engine, "tokenizer", None)
        use_ids = (tok is not None)
    except Exception:
        tok = None
        use_ids = False
    encoded = build_prompt(prompt, resolve_voice(voice))  # <- string, let TRT tokenizer encode

    # Defensive early-stop detection string for END_OF_SPEECH (id 128258)
    try:
        EOSPEECH_STR = (tok.decode([128258], skip_special_tokens=False) if tok is not None else "<|end_of_speech|>")
    except Exception:
        EOSPEECH_STR = "<|end_of_speech|>"

    async for out in engine.generate(encoded, sp, _random_uuid()):
        generation_step += 1
        if isinstance(out, dict):
            piece = out.get("text", "")
            delta_token_ids = out.get("token_ids_delta")
            codes = out.get("audio_codes_delta", [])
        else:
            # Back-compat: your old _CompatResult path
            outs = getattr(out, "outputs", []) or []
            piece = outs[0].text if outs else ""
            delta_token_ids = None
            codes = []  # no id path; we'll rely on regex
            
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

        # Primary path: use token IDs if provided
        global_codes_delta: list[int] = []
        if isinstance(delta_token_ids, (list, tuple)) and delta_token_ids:
            try:
                global_codes_delta = audio_codes_from_token_ids([int(x) for x in delta_token_ids])
            except Exception:
                global_codes_delta = []

        # Fallback path: regex extraction from delta text
        if (not global_codes_delta) and delta:
            try:
                for m in AUDIO_RE.finditer(delta):
                    tid = int(m.group(1))
                    if CODE_OFFSET <= tid < CODE_OFFSET + 7 * CODES_PER_LEVEL:
                        global_codes_delta.append(tid - CODE_OFFSET)
            except Exception:
                pass

        if global_codes_delta:
            try:
                logger.debug(f"[{session_id}] audio_tokens_in_delta={len(global_codes_delta)}")
            except Exception:
                pass
            buf_codes.extend(global_codes_delta)
            tokens_processed += len(global_codes_delta)

        # Extra sanity logging for first few steps
        if generation_step <= 3:
            total = len(buf_codes)
            rem = (total - emit_from) % tokens_per_frame
            logger.debug(f"[{session_id}] buffered_codes={total}, ready_frames={(total-emit_from)//tokens_per_frame}, leftover_tokens={rem}")

        # try to emit whole NEW frames only
        while (len(buf_codes) - emit_from) >= tokens_per_frame:
            frames_ready = (len(buf_codes) - emit_from) // tokens_per_frame
            if not first_emitted:
                need = FIRST_FRAMES
            else:
                need = SUBSEQ_FRAMES
            if frames_ready < need:
                break

            # Tentative slice — do NOT advance emit_from yet
            start = emit_from
            end   = emit_from + need * tokens_per_frame
            chunk_global_codes = buf_codes[start:end]

            # Guard: codes must form full frames (multiples of 7)
            if ((len(chunk_global_codes) % tokens_per_frame) != 0):
                logger.error(f"[{session_id}] Non-multiple-of-{TOKENS_PER_FRAME} codes; buffering more (len={len(chunk_global_codes)})")
                break
            F = len(chunk_global_codes) // tokens_per_frame

            # Split into three residual streams using the correct interleaved mapping
            codes0_np, codes1_np, codes2_np = split_to_snac_lanes(chunk_global_codes)

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

            # Decode once with the correct mapping
            async def _decode_once(c0, c1, c2):
                a = await snacx.decode_codes([c0, c1, c2])
                return _normalize_audio(a)

            audio = await _decode_once(codes_0, codes_1, codes_2)

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
                # Apply crossfade to smooth transitions between chunks
                output_pcm, crossfade_state = _apply_crossfade(pcm, crossfade_state)
                logger.debug(f"[{session_id}] Yielding PCM chunk {pcm_count} ({len(output_pcm)} bytes)")
                yield output_pcm
        
        if saw_eos and pcm_count > 0:
            break  # ok to stop; we streamed at least one hop
        # if saw_eos but no frames yet, keep looping until we accumulate 28 tokens or hit generator end
    # flush any remaining full frames (ignore leftovers < 7 tokens)  
    try:
        pending = len(buf_codes) - emit_from
        if pending >= tokens_per_frame and pcm_count == 0:
            frames_to_emit = pending // tokens_per_frame
            start = emit_from
            end = emit_from + frames_to_emit * tokens_per_frame
            chunk_codes = buf_codes[start:end]
            
            # Split into three residual streams using the correct interleaved mapping
            codes0_np, codes1_np, codes2_np = split_to_snac_lanes(chunk_codes)

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
                    # Apply crossfade to final flush chunk too
                    output_pcm, crossfade_state = _apply_crossfade(pcm, crossfade_state)
                    logger.debug(f"[{session_id}] Yielding final flushed PCM chunk ({len(output_pcm)} bytes)")
                    yield output_pcm
    except Exception:
        pass

    logger.info(f"[{session_id}] PCM generation completed: {generation_step} generation steps, "
                f"{tokens_processed} tokens processed, {pcm_count} PCM chunks yielded")



import asyncio
import os
import uuid
from typing import Iterator, Any

import numpy as np
import torch

from .prompts import build_prompt, resolve_voice
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
            
    def _split_snac_lanes(arr, tokens_per_frame=7):
        """
        Split audio tokens into SNAC's 3 residual streams (q0|q1|q2).
        ORPHEUS_LANE_ORDER := "0|1,2|3,4,5,6" (contiguous default)
        Some checkpoints use interleaved: "0|1,4|2,3,5,6" 
        """
        order = os.getenv("ORPHEUS_LANE_ORDER", "0|1,2|3,4,5,6")
        try:
            groups = [list(map(int, g.split(","))) for g in order.split("|")]
            assert len(groups) == 3, f"Expected 3 groups, got {len(groups)}"
            assert len(groups[0]) == 1, f"Group 0 should have 1 element, got {len(groups[0])}"
            assert len(groups[1]) == 2, f"Group 1 should have 2 elements, got {len(groups[1])}"  
            assert len(groups[2]) == 4, f"Group 2 should have 4 elements, got {len(groups[2])}"
        except Exception as e:
            logger.warning(f"Invalid ORPHEUS_LANE_ORDER '{order}': {e}, using default")
            groups = [[0], [1, 2], [3, 4, 5, 6]]  # contiguous default
            
        codes0_np = arr[:, groups[0]].reshape(-1)       # [F] 
        codes1_np = arr[:, groups[1]].reshape(-1)       # [2F]
        codes2_np = arr[:, groups[2]].reshape(-1)       # [4F]
        return codes0_np, codes1_np, codes2_np
        
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
        
        # Auto-coerce if they look like token ids (>=4096) or raw <custom_token_N> numbers
        if tokens_in_delta and (max(tokens_in_delta) >= 4096 or min(tokens_in_delta) < 0):
            try:
                from .core.custom_tokens import build_audio_id_lookup, turn_token_into_id
                id2code = build_audio_id_lookup(tok) if tok is not None else {}
                fixed = []
                # total audio tokens seen so far determines lane = (global_index % 7)
                global_idx = tokens_processed  # we already increment this when appending
                for i, tid in enumerate(tokens_in_delta):
                    # Map token-id -> custom_token_N (if present)
                    n = id2code.get(int(tid))
                    if n is None:
                        continue
                    fixed.append(turn_token_into_id(n, index=global_idx + i))
                tokens_in_delta = fixed
                logger.debug(f"[{session_id}] Auto-coerced {len(codes)} token IDs to {len(fixed)} audio codes")
            except Exception as e:
                logger.warning(f"[{session_id}] Token ID coercion failed: {e}")
                pass  # fall back; validator below will catch range issues
                
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

            # Split into three residual streams using configurable lane order
            codes0_np, codes1_np, codes2_np = _split_snac_lanes(arr, TOKENS_PER_FRAME)

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
        if pending >= TOKENS_PER_FRAME and pcm_count == 0:
            frames_to_emit = pending // TOKENS_PER_FRAME
            start = emit_from
            end = emit_from + frames_to_emit * TOKENS_PER_FRAME
            chunk_codes = buf_codes[start:end]
            
            arr = np.asarray(chunk_codes, dtype=np.int64).reshape(-1, TOKENS_PER_FRAME)  # [F, 7]
            F = arr.shape[0]

            # Split into three residual streams using configurable lane order
            codes0_np, codes1_np, codes2_np = _split_snac_lanes(arr, TOKENS_PER_FRAME)

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



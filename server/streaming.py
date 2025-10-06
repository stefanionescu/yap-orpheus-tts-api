"""Refactored TRT-LLM streaming with clean class-based architecture."""

from server.prompts import build_prompt, resolve_voice
from server.config import settings
from server.streaming.audio_decoder import AudioDecoder, TokenProcessor


async def aiter_pcm_from_custom_tokens(engine, prompt: str, voice: str, sp):
    """
    TRT-LLM streaming: read token_ids deltas, not detokenized text.
    Map Orpheus audio token ids → 7-stream RVQ codes → SNAC decode.
    """
    from tensorrt_llm import SamplingParams as _SP  # type: ignore

    # Build TRT-LLM sampling params
    sp = _SP(
        temperature=float(getattr(sp, "temperature", settings.default_temperature)) if hasattr(sp, "temperature") else settings.default_temperature,
        top_p=float(getattr(sp, "top_p", settings.default_top_p)) if hasattr(sp, "top_p") else settings.default_top_p,
        repetition_penalty=float(getattr(sp, "repetition_penalty", settings.default_repetition_penalty)) if hasattr(sp, "repetition_penalty") else settings.default_repetition_penalty,
        max_tokens=int(getattr(sp, "max_tokens", settings.streaming_default_max_tokens)) if hasattr(sp, "max_tokens") else settings.streaming_default_max_tokens,
        stop_token_ids=list(settings.streaming_stop_token_ids),
        detokenize=settings.trt_detokenize,
        skip_special_tokens=settings.trt_skip_special_tokens,
        add_special_tokens=settings.trt_add_special_tokens,
        ignore_eos=settings.trt_ignore_eos,
    )

    # Build the Orpheus prompt
    formatted = build_prompt(prompt, resolve_voice(voice))

    # Initialize clean decoder and processor
    decoder = AudioDecoder()
    processor = TokenProcessor(decoder)
    prev_len = 0

    # Stream tokens and process audio codes
    async for chunk in engine.generate_async(formatted, sp, streaming=True):
        if not getattr(chunk, "outputs", None):
            continue

        out = chunk.outputs[0]
        tids = getattr(out, "token_ids", None)
        if not tids:
            # Some TRT builds deliver `output_token_ids` instead
            tids = getattr(out, "output_token_ids", None)
        if not tids:
            continue

        new_tokens = tids[prev_len:]
        if not new_tokens:
            continue
        prev_len = len(tids)

        # Process each new token
        for token_id in new_tokens:
            frames_ready = processor.process_token(token_id)
            if frames_ready is not None:
                chunk_bytes = await processor.emit_window(frames_ready)
                if chunk_bytes:
                    yield chunk_bytes

    # Emit final window for remaining frames
    final_bytes = await processor.emit_final_window()
    if final_bytes:
        yield final_bytes
import os
import asyncio
from typing import Generator, Optional, Dict, Any
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from .vllm_config import vllm_engine_kwargs
from .text_chunker import chunk_text

MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")

# Orpheus suggests repetition_penalty >= 1.1 for stability; speed up w/ higher temp if needed.
DEFAULT_PARAMS = dict(
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
    seed=42,
    num_predict=49152,
    num_ctx=8192,
)

VOICE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Female -> Tara
    "tara": {
        "voice": "tara",
        "temperature": 0.80,
        "top_p": 0.80,
        "repetition_penalty": 1.90,
        "seed": 42,
    },
    # Male -> Zac
    "zac": {
        "voice": "zac",
        "temperature": 0.40,
        "top_p": 0.80,
        "repetition_penalty": 1.85,
        "seed": 42,
    },
}

ALIASES = {
    "female": "tara",
    "male": "zac",
}

class OrpheusTTSEngine:
    def __init__(self):
        engine_kwargs = vllm_engine_kwargs()
        max_model_len = engine_kwargs.pop("max_model_len")
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=MODEL_ID,
                max_model_len=max_model_len,
                enforce_eager=True,
                **engine_kwargs,
            )
        )
        # Tokenizer for formatting prompts
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def stream_pcm_chunks(
        self,
        text: str,
        voice: str = "tara",
        chunk_chars: int = 450,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        num_predict: Optional[int] = None,
        **gen_kwargs,
    ) -> Generator[bytes, None, None]:
        """
        Yield PCM16 bytes chunks at 24kHz as soon as Orpheus emits them.
        For long texts, we sequentially stream chunks to keep latency predictable.
        """
        # Resolve voice alias and preset
        voice_key = ALIASES.get(voice.lower(), voice.lower())
        preset = VOICE_PRESETS.get(voice_key, VOICE_PRESETS["tara"]).copy()

        params = DEFAULT_PARAMS.copy()
        params.update(preset)

        # Override with explicit args if provided
        if temperature is not None: params["temperature"] = temperature
        if top_p is not None: params["top_p"] = top_p
        if repetition_penalty is not None: params["repetition_penalty"] = repetition_penalty
        if seed is not None: params["seed"] = seed
        # num_predict is fixed by policy; do not allow override from request
        params.update(gen_kwargs or {})

        def format_prompt(prompt_text: str, voice_name: str) -> str:
            adapted = f"{voice_name}: {prompt_text}"
            ids = self.tokenizer(adapted, return_tensors="pt").input_ids
            # Insert BOS/EOS-like special tokens expected by model
            import torch
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
            all_ids = torch.cat([start_token, ids, end_tokens], dim=1)
            return self.tokenizer.decode(all_ids[0])

        for piece in chunk_text(text, target_chars=chunk_chars):
            prompt_string = format_prompt(piece, preset["voice"])
            sampling_params = SamplingParams(
                temperature=params.get("temperature", 0.8),
                top_p=params.get("top_p", 0.9),
                max_tokens=params.get("num_predict", 49152),
                repetition_penalty=params.get("repetition_penalty", 1.2),
                detokenize=True,
                skip_special_tokens=False,
            )

            # Convert async token stream to sync tokens via a background thread + Queue
            import threading, queue
            token_q: "queue.Queue[Optional[str]]" = queue.Queue()

            async def produce_tokens():
                prev_text = ""
                async for out in self.engine.generate(prompt=prompt_string, sampling_params=sampling_params):
                    text_chunk = out.outputs[0].text if out.outputs else ""
                    if not text_chunk:
                        continue
                    delta = text_chunk[len(prev_text):]
                    prev_text = text_chunk
                    if delta:
                        token_q.put(delta)
                token_q.put(None)

            def runner():
                asyncio.run(produce_tokens())

            th = threading.Thread(target=runner, daemon=True)
            th.start()

            def sync_token_iter():
                while True:
                    item = token_q.get()
                    if item is None:
                        break
                    yield item

            for audio_chunk in tokens_decoder_sync(sync_token_iter()):
                if audio_chunk:
                    yield audio_chunk

    def synthesize_wav_bytes(
        self,
        text: str,
        voice: str = "tara",
        chunk_chars: int = 450,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        num_predict: Optional[int] = None,
        **gen_kwargs,
    ) -> bytes:
        """
        Non-streaming: collect all chunks → return raw WAV bytes (PCM16/24kHz).
        """
        import io, wave
        pcm_iter = self.stream_pcm_chunks(
            text,
            voice,
            chunk_chars,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            num_predict=num_predict,
            **gen_kwargs,
        )

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            for chunk in pcm_iter:
                wf.writeframes(chunk)
        return buf.getvalue()



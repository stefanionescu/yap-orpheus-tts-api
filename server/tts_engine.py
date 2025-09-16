import os
from typing import Generator, Optional, Dict, Any
from orpheus_tts import OrpheusModel
from .vllm_config import vllm_engine_kwargs
from .text_chunker import chunk_text

MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")

# Orpheus suggests repetition_penalty >= 1.1 for stability; speed up w/ higher temp if needed.
DEFAULT_PARAMS = dict(
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
    seed=42,
    num_predict=None,
)

VOICE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Female -> Tara
    "tara": {
        "voice": "tara",
        "temperature": 0.80,
        "top_p": 0.80,
        "repetition_penalty": 1.90,
        "seed": 42,
        "num_predict": None,
    },
    # Male -> Zac
    "zac": {
        "voice": "zac",
        "temperature": 0.40,
        "top_p": 0.80,
        "repetition_penalty": 1.85,
        "seed": 42,
        "num_predict": None,
    },
}

ALIASES = {
    "female": "tara",
    "male": "zac",
}

class OrpheusTTSEngine:
    def __init__(self):
        # Forward key vLLM tuning knobs to Orpheus
        engine_kwargs = vllm_engine_kwargs()
        # OrpheusModel constructor accepts model_name + core args; unknown kwargs are forwarded
        self.model = OrpheusModel(
            model_name=MODEL_ID,
            max_model_len=engine_kwargs.pop("max_model_len"),
            **engine_kwargs,
        )

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
        if num_predict is not None: params["num_predict"] = num_predict
        params.update(gen_kwargs or {})

        for piece in chunk_text(text, target_chars=chunk_chars):
            # Orpheus returns a generator of PCM16 byte chunks
            for audio_chunk in self.model.generate_speech(prompt=piece, voice=preset["voice"], **params):
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



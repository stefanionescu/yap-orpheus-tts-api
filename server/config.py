import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    """Centralized configuration for the TTS server.

    Reads values from environment variables with safe defaults. Logic must
    remain identical to pre-refactor behavior: only centralize references.
    """

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Model / Tokenizer
    model_id: str = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")
    hf_token: str | None = os.environ.get("HF_TOKEN")

    # Engine (TensorRT-LLM)
    trtllm_engine_dir: str = os.getenv("TRTLLM_ENGINE_DIR", "").strip()
    kv_free_gpu_frac: str | None = os.getenv("KV_FREE_GPU_FRAC")
    kv_enable_block_reuse: bool = os.getenv("KV_ENABLE_BLOCK_REUSE", "0") == "1"

    # TTS / Streaming
    orpheus_max_tokens: int = int(os.getenv("ORPHEUS_MAX_TOKENS", "1024"))
    # Default sampling parameters used when client omits values
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.40"))
    default_top_p: float = float(os.getenv("DEFAULT_TOP_P", "0.9"))
    default_repetition_penalty: float = float(os.getenv("DEFAULT_REPETITION_PENALTY", "1.25"))
    # Stop-token policy for the server-side SamplingParams (non-streaming path)
    server_stop_token_ids: tuple[int, ...] = (128009, 128260)

    # SNAC / Audio
    snac_device: str | None = None  # resolved at runtime (cuda/cpu)
    snac_max_batch: int = int(os.getenv("SNAC_MAX_BATCH", "64"))
    snac_batch_timeout_ms: int = int(os.getenv("SNAC_BATCH_TIMEOUT_MS", "2"))
    snac_sr: int = int(os.getenv("SNAC_SR", "24000"))
    tts_decode_window: int = max(int(os.getenv("TTS_DECODE_WINDOW", "28")), 28)
    tts_max_sec: float = float(os.getenv("TTS_MAX_SEC", "0"))
    snac_torch_compile: bool = bool(int(os.getenv("SNAC_TORCH_COMPILE", "0")))

    # WebSocket protocol defaults
    ws_end_sentinel: str = os.getenv("WS_END_SENTINEL", "__END__")
    ws_close_busy_code: int = int(os.getenv("WS_CLOSE_BUSY_CODE", "1013"))
    ws_close_internal_code: int = int(os.getenv("WS_CLOSE_INTERNAL_CODE", "1011"))
    ws_queue_maxsize: int = int(os.getenv("WS_QUEUE_MAXSIZE", "128"))
    default_voice: str = os.getenv("DEFAULT_VOICE", "tara")
    # API key for simple auth (override in production); single source: YAP_API_KEY
    api_key: str = os.getenv("YAP_API_KEY", "yap_api_key")
    ws_meta_keys: tuple[str, ...] = (
        "voice", "temperature", "top_p", "repetition_penalty", "trim_silence"
    )

    # TRT-LLM streaming SamplingParams policy (identical behavior centralized)
    trt_detokenize: bool = False
    trt_skip_special_tokens: bool = False
    trt_add_special_tokens: bool = False
    trt_ignore_eos: bool = False
    streaming_stop_token_ids: tuple[int, ...] = (128258, 128262, 128009)  # EOS(speech), EOA, EOT
    streaming_default_max_tokens: int = int(os.getenv("STREAMING_DEFAULT_MAX_TOKENS", "1024"))

    # Orpheus audio code layout
    code_offset: int = 128266
    code_size: int = 4096
    frame_substreams: int = 7
    min_window_frames: int = 4
    snac_groups: tuple[tuple[int, ...], ...] = ((0,), (1, 4), (2, 3, 5, 6))

    # Event loop yield (0.0 keeps behavior identical while allowing tuning)
    yield_sleep_seconds: float = float(os.getenv("YIELD_SLEEP_SECONDS", "0"))

    # Audio post-processing
    trim_leading_silence: bool = os.getenv("TRIM_LEADING_SILENCE", "1") == "1"
    silence_rms_threshold: float = float(os.getenv("SILENCE_RMS_THRESHOLD", "0.0026"))
    silence_activation_ms: float = float(os.getenv("SILENCE_ACTIVATION_MS", "8"))
    silence_prespeech_pad_ms: float = float(os.getenv("SILENCE_PRESPEECH_PAD_MS", "80"))
    silence_max_leading_sec: float = float(os.getenv("SILENCE_MAX_LEADING_SEC", "0.6"))


settings = Settings()

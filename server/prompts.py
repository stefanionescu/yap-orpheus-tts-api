from transformers import AutoTokenizer
from server.config import settings
from server.voices import resolve_voice

MODEL_ID = settings.model_id

# Cache tokenizer at import time
_tok = AutoTokenizer.from_pretrained(MODEL_ID)


def build_prompt(text: str, voice: str = "tara") -> str:
    # Accept internal names directly; resolve only external aliases
    v = voice if voice in ("tara", "zac") else resolve_voice(voice)
    # Orpheus-official priming prompt: do not rely on tokenizer decoding
    return (
        f"<custom_token_3><|begin_of_text|>{v}: {text}"
        f"<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
    )



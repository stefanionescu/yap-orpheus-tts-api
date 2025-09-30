import os
from transformers import AutoTokenizer

ALIASES = {
    "female": "tara",
    "male": "zac",
}

MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")

# Cache tokenizer at import time
_tok = AutoTokenizer.from_pretrained(MODEL_ID)

def resolve_voice(v: str) -> str:
    if not v:
        return "tara"
    key = v.strip().lower()
    return ALIASES.get(key, key if key in ("tara", "zac") else "tara")

def build_prompt(text: str, voice: str = "tara") -> str:
    v = resolve_voice(voice)
    # Orpheus-official priming prompt: do not rely on tokenizer decoding
    return (
        f"<custom_token_3><|begin_of_text|>{v}: {text}"
        f"<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
    )



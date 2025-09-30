import os
from transformers import AutoTokenizer

ALIASES = {
    "female": "tara",
    "male": "zac",
}

MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")

# Cache tokenizer at import time
_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_SOH = _tok.decode([128259]) + (_tok.bos_token or "")
_END = _tok.decode([128260, 128261])  # EOH closers only; no EOS, no START-OF-AUDIO

def resolve_voice(v: str) -> str:
    if not v:
        return "tara"
    key = v.strip().lower()
    return ALIASES.get(key, key if key in ("tara", "zac") else "tara")

def build_prompt(text: str, voice: str = "tara") -> str:
    v = resolve_voice(voice)
    # Result is a *string* that includes decoded special tokens.
    return f"{_SOH}{v}: {text}{_END}"



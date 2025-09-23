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
# Do not include EOS (128009) here to avoid early termination before audio tokens
_END = _tok.decode([128260, 128261, 128257])  # START-OF-AUDIO triggers only

# Raw id constants for building prompt ids directly (preferred for TRT)
SOH_ID = 128259
START_OF_AUDIO_IDS = [128260, 128261, 128257]
EOS_ID = 128009

def resolve_voice(v: str) -> str:
    if not v:
        return "tara"
    key = v.strip().lower()
    return ALIASES.get(key, key if key in ("tara", "zac") else "tara")

def build_prompt(text: str, voice: str = "tara") -> str:
    v = resolve_voice(voice)
    # Result is a *string* that includes decoded special tokens.
    return f"{_SOH}{v}: {text}{_END}"


def build_prompt_ids(text: str, voice: str, tok: AutoTokenizer) -> list[int]:
    v = resolve_voice(voice)
    bos = tok.bos_token_id if tok.bos_token_id is not None else None
    pre = ([bos] if bos is not None else []) + [SOH_ID]
    voice_ids = tok.encode(f"{v}: ", add_special_tokens=False)
    msg_ids = tok.encode(text, add_special_tokens=False)
    ids = pre + voice_ids + msg_ids + START_OF_AUDIO_IDS
    # Strip trailing EOS if injected for any reason
    while ids and ids[-1] == EOS_ID:
        ids.pop()
    return ids



import os
from transformers import AutoTokenizer

ALIASES = {
    "female": "tara",
    "male": "zac",
}

MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")

# Cache tokenizer at import time
_tok = AutoTokenizer.from_pretrained(MODEL_ID)

# Canonical special IDs
SOH_ID = 128259           # START_OF_HUMAN (not used in new prompt)
SOT_ID = 128000           # <|begin_of_text|>
EOTXT_ID = 128009         # <|eot_id|>; same numeric as EOS for Llama-3 style
EOH_ID = 128260           # END_OF_HUMAN (not used in new prompt)
SOAI_ID = 128261          # START_OF_AI (not used in new prompt)
SOSPEECH_ID = 128257      # START_OF_SPEECH (not used in new prompt)
EO_SPEECH_ID = 128258     # <|end_of_speech|>
EOS_ID = 128009

PRIME = [3, 4, 5, 1]       # tiny kick to enter audio manifold

def resolve_voice(v: str) -> str:
    if not v:
        return "tara"
    key = v.strip().lower()
    return ALIASES.get(key, key if key in ("tara", "zac") else "tara")

def build_prompt(text: str, voice: str = "tara") -> str:
    v = resolve_voice(voice)
    # String form using proper speech boundary format
    from .core.custom_tokens import turn_token_into_id
    # Build token IDs and then decode for string compatibility
    ids: list[int] = []
    ids.append(SOT_ID)
    ids += _tok.encode(f"{v}: {text}", add_special_tokens=False)
    ids.append(EOTXT_ID)
    ids.append(SOSPEECH_ID)                       # <-- OPEN speech section
    for i, n in enumerate(PRIME):
        ids.append(turn_token_into_id(n, i))      # small audio seed AFTER SOS
    return _tok.decode(ids, skip_special_tokens=False)


def build_prompt_ids(text: str, voice: str, tok) -> list[int]:
    from .core.custom_tokens import turn_token_into_id
    v = resolve_voice(voice)
    ids: list[int] = []
    ids.append(SOT_ID)
    ids += tok.encode(f"{v}: {text}", add_special_tokens=False)
    ids.append(EOTXT_ID)
    ids.append(SOSPEECH_ID)                       # <-- OPEN speech section
    for i, n in enumerate(PRIME):
        ids.append(turn_token_into_id(n, i))      # small audio seed AFTER SOS
    return ids



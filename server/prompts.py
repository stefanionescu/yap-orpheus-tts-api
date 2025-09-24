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

PRIME_L = 3   # <custom_token_3>
PRIME_R = [4, 5, 1]  # post-text seed tokens

def resolve_voice(v: str) -> str:
    if not v:
        return "tara"
    key = v.strip().lower()
    return ALIASES.get(key, key if key in ("tara", "zac") else "tara")

def build_prompt(text: str, voice: str = "tara") -> str:
    v = resolve_voice(voice)
    # String form using new prompt format: <custom_token_3><|begin_of_text|>voice: text<|eot_id|><custom_token_4><custom_token_5><custom_token_1>
    from .core.custom_tokens import turn_token_into_id
    # Build token IDs and then decode for string compatibility
    ids = [turn_token_into_id(PRIME_L, 0)]  # <custom_token_3>
    ids += [SOT_ID]                         # <|begin_of_text|>
    ids += _tok.encode(f"{v}: {text}", add_special_tokens=False)
    ids += [EOTXT_ID]
    for i, n in enumerate(PRIME_R):
        ids.append(turn_token_into_id(n, i+1))  # <custom_token_4><_5><_1>
    return _tok.decode(ids, skip_special_tokens=False)


def build_prompt_ids(text: str, voice: str, tok) -> list[int]:
    from .core.custom_tokens import turn_token_into_id
    v = resolve_voice(voice)
    ids: list[int] = []
    # pre-seed one audio token + BOS
    ids.append(turn_token_into_id(PRIME_L, 0))   # <custom_token_3>
    ids.append(SOT_ID)                           # <|begin_of_text|>
    # "tara: {text}"
    ids += tok.encode(f"{v}: {text}", add_special_tokens=False)
    # close text and sprinkle a few audio primes
    ids.append(EOTXT_ID)                         # <|end_of_text|>
    for i, n in enumerate(PRIME_R, start=1):
        ids.append(turn_token_into_id(n, i))     # <custom_token_4><_5><_1>
    return ids



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
SOH_ID = 128259           # START_OF_HUMAN
SOT_ID = 128000           # START_OF_TEXT (often equals BOS)
EOTXT_ID = 128009         # END_OF_TEXT (same numeric as EOS)
EOH_ID = 128260           # END_OF_HUMAN
SOAI_ID = 128261          # START_OF_AI
SOSPEECH_ID = 128257      # START_OF_SPEECH
EO_SPEECH_ID = 128258     # END_OF_SPEECH
EOS_ID = 128009

# String segments for engines that accept string prompts (e.g., vLLM path)
_SOH = _tok.decode([SOH_ID, SOT_ID])
_END = _tok.decode([EOTXT_ID, EOH_ID, SOAI_ID, SOSPEECH_ID])

def resolve_voice(v: str) -> str:
    if not v:
        return "tara"
    key = v.strip().lower()
    return ALIASES.get(key, key if key in ("tara", "zac") else "tara")

def build_prompt(text: str, voice: str = "tara") -> str:
    v = resolve_voice(voice)
    # String form: [SOH][SOT] v:
    # text [EOTXT][EOH][SOAI][SOSPEECH]
    return f"{_SOH}{v}: {text}{_END}"


def build_prompt_ids(text: str, voice: str, tok: AutoTokenizer) -> list[int]:
    v = resolve_voice(voice)
    bos = tok.bos_token_id if tok.bos_token_id is not None else None
    # Avoid duplicate BOS/SOT when BOS equals SOT (128000 for Orpheus)
    pre: list[int] = []
    if (bos is not None) and (bos != SOT_ID):
        pre.append(int(bos))
    pre += [SOH_ID, SOT_ID]
    voice_and_text_ids = tok.encode(f"{v}: {text}", add_special_tokens=False)
    ids = pre + voice_and_text_ids + [EOTXT_ID, EOH_ID, SOAI_ID, SOSPEECH_ID]
    return ids



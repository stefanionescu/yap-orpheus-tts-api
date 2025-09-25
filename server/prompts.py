import os
from transformers import AutoTokenizer

ALIASES = {
    "female": "tara",
    "male": "zac",
}

MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")

# Use TOKENIZER_DIR to match engine build
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", os.getenv("MODEL_LOCAL_DIR", MODEL_ID))

# Cache tokenizer at import time from correct location
_tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

# Derive special token IDs from tokenizer (robust approach)
def _special_token_id(token_string: str) -> int:
    """Get token ID for special token, with fallback if not found"""
    try:
        return int(_tok.convert_tokens_to_ids(token_string))
    except Exception:
        # Fallback: try to find in added vocab
        added_vocab = getattr(_tok, "get_added_vocab", lambda: {})()
        if token_string in added_vocab:
            return int(added_vocab[token_string])
        raise RuntimeError(f"Special token '{token_string}' not found in tokenizer")

# Robust special token IDs derived from tokenizer
SOT_ID = _tok.bos_token_id if _tok.bos_token_id is not None else _special_token_id("<|begin_of_text|>")
EOTXT_ID = _special_token_id("<|eot_id|>")          # llama3-specific eot id  
SOSPEECH_ID = _special_token_id("<|start_of_speech|>")
EO_SPEECH_ID = _special_token_id("<|end_of_speech|>")
EOS_ID = EOTXT_ID  # Same as eot_id for Llama-3 style

# Legacy IDs (not used in new prompt but kept for compatibility)
SOH_ID = 128259           # START_OF_HUMAN 
EOH_ID = 128260           # END_OF_HUMAN 
SOAI_ID = 128261          # START_OF_AI 

# Validate special tokens decode correctly
try:
    assert _tok.decode([EO_SPEECH_ID]) == "<|end_of_speech|>", f"EO_SPEECH_ID decode failed: {_tok.decode([EO_SPEECH_ID])}"
    assert _tok.decode([SOSPEECH_ID]) == "<|start_of_speech|>", f"SOSPEECH_ID decode failed: {_tok.decode([SOSPEECH_ID])}"
    assert _tok.decode([EOTXT_ID]) == "<|eot_id|>", f"EOTXT_ID decode failed: {_tok.decode([EOTXT_ID])}"
    print(f"✓ Special tokens validated: tokenizer_dir={TOKENIZER_DIR}")
    print(f"  EO_SPEECH_ID={EO_SPEECH_ID} -> '{_tok.decode([EO_SPEECH_ID])}'")
    print(f"  SOSPEECH_ID={SOSPEECH_ID} -> '{_tok.decode([SOSPEECH_ID])}'") 
    print(f"  EOTXT_ID={EOTXT_ID} -> '{_tok.decode([EOTXT_ID])}'")
except Exception as e:
    print(f"⚠️  Special token validation failed: {e}")
    raise

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



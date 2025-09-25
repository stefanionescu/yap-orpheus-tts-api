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

# Orpheus special token IDs (use numeric IDs, not string lookups that crash!)
SOT_ID = 128000           # <|begin_of_text|>
EOTXT_ID = 128009         # <|eot_id|>
SOSPEECH_ID = 128257      # <custom_token_1> == Start of Speech in Orpheus  
EO_SPEECH_ID = 128258     # End of Speech
EOS_ID = EOTXT_ID         # Same as eot_id for Llama-3 style

# Legacy IDs (not used in new prompt but kept for compatibility)
SOH_ID = 128259           # START_OF_HUMAN 
EOH_ID = 128260           # END_OF_HUMAN 
SOAI_ID = 128261          # START_OF_AI 

# Validate tokenizer has the expected tokens at startup
try:
    # Check that these IDs decode to expected forms
    sospeech_decoded = _tok.decode([SOSPEECH_ID])
    eos_decoded = _tok.decode([EO_SPEECH_ID]) 
    eotxt_decoded = _tok.decode([EOTXT_ID])
    
    print(f"✓ Orpheus special tokens validated: tokenizer_dir={TOKENIZER_DIR}")
    print(f"  SOSPEECH_ID={SOSPEECH_ID} -> '{sospeech_decoded}'")  # Should be <custom_token_1>
    print(f"  EO_SPEECH_ID={EO_SPEECH_ID} -> '{eos_decoded}'")
    print(f"  EOTXT_ID={EOTXT_ID} -> '{eotxt_decoded}'")
    
    # Verify we have thousands of custom tokens (critical for audio)
    added_vocab = getattr(_tok, "get_added_vocab", lambda: {})()
    custom_count = sum(k.startswith("<custom_token_") for k in added_vocab)
    print(f"✓ Found {custom_count} <custom_token_*> entries in tokenizer")
    if custom_count < 1000:
        print(f"⚠️  Expected thousands of custom tokens, only found {custom_count}")
        
except Exception as e:
    print(f"⚠️  Tokenizer validation failed: {e}")
    # Don't crash on validation - let the engine handle it

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



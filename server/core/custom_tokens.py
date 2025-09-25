import re
from typing import Optional, Tuple, Pattern, Any
import os

_AUDIO_RX: Optional[Pattern[str]] = None
_AUDIO_ID_TO_CODE = None


def _family_key(s: str) -> Optional[Tuple[str, str]]:
    # Split into prefix, digits, suffix (e.g., "<ct_", "12345", ">")
    m = re.match(r"^(.*?)(\d+)(.*?)$", s)
    if not m:
        return None
    return (m.group(1), m.group(3))


def get_audio_token_regex(tokenizer: Any = None) -> Pattern[str]:
    """
    Inspect the tokenizer's added vocab and choose the largest 'numeric family'
    (prefix + digits + suffix). That is almost always the audio code family.
    Result is memoized per process (single tokenizer assumed).
    """
    global _AUDIO_RX
    if _AUDIO_RX is not None:
        return _AUDIO_RX

    # Allow explicit override via environment for fast debugging
    override = os.getenv("ORPHEUS_AUDIO_TOKEN_REGEX")
    if override:
        try:
            _AUDIO_RX = re.compile(override)
            return _AUDIO_RX
        except Exception:
            pass

    # Fallback if tokenizer isn't provided or get_added_vocab not available
    default_rx = re.compile(r"<custom_token_(\d+)>")
    if tokenizer is None:
        _AUDIO_RX = default_rx
        return _AUDIO_RX

    try:
        added = getattr(tokenizer, "get_added_vocab", lambda: {})()
    except Exception:
        added = {}

    # Heuristic: pick the family (prefix/suffix around digits) with the most members
    buckets: dict[Tuple[str, str], int] = {}
    for k in added.keys():
        fk = _family_key(k)
        if fk is None:
            continue
        buckets[fk] = buckets.get(fk, 0) + 1

    if not buckets:
        _AUDIO_RX = default_rx
        return _AUDIO_RX

    (pref, suff), _ = max(buckets.items(), key=lambda kv: kv[1])
    escaped_pref = re.escape(pref)
    escaped_suff = re.escape(suff)
    pat = rf"{escaped_pref}(\d+){escaped_suff}"
    _AUDIO_RX = re.compile(pat)
    return _AUDIO_RX


def build_audio_id_lookup(tokenizer) -> dict[int, int]:
    """
    Map *token_id* -> *audio_code* using the tokenizer's added vocab.
    This is stable across save/restore and avoids ID-range guesses.
    
    Returns dict mapping token_id to audio_code (e.g., 3929).
    """
    global _AUDIO_ID_TO_CODE
    if _AUDIO_ID_TO_CODE is not None:
        return _AUDIO_ID_TO_CODE
    
    # Use added_vocab to build precise mapping instead of scanning ID ranges
    try:
        added_vocab = getattr(tokenizer, "get_added_vocab", lambda: {})()
    except Exception:
        added_vocab = {}
    
    id2code = {}
    for tok_str, tok_id in added_vocab.items():
        if tok_str.startswith("<custom_token_") and tok_str.endswith(">"):
            try:
                code = int(tok_str[len("<custom_token_"):-1])
                id2code[int(tok_id)] = code
            except Exception:
                continue
    
    if not id2code:
        raise RuntimeError(
            f"No <custom_token_*> entries found in tokenizer added_vocab; "
            f"tokenizer/engine mismatch. Found {len(added_vocab)} total added tokens."
        )
    
    _AUDIO_ID_TO_CODE = id2code
    print(f"✓ Built audio ID lookup: {len(id2code)} custom tokens mapped")
    return id2code


def turn_token_into_id(token_number: int, index: int) -> int:
    # Map to lane-local 12-bit code (wrap within [0, 4096)) to avoid negatives/overflow
    lane = index % 7
    return int(((token_number - 10) - lane * 4096) & 0xFFF)



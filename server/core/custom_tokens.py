import re
from typing import Optional, Tuple, Pattern, Any

_AUDIO_RX: Optional[Pattern[str]] = None


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


def split_custom_tokens(s: str, tokenizer: Any = None) -> list[int]:
    """
    Extract the numeric part from whatever the tokenizer uses for audio codes.
    Provide `tokenizer` when available to auto-detect the correct pattern.
    """
    rx = get_audio_token_regex(tokenizer)
    return [int(x) for x in rx.findall(s) if x != "0"]


def turn_token_into_id(token_number: int, index: int) -> int:
    # Baseten’s exact rule
    return token_number - 10 - ((index % 7) * 4096)



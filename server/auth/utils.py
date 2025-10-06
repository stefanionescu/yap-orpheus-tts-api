from typing import Optional
from huggingface_hub import login
from server.config import settings


def ensure_hf_login():
    tok = settings.hf_token
    if not tok:
        raise RuntimeError("HF_TOKEN not set")
    login(token=tok, add_to_git_credential=False)


def extract_api_key_from_ws_headers(raw_headers) -> Optional[str]:
    """
    Extract API key from WebSocket headers.
    Only supports Authorization: Bearer <token>.
    Returns key string or None.
    """
    try:
        headers = {k.decode().lower(): v.decode() for k, v in raw_headers if isinstance(k, (bytes, bytearray))}
    except Exception:
        return None

    auth_header = headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return None


def is_api_key_authorized(provided_key: Optional[str]) -> bool:
    expected = settings.api_key or ""
    got = (provided_key or "").strip()
    # If expected is empty, treat as disabled auth? We default to yap_api_key, so compare stringly.
    return got == expected


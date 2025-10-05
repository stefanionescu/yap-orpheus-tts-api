from huggingface_hub import login
from ..config import settings


def ensure_hf_login():
    tok = settings.hf_token
    if not tok:
        raise RuntimeError("HF_TOKEN not set")
    login(token=tok, add_to_git_credential=False)



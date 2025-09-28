import os
from huggingface_hub import login


def ensure_hf_login():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN not set")
    login(token=tok, add_to_git_credential=False)



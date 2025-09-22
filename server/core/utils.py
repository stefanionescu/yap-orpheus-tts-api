import os
from huggingface_hub import login

from .logging_config import get_logger

logger = get_logger(__name__)


def ensure_hf_login():
    logger.debug("Checking HuggingFace token...")
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        logger.error("HF_TOKEN environment variable not set")
        raise RuntimeError("HF_TOKEN not set")
    
    logger.info("Logging into HuggingFace...")
    try:
        login(token=tok, add_to_git_credential=False)
        logger.info("HuggingFace login successful")
    except Exception as e:
        logger.error(f"HuggingFace login failed: {e}", exc_info=True)
        raise



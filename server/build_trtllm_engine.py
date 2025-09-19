#!/usr/bin/env python
import os
from huggingface_hub import snapshot_download  # type: ignore
from tensorrt_llm import LLM  # type: ignore


def main() -> None:
    model_id = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")
    out_dir = os.getenv("ENGINE_DIR", "engine/orpheus_a100_fp16_kvint8")

    print(f"[build-trtllm] model={model_id}")
    print(f"[build-trtllm] output dir={out_dir}")

    # Prefer using a pre-downloaded minimal local copy to avoid huge training artifacts
    local_model_dir = os.getenv("MODEL_LOCAL_DIR", os.path.join("models", "orpheus_hf"))
    os.makedirs(local_model_dir, exist_ok=True)
    allow = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "*.safetensors",
        "model.safetensors*",
    ]
    print(f"[build-trtllm] ensuring minimal HF files at {local_model_dir}")
    snapshot_download(
        repo_id=model_id,
        token=os.getenv("HF_TOKEN"),
        local_dir=local_model_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow,
    )

    batch = int(os.getenv("TRTLLM_MAX_BATCH", "24"))
    inp = int(os.getenv("TRTLLM_MAX_INPUT", "160"))
    out = int(os.getenv("TRTLLM_MAX_OUTPUT", "2048"))
    seq = int(os.getenv("TRTLLM_MAX_SEQ", "3072"))

    llm = None
    try:
        # Try with shape caps (no plugin-specific args)
        llm = LLM(
            model=local_model_dir,
            dtype="float16",
            max_batch_size=batch,
            max_input_len=inp,
            max_seq_len=seq,
        )
    except Exception:
        # Minimal constructor fallback
        llm = LLM(
            model=local_model_dir,
            dtype="float16",
        )
    os.makedirs(out_dir, exist_ok=True)
    llm.save_engine(out_dir)
    print("[build-trtllm] Done.")


if __name__ == "__main__":
    main()



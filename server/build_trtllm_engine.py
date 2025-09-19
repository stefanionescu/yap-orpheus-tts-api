#!/usr/bin/env python
import os
from tensorrt_llm import LLM  # type: ignore


def main() -> None:
    model_id = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")
    out_dir = os.getenv("ENGINE_DIR", "engine/orpheus_a100_fp16_kvint8")

    print(f"[build-trtllm] model={model_id}")
    print(f"[build-trtllm] output dir={out_dir}")

    batch = int(os.getenv("TRTLLM_MAX_BATCH", "24"))
    inp = int(os.getenv("TRTLLM_MAX_INPUT", "160"))
    out = int(os.getenv("TRTLLM_MAX_OUTPUT", "2048"))
    seq = int(os.getenv("TRTLLM_MAX_SEQ", "3072"))

    llm = None
    try:
        # Try with shape caps (no plugin-specific args)
        llm = LLM(
            model=model_id,
            dtype="float16",
            max_batch_size=batch,
            max_input_len=inp,
            max_seq_len=seq,
        )
    except Exception:
        # Minimal constructor fallback
        llm = LLM(
            model=model_id,
            dtype="float16",
        )
    os.makedirs(out_dir, exist_ok=True)
    llm.save_engine(out_dir)
    print("[build-trtllm] Done.")


if __name__ == "__main__":
    main()



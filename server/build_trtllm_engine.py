#!/usr/bin/env python
import os
from tensorrt_llm import LLM  # type: ignore


def main() -> None:
    model_id = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")
    out_dir = os.getenv("ENGINE_DIR", "engine/orpheus_a100_fp16_kvint8")

    print(f"[build-trtllm] model={model_id}")
    print(f"[build-trtllm] output dir={out_dir}")

    llm = LLM(
        model=model_id,
        dtype="float16",
        max_batch_size=int(os.getenv("TRTLLM_MAX_BATCH", "24")),
        max_input_len=int(os.getenv("TRTLLM_MAX_INPUT", "160")),
        max_output_len=int(os.getenv("TRTLLM_MAX_OUTPUT", "2048")),
        max_seq_len=int(os.getenv("TRTLLM_MAX_SEQ", "3072")),
        gpt_attention_plugin="auto",
        gemm_plugin="auto",
        use_paged_context_fmha=True,
        kv_cache_quantization="int8",
        enable_chunked_context=True,
    )
    os.makedirs(out_dir, exist_ok=True)
    llm.save_engine(out_dir)
    print("[build-trtllm] Done.")


if __name__ == "__main__":
    main()



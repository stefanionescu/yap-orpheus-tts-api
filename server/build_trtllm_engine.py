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
        llm = LLM(
            model=model_id,
            dtype="float16",
            max_batch_size=batch,
            max_input_len=inp,
            max_output_len=out,
            max_seq_len=seq,
            gpt_attention_plugin="auto",
            gemm_plugin="auto",
            use_paged_context_fmha=True,
            kv_cache_quantization="int8",
            enable_chunked_context=True,
        )
    except Exception:
        try:
            # Some TRT-LLM versions use different arg names; try without max_output_len
            llm = LLM(
                model=model_id,
                dtype="float16",
                max_batch_size=batch,
                max_input_len=inp,
                max_seq_len=seq,
                gpt_attention_plugin="auto",
                gemm_plugin="auto",
                use_paged_context_fmha=True,
                kv_cache_quantization="int8",
                enable_chunked_context=True,
            )
        except Exception:
            # Fallback to from_hugging_face API which is more permissive across versions
            from tensorrt_llm import LLM as _LLM
            print("[build-trtllm] Constructor failed; falling back to LLM.from_hugging_face()")
            llm = _LLM.from_hugging_face(
                model=model_id,
                dtype="float16",
                max_batch_size=batch,
                max_input_len=inp,
                max_output_len=out,
                max_seq_len=seq,
                gpt_attention_plugin="auto",
                use_paged_context_fmha=True,
                kv_cache_quantization="int8",
                enable_chunked_context=True,
            )
    os.makedirs(out_dir, exist_ok=True)
    llm.save_engine(out_dir)
    print("[build-trtllm] Done.")


if __name__ == "__main__":
    main()



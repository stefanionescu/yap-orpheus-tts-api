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
    # Build with TF32 enabled to match runtime default and avoid Myelin mismatch
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
    try:
        llm = LLM(
            model=local_model_dir,
            dtype="float16",
            max_batch_size=batch,
            max_input_len=inp,
            max_seq_len=seq,
        )
    except Exception:
        llm = LLM(
            model=local_model_dir,
            dtype="float16",
        )
    # Persist engine to out_dir across TRT-LLM versions
    os.makedirs(out_dir, exist_ok=True)
    saved = False
    # 1) Newer API
    if hasattr(llm, "save_engine"):
        try:
            llm.save_engine(out_dir)  # type: ignore[attr-defined]
            saved = True
        except Exception:
            pass
    # 2) Some versions expose .save()
    if (not saved) and hasattr(llm, "save"):
        try:
            llm.save(out_dir)  # type: ignore[attr-defined]
            saved = True
        except Exception:
            pass
    # 3) Fallback: copy from resolved engine_dir
    if not saved:
        import shutil
        engine_src = None
        for attr in ("get_engine_dir", "engine_dir", "_engine_dir"):
            v = getattr(llm, attr, None)
            if callable(v):
                try:
                    engine_src = v()
                except Exception:
                    continue
            elif isinstance(v, str):
                engine_src = v
            if engine_src:
                break
        if not engine_src:
            raise AttributeError("LLM engine save not supported by this version and engine_dir is unknown")
        if os.path.abspath(engine_src) != os.path.abspath(out_dir):
            shutil.copytree(engine_src, out_dir, dirs_exist_ok=True)
        saved = True

    print("[build-trtllm] Engine saved to:", out_dir)


if __name__ == "__main__":
    main()



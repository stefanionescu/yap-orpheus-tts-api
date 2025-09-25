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
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
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

    # Use existing config.json vocab_size that matches tokenizer length (156940)
    # TRT-LLM 0.21.0+ will read vocab_size from config.json automatically

    batch = int(os.getenv("TRTLLM_MAX_BATCH", "24"))
    inp = int(os.getenv("TRTLLM_MAX_INPUT", "160"))
    out = int(os.getenv("TRTLLM_MAX_OUTPUT", "2048"))
    seq = int(os.getenv("TRTLLM_MAX_SEQ", "3072"))

    llm = None
    # Build with TF32 enabled and pass to executor environment so workers inherit it
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
    os.environ.setdefault("TRTLLM_MPI_ENV_VARS", "NVIDIA_TF32_OVERRIDE")
    
    llm = LLM(
        model=local_model_dir,
        dtype="float16",
        max_batch_size=batch,
        max_input_len=inp,
        max_seq_len=seq,
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

    # Optional verification after saving the engine
    try:
        import json
        cfg_files = ["config.json", "engine_config.json"]
        for name in cfg_files:
            p = os.path.join(out_dir, name)
            if os.path.exists(p):
                with open(p, "r") as f:
                    ecfg = json.load(f)
                vs = ecfg.get("vocab_size") or ecfg.get("builder_config", {}).get("vocab_size")
                if vs:
                    print(f"[build-trtllm] engine vocab_size = {vs}")
    except Exception as e:
        print("[build-trtllm] (warn) could not read engine config:", e)


if __name__ == "__main__":
    main()



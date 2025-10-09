---
license: {{license}}
base_model: {{base_model}}
tags:
- awq
- quantized
- {{w_bit}}-bit
- tensorrt-llm
- orpheus-tts
language:
- en
pipeline_tag: text-to-speech
---

<div align="center">
  <h1 style="font-size: 48px; color: #2E86AB; font-weight: bold;">
    {{model_name}} — INT{{w_bit}} AWQ Quantized
  </h1>
  <p style="font-size: 18px; color: #666;">
    High-performance INT{{w_bit}} AWQ quantization for production TTS with TensorRT-LLM
  </p>
</div>

---

<div align="center">
  <img src="https://img.shields.io/badge/Quantization-AWQ-blue?style=for-the-badge" alt="AWQ">
  <img src="https://img.shields.io/badge/Precision-INT{{w_bit}}-green?style=for-the-badge" alt="INT{{w_bit}}">
  <img src="https://img.shields.io/badge/Framework-TensorRT--LLM-red?style=for-the-badge" alt="TensorRT-LLM">
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow?style=for-the-badge" alt="License">
</div>

---

## Model Overview

This is a production-grade INT{{w_bit}} AWQ quantized version of {{source_model_link}}, optimized for low-latency text-to-speech with TensorRT-LLM.

**Key Features:**
- **Optimized for Production**: Built for high-throughput, low-latency TTS serving
- **Faster Inference**: Up to ~3x faster than FP16 with minimal perceived quality loss
- **Memory Efficient**: ≈4x smaller weights vs. FP16 (INT{{w_bit}})
- **Ready for Streaming**: Designed for real-time streaming TTS backends
- **Calibrated**: Quantized with high-quality calibration data using AWQ

---

## Technical Specifications

| Specification | Details |
|---------------|---------|
| **Source Model** | {{source_model_link}} |
| **Quantization Method** | AWQ (Activation-aware Weight Quantization) |
| **Precision** | INT{{w_bit}} weights, INT8 KV cache |
| **AWQ Group/Block Size** | {{q_group_size}} |
| **TensorRT-LLM Version** | `{{awq_version}}` |
| **Generated** | {{generated_at}} |
| **Pipeline** | TensorRT-LLM AWQ Quantization |

### Artifact Layout

```
trt-llm/
  checkpoints/            # Quantized TRT-LLM checkpoints (portable)
    *.safetensors
    config.json
  engines/{{engine_label}}/     # Built TensorRT-LLM engines (hardware-specific)
    rank*.engine
    build_metadata.json
    build_command.sh
```

### Size Comparison

| Version | Size | Memory Usage | Speed |
|---------|------|--------------|-------|
| Original FP16 | {{original_size_gb}} GB | ~{{original_size_gb}} GB VRAM | 1x |
| **AWQ INT{{w_bit}}** | **{{quantized_size_gb}} GB** | **~{{quantized_size_gb}} GB VRAM** | **up to ~3x** |

> Note: Engine file sizes vary by GPU architecture and build options.

---

## Quick Start

### Download Artifacts (Python)

```python
from huggingface_hub import snapshot_download

# Download quantized checkpoints (portable)
ckpt_path = snapshot_download(
    repo_id="{{repo_name}}",
    allow_patterns=["trt-llm/checkpoints/**"],
)

# Or download TensorRT-LLM engines for a specific build label
eng_path = snapshot_download(
    repo_id="{{repo_name}}",
    allow_patterns=["trt-llm/engines/{{engine_label}}/**"],
)
print("checkpoints:", ckpt_path)
print("engines:", eng_path)
```

### Run a Streaming TTS Server (TensorRT-LLM)

```bash
# Point your server to the downloaded engines
export TRTLLM_ENGINE_DIR=/path/to/trt-llm/engines/{{engine_label}}

# Start your TTS server (example: FastAPI + WebSocket)
python -m server.server
```

### Minimal WebSocket Client (Python)

```python
import asyncio, json, websockets

async def main():
    url = "ws://127.0.0.1:8000/ws/tts"
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    async with websockets.connect(url, extra_headers=headers, max_size=None) as ws:
        await ws.send(json.dumps({"voice": "female"}))
        await ws.send(json.dumps({"text": "The future of voice is real-time."}))
        await ws.send("__END__")
        # Receive streaming PCM16 chunks (bytes)
        total = 0
        try:
            while True:
                msg = await ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    total += len(msg)
        except Exception:
            pass
        print("received bytes:", total)

asyncio.run(main())
```

---

## Quantization Details

{{calib_section}}

### Configuration Summary

```json
{{quant_summary}}
```

---

## Performance Notes

| Metric | FP16 | INT{{w_bit}} AWQ | Improvement |
|--------|------|------------------|-------------|
| Memory Usage | 100% | ~25% | ~4x smaller |
| Latency | 1x | ~0.3–0.5x | up to ~3x faster |
| Quality | Baseline | Near-parity | Minimal perceptual loss |

> Benchmarks depend on GPU, SM arch, TRT/CUDA versions, and server integration.

---

## Use Cases

- **Realtime Voice**: assistants, product demos, interactive agents
- **High-throughput Serving**: batch TTS pipelines, APIs
- **Edge & Cost-sensitive**: limited VRAM environments

---

## Advanced Configuration (Build-time)

- Max input length: tune `--max_input_len`
- Max output length: tune `--max_seq_len`
- Batch size: tune `--max_batch_size`
- Plugins: `--gpt_attention_plugin`, `--context_fmha`, `--paged_kv_cache`

---

## Requirements & Compatibility

### System Requirements
- **GPU**: NVIDIA, Compute Capability ≥ 8.0 (A100/RTX 40/H100 class recommended)
- **VRAM**: ≥ {{quantized_size_gb}} GB for INT{{w_bit}} engines (per GPU)
- **CUDA**: 12.x recommended
- **Python**: 3.10+

### Framework Compatibility
- **TensorRT-LLM** (engines), version `{{awq_version}}`
- **TRT-LLM Checkpoints** are portable across systems; engines are not

### Installation

```bash
pip install huggingface_hub
# Install TensorRT-LLM per NVIDIA docs
# https://nvidia.github.io/TensorRT-LLM/
```

---

## Troubleshooting

<details>
<summary><b>Engine not portable</b></summary>
Engines are specific to GPU SM and TRT/CUDA versions. Rebuild on the target
system or download a matching `engines/{{engine_label}}` variant if provided.
</details>

<details>
<summary><b>OOM or Slow Loading</b></summary>
Reduce `max_seq_len`, lower `max_batch_size`, and ensure `gpu_memory_utilization`
on your server is tuned to your GPU.
</details>

---

## Additional Resources

- **TensorRT-LLM Docs**: https://nvidia.github.io/TensorRT-LLM/
- **Activation-aware Weight Quantization (AWQ)**: https://github.com/mit-han-lab/llm-awq

---

## License

This quantized model inherits the license from the original model: **Apache 2.0**



# Stable Environment Configuration

**System:** Ubuntu 24.04 LTS
**GPU:** NVIDIA RTX 4090 (24GB VRAM)
**Driver Version:** 575.57.08
**CUDA Version:** 12.9 (Compatible with Torch +cu124)

## Verified Dependencies

- **Python:** 3.11
- **PyTorch:** 2.6.0+cu124
- **Diffusers:** Optimized for SDPA fallback
- **LM:** 0.6B V4-fix (requires ~10GB VRAM headroom)

## Required Environment Variables

```bash
# Optimized for RTX 4090 memory segmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Simulated VRAM cap if needed for multi-instance stability
export MAX_CUDA_VRAM=10
```

## Patch Status

- [x] `pyproject.toml` downgraded to stable Torch.
- [x] `acestep/third_parts/nano-vllm/pyproject.toml` removed `flash-attn` wheels.

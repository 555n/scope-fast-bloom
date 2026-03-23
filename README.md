# Scope Bloom

GPU bloom/glow postprocessor for [Daydream Scope](https://daydream.live).

Extracts bright areas above a threshold, applies fast downsample-upsample blur, and blends back as additive glow. Classic bloom effect, real-time on MPS and CUDA.

## Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Amount | 0-2.0 | 0.3 | Glow intensity |
| Radius | 1-8 | 4 | Glow spread (1=tight, 8=wide) |
| Threshold | 0-1.0 | 0.7 | Brightness cutoff — only pixels above this glow |

## Install

In Scope **Settings > Nodes**, paste:
```
git+https://github.com/555n/scope-bloom.git
```

## Notes

- Postprocessor — add after diffusion, before RIFE
- All parameters runtime-controllable and sequenceable
- Fast bilinear downsample/upsample — no convolution kernels
- MPS-safe (Apple Silicon compatible)

MIT License

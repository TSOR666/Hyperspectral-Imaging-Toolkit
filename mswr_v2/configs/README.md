# MSWR Training Configs

Use one training entry point and one default config:

```bash
python train_mswr_v212_logging.py \
  --config configs/train.yaml \
  --data_root /path/to/ARAD_1K
```

`train.yaml` is the maintained recipe for new runs. It uses the base model,
spectral attention, one wavelet level per stage, MRAE loss, AdamW, AMP, and EMA.
CLI arguments may override individual YAML values.

The files under `experiments/` are frozen comparison recipes:

| Config | Purpose |
| --- | --- |
| `baseline_mstpp.yaml` | Strict MST++-style optimizer and no-EMA control. |
| `ablation_regularized_spatial.yaml` | Regularization/EMA ablation without spectral attention. |
| `ablation_spectral_default_wavelets.yaml` | Spectral attention with the older `[1, 2, 3]` wavelet schedule. |
| `sota_mrae_ssim.yaml` | MRAE-primary multi-objective recipe with light SSIM/SAM regularization and auto raw/EMA selection. |
| `ablation_landmark_adaptive.yaml` | Canonical recipe with `landmark_pooling: adaptive` (content-dependent global mixing) vs the default static `learned`. |
| `ablation_wavelet_detail.yaml` | Canonical recipe with `wavelet_detail_processing: true` (lightweight depthwise residual on the LH/HL/HH detail bands). |
| `round5_wavelet_detail.yaml` | Canonical MRAE-only recipe + `wavelet_detail_processing` (capacity-flat, +0.17% params) + fp16 host cache. The param-light round-5 quality-ceiling experiment. |
| `robust_mrae.yaml` | Robust anti-overfit MRAE-only recipe: EMA + `drop_path`/`attention_dropout` + `wavelet_detail` + fp32 honest best-EMA selection + shorter (220-epoch) cosine. Config-reachable ceiling ~0.235–0.250 val MRAE; the remaining gap to MST++ 0.165 is architectural, not tuning. The CONTROL arm for the Rung-0 spectral A/B. |
| `robust_mrae_fullrank.yaml` | TREATMENT arm: byte-identical to `robust_mrae.yaml` except `spectral_attn_heads: 1`, which makes the existing band-to-band S-MSA full-rank (C×C) instead of 8-head block-diagonal. ~0 added params (35 fewer). Isolates whether spectral-attention rank is the architectural deficit. |

Do not choose an experiment config for ordinary training unless you are
reproducing that specific comparison.

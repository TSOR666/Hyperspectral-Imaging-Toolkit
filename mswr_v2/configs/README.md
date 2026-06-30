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
| `robust_mrae_fullrank.yaml` | `robust_mrae.yaml` + `spectral_attn_heads: 1` → full-rank (C×C) band-to-band S-MSA instead of 8-head block-diagonal. ~0 added params (35 fewer). **WON the Rung-0 A/B (0.250→0.229 val MRAE)** and is now the baseline recipe going forward. |
| `robust_mrae_spectralffn.yaml` | `robust_mrae_fullrank.yaml` + `spectral_ffn: true` (`spectral_ffn_mult: 2`): a zero-init-gated GDFN spectral feed-forward residual per block — the MSAB FFN the spectral branch lacked, and the non-redundant addition once attention is full-rank. ~+0.44M params (+14.5%), identity at init, checkpoint-safe. Won the Rung-1 A/B (−0.01..−0.018 MRAE); current working baseline. |
| `robust_mrae_shortcos.yaml` | Overfit attack (REFUTED 2026-06-25): `robust_mrae_spectralffn.yaml` with `end_epoch: 140` (was 220). Made it **~0.03 worse** (best EMA 0.269 vs 0.240) — this model needs the long high-LR schedule to reach its minimum; the shorter cosine froze it in a shallow one. Kept as a documented negative result; use the 220-epoch `robust_mrae_spectralffn.yaml` as the baseline. |
| `robust_mrae_multistage.yaml` | `robust_mrae_spectralffn.yaml` + `multistage_refine: true` (`refine_hidden: 64`): an MPRNet-style coarse-to-fine refinement stage on the final output — a small conv stage refines the stage-1 reconstruction conditioned on the original RGB, as a zero-init-gated residual. ~+47K params (+1.4%), identity at init, checkpoint-safe. Rung-2 single-variable A/B vs `robust_mrae_spectralffn.yaml` (keep 220ep). |

Do not choose an experiment config for ordinary training unless you are
reproducing that specific comparison.

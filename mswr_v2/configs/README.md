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

Do not choose an experiment config for ordinary training unless you are
reproducing that specific comparison.

# Hyperspectral Imaging Toolkit

This monorepo hosts production-ready code and utilities for hyperspectral image (HSI) reconstruction, evaluation, and visualization. Each project started life as an independent package and has been curated here with the fixes that were applied in practice. Use this README as an entry point for understanding what lives where and how to get started quickly.

## Repository layout

| Path | Description |
| --- | --- |
| [`CSWIN v2/`](CSWIN%20v2/README.md) | Sinkhorn-GAN training pipelines for a noise-robust CSWin transformer. |
| [`HSIFUSION&SHARP/`](HSIFUSION&SHARP/README.md) | Transformer baselines featuring HSIFusionNet v2.5.3 and SHARP v3.2.2 with hardened training scripts. |
| [`hsi_viz_suite/`](hsi_viz_suite/README.md) | Stand-alone visualization suite for turning reconstruction results into publication-ready figures. |
| [`mswr_v2/`](mswr_v2/README.md) | Training and inference scripts for the MSWR-Net v2.1.2 architecture with robustness patches. |

> 💡 The directory names are preserved from their original projects. Scripts assume you execute them from inside their respective folders (for example `cd mswr_v2` before running a training command).

## Prerequisites

All projects target **Python 3.9+** and PyTorch environments with CUDA acceleration. Recommended setup:

1. Create an isolated environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
2. Install per-project requirements from the provided `requirements.txt` (where available) or your curated environment modules.
3. Ensure the ARAD-1K dataset or compatible hyperspectral data is available locally. Training scripts default to `./data/ARAD_1K` but accept overrides via configuration flags or environment variables.

## Quick start per project

### CSWIN v2

```bash
cd "CSWIN v2"
pip install -r requirements.txt  # if you have a curated requirements file
python -m pip install hydra-core torch torchvision  # minimal dependencies
python src/hsi_model/training_script_fixed.py --config-name config
```

Key environment variables:

- `HSI_DATA_DIR` – dataset root (defaults to `./data/ARAD_1K`).
- `HSI_LOG_DIR` / `HSI_CKPT_DIR` – custom log & checkpoint destinations.
- `PYTORCH_CUDA_ALLOC_CONF` – set to `expandable_segments:True,max_split_size_mb:256` to mirror the memory tweaks baked into the scripts.

See the [CSWIN v2 README](CSWIN%20v2/README.md) for distributed training tips and an in-depth feature tour.

### HSIFusion & SHARP

```bash
cd "HSIFUSION&SHARP"
python dataset_setup.py --arad-root /path/to/raw/ARAD_1K --output-root ./data/ARAD_1K
python hsifusion_training.py --data_root ./data/ARAD_1K --model_size base
python sharp_training_script_fixed.py --data_root ./data/ARAD_1K --model_size base
```

Highlights:

- Shared MST++-style dataloaders (`optimized_dataloader.py`) work across both transformer trainers.
- `sharp_inference.py` can tile large RGB inputs and export `.npy` hyperspectral reconstructions for downstream evaluation.
- SLURM examples (`train_job_HSI.sh`, `train_job_SHARP.sh`) show how to schedule multi-GPU jobs with consistent logging dirs.

See the [HSIFusion & SHARP README](HSIFUSION&SHARP/README.md) for dataset staging, CLI options, and inference details.

### HSI visualization suite

```bash
cd hsi_viz_suite
pip install -r requirements.txt
python scripts/generate_all_visualizations.py \
  --results /path/to/model_outputs \
  --output figs
```

Point `--results` at a folder that contains `hsi/*.npy` reconstructions and optional `metrics.json`. The suite produces PNG/PDF figures for qualitative, quantitative, and spectral comparisons. More options are described in the [suite README](hsi_viz_suite/README.md).

### MSWR v2

```bash
cd mswr_v2
pip install -r requirements.txt  # supply your own file if needed
python train_mswr_v212_logging.py --model_size base --data_root /path/to/ARAD_1K
```

MSWR scripts expect the legacy `dataloader.py` module on the Python path. The training driver enables EMA, SAM loss, and extensive logging by default; refer to the [MSWR README](mswr_v2/README.md) for CLI flags and inference notes.

## Contributing

- Each subproject retains its own logging directories and checkpoints. Please keep changes scoped to the relevant folder to avoid cross-project regressions.
- Update the individual README files if you touch training defaults, configuration schemas, or output formats.
- Use conventional Git workflows (`feature` branches + pull requests) to keep history readable.

## Support

Issues and improvements typically surface from training runs or visualization gaps. When reporting problems, include:

1. The project folder (`CSWIN v2`, `hsi_viz_suite`, or `mswr_v2`).
2. The command (with arguments) you ran and the environment description (CUDA version, GPU model).
3. Relevant log excerpts from `artifacts/logs/` or generated figure paths.

This context speeds up reproductions and ensures fixes land in the correct package.

## License

The Hyperspectral Imaging Toolkit is released under the [MIT License](LICENSE). Contributions are accepted under the same terms.

## Models At A Glance

- MSWR-Net v2.1.2 (mswr_v2)
  - Architecture: Dual-attention U-Net with CNN-based wavelet branches. Each stage combines window attention and landmark/global attention, optional multi-level DWT gating, and an optimized FFN. Encoder–decoder with skip connections; LayerNorm2d/AdaptiveNorm2d fixes throughout. See `mswr_v2/model/mswr_net_v212.py`.
  - Training: Enhanced loss (L1 + SSIM + SAM + gradient) with warmup, AMP, EMA, Cosine/Warmup schedulers, gradient checkpointing, flash attention. Entry: `mswr_v2/train_mswr_v212_logging.py`.
  - Configuration: `MSWRDualConfig` controls `input_channels`, `output_channels`, `base_channels`, `num_stages`, `num_heads`, `window_size`, `num_landmarks`, `use_wavelet`, `wavelet_type`, `mlp_ratio`, `ffn_type`, `drop_path`, `norm_type`, `use_checkpoint`, `use_flash_attn`, `mixed_precision`.

- HSIFusionNet v2.5.3 and SHARP v3.2.2 (HSIFUSION&SHARP)
  - HSIFusionNet Architecture: Encoder–decoder with LightningPro blocks that combine sliding-window RoPE attention, spectral attention, optional MoE, and cross-attention fusion in the decoder. Optional uncertainty head. See `HSIFUSION&SHARP/hsifusion_v252_complete.py`.
  - SHARP Architecture: Hierarchical transformer with multi-scale attention + streaming sparse attention (top-k/local window fallbacks), ChannelRMSNorm, and cross-attention fusion. Configurable RBF key projection. See `HSIFUSION&SHARP/sharp_v322_hardened.py`.
  - Training: AMP, cosine LR with warmup, AdamW, gradient clipping/accumulation, optional `torch.compile`, EMA (SHARP). Entries: `hsifusion_training.py`, `sharp_training_script_fixed.py`.

- CSWIN v2 (CSWIN v2)
  - Architecture: U-Net style generator with CSWin spatial attention + spectral attention + adaptive GroupNorm + NaN‑safe blocks; SN transformer discriminator with spectral normalization. See `CSWIN v2/src/hsi_model/models`.
  - Training: Sinkhorn‑GAN pipeline with R1 regularization, mixed precision, EMA logging, Hydra configuration (`CSWIN v2/src/configs/config.yaml`). Entries: `training_script_fixed.py`, `train_optimized.py`.

- WaveDiff (Diffusion/WaveDiff)
  - Architecture: Latent diffusion models augmented with wavelet transforms (standard/learnable/adaptive) and spectral/pixel refinement heads. See `Diffusion/WaveDiff/modules` and `Diffusion/WaveDiff/models`.
  - Training: JSON‑driven config (`Diffusion/WaveDiff/configs/example_config.json`), cosine scheduling, combined spectral losses, curriculum masking, visualization hooks. Entry: `Diffusion/WaveDiff/train.py`.

See each subfolder README for code‑level details and examples.

# Hyperspectral Imaging Toolkit

This monorepo hosts production-ready code and utilities for hyperspectral image (HSI) reconstruction, evaluation, and visualization. Each project started life as an independent package and has been curated here with the fixes that were applied in practice. Use this README as an entry point for understanding what lives where and how to get started quickly.

## Repository layout

| Path | Description |
| --- | --- |
| [`CSWIN v2/`](CSWIN%20v2/README.md) | Generator-only U-Net/Transformer hybrid with CSWin cross-shaped attention and spectral self-attention (legacy Sinkhorn-GAN trainers retained). |
| [`HSIFUSION&SHARP/`](HSIFUSION&SHARP/README.md) | Transformer baselines featuring HSIFusionNet v2.5.3 and SHARP v3.2.2, with unified MST++-faithful training/inference entry points. |
| [`hsi_viz_suite/`](hsi_viz_suite/README.md) | Stand-alone visualization suite for turning reconstruction results into publication-ready figures. |
| [`mswr_v2/`](mswr_v2/README.md) | Training and inference scripts for the MSWR-Net v2.1.2 architecture with robustness patches. |
| [`SSTrans/`](SSTrans/README.md) | HSIFormer — a self-contained spectral-spatial transformer (CSWin spatial + spectral MSA + CAT blocks) retraining pipeline for ARAD-1K. |
| [`WaveDiff/`](WaveDiff/README.md) | Latent diffusion-based HSI reconstruction with wavelet modules and spectral refinement. |
| [`hsi_benchmark/`](hsi_benchmark) | Shared benchmarking package (metrics, model loading, datasets, reports) behind [`benchmark_hsi.py`](benchmark_hsi.py). |

> 💡 The directory names are preserved from their original projects. Scripts assume you execute them from inside their respective folders (for example `cd mswr_v2` before running a training command).

## Prerequisites

All projects target **Python 3.9+** and PyTorch environments with CUDA acceleration. Recommended setup:

1. Create an isolated environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
2. Install per-project requirements from the provided `requirements.txt` (where available) or your curated environment modules.
3. Ensure the ARAD-1K dataset or compatible hyperspectral data is available locally. Training scripts default to `./data/ARAD_1K` but accept overrides via configuration flags or environment variables.

## Quick start per project

### Unified checkpoint benchmark

Use [`benchmark_hsi.py`](benchmark_hsi.py) to evaluate native checkpoints and
official MST++ model-zoo checkpoints on CAVE, ICVL, BGU, or custom datasets:

```bash
python benchmark_hsi.py \
  --model "MSWR=mswr:base@checkpoints/mswr_best.pth" \
  --model "MST++=mst:mst_plus_plus@model_zoo/mst_plus_plus.pth" \
  --mst-root ../MST-plus-plus \
  --dataset "cave=/data/CAVE" \
  --dataset "icvl=/data/ICVL" \
  --dataset "bgu=/data/BGU" \
  --output results/paper_benchmark
```

The command exports per-scene and per-band metrics, 95% confidence intervals,
runtime/memory measurements, PNG/PDF qualitative figures, hsi-viz-compatible
arrays, and CSV/Markdown/LaTeX paper tables. See
[`BENCHMARKING.md`](BENCHMARKING.md) for supported checkpoint formats,
manifests, wavelength calibration, RGB synthesis, and reproducibility notes.

### CSWIN v2

```bash
cd "CSWIN v2"
pip install torch torchvision torchaudio
pip install -r requirements.txt
python src/hsi_model/train_generator.py --config-name config data_dir=/path/to/ARAD_1K
```

The generator-only trainer (`train_generator.py`) is the active entry point; the legacy Sinkhorn-GAN trainers (`training_script_fixed.py`, `train_optimized.py`) remain for legacy experiments only.

Key environment variables:

- `HSI_DATA_DIR` – dataset root (defaults to `./data/ARAD_1K`).
- `HSI_LOG_DIR` / `HSI_CKPT_DIR` – custom log & checkpoint destinations.
- `PYTORCH_CUDA_ALLOC_CONF` – set to `expandable_segments:True,max_split_size_mb:256` to mirror the memory tweaks baked into the scripts.

See the [CSWIN v2 README](CSWIN%20v2/README.md) for distributed training tips and an in-depth feature tour.

### HSIFusion & SHARP

```bash
cd "HSIFUSION&SHARP"
python dataset_setup.py /path/to/ARAD_1K --train-ratio 0.95
python unified_training.py --model hsifusion --data_root ./data/ARAD_1K --model_size base
python unified_training.py --model sharp --data_root ./data/ARAD_1K --model_size base
python unified_inference.py experiments/unified/best_model.pth --data_root ./data/ARAD_1K
```

Highlights:

- `unified_training.py` / `unified_inference.py` are the canonical entry points: one `--model {hsifusion,sharp}` flag, MST++/NTIRE-faithful defaults, and 6-metric evaluation (MRAE/RMSE/PSNR/SAM/SSIM/MAE) via the shared `hsi_benchmark` package under both crop and full-frame protocols.
- Shared MST++-style dataloaders (`optimized_dataloader.py`) work across both transformer trainers; the legacy per-model trainers remain for the pinned regression suites.
- LSF job examples (`train_job_HSI.sh`, `train_job_SHARP.sh`) show how to schedule GPU jobs with consistent logging dirs.

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
python train_mswr_v212_logging.py --config configs/train.yaml --data_root /path/to/ARAD_1K
```

MSWR scripts expect the legacy `dataloader.py` module on the Python path. The training driver enables EMA, SAM loss, and extensive logging by default; refer to the [MSWR README](mswr_v2/README.md) for CLI flags and inference notes.

### SSTrans (HSIFormer)

```bash
cd SSTrans
uv sync --extra dev   # or: python -m pip install -e ".[dev]"
python scripts/train.py --data-root /path/to/ARAD_1K --output-dir runs/hsiformer_arad1k --device cuda
python scripts/test_ntire.py --checkpoint runs/hsiformer_arad1k/checkpoints/best.pt --data-root /path/to/ARAD_1K --output-dir outputs/public_test
```

A self-contained, author-faithful HSIFormer retraining pipeline: L1 objective, per-iteration cosine decay, progressive 128→256→512 stages, bf16 AMP with gradient clipping and non-finite guards. See the [SSTrans README](SSTrans/README.md) for presets and NTIRE cube export.

### WaveDiff

```bash
cd WaveDiff
pip install -r requirements.txt
# Training (via config)
python train.py --config configs/example_config.json
# or minimal CLI example
python train.py --model_type adaptive_wavelet --train_dir data/ARAD1K/train --val_dir data/ARAD1K/val

# Inference on a single RGB image
python inference.py --checkpoint checkpoints/<run_id>/final_model.pt --image path/to/rgb.png --output_dir results/
```

See the [WaveDiff README](WaveDiff/README.md) and [Quick Start](WaveDiff/QUICK_START.md) for detailed setup, configuration, and evaluation.

## Contributing

- Each subproject retains its own logging directories and checkpoints. Please keep changes scoped to the relevant folder to avoid cross-project regressions.
- Update the individual README files if you touch training defaults, configuration schemas, or output formats.
- Use conventional Git workflows (`feature` branches + pull requests) to keep history readable.

## Support

Issues and improvements typically surface from training runs or visualization gaps. When reporting problems, include:

1. The project folder (`CSWIN v2`, `HSIFUSION&SHARP`, `hsi_viz_suite`, `mswr_v2`, `SSTrans`, or `WaveDiff`).
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
  - SHARP Architecture: Hierarchical transformer with multi-scale attention + streaming sparse attention (top-k/local window fallbacks), ChannelRMSNorm, cross-attention fusion, and a rank-limited spectral-basis refinement head. Sigmoid output and MRAE loss by default; linear RBF key projection. See `HSIFUSION&SHARP/sharp_v322_hardened.py`.
  - Training: Canonical entry `unified_training.py` (`--model hsifusion|sharp`, MST++-faithful Adam/cosine/MRAE recipe, AMP, optional EMA); legacy entries `hsifusion_training.py`, `sharp_training_script_fixed.py`.

- CSWIN v2 (CSWIN v2)
  - Architecture: Generator-only U-Net/Transformer hybrid with configurable spatial attention (`local_global` default, head-split `cswin` cross-shaped stripes, experimental `axial`), spectral self-attention, and learned PixelShuffle sampling (~11.4M trainable params). See `CSWIN v2/src/hsi_model/models`.
  - Training: Annealed pure-MRAE objective, Adam 4e-4 with 300k-step cosine decay, EMA validation, Hydra configuration (`CSWIN v2/src/configs/config.yaml`). Entry: `train_generator.py`; legacy Sinkhorn-GAN entries `training_script_fixed.py`, `train_optimized.py`.

- SSTrans / HSIFormer (SSTrans)
  - Architecture: Spectral-spatial transformer combining CSWin spatial cross-attention, spectral multi-head self-attention, and CAT blocks, packaged as the `hsiformer` Python package with selectable presets (`recommended_retrain` default). See `SSTrans/src/hsiformer`.
  - Training: Author-faithful recipe — L1 loss, Adam, per-iteration cosine to 1e-6, progressive 128→256→512 stages (batch 32→8→1), bf16 AMP, grad clip 1.0, non-finite loss/resume guards. Entry: `SSTrans/scripts/train.py` (`hsiformer-train`).

- WaveDiff (WaveDiff)
  - Architecture: Latent diffusion models augmented with wavelet transforms (standard/learnable/adaptive) and spectral/pixel refinement heads. See `WaveDiff/modules` and `WaveDiff/models`.
  - Training: JSON-driven config (`WaveDiff/configs/example_config.json`), cosine scheduling, combined spectral losses, curriculum masking, visualization hooks. Entry: `WaveDiff/train.py`.

See each subfolder README for code‑level details and examples.

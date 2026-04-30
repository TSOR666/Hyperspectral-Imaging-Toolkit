# Hyperspectral Imaging Toolkit

A monorepo of production-ready models and utilities for reconstructing 31-band hyperspectral images (HSI) from standard RGB inputs. Each project targets the ARAD-1K benchmark and is independently runnable, sharing only dataset conventions and the visualization suite.

---

## Table of Contents

1. [What This Toolkit Does](#what-this-toolkit-does)
2. [Repository Layout](#repository-layout)
3. [Choosing a Model](#choosing-a-model)
4. [System Requirements](#system-requirements)
5. [Dataset Setup (ARAD-1K)](#dataset-setup-arad-1k)
6. [Quick Start per Project](#quick-start-per-project)
   - [CSWIN v2](#cswin-v2-sinkhorn-gan)
   - [HSIFusion & SHARP](#hsifusion--sharp)
   - [MSWR v2](#mswr-v2)
   - [WaveDiff](#wavediff)
   - [Visualization Suite](#hsi-visualization-suite)
7. [Models at a Glance](#models-at-a-glance)
8. [Evaluation Metrics](#evaluation-metrics)
9. [End-to-End Workflow](#end-to-end-workflow)
10. [Contributing](#contributing)
11. [License](#license)

---

## What This Toolkit Does

The toolkit trains and evaluates models that take an RGB image (3 channels, uint8) and predict its full hyperspectral representation (31 bands, 400–700 nm). All models are evaluated on **ARAD-1K** using the MST++ centre-crop protocol (patch size 128, validation stride 8, crop 226×256 from 482×512).

Input → Output summary:

| Item | Details |
|---|---|
| Input | RGB image, 3 channels, any resolution |
| Output | HSI cube, 31 bands, 10 nm spacing 400–700 nm |
| Dataset | ARAD-1K (900 train / 100 val images at 482×512) |
| Metrics | MRAE, RMSE, PSNR, SSIM, SAM |

---

## Repository Layout

| Path | Model | Type | GPU Memory |
|---|---|---|---|
| [`CSWIN v2/`](CSWIN%20v2/README.md) | NoiseRobustCSWinGenerator + Discriminator | Sinkhorn-GAN | ~24 GB |
| [`HSIFUSION&SHARP/`](HSIFUSION%26SHARP/README.md) | HSIFusionNet v2.5.3 + SHARP v3.2.2 | Transformer | 8–18 GB |
| [`mswr_v2/`](mswr_v2/README.md) | MSWR-Net v2.1.2 | CNN + Attention | ~14 GB |
| [`WaveDiff/`](WaveDiff/README.md) | WaveDiff (base / wavelet / adaptive) | Latent Diffusion | ~12 GB |
| [`hsi_viz_suite/`](hsi_viz_suite/README.md) | — | Visualization utility | CPU only |

> Each subdirectory is self-contained. Run all commands from inside the relevant folder (`cd mswr_v2`, etc.).

---

## Choosing a Model

| Situation | Recommended Model |
|---|---|
| Best overall quality, adversarial training | **CSWIN v2** |
| Fastest convergence, AMP-friendly, easy to extend | **HSIFusionNet v2.5.3** |
| Memory-efficient sparse attention, production hardening | **SHARP v3.2.2** |
| Strong CNN baseline with wavelet branches | **MSWR-Net v2.1.2** |
| Generative reconstruction, uncertainty estimation | **WaveDiff** |
| Low GPU memory (<12 GB) | **HSIFusionNet tiny** or **MSWR tiny** |
| Multi-GPU cluster, >4 GPUs | **CSWIN v2** or **SHARP v3.2.2** |

---

## System Requirements

**Minimum**

- Python 3.9+
- PyTorch 1.13+ with CUDA 11.8 or 12.1
- NVIDIA GPU with ≥10 GB VRAM (use `tiny` model sizes)
- 32 GB host RAM, 200 GB disk (ARAD-1K dataset + checkpoints)

**Recommended**

- NVIDIA A100 (80 GB) or RTX 3090/4090 (24 GB)
- CUDA 12.1 + cuDNN 8.9
- 64 GB host RAM

**Environment setup (one-time, any project)**

```bash
# Create and activate a dedicated environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Install PyTorch (CUDA 11.8 example — adjust for your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Per-project dependencies are listed in each folder's requirements.txt
```

**Recommended CUDA allocator settings** (export before training to avoid OOM fragmentation):

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export OMP_NUM_THREADS=2
```

---

## Dataset Setup (ARAD-1K)

All models train on [ARAD-1K](https://www.cs.ubc.ca/labs/imager/tr/arad1k/). Obtain the dataset and place it as follows:

```
data/ARAD_1K/               # (or any path — set HSI_DATA_DIR)
├── Train_RGB/              # 900 JPEG/PNG RGB images, 482×512
├── Train_Spec/             # 900 spectral cubes  (.mat or .npy, 31 bands)
├── split_txt/              # train.txt and valid.txt split files
├── statistics/             # per-image min/max normalization stats
└── channel_metadata/       # wavelength metadata for 31 bands
```

For HSIFusion & SHARP, run the bundled helper to produce MST++-style crops:

```bash
cd "HSIFUSION&SHARP"
python dataset_setup.py \
  --arad-root /path/to/ARAD_1K_raw \
  --output-root ./data/ARAD_1K \
  --patch-size 128 \
  --stride 8 \
  --workers 8
```

**HuggingFace alternative** (CSWIN only): the CSWIN trainer can load directly from the `mhmdjouni/arad_hsdb` dataset on HuggingFace Hub. Set `dataset_source: huggingface` in `src/configs/config.yaml`.

**Environment variable**: set `HSI_DATA_DIR` once instead of repeating the path on every command:

```bash
export HSI_DATA_DIR=/path/to/ARAD_1K
```

---

## Quick Start per Project

### CSWIN v2 (Sinkhorn-GAN)

Sinkhorn-GAN with a noise-robust CSWin transformer generator and a spectral-normalized discriminator. Best quality on ARAD-1K.

```bash
cd "CSWIN v2"
pip install torch torchvision hydra-core einops numpy h5py tqdm psutil

# Single GPU
python src/hsi_model/training_script_fixed.py \
  --config-name config \
  data_dir=/path/to/ARAD_1K

# Memory-optimized (keeps peak VRAM <30 GB on 80 GB A100)
python src/hsi_model/train_optimized.py \
  --config-name config \
  data_dir=/path/to/ARAD_1K

# Multi-GPU (4 GPUs)
python -m torch.distributed.run --nproc_per_node=4 \
  src/hsi_model/training_script_fixed.py \
  --config-name config \
  data_dir=/path/to/ARAD_1K
```

Common overrides (Hydra dot-notation):

```bash
python src/hsi_model/training_script_fixed.py \
  data_dir=/path/to/ARAD_1K \
  batch_size=16 \
  generator_lr=1e-4 \
  discriminator_lr=5e-5 \
  lambda_adversarial=0.15 \
  epochs=500
```

See [CSWIN v2 README](CSWIN%20v2/README.md) and [Quick Start](CSWIN%20v2/QUICK_START.md) for the full configuration reference.

---

### HSIFusion & SHARP

Two complementary transformer models that share a dataset pipeline.

```bash
cd "HSIFUSION&SHARP"
pip install torch torchvision einops numpy h5py tqdm tensorboard

# Stage dataset (first time only)
python dataset_setup.py --arad-root /path/to/ARAD_1K --output-root ./data/ARAD_1K

# Train HSIFusionNet v2.5.3 (fast, AMP-friendly)
python hsifusion_training.py \
  --data_root ./data/ARAD_1K \
  --model_size base \
  --batch_size 12 \
  --use_amp \
  --compile_model

# Train SHARP v3.2.2 (sparse attention, more memory)
python sharp_training_script_fixed.py \
  --data_root ./data/ARAD_1K \
  --model_size base \
  --batch_size 20 \
  --sparse_sparsity_ratio 0.9 \
  --use_amp

# SHARP inference with overlap tiling
python sharp_inference.py \
  --checkpoint experiments/sharp/best.ckpt \
  --input path/to/rgb.png \
  --output outputs/hsi.npy \
  --patch-size 256
```

See [HSIFusion & SHARP README](HSIFUSION%26SHARP/README.md) for distributed training, SLURM templates, and all CLI flags.

---

### MSWR v2

CNN-based dual-attention encoder-decoder with optional wavelet branches.

```bash
cd mswr_v2
pip install torch torchvision numpy h5py tqdm psutil pyyaml opencv-python scipy

# Train (base model, wavelet + enhanced loss enabled by default)
python train_mswr_v212_logging.py \
  --model_size base \
  --data_root /path/to/ARAD_1K

# Train with explicit options
python train_mswr_v212_logging.py \
  --model_size base \
  --data_root /path/to/ARAD_1K \
  --batch_size 8 \
  --end_epoch 300 \
  --init_lr 2e-4 \
  --use_wavelet \
  --wavelet_type db2 \
  --use_enhanced_loss \
  --use_amp \
  --use_checkpoint \
  --use_flash_attn

# Inference
python mswr_inference.py \
  --checkpoint experiments/checkpoints/best_model.pth \
  --data-root /path/to/ARAD_1K \
  --output-dir outputs/mswr
```

See [MSWR v2 README](mswr_v2/README.md) for the full list of CLI flags and architecture details.

---

### WaveDiff

Latent diffusion models with wavelet-enhanced denoisers. Three variants: `base`, `wavelet`, `adaptive_wavelet`.

```bash
cd WaveDiff
pip install torch torchvision numpy scipy pywavelets tensorboard einops

# Train (adaptive_wavelet recommended for best quality)
python train.py \
  --model_type adaptive_wavelet \
  --train_dir data/ARAD1K/train \
  --val_dir data/ARAD1K/val \
  --batch_size 8 \
  --num_epochs 100

# Train from a config file
python train.py --config configs/example_config.json

# Inference — single image
python inference.py \
  --checkpoint checkpoints/<run_id>/final_model.pt \
  --image path/to/rgb.png \
  --output_dir results/

# Inference — full directory with metrics
python inference.py \
  --checkpoint checkpoints/<run_id>/final_model.pt \
  --input_dir data/ARAD1K/test/RGB \
  --output_dir results/
```

See [WaveDiff README](WaveDiff/README.md) and [Quick Start](WaveDiff/QUICK_START.md) for the config schema and sampling options.

---

### HSI Visualization Suite

Converts `.npy` reconstructions into publication-quality figures.

```bash
cd hsi_viz_suite
pip install -r requirements.txt

# Generate all figures for one method
python scripts/generate_all_visualizations.py \
  --results /path/to/method_outputs \
  --output figs \
  --dpi 300 \
  --style paper

# Side-by-side comparison of multiple methods
python scripts/generate_all_visualizations.py \
  --results outputs/ours \
  --methods outputs/baseline_a outputs/baseline_b \
  --method-names "Ours" "Baseline A" "Baseline B" \
  --output figs
```

Output folders under `figs/`:
- `main_figures/` — qualitative reconstruction grids
- `error_maps/` — per-pixel error heatmaps
- `spectral_analysis/` — spectral curve plots
- `statistics/` — PSNR/SAM distribution charts
- `comparison_grids/` — side-by-side method comparisons

See [HSI Viz Suite README](hsi_viz_suite/README.md) for the full options.

---

## Models at a Glance

### MSWR-Net v2.1.2

| Item | Details |
|---|---|
| Architecture | CNN encoder-decoder, dual attention (window + landmark), optional DWT/IDWT gating |
| Key file | `mswr_v2/model/mswr_net_v212.py` |
| Config class | `MSWRDualConfig` |
| Loss | L1 + SSIM + SAM + gradient (EnhancedMSWRLoss) |
| Regularization | Drop-path, gradient checkpointing, EMA, AMP |
| GPU memory | ~12–16 GB (base, batch 8) |
| Training entry | `train_mswr_v212_logging.py` |
| Inference entry | `mswr_inference.py` |

Unique features: symmetric reflect padding in wavelet branches (eliminates padding crashes on small maps), `AdaptiveNorm2d` for mixed CNN/transformer blocks, filter caching for DWT.

---

### HSIFusionNet v2.5.3 ("Lightning Pro")

| Item | Details |
|---|---|
| Architecture | ViT-style encoder-decoder, LightningProBlock (RoPE sliding-window + spectral attention + optional MoE) |
| Key file | `HSIFUSION&SHARP/hsifusion_v252_complete.py` |
| Config class | `LightningProConfig` |
| Model sizes | tiny / small / base / large |
| Loss | L1 + spectral consistency |
| GPU memory | ~8–12 GB (base, batch 12) |
| Training entry | `hsifusion_training.py` |

Unique features: `torch.compile` compatible, channels-last layout, cross-attention fusion in decoder, optional uncertainty head.

---

### SHARP v3.2.2

| Item | Details |
|---|---|
| Architecture | Hierarchical transformer, streaming sparse attention (top-k + local window fallback), RBF key projection |
| Key file | `HSIFUSION&SHARP/sharp_v322_hardened.py` |
| Config class | `SHARPv32Config` |
| Model sizes | tiny / small / base / large |
| Loss | L1 + spectral basis regularization |
| GPU memory | ~14–18 GB (base, batch 20) |
| Training entry | `sharp_training_script_fixed.py` |
| Inference entry | `sharp_inference.py` |

Unique features: 90% token sparsity by default, EMA weight tracking, `ChannelRMSNorm` for stability, overlap-blend tiling in inference.

---

### CSWIN v2 (Sinkhorn-GAN)

| Item | Details |
|---|---|
| Architecture | U-Net generator (CSWin spatial + spectral attention, adaptive GroupNorm, noise-aware gating) + SN transformer discriminator |
| Key files | `CSWIN v2/src/hsi_model/models/generator_v3.py`, `discriminator_v2.py` |
| Config | `src/configs/config.yaml` (Hydra) |
| Loss | Sinkhorn OT + L1 + SAM + R1 regularization |
| GPU memory | ~20–25 GB (batch 20) |
| Training entry | `training_script_fixed.py` / `train_optimized.py` |

Unique features: Sinkhorn optimal-transport loss with FP32 fallback, R1 gradient penalty, NaN-safe attention clamping, HuggingFace dataset adapter.

---

### WaveDiff

| Item | Details |
|---|---|
| Architecture | RGB encoder → latent UNet denoiser → HSI decoder, with Haar/learnable/adaptive wavelets |
| Key files | `WaveDiff/models/`, `WaveDiff/modules/` |
| Model types | `base` / `wavelet` / `adaptive_wavelet` |
| Loss | Diffusion + L1 + cycle + wavelet |
| GPU memory | ~10–14 GB (batch 8) |
| Training entry | `train.py` |
| Inference entry | `inference.py` |

Unique features: DPM-OT optimal-transport sampling, curriculum masking (random → block → spectral → combined), trainable wavelet thresholds, `SpectralNoiseSchedule`.

---

## Evaluation Metrics

All projects report the same five metrics on the ARAD-1K validation set. Lower is better for MRAE, RMSE, SAM; higher is better for PSNR, SSIM.

| Metric | Formula | Typical range |
|---|---|---|
| MRAE | mean(|pred−gt| / (gt + ε)) | 0.01–0.08 |
| RMSE | sqrt(mean((pred−gt)²)) | 0.005–0.04 |
| PSNR | 10·log₁₀(1/MSE) | 34–42 dB |
| SSIM | Structural similarity | 0.90–0.99 |
| SAM | Spectral angle (radians) | 0.02–0.12 |

The validation protocol crops a 226×256 centre region from each 482×512 image, matching the MST++ benchmark convention.

---

## End-to-End Workflow

```
1. Prepare dataset
   └─ dataset_setup.py  (HSIFUSION&SHARP) or manual layout

2. Train a model
   ├─ CSWIN v2/src/hsi_model/training_script_fixed.py
   ├─ HSIFUSION&SHARP/hsifusion_training.py
   ├─ HSIFUSION&SHARP/sharp_training_script_fixed.py
   ├─ mswr_v2/train_mswr_v212_logging.py
   └─ WaveDiff/train.py

3. Run inference → .npy files
   ├─ mswr_v2/mswr_inference.py
   ├─ HSIFUSION&SHARP/sharp_inference.py
   └─ WaveDiff/inference.py

4. Visualize & compare
   └─ hsi_viz_suite/scripts/generate_all_visualizations.py
```

Checkpoints, logs, and visualizations are stored under `artifacts/` or `experiments/` inside each project folder. Set the `HSI_LOG_DIR` and `HSI_CKPT_DIR` environment variables to redirect them to a shared location.

---

## Contributing

- Keep changes scoped to the relevant subdirectory. Cross-project imports are intentionally avoided.
- Update the sub-project README when you change CLI defaults, config schemas, or output formats.
- Use feature branches and pull requests against `main`.
- When reporting an issue, include: the project folder, the exact command, your GPU/CUDA version, and a relevant log excerpt from `artifacts/logs/`.

---

## License

Released under the [MIT License](LICENSE). Contributions are accepted under the same terms.

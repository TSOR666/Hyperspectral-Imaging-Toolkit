# WaveDiff — Latent Diffusion for Hyperspectral Image Reconstruction

Reconstructs 31-band hyperspectral images from RGB inputs using latent diffusion models augmented with multi-scale wavelet processing. Three model variants are available:

| Variant | Description | Best For |
|---|---|---|
| `base` | Standard latent diffusion (RGB encoder → latent UNet → HSI decoder) | Baseline, fastest training |
| `wavelet` | Adds Haar wavelet transforms to encoder, denoiser, and decoder | Better detail / edge preservation |
| `adaptive_wavelet` | Learnable wavelet with adaptive soft/hard thresholding | Best quality, noisy inputs |

---

## Table of Contents

1. [Architecture](#architecture)
   - [Model Variants](#model-variants)
   - [Key Modules](#key-modules)
   - [Diffusion Process](#diffusion-process)
2. [Environment Setup](#environment-setup)
3. [Dataset Layout](#dataset-layout)
4. [Training](#training)
5. [Configuration Reference](#configuration-reference)
6. [Inference](#inference)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Tests](#tests)
9. [Troubleshooting](#troubleshooting)
10. [Project Structure](#project-structure)

---

## Architecture

### Model Variants

All three variants share the same encoder-diffusion-decoder pipeline; wavelets are injected at the encoder, denoiser, and decoder stages.

```
RGB (3ch) ──→ [RGB Encoder] ──→ latent z ──→ [UNet Denoiser] (T steps)
                                                       ↓
                                             z_0 (denoised) ──→ [HSI Decoder] ──→ HSI (31ch)
                                                                        ↓
                                                              [Spectral Refinement Head]
                                                                        ↓
                                                              [Pixel Refinement Head] (optional)
```

#### `base` — `HSILatentDiffusionModel`

File: `models/base_model.py`

- `RGBEncoder`: 3-channel → latent dimension via convolutional blocks with BatchNorm
- `UNetDenoiser`: reverse diffusion in latent space with timestep embedding
- `HSIDecoder`: latent → 31-band output
- `SpectralRefinementHead`: post-processing for spectral consistency
- Optional `PixelRefinementHead` for spatial cleanup

#### `wavelet` — `WaveletHSILatentDiffusionModel`

File: `models/wavelet_model.py`

- `WaveletRGBEncoder`: Haar wavelet sub-band decomposition before each conv block
- `WaveletUNetDenoiser`: frequency-aware denoising via per-band attention
- `WaveletHSIDecoder`: inverse wavelet synthesis before final conv

#### `adaptive_wavelet` — `AdaptiveWaveletHSILatentDiffusionModel`

File: `models/adaptive_model.py`

- All wavelet model features plus:
- **Learnable wavelet filters** (gradient-trainable)
- **Adaptive thresholding**: soft / hard; threshold can be fixed or trainable (`trainable_threshold=true`)
- Per-frequency-band learnable scale/bias

---

### Key Modules

#### Encoders — `modules/encoders.py`

| Class | Input | Output | Description |
|---|---|---|---|
| `RGBEncoder` | (B, 3, H, W) | (B, latent_dim, H', W') | Conv blocks with BN |
| `WaveletRGBEncoder` | (B, 3, H, W) | (B, latent_dim, H', W') | Haar DWT before each block |

#### Denoisers — `modules/denoisers.py`

| Class | Description |
|---|---|
| `UNetDenoiser` | Standard UNet with timestep embedding (sinusoidal + MLP) |
| `WaveletUNetDenoiser` | Adds per-resolution wavelet gating in skip connections |

Both accept `(x_t, t)` where `t` is the integer diffusion timestep.

#### Decoders — `modules/decoders.py`

| Class | Input | Output | Description |
|---|---|---|---|
| `HSIDecoder` | latent | (B, 31, H, W) | Transposed conv blocks |
| `WaveletHSIDecoder` | latent | (B, 31, H, W) | Inverse DWT synthesis |
| `HSI2RGBConverter` | (B, 31, H, W) | (B, 3, H, W) | Differentiable spectral integral for cycle loss |

#### Wavelet Transforms — `transforms/haar_wavelet.py`

- `HaarWavelet2D`: multi-level 2D DWT/IDWT via lifting scheme
- `AdaptiveWaveletThreshold`: learnable soft/hard threshold with temperature

#### Diffusion — `diffusion/`

| Module | Description |
|---|---|
| `noise_schedule.py` | Beta schedules: `linear`, `cosine`, `spectral` |
| `dpm_ot.py` | DPM-OT sampler with cached optimal-transport statistics |
| `enhanced_dpm_ot.py` | Extended DPM-OT with spectral consistency guidance |
| `SpectralNoiseSchedule` | Frequency-aware noise scheduling per spectral band |

#### Masking — `utils/masking.py`

`MaskingManager` implements curriculum masking for regularization:

| Strategy | Description |
|---|---|
| `random` | Uniformly random pixel masks |
| `block` | Contiguous rectangular masks |
| `spectral` | Entire spectral bands masked |
| `combined` | Mix of all strategies |
| `curriculum` | Progresses through `curriculum_strategies` during training |

---

### Diffusion Process

The model uses a **forward diffusion** process that adds noise over `T=1000` timesteps:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) x_{t-1}, β_t I)
```

Training minimizes the denoising objective in latent space. At inference, the reverse process generates `z_0` in `sampling_steps` steps (default 20 with DPM-OT, much fewer than 1000).

**DPM-OT sampling**: uses optimal-transport step scheduling to minimize the transport distance between noise and data distributions, enabling high-quality reconstruction in 10–50 steps.

---

## Environment Setup

```bash
cd WaveDiff
python -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\activate             # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

pip install -r requirements.txt
```

Core requirements: `numpy`, `scipy`, `pywavelets`, `einops`, `tensorboard`, `tqdm`, `h5py`

---

## Dataset Layout

```
data/
└── ARAD1K/
    ├── train/
    │   ├── RGB/          # .png or .jpg images (any resolution)
    │   │   ├── img001.png
    │   │   └── ...
    │   └── HSI/          # .npy or .mat cubes (31 × H × W)
    │       ├── img001.npy
    │       └── ...
    ├── val/
    │   ├── RGB/
    │   └── HSI/
    └── test/
        ├── RGB/
        └── HSI/          # optional — used for evaluation metrics
```

Each RGB filename must match its HSI counterpart (same base name, any extension). The dataloader automatically resolves `.mat` to `.npy` conversion if needed.

If you have the raw ARAD-1K archive, symlink or copy the splits into the above layout, or use `dataset_setup.py` from the `HSIFUSION&SHARP` folder to produce MST++ style patches.

---

## Training

### Minimal command

```bash
cd WaveDiff
python train.py \
  --model_type adaptive_wavelet \
  --train_dir data/ARAD1K/train \
  --val_dir data/ARAD1K/val
```

### From a config file (recommended for reproducibility)

```bash
python train.py --config configs/example_config.json
```

### Typical command with common options

```bash
python train.py \
  --model_type adaptive_wavelet \
  --train_dir data/ARAD1K/train \
  --val_dir data/ARAD1K/val \
  --latent_dim 64 \
  --timesteps 1000 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --use_masking \
  --mask_strategy curriculum \
  --threshold_method soft \
  --trainable_threshold
```

### All CLI flags

#### Model

| Flag | Default | Description |
|---|---|---|
| `--model_type` | `adaptive_wavelet` | `base` / `wavelet` / `adaptive_wavelet` |
| `--latent_dim` | `64` | Latent space dimensionality |
| `--timesteps` | `1000` | Forward diffusion timesteps |
| `--image_size` | `256` | Spatial resolution for training patches |
| `--use_batchnorm` | `true` | Use BatchNorm in encoder/decoder |

#### Training

| Flag | Default | Description |
|---|---|---|
| `--train_dir` | `data/ARAD1K/train` | Training data directory |
| `--val_dir` | `data/ARAD1K/val` | Validation data directory |
| `--batch_size` | `8` | Batch size |
| `--num_epochs` | `100` | Training epochs |
| `--num_workers` | `4` | DataLoader workers |
| `--learning_rate` | `1e-4` | Peak learning rate |
| `--min_lr` | `1e-6` | Minimum LR for cosine schedule |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--max_grad_norm` | `1.0` | Gradient clipping norm |

#### Loss weights

| Flag | Default | Description |
|---|---|---|
| `--diffusion_loss_weight` | `1.0` | Denoising objective weight |
| `--l1_loss_weight` | `1.0` | Reconstruction L1 weight |
| `--cycle_loss_weight` | `0.8` | Cycle-consistency (HSI→RGB→HSI) weight |
| `--wavelet_loss_weight` | `0.5` | Wavelet-domain loss (wavelet models only) |

#### Masking / curriculum

| Flag | Default | Description |
|---|---|---|
| `--use_masking` | `true` | Enable masking regularization |
| `--mask_strategy` | `curriculum` | `random` / `block` / `spectral` / `combined` / `curriculum` |
| `--initial_mask_ratio` | `0.1` | Starting mask coverage |
| `--final_mask_ratio` | `0.7` | Final mask coverage (for curriculum) |

#### Wavelet thresholding

| Flag | Default | Description |
|---|---|---|
| `--threshold_method` | `soft` | `soft` / `hard` |
| `--init_threshold` | `0.1` | Initial threshold value |
| `--trainable_threshold` | `true` | Learn threshold during training |

#### Output

| Flag | Default | Description |
|---|---|---|
| `--checkpoint_dir` | `checkpoints/adaptive_wavelet` | Where to save checkpoints |
| `--visualization_dir` | `visualizations/adaptive_wavelet` | Where to save training vis |
| `--log_interval` | `100` | Logging interval in steps |
| `--resume_from_checkpoint` | `null` | Path to resume checkpoint |
| `--config` | `null` | Path to JSON config file |

---

## Configuration Reference

Full contents of `configs/example_config.json`:

```json
{
  "model_type": "adaptive_wavelet",
  "latent_dim": 64,
  "timesteps": 1000,
  "use_batchnorm": true,
  "batch_size": 8,
  "learning_rate": 1e-4,
  "min_lr": 1e-6,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "num_epochs": 100,
  "num_workers": 4,
  "diffusion_loss_weight": 1.0,
  "l1_loss_weight": 1.0,
  "cycle_loss_weight": 0.8,
  "wavelet_loss_weight": 0.5,
  "train_dir": "data/ARAD1K/train",
  "val_dir": "data/ARAD1K/val",
  "image_size": 256,
  "use_masking": true,
  "mask_strategy": "curriculum",
  "curriculum_strategies": ["random", "block", "spectral", "combined"],
  "checkpoint_dir": "checkpoints/adaptive_wavelet",
  "visualization_dir": "visualizations/adaptive_wavelet",
  "resume_from_checkpoint": null,
  "log_interval": 100,
  "threshold_method": "soft",
  "init_threshold": 0.1,
  "trainable_threshold": true,
  "initial_mask_ratio": 0.1,
  "final_mask_ratio": 0.7
}
```

CLI flags override any key in the config file when both are provided.

---

## Inference

### Single image

```bash
python inference.py \
  --checkpoint checkpoints/adaptive_wavelet/final_model.pt \
  --image path/to/rgb.png \
  --output_dir results/
```

### Full directory with evaluation

```bash
python inference.py \
  --checkpoint checkpoints/adaptive_wavelet/final_model.pt \
  --input_dir data/ARAD1K/test/RGB \
  --output_dir results/
```

When a sibling `HSI/` directory is present, the script automatically computes MRAE, RMSE, PSNR, SSIM, and SAM and writes `results/metrics.json`.

### Adaptive threshold inference options

```bash
python inference.py \
  --checkpoint checkpoints/adaptive_wavelet/final_model.pt \
  --image path/to/rgb.png \
  --output_dir results/ \
  --sampling_steps 20 \
  --no_adaptive_threshold   # disable learned threshold, use fixed
```

### All inference flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.pt` checkpoint |
| `--image` | — | Single image path |
| `--input_dir` | — | Directory of RGB images |
| `--output_dir` | `results/` | Output directory |
| `--sampling_steps` | `20` | DPM-OT sampling steps |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--no_adaptive_threshold` | disabled | Disable adaptive thresholding |
| `--batch_size` | `1` | Batch size for directory mode |

### Quick demo

```bash
python demo.py
```

Prints a minimal end-to-end example for a synthetic RGB input without requiring a real dataset.

---

## Evaluation Metrics

The inference script reports:

| Metric | Description | Lower/Higher is better |
|---|---|---|
| MRAE | Mean Relative Absolute Error | Lower |
| RMSE | Root Mean Squared Error | Lower |
| PSNR (dB) | Peak Signal-to-Noise Ratio | Higher |
| SSIM | Structural Similarity | Higher |
| SAM (rad) | Spectral Angle Mapper | Lower |

Results are printed per image and aggregated. When `--input_dir` is used, a `metrics.json` summary is written to `--output_dir`.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Test suites:
- `tests/test_runtime_audit.py` — forward pass shapes, dtype consistency, gradient flow
- `tests/test_verification.py` — checkpoint save/load round-trip

---

## Troubleshooting

### Training loss is NaN from epoch 1

Usually indicates a numerical issue in the diffusion schedule. Try:
- `--use_batchnorm true` (default)
- Lower `--learning_rate 5e-5`
- Reduce `--cycle_loss_weight 0.3`

### `KeyError: 'hsi'` in dataloader

The dataloader expects HSI files to share the same base filename as their RGB counterparts. Verify that `train/RGB/img001.png` and `train/HSI/img001.npy` (or `.mat`) exist with the same stem.

### Slow inference (>1 minute per image)

Reduce `--sampling_steps`. The DPM-OT sampler achieves good quality at 20 steps. If quality suffers, try 50 steps. Values above 100 rarely improve results.

### Old checkpoint fails to load

```
RuntimeError: Error(s) in loading state_dict...
```

If you have a checkpoint from before the audit fixes (using `weights_only=False`):

```python
ckpt = torch.load('path.pt', weights_only=False, map_location='cpu')
```

All new checkpoints are saved with `weights_only=True`.

### CUDA OOM during training

Try:
1. Reduce `--batch_size 4`
2. Reduce `--latent_dim 32`
3. Reduce `--image_size 128`
4. Reduce `--timesteps 500`

---

## Project Structure

```
WaveDiff/
├── configs/
│   └── example_config.json           # Reference training configuration
├── diffusion/
│   ├── noise_schedule.py             # Beta schedules (linear, cosine, spectral)
│   ├── dpm_ot.py                     # DPM-OT sampler
│   └── enhanced_dpm_ot.py            # Extended DPM-OT with spectral guidance
├── losses/
│   └── spectral_consistency.py       # CombinedSpectralLoss, frequency-domain matching
├── models/
│   ├── base_model.py                 # HSILatentDiffusionModel
│   ├── wavelet_model.py              # WaveletHSILatentDiffusionModel
│   └── adaptive_model.py             # AdaptiveWaveletHSILatentDiffusionModel
├── modules/
│   ├── encoders.py                   # RGBEncoder, WaveletRGBEncoder
│   ├── decoders.py                   # HSIDecoder, WaveletHSIDecoder, HSI2RGBConverter
│   ├── denoisers.py                  # UNetDenoiser, WaveletUNetDenoiser
│   └── attention.py                  # Attention primitives
├── transforms/
│   └── haar_wavelet.py               # HaarWavelet2D, AdaptiveWaveletThreshold
├── utils/
│   ├── masking.py                    # MaskingManager, curriculum strategies
│   ├── metrics.py                    # MRAE, RMSE, PSNR, SSIM, SAM
│   ├── logging.py                    # TensorBoard + JSON logging helpers
│   └── visualization.py             # False-color rendering, spectral plots
├── tests/
│   ├── test_runtime_audit.py
│   └── test_verification.py
├── train.py                          # Training entry point
├── inference.py                      # Inference + evaluation entry point
├── demo.py                           # Minimal single-image demo
├── QUICK_START.md                    # Step-by-step setup guide
└── README.md
```

---

## Model Selection Guide

| Scenario | Recommended |
|---|---|
| First run / sanity check | `base` |
| Noisy or compressed RGB input | `adaptive_wavelet` |
| Best reconstruction quality | `adaptive_wavelet` (300+ epochs) |
| Fastest training time | `base` |
| Detail / edge preservation matters | `wavelet` or `adaptive_wavelet` |
| Limited GPU memory | `base` with `--latent_dim 32` |

---

## Related Projects

- [`../CSWIN v2`](../CSWIN%20v2/README.md) — adversarial (GAN) alternative
- [`../HSIFUSION&SHARP`](../HSIFUSION%26SHARP/README.md) — transformer baselines
- [`../mswr_v2`](../mswr_v2/README.md) — CNN baseline
- [`../hsi_viz_suite`](../hsi_viz_suite/README.md) — visualization suite for outputs

---

## Citation

If this code is useful for your research, please cite:

```bibtex
@software{hsi_wavelet_ldm,
  title  = {Latent Diffusion Models for Hyperspectral Image Reconstruction},
  author = {Thierry Silvio Claude Soreze},
  year   = {2024},
  url    = {https://github.com/TSOR666/hsi-wavelet-diffusion}
}
```

---

## License

Distributed under the [MIT License](../LICENSE).

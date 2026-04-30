# CSWIN v2 — Sinkhorn-GAN for Noise-Robust HSI Reconstruction

Trains a **noise-robust CSWin transformer generator** adversarially against a **spectral-normalized transformer discriminator** using Sinkhorn optimal-transport loss and R1 gradient regularization. Produces 31-band hyperspectral cubes from RGB inputs.

Two entry points are provided:

| Script | Description |
|---|---|
| `src/hsi_model/training_script_fixed.py` | Production Sinkhorn-GAN trainer with R1 regularization and EMA logging |
| `src/hsi_model/train_optimized.py` | Memory-optimized MST++ trainer; keeps peak VRAM < 30 GB on 80 GB A100 |

Both share `src/configs/config.yaml` and all utilities under `src/hsi_model/utils/`.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Environment Setup](#environment-setup)
3. [Dataset Setup](#dataset-setup)
4. [Training](#training)
5. [Configuration Reference](#configuration-reference)
6. [Inference](#inference)
7. [Tests](#tests)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)
10. [Project Structure](#project-structure)

---

## Architecture

### Generator — `NoiseRobustCSWinGenerator`

File: `src/hsi_model/models/generator_v3.py`

```
RGB (3ch) → Encoder Stage 1 → Encoder Stage 2 → Transformer Bottleneck
           ↑ skip                ↑ skip                ↓
          Decoder Stage 2 ← Decoder Stage 1 ← Output Head → HSI (31ch)
```

Each encoder/decoder block contains:
- **CSWin spatial attention**: cross-shaped window self-attention with adaptive GroupNorm
- **Spectral attention**: channel-wise attention over the 31-band dimension
- **Noise-aware gating**: learned gate that suppresses noise artifacts during upsampling
- Safe clamping (`max=50.0`) prevents attention overflow; NaN-fail-fast in training mode

Output activation options: `none` (default) / `sigmoid` / `tanh` / `delayed_sigmoid`

### Discriminator — `SpectralNormalizedTransformerDiscriminator`

File: `src/hsi_model/models/discriminator_v2.py`

- Input: concatenated RGB (3ch) + HSI (31ch) = 34-channel input
- Spectral normalization on all linear and convolutional layers
- Spectral self-attention with learnable temperature scaling
- NaN-safe logits (replaces `inf` with large finite values)
- Outputs spatial feature maps (no global pooling — preserves spatial discrimination)

### Loss Functions

File: `src/hsi_model/models/losses_consolidated.py`

| Loss | Weight | Description |
|---|---|---|
| Reconstruction (L1) | `lambda_rec=1.0` | Pixel-wise MAE on 31-band output |
| SAM | `lambda_sam=0.05` | Spectral Angle Mapper (radians); cosine clamped at 0.999 |
| Sinkhorn (adversarial) | `lambda_adversarial=0.1` | Optimal-transport matching; forced FP32 |
| Perceptual | `lambda_perceptual=0.0` | Feature-level loss (disabled by default) |
| R1 regularization | `r1_gamma=10.0` | Gradient penalty on discriminator real samples |

Sinkhorn algorithm parameters: `epsilon=0.1`, `iters=50`, `force_fp32=True`, `loss_clip=5.0`.

---

## Environment Setup

```bash
cd "CSWIN v2"
python -m venv .venv

# Activate
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate            # Windows

# Install PyTorch — choose the correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# Core dependencies
pip install hydra-core einops numpy h5py psutil tqdm packaging

# Optional but recommended
pip install wandb tensorboard datasets    # experiment tracking, HuggingFace loader
```

Register the source as an editable package if you plan to import from other scripts:

```bash
pip install -e src
```

**Environment variables** — set these once in your shell profile:

| Variable | Purpose | Default |
|---|---|---|
| `HSI_DATA_DIR` | Root of the ARAD-1K (or compatible) dataset | `./data/ARAD_1K` |
| `HSI_LOG_DIR` | Log output directory | `./artifacts/logs` |
| `HSI_CKPT_DIR` | Checkpoint directory | `./artifacts/checkpoints` |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA allocator — reduces fragmentation | `expandable_segments:True,max_split_size_mb:256` |
| `OMP_NUM_THREADS` | DataLoader worker threads | `2` |
| `HYDRA_FULL_ERROR` | Surface full Hydra stack traces | `0` (set to `1` for debug) |

---

## Dataset Setup

### Option A — Local MST++ layout

```
data/ARAD_1K/
├── Train_RGB/          # 900 JPEG/PNG RGB images (482×512)
├── Train_Spec/         # 900 spectral cubes (.mat or .npy, 31 bands)
├── split_txt/
│   ├── train.txt       # filenames for training split
│   └── valid.txt       # filenames for validation split
├── statistics/         # per-image min/max for normalization
└── channel_metadata/   # wavelength info for 31 bands
```

Set the path:

```bash
export HSI_DATA_DIR=/path/to/ARAD_1K
```

### Option B — HuggingFace Hub

Set `dataset_source: huggingface` in `config.yaml`. The trainer will load `mhmdjouni/arad_hsdb` directly:

```yaml
dataset_source: huggingface
hf_dataset_name: mhmdjouni/arad_hsdb
hf_split: train
hf_train_label_filter: train
hf_val_label_filter: validation
hf_patches_per_image: 1       # patches sampled per image per epoch
hf_max_train_samples: null    # null = use all
hf_max_val_samples: null
hf_allow_pseudo_hsi: false    # set true to allow RGB-only samples
```

Override individual fields on the CLI:

```bash
python src/hsi_model/training_script_fixed.py \
  dataset_source=huggingface \
  hf_max_train_samples=500
```

---

## Training

### Single GPU

```bash
cd "CSWIN v2"
python src/hsi_model/training_script_fixed.py \
  --config-name config \
  data_dir=/path/to/ARAD_1K
```

### Multi-GPU (DDP)

```bash
python -m torch.distributed.run \
  --nproc_per_node=4 \
  src/hsi_model/training_script_fixed.py \
  --config-name config \
  data_dir=/path/to/ARAD_1K
```

On multi-node clusters, add `NCCL_P2P_DISABLE=1` if you observe NCCL timeouts. Hydra expands log and checkpoint directories per-rank automatically.

### Memory-Optimized Trainer

For 24 GB GPUs or when the standard trainer runs out of memory:

```bash
python src/hsi_model/train_optimized.py \
  --config-name config \
  data_dir=/path/to/ARAD_1K \
  batch_size=8 \
  memory_mode=lazy
```

### Resuming from Checkpoint

Checkpoints are saved rolling under `${HSI_CKPT_DIR}`. The trainer automatically resumes from the latest checkpoint when it finds one in the output directory. To restart from a specific checkpoint:

```bash
python src/hsi_model/training_script_fixed.py \
  --config-name config \
  data_dir=/path/to/ARAD_1K
  # Place the checkpoint in ${HSI_CKPT_DIR} — the trainer picks up the most recent one
```

### Hydra Configuration Overrides

Any parameter in `config.yaml` can be overridden on the command line using dot-notation:

```bash
python src/hsi_model/training_script_fixed.py \
  data_dir=/datasets/ARAD_1K \
  batch_size=16 \
  epochs=500 \
  generator_lr=1e-4 \
  discriminator_lr=5e-5 \
  lambda_adversarial=0.15 \
  lambda_sam=0.1 \
  sinkhorn_epsilon=0.05 \
  sinkhorn_iters=100 \
  r1_gamma=5.0 \
  mixed_precision=true \
  memory_mode=standard
```

---

## Configuration Reference

Full contents of `src/configs/config.yaml`:

### Reproducibility

| Key | Default | Description |
|---|---|---|
| `seed` | `42` | Global random seed for deterministic runs |

### Dataset

| Key | Default | Description |
|---|---|---|
| `dataset_source` | `mst` | `mst` (local files) or `huggingface` |
| `data_dir` | `./data/ARAD_1K` | Root path for MST++ layout datasets |
| `hf_dataset_name` | `mhmdjouni/arad_hsdb` | HuggingFace dataset identifier |
| `hf_split` | `train` | HuggingFace split to use |
| `hf_train_label_filter` | `train` | Label prefix to select training samples |
| `hf_val_label_filter` | `validation,val` | Label prefix(es) for validation |
| `hf_rgb_label` | `null` | Custom RGB column name (null = auto-detect) |
| `hf_hsi_label` | `null` | Custom HSI column name (null = auto-detect) |
| `hf_patches_per_image` | `1` | Patches sampled per image per epoch |
| `hf_max_train_samples` | `null` | Cap on training samples (null = all) |
| `hf_max_val_samples` | `null` | Cap on validation samples (null = all) |
| `hf_allow_pseudo_hsi` | `false` | Allow RGB-only records without real HSI |

### Data Loading

| Key | Default | Description |
|---|---|---|
| `batch_size` | `20` | Training batch size |
| `val_batch_size` | `1` | Validation batch size |
| `patch_size` | `128` | Spatial patch side length (pixels) |
| `stride` | `8` | Stride between validation patches |
| `num_workers` | `2` | DataLoader worker processes |
| `memory_mode` | `standard` | `standard`, `float16`, or `lazy` |

### Training Schedule

| Key | Default | Description |
|---|---|---|
| `epochs` | `300` | Total training epochs |
| `iterations_per_epoch` | `1000` | Generator update steps per epoch |
| `gradient_accumulation_steps` | `2` | Steps before an optimizer update |
| `mixed_precision` | `true` | Automatic mixed precision (AMP) |

### Optimizer

| Key | Default | Description |
|---|---|---|
| `generator_lr` | `0.0002` | Generator Adam learning rate |
| `discriminator_lr` | `0.00005` | Discriminator Adam learning rate |
| `warmup_steps` | `2000` | Linear warmup steps at the start |

### Adversarial Training

| Key | Default | Description |
|---|---|---|
| `n_critic` | `1` | Discriminator updates per generator update |
| `use_r1_regularization` | `true` | Enable R1 gradient penalty |
| `r1_gamma` | `10.0` | R1 penalty weight |

### Sinkhorn Loss

| Key | Default | Description |
|---|---|---|
| `sinkhorn_epsilon` | `0.1` | Entropic regularization (smaller = sharper OT) |
| `sinkhorn_iters` | `50` | Sinkhorn iteration count |
| `sinkhorn_flatten_spatial` | `true` | Flatten H×W before computing OT |
| `sinkhorn_max_points` | `1024` | Maximum points per Sinkhorn call |
| `sinkhorn_kernel_clamp` | `60.0` | Cost matrix clamping for numerical stability |
| `sinkhorn_force_fp32` | `true` | Compute Sinkhorn in FP32 even under AMP |
| `sinkhorn_loss_clip` | `5.0` | Max Sinkhorn loss value (clips gradients) |

### Loss Weights

| Key | Default | Description |
|---|---|---|
| `lambda_rec` | `1.0` | Reconstruction (L1) loss weight |
| `lambda_perceptual` | `0.0` | Perceptual (feature) loss weight |
| `lambda_adversarial` | `0.1` | Sinkhorn adversarial loss weight |
| `lambda_sam` | `0.05` | Spectral Angle Mapper loss weight |

### Output & Logging

| Key | Default | Description |
|---|---|---|
| `log_dir` | `./artifacts/logs` | Log output directory |
| `checkpoint_dir` | `./artifacts/checkpoints` | Checkpoint output directory |
| `checkpoint_keep` | `5` | Number of checkpoints to retain |

---

## Inference

CSWIN v2 does not currently ship a standalone inference script. To reconstruct HSI from a trained checkpoint:

```python
import sys
sys.path.insert(0, 'src')
import torch
from hsi_model.models import NoiseRobustCSWinModel
from omegaconf import OmegaConf

cfg = OmegaConf.load('src/configs/config.yaml')
model = NoiseRobustCSWinModel(cfg)

ckpt = torch.load('artifacts/checkpoints/generator_best.pt', weights_only=True)
model.generator.load_state_dict(ckpt['generator'])
model.eval()

rgb = torch.rand(1, 3, 128, 128)   # replace with your image
with torch.no_grad():
    hsi = model.generator(rgb)     # shape: (1, 31, 128, 128)
```

Pass the output `.npy` files to the [`hsi_viz_suite`](../hsi_viz_suite/README.md) for plots and metrics.

---

## Tests

```bash
cd "CSWIN v2"
pip install pytest

# Run all tests
pytest tests/ -v

# Individual test suites
pytest tests/test_models.py      # gradient flow through generator and discriminator
pytest tests/test_attention.py   # edge cases: small maps, NaN injection, wrong dims
pytest tests/test_datasets.py    # DataLoader correctness, padding behavior
pytest tests/test_losses.py      # Sinkhorn gradcheck, gradient stability
pytest tests/test_integration.py # single-batch overfit, determinism
```

---

## Troubleshooting

### `padding (...) at dimension 3` error

This is a known issue in older versions and is already patched. Ensure you are using the latest code from this repository. The fix uses symmetric reflect padding that splits evenly between left/right and top/bottom.

### DataLoader stalls or slow I/O

```bash
python src/hsi_model/training_script_fixed.py \
  --config-name config \
  memory_mode=lazy \
  num_workers=2
```

`lazy` mode streams data instead of pre-loading it into host RAM.

### CUDA out of memory

Try in order:
1. Reduce `batch_size` (e.g., `batch_size=8`)
2. Increase `gradient_accumulation_steps` (e.g., `gradient_accumulation_steps=4`)
3. Switch to `train_optimized.py`
4. Set `memory_mode=float16`
5. Reduce `sinkhorn_max_points=512`

### Hydra config not found

```
Error: ConfigCompositionException
```

Always run from inside the `CSWIN v2/` directory and include `--config-name config` (without the `.yaml` suffix). Hydra searches `src/configs/` for the config.

### NaN loss during training

This usually means the Sinkhorn computation overflowed. Try:
- `sinkhorn_force_fp32=true` (already default)
- Lower `sinkhorn_epsilon=0.05`
- Reduce `generator_lr=1e-4`
- Verify `lambda_adversarial` is not too large (keep ≤0.2)

### Full stack traces

```bash
HYDRA_FULL_ERROR=1 python src/hsi_model/training_script_fixed.py --config-name config ...
```

---

## Performance Tips

**Speed**
- Enable AMP: `mixed_precision=true` (default)
- Increase `num_workers=4` if disk is fast
- Use `gradient_accumulation_steps=2` (default) to keep GPU busy

**Memory**
- Use `train_optimized.py` for 24 GB GPUs
- Set `memory_mode=float16` or `memory_mode=lazy`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`

**Quality**
- Train for 500 epochs (`epochs=500`)
- Lower `sinkhorn_epsilon=0.05` for sharper OT matching
- Increase `lambda_sam=0.1` to penalize spectral distortion
- Use `r1_gamma=10.0` (default) to stabilize discriminator

---

## Project Structure

```
CSWIN v2/
├── src/
│   ├── configs/
│   │   └── config.yaml              # Shared Hydra configuration
│   └── hsi_model/
│       ├── constants.py             # Dataset metadata and global defaults
│       ├── training_script_fixed.py # Sinkhorn-GAN production trainer
│       ├── train_optimized.py       # Memory-optimized MST++ trainer
│       ├── models/
│       │   ├── generator_v3.py      # NoiseRobustCSWinGenerator
│       │   ├── discriminator_v2.py  # SpectralNormalizedTransformerDiscriminator
│       │   ├── losses_consolidated.py  # Sinkhorn, SAM, perceptual losses
│       │   ├── attention.py         # CSWin attention primitives
│       │   └── model.py             # Top-level NoiseRobustCSWinModel wrapper
│       └── utils/
│           ├── checkpoint.py        # Safe checkpoint load/save helpers
│           ├── logging.py           # MetricsLogger, structured JSON logs
│           ├── metrics.py           # PSNR, SSIM, SAM, MRAE
│           ├── patch_inference.py   # Tiling-based inference utility
│           └── data/                # Dataset adapters (MST++, HuggingFace)
├── tests/
│   ├── test_models.py
│   ├── test_attention.py
│   ├── test_datasets.py
│   ├── test_losses.py
│   └── test_integration.py
├── QUICK_START.md                   # Step-by-step 10-minute setup guide
├── AUDIT_FIXES_SUMMARY.md           # Log of all 28 robustness fixes
└── README.md
```

---

## Related Projects

- [`../HSIFUSION&SHARP`](../HSIFUSION%26SHARP/README.md) — transformer baselines that share the same dataset utilities
- [`../mswr_v2`](../mswr_v2/README.md) — CNN baseline with SAM loss and wavelet branches
- [`../hsi_viz_suite`](../hsi_viz_suite/README.md) — visualization suite for reconstruction outputs
- [`../WaveDiff`](../WaveDiff/README.md) — latent diffusion alternative

---

## License

Distributed under the [MIT License](../LICENSE).

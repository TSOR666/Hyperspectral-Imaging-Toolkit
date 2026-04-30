# HSIFusion & SHARP — Transformer Baselines for HSI Reconstruction

Two production-hardened transformer models for reconstructing 31-band hyperspectral images from RGB inputs, both operating on ARAD-1K style datasets and sharing the same data pipeline.

| Model | Version | Type | Best For |
|---|---|---|---|
| **HSIFusionNet ("Lightning Pro")** | v2.5.3 | Lightweight ViT | Fast convergence, AMP-friendly, easy to extend |
| **SHARP (Hardened)** | v3.2.2 | Sparse hierarchical transformer | Memory-efficient, production-stable, best PSNR |

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
   - [HSIFusionNet v2.5.3](#hsifusionnet-v253-lightning-pro)
   - [SHARP v3.2.2](#sharp-v322-hardened)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training HSIFusionNet](#training-hsifusionnet-v253)
5. [Training SHARP](#training-sharp-v322)
6. [SHARP Inference](#sharp-inference)
7. [Configuration Reference](#configuration-reference)
   - [HSIFusionNet Config](#hsifusionnet-lightningproconfig)
   - [SHARP Config](#sharp-sharpv32config)
8. [Distributed Training](#distributed-training)
9. [SLURM Batch Jobs](#slurm-batch-jobs)
10. [Tests](#tests)
11. [Troubleshooting](#troubleshooting)
12. [Project Structure](#project-structure)

---

## Architecture Overview

### HSIFusionNet v2.5.3 ("Lightning Pro")

File: `hsifusion_v252_complete.py` — class `LightningProConfig`, factory `create_hsifusion_lightning_pro`

```
RGB (3ch) → Patch Embed → Encoder Stages → Decoder Stages → Conv Head → HSI (31ch)
                               │                  ↑
                         LightningProBlocks   Cross-Attention Fusion
```

**LightningProBlock internals:**
1. Sliding-window self-attention with **Rotary Position Embedding (RoPE)**
2. **Spectral attention** — channel-wise over the 31-band dimension
3. Optional **Mixture of Experts (MoE)** FFN
4. GELU MLP with LayerScale + DropPath
5. GroupNorm in encoder stages; optional uncertainty head in decoder

**Additional features:**
- `torch.compile` compatible (version-gated, auto-disables on older PyTorch)
- `channels_last` memory format for faster convolutions on modern GPUs
- Optional uncertainty estimation head for confidence maps
- AMP/bfloat16 safe throughout

---

### SHARP v3.2.2 (Hardened)

File: `sharp_v322_hardened.py` — class `SHARPv32Config`, factory `create_sharp_v32`

```
RGB (3ch) → Multi-scale Encoder (streaming sparse attention)
           → Bottleneck
           → Decoder (cross-attention fusion) → Spectral Head → HSI (31ch)
```

**Attention mechanism:**
- `sparse_attention_topk_streaming`: retains top-k tokens + local window fallback
- Default sparsity: `sparse_sparsity_ratio=0.9` — 90% of tokens pruned per head
- RBF key projection modes: `mean` (default) / `linear` / `none`
- `ChannelRMSNorm` with eval-time caches for numerical stability

**Additional features:**
- Spectral basis regularization in the reconstruction head
- **EMA (Exponential Moving Average)** weight tracking: `ema_decay=0.999`
- `torch.compile` support (version-gated)
- Overlap-and-blend tiling in `sharp_inference.py` for arbitrary image sizes

---

## Environment Setup

```bash
cd "HSIFUSION&SHARP"
python -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\activate             # Windows

# PyTorch — choose CUDA version matching your driver
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# Dependencies
pip install einops numpy h5py psutil tqdm tensorboard pyyaml
```

**Environment variables:**

| Variable | Purpose | Default |
|---|---|---|
| `HSI_DATA_DIR` | Dataset root containing `train/`, `val/` | `./data/ARAD_1K` |
| `HSI_LOG_DIR` | TensorBoard + JSON log directory | `./artifacts/logs` |
| `HSI_CKPT_DIR` | Checkpoint directory | `./artifacts/checkpoints` |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA allocator tweak | `expandable_segments:True,max_split_size_mb:256` |

---

## Dataset Preparation

Use the bundled `dataset_setup.py` to produce MST++ style crops from raw ARAD-1K files:

```bash
python dataset_setup.py \
  --arad-root /path/to/ARAD_1K_raw \
  --output-root ./data/ARAD_1K \
  --patch-size 128 \
  --stride 8 \
  --workers 8
```

This populates:

```
data/ARAD_1K/
├── train/
│   ├── RGB/        # cropped 128×128 RGB patches (.png)
│   └── HSI/        # corresponding 31-band cubes (.npy)
└── val/
    ├── RGB/
    └── HSI/
```

Skip this step if you already have a staged dataset from MSWR or CSWIN in the repository root — all four models share the same layout.

---

## Training HSIFusionNet v2.5.3

### Minimal command

```bash
cd "HSIFUSION&SHARP"
python hsifusion_training.py \
  --data_root ./data/ARAD_1K
```

### Full command with common options

```bash
python hsifusion_training.py \
  --data_root ./data/ARAD_1K \
  --model_size base \
  --batch_size 12 \
  --accumulate_steps 2 \
  --warmup_epochs 5 \
  --memory_mode float16 \
  --use_amp \
  --compile_model \
  --use_channels_last
```

### Resume from checkpoint

```bash
python hsifusion_training.py \
  --data_root ./data/ARAD_1K \
  --resume_from experiments/hsifusion_base/checkpoint_epoch050.pt
```

### All HSIFusionNet CLI flags

| Flag | Default | Description |
|---|---|---|
| `--data_root` | `./data/ARAD_1K` | Dataset root (train/ and val/ subfolders) |
| `--model_size` | `base` | `tiny` / `small` / `base` / `large` |
| `--batch_size` | `12` | Training batch size per GPU |
| `--accumulate_steps` | `1` | Gradient accumulation steps |
| `--warmup_epochs` | `5` | Cosine warm-up epochs |
| `--memory_mode` | `float16` | `standard` / `float16` / `lazy` |
| `--use_amp` / `--no_use_amp` | enabled | Automatic mixed precision |
| `--compile_model` / `--no_compile_model` | enabled | `torch.compile` for the forward pass |
| `--use_channels_last` / `--no_use_channels_last` | disabled | NHWC memory format |
| `--resume_from` | `None` | Path to checkpoint to resume from |
| `--log_dir` | `./experiments/hsifusion_*` | Output directory |

---

## Training SHARP v3.2.2

### Minimal command

```bash
python sharp_training_script_fixed.py \
  --data_root ./data/ARAD_1K
```

### Full command

```bash
python sharp_training_script_fixed.py \
  --data_root ./data/ARAD_1K \
  --model_size base \
  --batch_size 20 \
  --sparse_sparsity_ratio 0.9 \
  --sparse_block_size 64 \
  --sparse_q_block_size 32 \
  --sparse_max_tokens 256 \
  --sparse_window_size 8 \
  --rbf_centers_per_head 8 \
  --key_rbf_mode mean \
  --ema_decay 0.999 \
  --ema_update_every 10 \
  --use_amp
```

### All SHARP CLI flags

| Flag | Default | Description |
|---|---|---|
| `--data_root` | `./data/ARAD_1K` | Dataset root |
| `--model_size` | `base` | `tiny` / `small` / `base` / `large` |
| `--batch_size` | `20` | Training batch size per GPU |
| `--sparse_sparsity_ratio` | `0.9` | Fraction of tokens pruned (0 = dense) |
| `--sparse_block_size` | `64` | Key/value block size for streaming attention |
| `--sparse_q_block_size` | `32` | Query block size |
| `--sparse_max_tokens` | `256` | Maximum tokens retained after pruning |
| `--sparse_window_size` | `8` | Local window size for fallback attention |
| `--rbf_centers_per_head` | `8` | RBF kernel centers per attention head |
| `--key_rbf_mode` | `mean` | Key projection mode: `mean` / `linear` / `none` |
| `--ema_decay` | `0.999` | EMA decay factor |
| `--ema_update_every` | `10` | Steps between EMA updates |
| `--use_amp` / `--no_use_amp` | enabled | Automatic mixed precision |
| `--memory_mode` | `float16` | `standard` / `float16` / `lazy` |
| `--compile_model` / `--no_compile_model` | version-gated | `torch.compile` |
| `--gradient_clip` | `1.0` | Max gradient norm |
| `--resume_from` | `None` | Checkpoint to resume |

> Setting `--sparse_sparsity_ratio 0` disables token pruning and falls back to dense attention. The `k_cap` parameter is automatically disabled in this mode.

---

## SHARP Inference

`sharp_inference.py` loads a trained checkpoint and reconstructs HSI from a single RGB image or a batch. It supports overlap-and-blend tiling to handle images larger than the training patch size.

### Single image

```bash
python sharp_inference.py \
  --checkpoint experiments/sharp/best.ckpt \
  --input path/to/rgb.png \
  --output outputs/hsi.npy \
  --device cuda
```

### Tiled inference for large images

```bash
python sharp_inference.py \
  --checkpoint experiments/sharp/best.ckpt \
  --input path/to/high_res_rgb.png \
  --output outputs/hsi.npy \
  --patch-size 256 \
  --overlap 32 \
  --device cuda
```

### Batch directory

```bash
python sharp_inference.py \
  --checkpoint experiments/sharp/best.ckpt \
  --input-dir data/test/RGB \
  --output-dir outputs/hsi \
  --patch-size 256
```

### All inference flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.ckpt` file |
| `--input` | — | Single RGB image path |
| `--input-dir` | — | Directory of RGB images (batch mode) |
| `--output` | — | Single output `.npy` path |
| `--output-dir` | — | Directory for batch outputs |
| `--patch-size` | `None` | Patch size for tiling (None = no tiling) |
| `--overlap` | `16` | Overlap in pixels between tiles |
| `--device` | `cuda` | `cuda` or `cpu` |

Outputs are `.npy` files of shape `(H, W, 31)` compatible with the [HSI Viz Suite](../hsi_viz_suite/README.md).

---

## Configuration Reference

### HSIFusionNet `LightningProConfig`

```python
LightningProConfig(
    # I/O
    in_channels=3,          # RGB input
    out_channels=31,        # HSI output bands

    # Model scale
    base_channels=64,       # Channel width at first stage
    depths=[2, 2, 6, 2],    # Transformer blocks per stage
    num_heads=[2, 4, 8, 16],

    # Attention
    window_size=8,          # Sliding window size
    use_rope=True,          # Rotary position embedding
    use_sliding_window=True,
    use_sparse_attention=False,

    # Spectral
    enable_spectral=True,   # Spectral attention head

    # MoE
    use_moe=False,          # Mixture of Experts FFN
    num_experts=4,

    # FFN
    mlp_ratio=4.0,

    # Regularization
    drop_path=0.1,
    dropout=0.1,
    auxiliary_loss_weight=0.4,

    # Decoder
    use_channels_last=False,
    min_input_size=32,      # Minimum accepted spatial resolution
)
```

Model size presets (automatically set `depths`, `num_heads`, `base_channels`):

| Size | Params | `base_channels` | `depths` |
|---|---|---|---|
| `tiny` | ~6 M | 32 | [2, 2, 2, 2] |
| `small` | ~15 M | 48 | [2, 2, 4, 2] |
| `base` | ~35 M | 64 | [2, 2, 6, 2] |
| `large` | ~80 M | 96 | [2, 2, 8, 2] |

---

### SHARP `SHARPv32Config`

```python
SHARPv32Config(
    # I/O
    in_channels=3,
    out_channels=31,

    # Model scale
    base_dim=64,
    depths=[2, 2, 6, 2],
    heads=[2, 4, 8, 16],
    mlp_ratios=[4, 4, 4, 4],

    # Sparse attention
    sparse_block_size=64,
    sparse_q_block_size=32,
    sparse_max_tokens=256,
    sparse_window_size=8,
    sparse_k_cap=None,          # auto-computed from sparsity_ratio
    sparse_sparsity_ratio=0.9,  # 90% tokens pruned
    sparsemax_pad_value=-1e9,   # padding for pruned positions

    # RBF key projection
    rbf_centers_per_head=8,
    key_rbf_mode='mean',        # 'mean' | 'linear' | 'none'

    # Regularization
    drop_path_rate=0.1,

    # Memory
    use_checkpoint=True,        # gradient checkpointing

    # Runtime
    compile_mode=None,          # None | 'default' | 'reduce-overhead'
    ema_update_every=10,
)
```

---

## Distributed Training

Both models support DDP via `torch.distributed.run`:

```bash
# HSIFusion on 4 GPUs
python -m torch.distributed.run --nproc_per_node=4 \
  hsifusion_training.py \
  --data_root ./data/ARAD_1K \
  --model_size base

# SHARP on 4 GPUs
python -m torch.distributed.run --nproc_per_node=4 \
  sharp_training_script_fixed.py \
  --data_root ./data/ARAD_1K \
  --model_size base
```

---

## SLURM Batch Jobs

Ready-to-use job templates are provided:

```bash
# HSIFusion
sbatch train_job_HSI.sh

# SHARP
sbatch train_job_SHARP.sh
```

Edit the templates to set your cluster account, partition, node count, and data paths before submitting.

---

## Tests

Tests are in the project-level `tests/` directory:

```bash
pip install pytest
pytest tests/ -v
```

---

## Troubleshooting

### `torch.compile` crashes on older PyTorch

Both models auto-detect the PyTorch version and disable `torch.compile` when it is not stable. If you see compile-related errors, add `--no_compile_model` to disable it manually.

### Memory errors with SHARP

SHARP is memory-intensive with its multi-scale architecture. Try:
1. Reduce `--batch_size 8`
2. Lower `--sparse_max_tokens 128`
3. Use `--memory_mode lazy`
4. Reduce `--model_size small`

### SHARP sparse config warnings

```
UserWarning: k_cap was set but sparse_sparsity_ratio=0 — disabling k_cap
```

This is expected when you set `--sparse_sparsity_ratio 0` for dense evaluation. The warning is informational.

### HSIFusion `min_input_size` error

```
ValueError: Input spatial size (H x W) is below min_input_size=32
```

This means the encoder down-sampled the feature map below the minimum window size. Either use larger input patches (`--patch-size 64` minimum) or reduce `--model_size tiny`.

### Checkpoint loading fails

Checkpoints saved with `torch.save(..., weights_only=False)` may fail to load on newer PyTorch versions. Both trainers now use `weights_only=True` for secure loading. If you have an older checkpoint, load it with:

```python
ckpt = torch.load('path/to/ckpt.pt', weights_only=False, map_location='cpu')
```

---

## Project Structure

```
HSIFUSION&SHARP/
├── hsifusion_training.py          # HSIFusionNet trainer
├── hsifusion_v252_complete.py     # LightningProConfig + model factory
├── sharp_training_script_fixed.py # SHARP trainer (hardened)
├── sharp_inference.py             # Overlap-blend tiling inference
├── sharp_v322_hardened.py         # SHARPv32Config + model factory
├── optimized_dataloader.py        # MST++ dataloaders + EnhancedMSWRLoss
├── common_utils_v32.py            # Shared metrics, logging, helpers
├── dataset_setup.py               # ARAD-1K staging utility
├── train_job_HSI.sh               # SLURM launcher for HSIFusion
├── train_job_SHARP.sh             # SLURM launcher for SHARP
└── README.md
```

---

## Related Projects

- [`../CSWIN v2`](../CSWIN%20v2/README.md) — Sinkhorn-GAN with similar MST++ data pipeline
- [`../mswr_v2`](../mswr_v2/README.md) — CNN baseline, shares same dataset layout
- [`../hsi_viz_suite`](../hsi_viz_suite/README.md) — visualization suite for outputs
- [`../WaveDiff`](../WaveDiff/README.md) — latent diffusion alternative

---

## License

Distributed under the [MIT License](LICENSE).

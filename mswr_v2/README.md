# MSWR-Net v2.1.2

A patched and production-hardened CNN encoder-decoder with dual attention (local window + global landmark) and optional multi-level wavelet branches. Reconstructs 31-band hyperspectral images from RGB inputs on the ARAD-1K benchmark.

**What was fixed in v2.1.2:**
- Symmetric reflect padding in wavelet branches (eliminates `padding size should be less than the input dimension` crash on small feature maps)
- `LayerNorm2d` and `AdaptiveNorm2d` wrappers for correct normalization in CNN contexts
- Non-contiguous tensor handling (`reshape` instead of `view`) in loss functions
- QKV fusion for small feature maps (proper 5D→4D reshaping)
- Logger lifecycle management (handler cleanup, `propagate=False`)
- EMA weights and early-stopping state persisted in checkpoints

---

## Table of Contents

1. [Architecture](#architecture)
2. [Environment Setup](#environment-setup)
3. [Dataset Layout](#dataset-layout)
4. [Training](#training)
5. [Configuration Reference](#configuration-reference)
6. [Inference](#inference)
7. [Evaluation (NTIRE)](#evaluation-ntire)
8. [Tests](#tests)
9. [Troubleshooting](#troubleshooting)
10. [Performance Tips](#performance-tips)
11. [Project Structure](#project-structure)

---

## Architecture

File: `model/mswr_net_v212.py` — class `IntegratedMSWRNet`, config class `MSWRDualConfig`

### Backbone

```
RGB (3ch) ──→ [Stage 1 Encoder] ──→ [Stage 2 Encoder] ──→ [Stage 3 Encoder]
                      ↓ skip                ↓ skip                ↓
             [Stage 3 Decoder] ←── [Stage 2 Decoder] ←── [Stage 1 Decoder]
                                                                   ↓
                                                            HSI Head → HSI (31ch)
```

Each encoder/decoder stage contains:
1. **OptimizedWindowAttention2D** — local window self-attention with relative position bias
   - Optional Flash attention (`use_flash_attn=True`)
   - Window size `window_size=8` by default
2. **OptimizedLandmarkAttention** — global context via learned landmarks
   - Pooling modes: `learned` / `uniform` / `adaptive`
   - `num_landmarks=64` by default
3. **OptimizedCNNWaveletTransform** (optional) — multi-level DWT/IDWT
   - Filter types: `haar`, `db1`, `db2`, `db3`, `db4`
   - Learnable gating of high-frequency bands (`wavelet_gate_reuse=True`)
   - Filter caching avoids repeated recomputation
4. **Feedforward network** — 1×1 conv MLP with optional gated variant
5. **LayerNorm2d / AdaptiveNorm2d** — patched normalization for NCHW tensors

### Loss Function

`EnhancedMSWRLoss` (in `utils.py`) combines:

| Component | Weight | Description |
|---|---|---|
| L1 (MAE) | 1.0 | Pixel-level reconstruction |
| SSIM | 0.1 | Structural similarity |
| SAM | 0.05 | Spectral Angle Mapper (radians) |
| Gradient | 0.05 | Edge-preserving spatial gradient loss |

Loss weights apply a cosine warm-up schedule over the first `warmup_epochs` epochs.

---

## Environment Setup

```bash
cd mswr_v2
python -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\activate             # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

pip install numpy h5py tqdm psutil pyyaml scipy
pip install opencv-python hdf5storage matplotlib seaborn pandas scikit-learn
```

Optional:
```bash
pip install wandb fvcore plotly    # experiment tracking and model analysis
```

**Environment variables:**

| Variable | Purpose | Default |
|---|---|---|
| `MSWR_DATA_ROOT` | Dataset root | `./data/ARAD_1K` |
| `MSWR_EXPERIMENTS_ROOT` | Checkpoint + log output | `./experiments` |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA allocator | `expandable_segments:True,max_split_size_mb:128` |

---

## Dataset Layout

```
data/ARAD_1K/               # or any path passed to --data_root
├── Train_RGB/              # 900 JPEG/PNG RGB images (482×512)
├── Train_Spec/             # 900 spectral cubes (.mat or .npy, shape 31×H×W)
├── split_txt/
│   ├── train.txt           # training split filenames
│   └── valid.txt           # validation split filenames
├── statistics/             # per-image min/max normalization stats
└── channel_metadata/       # wavelength metadata for the 31 bands
```

The `dataloader.py` module handles patch extraction with configurable `patch_size` and `stride`, optional BGR→RGB flipping, and min-max spectral normalization. No separate staging step is required.

---

## Training

### Minimal command

```bash
cd mswr_v2
python train_mswr_v212_logging.py \
  --data_root /path/to/ARAD_1K
```

### Typical training command

```bash
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
```

### Training with early stopping

```bash
python train_mswr_v212_logging.py \
  --model_size base \
  --data_root /path/to/ARAD_1K \
  --early_stopping_mode min \
  --early_stopping_patience 30 \
  --ema_eval_mode ema
```

### Resume from checkpoint

```bash
python train_mswr_v212_logging.py \
  --model_size base \
  --data_root /path/to/ARAD_1K \
  --resume experiments/checkpoints/checkpoint_epoch100.pth
```

### All training CLI flags

#### Core

| Flag | Default | Description |
|---|---|---|
| `--model_size` | `base` | `tiny` / `small` / `base` / `large` |
| `--data_root` | `./data/ARAD_1K` | Path to ARAD-1K dataset |
| `--batch_size` | `8` | Training batch size |
| `--end_epoch` | `300` | Number of training epochs |
| `--init_lr` | `2e-4` | Initial learning rate |
| `--patch_size` | `128` | Spatial patch side length |
| `--stride` | `8` | Validation stride |
| `--num_workers` | `4` | DataLoader workers |

#### Model features

| Flag | Default | Description |
|---|---|---|
| `--use_wavelet` / `--no_wavelet` | enabled | Enable wavelet branches |
| `--wavelet_type` | `db1` | `haar` / `db1` / `db2` / `db3` / `db4` |
| `--wavelet_levels` | `2` | DWT decomposition levels |
| `--num_stages` | `3` | Encoder/decoder depth |
| `--num_heads` | `8` | Attention heads |
| `--window_size` | `8` | Window attention size |
| `--num_landmarks` | `64` | Landmark attention tokens |
| `--base_channels` | `64` | Channel width at stage 1 |

#### Loss and optimizer

| Flag | Default | Description |
|---|---|---|
| `--use_enhanced_loss` / `--no_enhanced_loss` | enabled | Composite loss (L1+SSIM+SAM+grad) |
| `--optimizer` | `adamw` | `adamw` / `adam` / `sgd` |
| `--weight_decay` | `0.01` | Optimizer weight decay |
| `--scheduler` | `cosine` | `cosine` / `step` / `exponential` |
| `--warmup_epochs` | `10` | Cosine warm-up epochs |
| `--gradient_clip` | `1.0` | Max gradient norm |
| `--gradient_accumulation_steps` | `1` | Accumulation steps |

#### EMA

| Flag | Default | Description |
|---|---|---|
| `--use_ema` / `--no_ema` | enabled | Exponential moving average |
| `--ema_decay` | `0.999` | EMA decay factor |
| `--ema_eval_mode` | `ema` | `ema` / `model` / `both` |

#### Early stopping

| Flag | Default | Description |
|---|---|---|
| `--early_stopping_mode` | `off` | `off` / `min` / `max` |
| `--early_stopping_patience` | `50` | Epochs without improvement before stopping |
| `--early_stopping_metric` | `mrae` | Metric to monitor |

#### Performance and memory

| Flag | Default | Description |
|---|---|---|
| `--use_amp` / `--no_amp` | disabled | Automatic mixed precision |
| `--use_checkpoint` / `--no_checkpoint` | disabled | Gradient checkpointing |
| `--use_flash_attn` / `--no_flash_attn` | disabled | Flash attention (requires FlashAttn 2) |
| `--compile_model` | disabled | `torch.compile` |
| `--profile` | disabled | GPU/CPU profiling via `PerformanceMonitor` |

#### Output

| Flag | Default | Description |
|---|---|---|
| `--log_base` | `./experiments/logs` | Log directory |
| `--checkpoint_base` | `./experiments/checkpoints` | Checkpoint directory |
| `--resume` | `None` | Checkpoint to resume from |
| `--save_every` | `10` | Save checkpoint every N epochs |

---

## Configuration Reference

`MSWRDualConfig` parameters (set programmatically or mapped from CLI flags):

### Core architecture

| Parameter | Default | Description |
|---|---|---|
| `input_channels` | `3` | Input channels (RGB) |
| `output_channels` | `31` | Output HSI bands |
| `base_channels` | `64` | Channel width at first stage |
| `num_stages` | `3` | Number of encoder/decoder stages |
| `channel_expansion` | `2.0` | Channel multiplier between stages |

### Attention

| Parameter | Default | Description |
|---|---|---|
| `attention_type` | `dual` | `window` / `landmark` / `dual` / `hybrid` |
| `num_heads` | `8` | Number of attention heads |
| `window_size` | `8` | Local window size |
| `num_landmarks` | `64` | Landmark token count for global attention |
| `landmark_pooling` | `learned` | `learned` / `uniform` / `adaptive` |
| `local_global_fusion` | `True` | Fuse window and landmark outputs |
| `fuse_qkv_small_maps` | `True` | QKV fusion for spatial dim < window_size |

### Wavelets

| Parameter | Default | Description |
|---|---|---|
| `use_wavelet` | `True` | Enable CNN wavelet branches |
| `wavelet_type` | `db1` | Filter family |
| `wavelet_levels` | `[2, 2, 2]` | DWT levels per stage |
| `wavelet_gate_reuse` | `True` | Reuse gate parameters across levels |

### Network

| Parameter | Default | Description |
|---|---|---|
| `mlp_ratio` | `4.0` | FFN hidden dim / input dim |
| `ffn_type` | `standard` | `standard` / `gated` |

### Regularization

| Parameter | Default | Description |
|---|---|---|
| `dropout` | `0.0` | General dropout rate |
| `attention_dropout` | `0.0` | Attention weight dropout |
| `drop_path` | `0.1` | Stochastic depth rate |
| `layer_scale_init` | `1e-4` | Layer-scale initialization |

### Performance

| Parameter | Default | Description |
|---|---|---|
| `use_checkpoint` | `False` | Gradient checkpointing |
| `checkpoint_blocks` | `[True, True, False]` | Which stages to checkpoint |
| `use_flash_attn` | `False` | Flash attention |
| `compile_model` | `False` | `torch.compile` |
| `mixed_precision` | `False` | AMP |
| `memory_efficient` | `False` | Extra memory-saving options |

### Other

| Parameter | Default | Description |
|---|---|---|
| `norm_type` | `layer` | `layer` / `group` / `batch` / `none` |
| `use_multi_scale_input` | `False` | Multi-scale RGB input fusion |
| `use_skip_init` | `True` | Zero-init skip connections |
| `performance_monitoring` | `False` | Enable `PerformanceMonitor` |

---

## Inference

Run batch inference on the ARAD-1K validation set:

```bash
python mswr_inference.py \
  --checkpoint experiments/checkpoints/best_model.pth \
  --data-root /path/to/ARAD_1K \
  --output-dir outputs/mswr_inference
```

The script:
1. Loads the model configuration from checkpoint metadata
2. Processes validation images in patches (using the same stride as training)
3. Writes `.npy` reconstructions and optional false-color PNG visualizations to `--output-dir`
4. Reports MRAE, RMSE, PSNR, SSIM, SAM for the validation set

All inference flags:

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.pth` checkpoint file |
| `--data-root` | `./data/ARAD_1K` | Dataset root |
| `--output-dir` | `./outputs` | Directory for `.npy` outputs |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--batch-size` | `4` | Inference batch size |
| `--save-rgb` | disabled | Save false-color PNG visualizations |
| `--ema` | enabled | Use EMA weights if available |

Outputs are compatible with the [`hsi_viz_suite`](../hsi_viz_suite/README.md).

---

## Evaluation (NTIRE)

Use `mswr_test_ntire.py` for NTIRE-style evaluation on a test set without ground truth:

```bash
python mswr_test_ntire.py \
  --checkpoint experiments/checkpoints/best_model.pth \
  --test-root /path/to/NTIRE_test \
  --output-dir submissions/mswr_ntire
```

This produces a `submission.zip` in the expected NTIRE format.

---

## Tests

```bash
pip install pytest
pytest tests/ -v

# Individual suites
pytest tests/test_dataloader.py    # patch extraction, normalization, collation
pytest tests/test_metrics.py       # MRAE, RMSE, PSNR, SAM correctness
pytest tests/test_model_variants.py # tiny/small/base/large shapes
```

---

## Troubleshooting

### `padding size should be less than the input dimension`

This was the original bug in MSWR-Net. The patched code uses symmetric reflect padding. If you see this error, ensure you are on the latest version of this repository and that `use_wavelet=True` code paths are using `model/mswr_net_v212.py` (not an older copy).

### Logger writes to wrong file / duplicate log lines

Ensure you call only one training process at a time. The logger cleanup in `train_mswr_v212_logging.py` closes all handlers before reconfiguration. If you see duplicate lines, restart Python (stale handlers from a previous crashed session may persist in interactive environments).

### CUDA OOM

Try in order:
1. Reduce `--batch_size 4`
2. Enable `--use_checkpoint`
3. Enable `--use_amp`
4. Reduce `--model_size small`
5. Lower `--base_channels 32`

### Flash attention not available

Flash attention requires the `flash_attn` package and a compatible GPU (Ampere+). If it is not installed, the flag `--use_flash_attn` silently falls back to standard attention. Install it with:

```bash
pip install flash-attn --no-build-isolation
```

### Checkpoint resume misses optimizer state

Resume always restores model weights, optimizer state, scheduler state, and EMA. If optimizer state is missing (e.g., upgrading from an older checkpoint), the trainer starts with a fresh optimizer but retains the model weights.

---

## Performance Tips

**Speed**
- Enable `--use_amp` for ~1.5–2× throughput
- Enable `--use_flash_attn` if on Ampere+ GPU
- Set `--num_workers 8` with fast NVMe storage
- Use `--gradient_accumulation_steps 2` to saturate GPU

**Memory**
- Enable `--use_checkpoint` (saves ~40% VRAM at 20% speed cost)
- Reduce `--base_channels 32` for a smaller model
- Use `--model_size small` or `tiny`

**Quality**
- Train for 300+ epochs with `--scheduler cosine`
- Use `--wavelet_type db2` or `db3` (smoother filters than haar)
- Enable `--use_enhanced_loss` (always enabled by default)
- Set `--ema_eval_mode ema` for best validation metrics

---

## Project Structure

```
mswr_v2/
├── model/
│   └── mswr_net_v212.py      # IntegratedMSWRNet, MSWRDualConfig
├── utils.py                  # Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_SAM,
│                             # AverageMeter, EnhancedMSWRLoss, PerformanceMonitor
├── dataloader.py             # TrainDataset, ValidDataset, DatasetConfig
├── train_mswr_v212_logging.py # Main training entry point
├── mswr_inference.py         # Batch inference utility
├── mswr_test_ntire.py        # NTIRE evaluation script
├── smoke_train.py            # Quick 1-epoch sanity check
├── tests/
│   ├── test_dataloader.py
│   ├── test_metrics.py
│   └── test_model_variants.py
├── AUDIT_REPORT.md           # Robustness improvement log
└── README.md
```

---

## Related Projects

- [`../CSWIN v2`](../CSWIN%20v2/README.md) — Sinkhorn-GAN transformer baseline
- [`../HSIFUSION&SHARP`](../HSIFUSION%26SHARP/README.md) — lightweight and sparse transformer baselines
- [`../WaveDiff`](../WaveDiff/README.md) — latent diffusion alternative
- [`../hsi_viz_suite`](../hsi_viz_suite/README.md) — visualization suite for `.npy` outputs

---

## License

Distributed under the [MIT License](../LICENSE).

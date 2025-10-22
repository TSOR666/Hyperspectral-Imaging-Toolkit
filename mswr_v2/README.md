# MSWR-Net v2.1.2 (Fixed & Packaged)

This directory bundles the patched MSWR-Net v2.1.2 architecture, accompanying utilities, and hardened training/inference scripts used in production. The original release suffered from padding crashes, fragile logging, and non-contiguous tensor assumptions; those issues are resolved here while preserving the public API.

## Highlights

- **Symmetric reflect padding** inside the wavelet attention blocks eliminates the "padding size should be less than the input dimension" error on small feature maps.
- **Stabilised training driver** (`train_mswr_v212_logging.py`) with proper logger lifecycle management, EMA tracking, and SAM-augmented loss.
- **Inference helpers** (`mswr_inference.py`) for batch processing checkpoints across validation sets.
- **CLI ergonomics** such as early-stopping controls, EMA toggles, and diagnostics logging.

## Directory structure

```
mswr_v2/
├─ model/                        # Patched MSWR modules
│  └─ mswr_net_v212.py
├─ utils.py                      # Losses, metrics, and logging helpers
├─ train_mswr_v212_logging.py    # Main training entry point
├─ mswr_inference.py             # Offline inference utility
├─ mswr_test_ntire.py            # NTIRE evaluation script
└─ README.md
```

## Installation

1. Create/activate a Python 3.9+ environment with CUDA-enabled PyTorch.
2. Install project dependencies (adapt to your package manager):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install numpy h5py wandb tqdm psutil pyyaml
   ```
3. The legacy MST++ dataloader is now bundled as [`dataloader.py`](dataloader.py),
   so no additional files are required. Ensure the ARAD-1K dataset follows the
   `Train_RGB` / `Train_Spec` / `split_txt` layout before launching training.

## Training

```bash
cd mswr_v2
python train_mswr_v212_logging.py \
  --model_size base \
  --data_root /path/to/ARAD_1K \
  --use_wavelet \
  --use_enhanced_loss
```

Key options:

| Flag | Description | Default |
| --- | --- | --- |
| `--model_size {tiny,small,base,large}` | Selects the model constructor inside `model/mswr_net_v212.py`. | `base` |
| `--use_wavelet/--no_wavelet` | Enables the CNN-based wavelet branch. | Enabled |
| `--use_enhanced_loss/--no_enhanced_loss` | Toggles the composite loss (L1 + SAM + SSIM + gradient terms). | Enabled |
| `--use_ema/--no_ema` | Controls exponential moving average tracking. | Enabled |
| `--early_stopping_mode {off,min,max}` | Configure patience-based stopping. | `off` |
| `--ema_eval_mode {ema,model,both}` | Choose which weights drive validation metrics. | `ema` |

All CLI flags are documented near the bottom of [`train_mswr_v212_logging.py`](train_mswr_v212_logging.py). Logs and checkpoints are saved under timestamped folders in `./experiments/` unless overridden.

### Tips

- Set `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"` to mirror the allocator tweaks used when debugging memory spikes.
- Use `--gradient_accumulation_steps` to fit larger effective batch sizes on smaller GPUs.
- Enable `--profile` to emit GPU/CPU utilisation snapshots via `PerformanceMonitor`.

## Inference

Run inference against a trained checkpoint:

```bash
python mswr_inference.py \
  --checkpoint checkpoints/best_model.pth \
  --data-root /path/to/ARAD_1K \
  --output-dir outputs/inference
```

The script loads the appropriate model configuration from the checkpoint metadata and writes reconstructed `.npy` files alongside optional RGB visualisations. Reuse these outputs with the [visualisation suite](../hsi_viz_suite/README.md).

## Related projects

- [`../HSIFUSION&SHARP`](../HSIFUSION&SHARP/README.md) hosts the transformer-based HSIFusion and SHARP baselines that share the same dataset layout.
- [`../CSWIN v2`](../CSWIN%20v2/README.md) provides the Sinkhorn-GAN training pipelines for a complementary transformer model.

## Patch notes

- **Wavelet padding fix**: all call sites now split the padding between left/right and top/bottom to stay within the constraints of `mode="reflect"`.
- **Logger lifecycle**: every handler is closed before reconfiguration; warnings are redirected to the log file; an early bootstrap logger captures import issues.
- **Loss functions**: `Loss_SAM` and friends operate on non-contiguous tensors by using `reshape` instead of `view`.
- **Checkpoint hygiene**: EMA weights and early-stopping state are persisted, making resume runs deterministic.

If you encounter edge cases (tiny crops, unusual window sizes), consider switching the padding mode to `'circular'` in the wavelet path; the symmetric reflect fix resolves the known crashes without regressing quality in our tests.

## License

MSWR-Net v2.1.2 patches are released under the [MIT License](../LICENSE), consistent with the rest of the monorepo.

## Architecture

- Backbone: Encoder–decoder with `num_stages` stages and learned skip connections.
- Attention: Each block uses dual attention:
  - Local window attention with relative positional bias and optional flash attention.
  - Global/landmark attention with learned/adaptive landmark pooling.
- Wavelets: Optional CNN-based DWT/IDWT per stage with learnable gating of high‑frequency bands.
- Normalization: Fixed handling for CNN contexts via `LayerNorm2d` and `AdaptiveNorm2d` (BatchNorm/GroupNorm compatibility).
- FFN: 1x1 conv MLP with optional gated variant; drop‑path and layer‑scale supported.

Reference: `model/mswr_net_v212.py` (`IntegratedMSWRNet`, `MSWRDualConfig`).

## Training Overview

- Driver: `train_mswr_v212_logging.py` with robust logging and error capture.
- Loss: `EnhancedMSWRLoss` combines L1, SSIM, SAM (radians, logged in degrees), and gradient loss with warm‑up weighting.
- Optimizer: AdamW/Adam/SGD with grouped parameter decay; warm‑up + cosine/step/exponential scheduler via wrapper.
- Efficiency: AMP, gradient checkpointing, gradient accumulation, optional `torch.compile`, multi‑GPU (DDP), EMA.
- Data: `dataloader.py` (ARAD‑1K MST++ style). Patch extraction by `patch_size` and `stride`, optional RGB BGR→RGB, min/max scaling.

Example:

```bash
python train_mswr_v212_logging.py \
  --model_size base --data_root /path/to/ARAD_1K \
  --batch_size 8 --end_epoch 300 --init_lr 2e-4 \
  --use_wavelet --wavelet_type db2 --use_enhanced_loss \
  --use_amp --use_checkpoint --use_flash_attn
```

## Key Configuration (MSWRDualConfig)

- Core: `input_channels`, `output_channels`, `base_channels`, `num_stages`, `channel_expansion`.
- Attention: `attention_type` (window/dual/landmark/hybrid), `num_heads`, `window_size`, `num_landmarks`, `landmark_pooling`, `local_global_fusion`.
- Wavelets: `use_wavelet`, `wavelet_type` (haar/db1–db4), `wavelet_levels`, `wavelet_gate_reuse`.
- Network: `mlp_ratio`, `ffn_type` (standard/gated), `fuse_qkv_small_maps`.
- Regularization: `dropout`, `attention_dropout`, `drop_path`, `layer_scale_init`.
- Performance: `use_checkpoint`, `checkpoint_blocks`, `use_flash_attn`, `compile_model`, `mixed_precision`, `memory_efficient`.
- Other: `norm_type` (layer/group/batch/none), `use_multi_scale_input`, `use_skip_init`, `performance_monitoring`.

Tip: Set `--wavelet_levels` per stage (e.g., `--wavelet_levels 1 1 2`) or pass one value to repeat automatically.

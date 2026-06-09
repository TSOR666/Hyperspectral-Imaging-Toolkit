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
   pip install opencv-python hdf5storage fvcore scipy matplotlib seaborn pandas plotly scikit-learn
   ```
3. The legacy MST++ dataloader is now bundled as [`dataloader.py`](dataloader.py),
   so no additional files are required. Ensure the ARAD-1K dataset follows the
   `Train_RGB` / `Train_Spec` / `split_txt` layout before launching training.
4. (Optional) Point the tooling at custom folders by setting environment variables before launching:
   - `MSWR_DATA_ROOT=/path/to/ARAD_1K`
   - `MSWR_EXPERIMENTS_ROOT=/path/to/output_dir`

By default, logs and checkpoints are stored under `./experiments/{logs,checkpoints}` relative to this directory and the dataset path resolves to `./data/ARAD_1K`. Override them via the CLI flags `--data_root`, `--log_base`, and `--checkpoint_base` or by exporting the environment variables above.

When `ema_eval_mode: both`, training saves independent lightweight minima as
`best_raw_model.pth` and `best_ema_model.pth`. This prevents an improved raw
model checkpoint from being hidden by EMA-only ranking.

## Training

```bash
cd mswr_v2
python train_mswr_v212_logging.py \
  --model_size base \
  --data_root /path/to/ARAD_1K \
  --use_wavelet \
  --use_enhanced_loss
```

### Configuration workflow

Training options can be supplied directly on the CLI or through a YAML file:

```bash
python train_mswr_v212_logging.py --config configs/mswr_base.yaml --init_lr 1e-4
```

YAML keys match the CLI names without leading dashes. Explicit CLI values override YAML values, so the command above would keep the YAML setup but replace `init_lr`. Unknown YAML keys are ignored by the trainer. Logs and checkpoints are saved under timestamped folders in `./experiments/` unless `--log_base`, `--checkpoint_base`, or `MSWR_EXPERIMENTS_ROOT` are set.

By default, MSWR uses MST++-style logical epochs: `--steps_per_epoch 1000` caps each epoch to 1000 training batches even when dense patch extraction produces far more available batches. This keeps LR warmup, cosine decay, EMA start, and loss warmup on the intended iteration scale. Set `--steps_per_epoch 0` to consume the full DataLoader each epoch. When resuming older checkpoints, the trainer derives the logical epoch from the saved iteration count if the checkpoint epoch is behind.

### Model presets

`--model_size` selects one of the factory constructors in [`model/mswr_net_v212.py`](model/mswr_net_v212.py):

| `--model_size` | `base_channels` | `num_stages` | `num_heads` | `window_size` | `num_landmarks` | Suggested use |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tiny` | 32 | 2 | 4 | 4 | 32 | Fast smoke tests and memory-constrained runs. |
| `small` | 48 | 3 | 6 | 8 | 49 | Speed/quality compromise. |
| `base` | 64 | 3 | 8 | 8 | 64 | Default production setting. |
| `large` | 96 | 4 | 12 | 12 | 128 | Higher-capacity runs when memory allows. |

When `model_size` is one of these presets, the preset owns `base_channels`, `num_stages`, `num_heads`, `window_size`, and `num_landmarks`; passing those flags does not change the preset architecture. For custom architectures, use a YAML config with a non-preset `model_size` so the trainer falls back to `MSWRDualConfig`:

```yaml
model_size: custom
base_channels: 80
num_stages: 4
num_heads: 8
window_size: 8
num_landmarks: 96
wavelet_levels: [1, 1, 2, 2]
```

### Training CLI reference

Run and path options:

| Flag | Choices / type | Default | Notes |
| --- | --- | --- | --- |
| `--config` | path | `None` | Optional YAML config. CLI values take precedence. |
| `--pretrained_model_path` | path | `None` | Resume/load path stored as `resume_path`. |
| `--data_root` | path | `MSWR_DATA_ROOT` or `./data/ARAD_1K` | Dataset root with ARAD-style folders. |
| `--log_base` | path | `./experiments/logs` | Timestamped run log directory is created here. |
| `--checkpoint_base` | path | `./experiments/checkpoints` | Timestamped checkpoint directory is created here. |
| `--experiment_name` | string | `mswr_v212` | Used for logging/W&B run naming. |
| `--use_wandb` | flag | off | Enables Weights & Biases logging. |
| `--gpu_id` | string | `0` | Sets visible GPU(s) for single-process training. |

Data and iteration options:

| Flag | Type | Default | Notes |
| --- | --- | ---: | --- |
| `--batch_size` | int | 20 | Per-step batch size before gradient accumulation. |
| `--end_epoch` | int | 300 | Epoch limit; with dense patch extraction this can be very long. |
| `--steps_per_epoch` | int | 1000 | Training batches per logical epoch; set `<=0` for a full pass over all extracted patches. |
| `--patch_size` | int | 128 | Training crop size. |
| `--stride` | int | 8 | Patch extraction stride. |
| `--num_workers` | int | 4 | DataLoader workers. |
| `--pin_memory` | flag | on | Defaults to enabled; set `pin_memory: false` in YAML to disable. |
| `--save_frequency` | int | 5000 | Iteration interval for numbered checkpoints. |
| `--validate_frequency` | int | 1000 | Iteration interval for validation. |
| `--seed` | int | 42 | Python/NumPy/PyTorch seed. |
| `--deterministic` | flag | off | Slower reproducible mode; disables TF32. |

Architecture and attention options:

| Flag | Choices / type | Default | Notes |
| --- | --- | --- | --- |
| `--model_size` | `tiny`, `small`, `base`, `large` | `base` | Selects a preset factory. |
| `--attention_type` | `window`, `dual`, `landmark`, `hybrid` | `dual` | Attention block mode. |
| `--landmark_pooling` | `learned`, `uniform`, `adaptive` | `learned` | Landmark/global attention pooling. |
| `--use_checkpoint` | flag | off | Enables activation checkpointing in the training config. |
| `--use_flash_attn` | flag | on | Defaults to enabled; set `use_flash_attn: false` in YAML to disable. |
| `--base_channels` | int | 64 | Used only by the custom `MSWRDualConfig` path. |
| `--num_stages` | int | 3 | Used only by the custom `MSWRDualConfig` path. |
| `--num_heads` | int | 8 | Used only by the custom `MSWRDualConfig` path. |
| `--window_size` | int | 8 | Used only by the custom `MSWRDualConfig` path. |
| `--num_landmarks` | int | 64 | Used only by the custom `MSWRDualConfig` path. |

Wavelet options:

| Flag | Choices / type | Default | Notes |
| --- | --- | --- | --- |
| `--use_wavelet` | flag | on | CNN wavelet branch is enabled by default; set `use_wavelet: false` in YAML to disable. |
| `--wavelet_type` | `haar`, `db1`, `db2`, `db3`, `db4` | `db2` | Training-script default; `MSWRDualConfig` itself defaults to `db1`. |
| `--wavelet_levels` | int list | `None` | Per-stage DWT levels, e.g. `--wavelet_levels 1 1 2`; one value is repeated across stages. |

Loss options:

| Flag | Type | Default | Notes |
| --- | --- | ---: | --- |
| `--use_enhanced_loss` | flag | off | Enables `EnhancedMSWRLoss` (`L1 + MRAE + SSIM + SAM + Gradient`). |
| `--l1_weight` | float | 1.0 | L1 component weight. |
| `--mrae_weight` | float | 0.0 | MRAE component weight; useful for MRAE-focused fine-tuning. |
| `--ssim_weight` | float | 0.5 | SSIM component weight. |
| `--sam_weight` | float | 0.1 | SAM component weight; optimized in radians, logged in degrees. |
| `--gradient_weight` | float | 0.1 | Spatial gradient loss weight. |
| `--loss_warmup_epochs` | int | 10 | Warmup for auxiliary loss terms. |

Optimization and schedule options:

| Flag | Choices / type | Default | Notes |
| --- | --- | ---: | --- |
| `--optimizer` | `adam`, `adamw`, `sgd` | `adamw` | Optimizer family. |
| `--scheduler` | `cosine`, `step`, `exponential` | `cosine` | LR schedule. |
| `--init_lr` | float | 4e-4 | Initial/base learning rate. |
| `--min_lr` | float | 1e-6 | Lower LR bound for scheduled decay. |
| `--warmup_epochs` | int | 5 | LR warmup duration. |
| `--weight_decay` | float | 1e-4 | Weight decay. |
| `--gradient_clip` | float | 1.0 | Gradient clipping threshold. |
| `--gradient_accumulation_steps` | int | 1 | Effective batch multiplier. |

Runtime, precision, EMA, and stopping options:

| Flag | Choices / type | Default | Notes |
| --- | --- | --- | --- |
| `--use_amp` | flag | on | AMP is enabled by default; set `use_amp: false` in YAML to disable. |
| `--amp_dtype` | `auto`, `fp16`, `bf16` | `auto` | `auto` prefers BF16 when supported. |
| `--channels_last` / `--no_channels_last` | flag pair | on | CUDA memory format. |
| `--memory_monitoring` | flag | on | Enables memory/performance monitoring; set false in YAML to disable. |
| `--profile_model` | flag | off | Emits model profiling diagnostics. |
| `--mrae_diagnostics` | flag | off | Logs strict MRAE contribution by target-intensity bucket during validation. |
| `--use_ema` / `--no_ema` | flag pair | on | EMA tracking for model weights. |
| `--ema_decay` | float | 0.999 | EMA decay factor. |
| `--ema_start_epoch` | int | 5 | Delays EMA updates until this epoch. |
| `--ema_eval_mode` | `ema`, `model`, `both` | `ema` | Weight set used for validation metrics. |
| `--early_stopping_mode` | `off`, `min`, `max` | `off` | Disabled by default. |
| `--early_stopping_patience` | int | 50 | Patience once early stopping is active. |
| `--early_stopping_warmup` | int | 5 | Epochs skipped before early stopping can trigger. |

Distributed options:

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--distributed` | flag | off | Also auto-enabled when `WORLD_SIZE > 1` under `torchrun`. |
| `--local_rank` / `--local-rank` | int | 0 | Launcher rank compatibility. |
| `--ddp_find_unused_parameters` | flag | off | Debug option for dynamic graphs. |

### Tips

- Set `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"` to mirror the allocator tweaks used when debugging memory spikes.
- Use `--gradient_accumulation_steps` to fit larger effective batch sizes on smaller GPUs.
- Enable `--profile_model` to emit model profiling diagnostics.

## Inference

Run inference against a trained checkpoint:

```bash
python mswr_inference.py \
  --checkpoint checkpoints/best_model.pth \
  --data-root /path/to/ARAD_1K \
  --output-dir outputs/inference
```

The script loads the appropriate model configuration from the checkpoint metadata and writes reconstructed `.npy` files alongside optional RGB visualisations. Reuse these outputs with the [visualisation suite](../hsi_viz_suite/README.md).

## Testing and visualization export

`mswr_test_ntire.py` now supports real ARAD test samples when they are available. By default `--split auto` prefers `split_txt/test_list.txt` with `Test_RGB` / `Test_Spec`, falling back to validation splits when no test split exists. For the selected visualization samples it also exports `test_results/hsi/<sample>.npy`, `test_results/hsi/<sample>_target.npy`, and per-sample metric JSON files so `hsi_viz_suite/scripts/generate_all_visualizations.py` can consume the actual test samples directly.

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
- Optimizer: AdamW/Adam/SGD with grouped parameter decay; continuous warm‑up + cosine/step/exponential scheduler via `LambdaLR`.
- Efficiency: AMP with `--amp_dtype auto|fp16|bf16`, channels-last CUDA training, gradient checkpointing, gradient accumulation, optional `torch.compile`, multi‑GPU DDP, EMA.
- Data: `dataloader.py` (ARAD‑1K MST++ style). Patch extraction by `patch_size` and `stride`, optional RGB BGR→RGB, min/max scaling.

Example:

```bash
python train_mswr_v212_logging.py \
  --model_size base --data_root /path/to/ARAD_1K \
  --batch_size 8 --end_epoch 300 --init_lr 2e-4 \
  --use_wavelet --wavelet_type db2 --use_enhanced_loss \
  --use_amp --amp_dtype auto --channels_last \
  --use_checkpoint --use_flash_attn
```

## Full Model Configuration (`MSWRDualConfig`)

These fields live in [`MSWRDualConfig`](model/mswr_net_v212.py). The most common ones are exposed through `train_mswr_v212_logging.py`; lower-level architecture fields are useful when constructing the model directly. YAML configs only affect keys already known to the training parser.

| Group | Fields | Defaults / choices |
| --- | --- | --- |
| Core channels | `input_channels`, `output_channels`, `base_channels`, `channel_expansion`, `num_stages` | `3`, `31`, `64`, `2.0`, `3` |
| Attention | `attention_type`, `num_heads`, `window_size`, `num_landmarks`, `landmark_pooling`, `local_global_fusion` | `dual`; `8`; `8`; `64`; `learned`; `adaptive` |
| Attention choices | `attention_type`; `landmark_pooling`; `local_global_fusion` | `window`, `dual`, `landmark`, `hybrid`; `learned`, `uniform`, `adaptive`; `adaptive`, `concat`, `add`, `gated` |
| Wavelets | `use_wavelet`, `wavelet_type`, `wavelet_levels`, `wavelet_gate_reuse` | `True`; `db1` in the config class, `db2` in the trainer; `None` expands to `[1..num_stages]`; `False` |
| Network block | `mlp_ratio`, `ffn_type`, `fuse_qkv_small_maps` | `4.0`; `standard` or `gated`; `False` no-op compatibility knob |
| Regularization | `dropout`, `attention_dropout`, `drop_path`, `layer_scale_init` | `0.0`, `0.0`, `0.1`, `1e-4` |
| Performance | `use_checkpoint`, `checkpoint_blocks`, `use_flash_attn`, `compile_model`, `mixed_precision`, `memory_efficient` | `True`; auto-selected when checkpointing; `True`; `False`; `True`; `True` |
| Advanced features | `use_multi_scale_input`, `use_skip_init`, `norm_type`, `performance_monitoring` | `True`; `True`; `layer` (`layer`, `group`, `batch`, `none`); `True` |

Tip: set `--wavelet_levels` per stage, e.g. `--wavelet_levels 1 1 2`, or pass one value to repeat it across stages. For fields that are not wired to CLI flags, create the model through Python with `MSWRDualConfig`.

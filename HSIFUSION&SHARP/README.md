# HSIFusion & SHARP — Transformer Baselines for HSI Reconstruction

This directory packages the two transformer-based baselines we maintain alongside the CNN models in this monorepo:

- **HSIFusionNet v2.5.3 ("Lightning Pro")** – a lightweight spectrum-aware ViT that favours fast convergence and AMP-friendly kernels.
- **SHARP v3.2.2 (Hardened)** – a sparse attention reconstruction pipeline with the audit fixes applied for production use.

Both projects share the same data preparation code and operate on ARAD-1K style hyperspectral datasets (31 channels). The scripts here mirror the ones we run internally after incorporating stability fixes, deterministic logging, and memory-usage guards, plus the fixes from five audit passes (see `AUDIT_REPORT*.md`).

**The canonical entry points are now [`unified_training.py`](unified_training.py) and [`unified_inference.py`](unified_inference.py)**, which train and evaluate either model behind a single `--model {hsifusion,sharp}` flag with MST++/NTIRE-faithful defaults. The per-model scripts (`hsifusion_training.py`, `sharp_training_script_fixed.py`, `sharp_inference.py`) are retained because the audit regression tests pin their internals, but new runs should use the unified scripts.

## Directory structure

```
HSIFUSION&SHARP/
├─ unified_training.py             # Canonical trainer (--model hsifusion|sharp)
├─ unified_inference.py            # Canonical evaluation / reconstruction CLI
├─ test_unified_infra.py           # Tests for the unified infrastructure
├─ hsifusion_training.py           # Legacy HSIFusionNet Lightning Pro trainer
├─ hsifusion_v252_complete.py      # HSIFusion model factory (tiny/small/base/large)
├─ hsifusion_classifier_v253.py    # Classifier variant of the HSIFusion backbone
├─ sharp_training_script_fixed.py  # Legacy SHARP v3.2.2 hardened trainer
├─ sharp_inference.py              # Legacy inference / patch-based tiling utility
├─ sharp_v322_hardened.py          # SHARP model + trainer implementations
├─ sharp_config.ini                # SHARP config file (+ sharp_config_loader.py)
├─ optimized_dataloader.py         # Memory-efficient MST++ dataloaders + losses
├─ common_utils_v32.py             # Shared utilities for both models
├─ early_stopping.py               # Early-stopping helper
├─ dataset_setup.py                # Helper to stage ARAD-1K splits and caches
├─ smoke_train.py / smoke_infer.py / smoke_pipeline.py  # CPU smoke checks
├─ test_audit*.py, test_runtime_audit_regressions.py    # Audit regression suites
├─ AUDIT_REPORT*.md                # Findings from the audit passes
├─ train_job_HSI.sh                # Example LSF launcher for HSIFusion
├─ train_job_SHARP.sh              # Example LSF launcher for SHARP
└─ README.md
```

## Environment setup

1. Create / activate a Python 3.9+ environment with CUDA-enabled PyTorch 1.13 or newer.
2. Install the core dependencies used by both trainers:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install numpy hydra-core h5py psutil tqdm tensorboard einops
   ```
3. (Optional) Install extra logging / experiment tracking tools as required (e.g. `wandb`).

Set the following environment variables to keep outputs consistent with the rest of the toolkit:

| Variable | Purpose | Default |
| --- | --- | --- |
| `HSI_DATA_DIR` | Root folder that contains `train/`, `val/`, and `test/` HSI tiles. | `./data/ARAD_1K` |
| `HSI_LOG_DIR` | Where training logs (TensorBoard + JSON) will be written. | `./artifacts/logs` |
| `HSI_CKPT_DIR` | Folder to store checkpoints per experiment. | `./artifacts/checkpoints` |
| `PYTORCH_CUDA_ALLOC_CONF` | Recommended allocator tweak to avoid fragmentation. | `expandable_segments:True,max_split_size_mb:256` |

## Dataset preparation

Use the bundled `dataset_setup.py` helper to verify an existing ARAD-1K layout and create `split_txt/train_list.txt` plus `valid_list.txt`. Example:

```bash
cd "HSIFUSION&SHARP"
python dataset_setup.py /path/to/ARAD_1K --train-ratio 0.95 --samples 5
```

The dataset root must already contain `Train_RGB/` images and matching `Train_Spec/*.mat` cubes. Skip split creation with `--verify-only`.

## Unified training (canonical)

`unified_training.py` trains either model with MST++/NTIRE-faithful optimizer and data defaults: Adam (`--optimizer adam`), lr `4e-4` with per-iteration cosine decay to `--eta_min 1e-6`, batch 20, 300 epochs, 128×128 patches with stride 8, no weight decay, and no warmup. Training uses an MRAE continuation floor that anneals from `1e-2` to `1e-3`; validation and best-model selection retain the exact `1e-6` MRAE floor. This prevents dark/zero target pixels from producing unusably large mixed-precision gradients without changing the reported benchmark metric.

```bash
cd "HSIFUSION&SHARP"
python unified_training.py \
  --model sharp \
  --model_size base \
  --data_root ${HSI_DATA_DIR:-./data/ARAD_1K} \
  --output_dir ./experiments/unified
```

Key flags (see `unified_training.py --help` for the full list):

| Flag | Description | Default |
| --- | --- | --- |
| `--model {hsifusion,sharp}` | Which architecture to train. | `sharp` |
| `--model_size` | Backbone variant passed to the model factory. | `base` |
| `--model_kwargs` | JSON string of extra config overrides. | `{}` |
| `--batch_size` / `--patch_size` / `--stride` | MST++-style patch pipeline. | `20` / `128` / `8` |
| `--epochs` / `--lr` / `--eta_min` | Schedule length and cosine bounds. | `300` / `4e-4` / `1e-6` |
| `--optimizer {adam,adamw}` / `--weight_decay` | Optimizer family. | `adam` / `0.0` |
| `--amp {auto,bf16,fp16,off}` | Mixed precision (`auto` prefers BF16). | `auto` |
| `--train_mrae_eps_start` / `--train_mrae_eps_end` / `--train_mrae_eps_anneal_steps` | Stable training-only MRAE floor schedule. | `1e-2` / `1e-3` / `50000` |
| `--mrae_eps` | Validation/selection MRAE floor. | `1e-6` |
| `--ema_decay` | EMA of model weights (`0` disables). | `0.0` |
| `--memory_mode {standard,float16,lazy}` / `--cache_size` | Dataloader caching. | `float16` / `4` |
| `--accumulate_steps` / `--gradient_clip` | Effective batch / clipping. | `1` / `1.0` |
| `--val_interval` / `--val_crop_border` | Validation cadence and MST++ crop. | `10` / `128` |
| `--checkpoint_interval_steps` | Rolling mid-epoch resume checkpoint cadence (`0` disables). | `5000` |
| `--compile` | Enable `torch.compile`. | off |
| `--resume` | Trainer checkpoint to resume from. | – |

Validation reports MRAE, RMSE, PSNR, SAM, SSIM, and MAE via the shared [`hsi_benchmark`](../hsi_benchmark) metrics package, under **both** the MST++ center-crop protocol and full-frame evaluation. Inputs are automatically padded to a multiple of 8 for SHARP (ARAD frames are 482×512, which SHARP's decoder cannot handle unpadded) and cropped back after the forward pass.

`last.pth` is overwritten every 5,000 optimizer steps, after every epoch, and on recoverable training failure; `best.pth` is updated only by a finite validation-MRAE improvement. Periodic validation therefore no longer delays the first resume checkpoint.

## Unified inference (canonical)

`unified_inference.py` evaluates a checkpoint on a dataset split or reconstructs arbitrary RGB images. The architecture is auto-detected from checkpoint metadata when `--model` is omitted, and legacy SHARP checkpoints (pre-sigmoid output) load with a tanh fallback automatically.

```bash
# Evaluate on a dataset (crop + full protocols, 6 metrics)
python unified_inference.py experiments/unified/sharp_base/best.pth \
  --data_root ${HSI_DATA_DIR:-./data/ARAD_1K}

# Reconstruct RGB images to NTIRE-style .mat cubes
python unified_inference.py experiments/unified/sharp_base/best.pth \
  --rgb path/to/rgb_folder --out_dir outputs/cubes
```

Reconstruction writes NTIRE-compatible `.mat` files containing an `(H, W, 31)` `cube`. Use `--no_ema` to evaluate raw weights when the checkpoint carries EMA weights, and `--crop_border` (default 128) to adjust the MST++ crop protocol.

## Legacy per-model trainers

The scripts below predate the unified infrastructure and are kept because the audit regression tests pin their internals. Note their defaults differ from the unified recipe (e.g. `hsifusion_training.py` uses AdamW, lr `3e-4`, weight decay `1e-4`).

## Training HSIFusionNet v2.5.3

```bash
cd "HSIFUSION&SHARP"
python hsifusion_training.py \
  --data_root ${HSI_DATA_DIR:-./data/ARAD_1K} \
  --batch_size 12 \
  --model_size base \
  --cross_attention_max_tokens 1024
```

Key CLI flags exposed by the dataclass configuration:

| Flag | Description | Default |
| --- | --- | --- |
| `--model_size {tiny,small,base,large}` | Chooses the backbone variant in `hsifusion_v252_complete.py`. | `base` |
| `--memory_mode {standard,float16,lazy}` | Controls dataloader caching and precision. | `float16` |
| `--accumulate_steps` | Gradient accumulation steps to emulate larger batches. | `1` |
| `--warmup_epochs` | Number of cosine warm-up epochs. | `5` |
| `--no_compile` | Disable `torch.compile` for the forward pass. | Compilation enabled |
| `--cross_attention_max_tokens` | Caps decoder cross-attention key/value tokens; `0` restores full attention. | `1024` |

Checkpoints and TensorBoard logs are stored under `./experiments/hsifusion_*` by default. Resume training with `--resume path/to/checkpoint.pt`.

## Training SHARP v3.2.2 Hardened

```bash
cd "HSIFUSION&SHARP"
python sharp_training_script_fixed.py \
  --data_root ${HSI_DATA_DIR:-./data/ARAD_1K} \
  --batch_size 20 \
  --model_size base \
  --sparse_sparsity_ratio 0.9 \
  --sparse_exact_topk_max_tokens 1024 \
  --sparse_landmark_tokens 256 \
  --max_global_tokens 1024
```

SHARP now defaults to a **sigmoid** output activation and the **MRAE** loss (`--loss_type mrae`, matching the MST++ objective); the old tanh output and `l1_curvature` loss (L1 + 0.1× spectral curvature) remain available for ablations. Legacy checkpoints trained before this change load automatically with the tanh fallback.

Important parameters:

- `--sparse_exact_topk_max_tokens` bounds exact all-key top-k. Longer sequences use 2D local candidates plus `--sparse_landmark_tokens` pooled global candidates, reducing attention compute to `O(N * (window + landmarks))`.
- `--sparse_block_size`, `--sparse_q_block_size`, `--sparse_max_tokens`, and `--sparse_window_size` tune the sparse attention kernels and must respect GPU memory limits.
- `--max_global_tokens` bounds dense global and decoder cross-attention context without changing checkpoint parameter shapes.
- `--key_rbf_mode linear` preserves query-dependent sparse rankings and is the training default; `mean` remains available for legacy experiments.
- `--ema_decay` together with `--ema_update_every` mirrors the production EMA scheme – keep these defaults unless you benchmark alternatives.
- Set `--memory_mode lazy --cache_size 4` to stream tiles with a bounded per-worker float16 cache.

Multi-GPU SHARP training uses native DDP:

```bash
torchrun --standalone --nproc_per_node=4 sharp_training_script_fixed.py \
  --data_root ${HSI_DATA_DIR:-./data/ARAD_1K} \
  --batch_size 5 \
  --model_size base
```

`--batch_size` is per process. Training and validation use disjoint distributed
samplers; validation metrics are reduced globally, while logs and checkpoints
are written only by rank zero.

## SHARP inference (legacy utility)

Prefer `unified_inference.py` (above) for evaluation and reconstruction. The standalone `sharp_inference.py` utility remains for patch-based tiling workflows; it loads checkpoints (with or without embedded configs) and optionally tiles large RGB inputs.

```bash
python sharp_inference.py \
  experiments/sharp/best_model.pth \
  tests/rgb/frame.png \
  --output outputs/hsis/frame.npy \
  --patch_size 256 \
  --device cuda
```

When `--patch_size` is provided the script applies overlap-and-blend tiling to avoid seams. Outputs are compatible with the [`hsi_viz_suite`](../hsi_viz_suite/README.md) plotting scripts.

## Batch jobs

Two LSF job templates (`train_job_HSI.sh`, `train_job_SHARP.sh`) demonstrate single-GPU A100 runs. Adapt queue, account, module, and environment paths before use.

## Testing

The folder carries the regression suites from the five audit passes plus the unified-infrastructure tests (~118 tests total):

```bash
cd "HSIFUSION&SHARP"
python -m pytest -q
```

`test_unified_infra.py` covers the unified trainer/inference CLIs, checkpoint auto-detection, the SHARP pad-to-multiple path on 482×512 inputs, and the metric protocols. The `test_audit*.py` and `test_runtime_audit_regressions.py` files pin the behavior of the fixes documented in `AUDIT_REPORT*.md`. The `smoke_*.py` scripts run quick CPU-only end-to-end checks.

## Interoperability tips

- The dataloaders in `optimized_dataloader.py` match the MST++ patching logic used by CSWIN and MSWR, so you can reuse cached datasets and evaluation metrics across projects.
- Run [`../hsi_viz_suite/scripts/generate_all_visualizations.py`](../hsi_viz_suite/README.md) on SHARP or HSIFusion outputs to produce publication-grade figures.
- Compare transformer and CNN baselines by exporting checkpoints to the shared `artifacts/` directory and pointing the visualization suite at the combined results.

## License

The HSIFusion and SHARP implementations are distributed under the [MIT License](LICENSE). Contributions to this folder are accepted under the same terms.

## Architecture Details

- HSIFusionNet v2.5.3 (Lightning Pro)
  - Blocks: `LightningProBlock` with sliding‑window attention (RoPE), spectral attention, optional MoE, and GELU MLP; layer‑scale and drop‑path.
  - Topology: Encoder–decoder hierarchy with GroupNorm, staged down/upsampling, optional cross‑attention fusion, optional uncertainty head.
  - Robustness: Torch compile compatibility, safe sliding window merge, dtype handling, AMP/bfloat16 support.
  - Audit pass 4 repaired three inductive biases that had silently degraded to no-ops: RoPE is now applied in standard attention (`standard_attn_rope` defaults on), spectral attention groups along the true spectral axis (`spectral_min_bands_per_group=4`), and the decoder cross-attention residual path is wired correctly.
  - Reference: `hsifusion_v252_complete.py` (`LightningProConfig`, factory `create_hsifusion_lightning_pro`).

- SHARP v3.2.2 (Hardened)
  - Attention: Multi-scale attention plus exact top-k for short sequences and bounded 2D-local + pooled-landmark top-k for moderate/high resolutions; RBF query/key projection modes (`linear` default, `mean`/`none` legacy).
  - Norm: Channel RMSNorm with eval‑time caches; the second attention sublayer is pre-normed by default (`attn2_prenorm=True`); cross‑attention fusion in the decoder.
  - Topology: Hierarchical encoder–decoder; output head applies a smooth rank-limited spectral-basis refinement (`spectral_head_rank=8`) and a **sigmoid** output activation by default (tanh retained as legacy-checkpoint fallback).
  - MoE: Switch-style routing now includes a load-balancing auxiliary loss.
  - Reference: `sharp_v322_hardened.py` (`SHARPv32Config`, factory `create_sharp_v32`).

## Training Overview

- Unified (`unified_training.py`) — canonical
  - One trainer for both models with MST++/NTIRE-faithful defaults (Adam, lr 4e-4, per-iteration cosine, batch 20, 300 epochs, annealed-floor MRAE loss, no weight decay/warmup).
  - BF16 is preferred automatically; FP16 uses a conservative initial loss scale. Persistent non-finite updates abort with a recovery checkpoint instead of being skipped forever. Validation on crop + full protocols remains exact FP32 MRAE.

- HSIFusionNet (`hsifusion_training.py`)
  - Data: `optimized_dataloader.py` (MST++ compatible) with `memory_mode` (standard/float16/lazy).
  - Optimizer: AdamW, cosine LR with warmup (`LambdaLR`).
  - Runtime: AMP (`GradScaler/auto_cast`), optional `torch.compile`, channels_last, gradient accumulation, TB logging.
  - Common flags: `--model_size`, `--batch_size`, `--accumulate_steps`, `--warmup_epochs`, `--compile_model`, `--use_channels_last`.

- SHARP (`sharp_training_script_fixed.py`)
  - Sparse config: `--sparse_block_size`, `--sparse_q_block_size`, `--sparse_max_tokens`, `--sparse_exact_topk_max_tokens`, `--sparse_landmark_tokens`, `--sparse_window_size`, `--sparse_sparsity_ratio`, `--rbf_centers_per_head`, `--key_rbf_mode`.
  - Optimizer/Runtime: AdamW, AMP, gradient clipping, EMA with configurable `ema_update_every`, optional `torch.compile` (version‑gated).
  - Distributed: Native `torchrun`/DDP with rank-aware samplers, metrics, EMA, resume, and checkpoints.

## Key Configuration

- HSIFusion (Lightning Pro)
  - Model: `in_channels`, `out_channels`, `base_channels`, `depths`, `num_heads`, `window_size`, `mlp_ratio`.
  - Features: `enable_spectral`, `use_sparse_attention`, `use_sliding_window`, `use_moe`, `num_experts`, `use_rope`, `use_channels_last`.
  - Regularization: `drop_path`, `dropout`, `auxiliary_loss_weight`, `min_input_size`.

- SHARP v3.2.2
  - Core: `in_channels`, `out_channels`, `base_dim`, `depths`, `heads`, `mlp_ratios`, `drop_path_rate`, `use_checkpoint`.
  - Sparse: `sparse_block_size`, `sparse_max_tokens`, `sparse_exact_topk_max_tokens`, `sparse_landmark_tokens`, `sparse_window_size`, `sparse_k_cap`, `sparse_q_block_size`, `sparse_sparsity_ratio`, `rbf_centers_per_head`, `key_rbf_mode`, `sparsemax_pad_value`.
  - Runtime: `compile_mode`, `ema_update_every`.

Tip: With `sparse_sparsity_ratio=0`, SHARP auto-disables `k_cap`. Attention is exact only up to `sparse_exact_topk_max_tokens`; longer sequences still use bounded local and landmark candidates.

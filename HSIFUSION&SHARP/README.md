# HSIFusion & SHARP — Transformer Baselines for HSI Reconstruction

This directory packages the two transformer-based baselines we maintain alongside the CNN models in this monorepo:

- **HSIFusionNet v2.5.3 ("Lightning Pro")** – a lightweight spectrum-aware ViT that favours fast convergence and AMP-friendly kernels.
- **SHARP v3.2.2 (Hardened)** – a sparse attention reconstruction pipeline with the audit fixes applied for production use.

Both projects share the same data preparation code and operate on ARAD-1K style hyperspectral datasets (31 channels). The scripts here mirror the ones we run internally after incorporating stability fixes, deterministic logging, and memory-usage guards.

## Directory structure

```
HSIFUSION&SHARP/
├─ hsifusion_training.py           # HSIFusionNet Lightning Pro trainer
├─ hsifusion_v252_complete.py      # Model factory (tiny/small/base/large variants)
├─ sharp_training_script_fixed.py  # SHARP v3.2.2 hardened trainer
├─ sharp_inference.py              # Offline inference / patch-based tiling utility
├─ sharp_v322_hardened.py          # Model + trainer implementations
├─ optimized_dataloader.py         # Memory-efficient MST++ dataloaders + losses
├─ common_utils_v32.py             # Shared utilities for both models
├─ dataset_setup.py                # Helper to stage ARAD-1K splits and caches
├─ train_job_HSI.sh                # Example SLURM launcher for HSIFusion
├─ train_job_SHARP.sh              # Example SLURM launcher for SHARP
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

Use the bundled `dataset_setup.py` helper to stage ARAD-1K data with MST++ style crops, statistics, and channel metadata. Example:

```bash
cd "HSIFUSION&SHARP"
python dataset_setup.py \
  --arad-root /path/to/ARAD_1K_raw \
  --output-root ./data/ARAD_1K \
  --patch-size 128 \
  --stride 8 \
  --workers 8
```

The script populates `train/` and `val/` directories with `.npy` spectral tensors and generates lookup tables consumed by both trainers. Skip this step if you already have a curated dataset prepared for MSWR/CSWIN in the repository root.

## Training HSIFusionNet v2.5.3

```bash
cd "HSIFUSION&SHARP"
python hsifusion_training.py \
  --data_root ${HSI_DATA_DIR:-./data/ARAD_1K} \
  --batch_size 12 \
  --model_size base \
  --use_amp \
  --compile_model
```

Key CLI flags exposed by the dataclass configuration:

| Flag | Description | Default |
| --- | --- | --- |
| `--model_size {tiny,small,base,large}` | Chooses the backbone variant in `hsifusion_v252_complete.py`. | `base` |
| `--memory_mode {standard,float16,lazy}` | Controls dataloader caching and precision. | `float16` |
| `--accumulate_steps` | Gradient accumulation steps to emulate larger batches. | `1` |
| `--warmup_epochs` | Number of cosine warm-up epochs. | `5` |
| `--compile_model/--no-compile_model` | Toggle `torch.compile` for the forward pass. | Enabled |

Checkpoints and TensorBoard logs are stored under `./experiments/hsifusion_*` by default. Resume training with `--resume_from path/to/checkpoint.pt`.

## Training SHARP v3.2.2 Hardened

```bash
cd "HSIFUSION&SHARP"
python sharp_training_script_fixed.py \
  --data_root ${HSI_DATA_DIR:-./data/ARAD_1K} \
  --batch_size 20 \
  --model_size base \
  --sparse_sparsity_ratio 0.9 \
  --use_amp
```

Important parameters:

- `--sparse_block_size`, `--sparse_q_block_size`, `--sparse_max_tokens`, and `--sparse_window_size` tune the streaming attention kernels and must respect GPU memory limits.
- `--ema_decay` together with `--ema_update_every` mirrors the production EMA scheme – keep these defaults unless you benchmark alternatives.
- Set `--memory_mode lazy` if you need to stream tiles from slower storage without blowing host RAM.

Distributed training works by wrapping the invocation with `torch.distributed.run` in the same fashion as the CSWIN trainer.

## SHARP inference

The standalone inference utility loads checkpoints (with or without embedded configs) and optionally tiles large RGB inputs.

```bash
python sharp_inference.py \
  --checkpoint experiments/sharp/best.ckpt \
  --input tests/rgb/frame.png \
  --output outputs/hsis/frame.npy \
  --patch-size 256 \
  --device cuda
```

When `--patch-size` is provided the script applies overlap-and-blend tiling to avoid seams. Outputs are compatible with the [`hsi_viz_suite`](../hsi_viz_suite/README.md) plotting scripts.

## Batch jobs

Two SLURM-ready job templates (`train_job_HSI.sh`, `train_job_SHARP.sh`) demonstrate how we schedule multi-GPU experiments with pre-configured environment variables. Adapt them to your cluster (account names, partitions, `srun` args) before use.

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
  - Reference: `hsifusion_v252_complete.py` (`LightningProConfig`, factory `create_hsifusion_lightning_pro`).

- SHARP v3.2.2 (Hardened)
  - Attention: Multi‑scale attention + streaming sparse attention (`sparse_attention_topk_streaming`) with top‑k and local window fallback; RBF query/key projection modes (mean/linear/none).
  - Norm: Channel RMSNorm with eval‑time caches; cross‑attention fusion in the decoder.
  - Topology: Hierarchical encoder–decoder with ChannelRMSNorm, spectral basis regularization in the head.
  - Reference: `sharp_v322_hardened.py` (`SHARPv32Config`, factory `create_sharp_v32`).

## Training Overview

- HSIFusionNet (`hsifusion_training.py`)
  - Data: `optimized_dataloader.py` (MST++ compatible) with `memory_mode` (standard/float16/lazy).
  - Optimizer: AdamW, cosine LR with warmup (`LambdaLR`).
  - Runtime: AMP (`GradScaler/auto_cast`), optional `torch.compile`, channels_last, gradient accumulation, TB logging.
  - Common flags: `--model_size`, `--batch_size`, `--accumulate_steps`, `--warmup_epochs`, `--compile_model`, `--use_channels_last`.

- SHARP (`sharp_training_script_fixed.py`)
  - Sparse config: `--sparse_block_size`, `--sparse_q_block_size`, `--sparse_max_tokens`, `--sparse_window_size`, `--sparse_sparsity_ratio`, `--rbf_centers_per_head`, `--key_rbf_mode`.
  - Optimizer/Runtime: AdamW, AMP, gradient clipping, EMA with configurable `ema_update_every`, optional `torch.compile` (version‑gated).
  - Distributed: Wrapper via `torch.distributed.run` identical to CSWIN.

## Key Configuration

- HSIFusion (Lightning Pro)
  - Model: `in_channels`, `out_channels`, `base_channels`, `depths`, `num_heads`, `window_size`, `mlp_ratio`.
  - Features: `enable_spectral`, `use_sparse_attention`, `use_sliding_window`, `use_moe`, `num_experts`, `use_rope`, `use_channels_last`.
  - Regularization: `drop_path`, `dropout`, `auxiliary_loss_weight`, `min_input_size`.

- SHARP v3.2.2
  - Core: `in_channels`, `out_channels`, `base_dim`, `depths`, `heads`, `mlp_ratios`, `drop_path_rate`, `use_checkpoint`.
  - Sparse: `sparse_block_size`, `sparse_max_tokens`, `sparse_window_size`, `sparse_k_cap`, `sparse_q_block_size`, `sparse_sparsity_ratio`, `rbf_centers_per_head`, `key_rbf_mode`, `sparsemax_pad_value`.
  - Runtime: `compile_mode`, `ema_update_every`.

Tip: With `sparse_sparsity_ratio=0`, SHARP auto‑disables `k_cap` for dense attention; windowed fallback is used for very long sequences.

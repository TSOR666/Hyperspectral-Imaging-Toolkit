# CSWIN v2 — Sinkhorn-GAN Training Pipelines

This package contains the latest training code for the noise-robust CSWin transformer that reconstructs hyperspectral imagery from RGB measurements. Two complementary entry points are provided:

- [`src/hsi_model/training_script_fixed.py`](src/hsi_model/training_script_fixed.py) — the production Sinkhorn-GAN trainer with R1 regularization and EMA logging.
- [`src/hsi_model/train_optimized.py`](src/hsi_model/train_optimized.py) — a heavily memory-optimized MST++ trainer that keeps peak GPU RAM < 30 GB on 80 GB A100 GPUs.

Both scripts share the same configuration schema (`src/configs/config.yaml`) and utilities (`src/hsi_model/utils`).

## Environment setup

1. Create a virtual environment and install the core dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install hydra-core numpy h5py psutil tqdm wandb
   ```
2. (Optional) Register the project as an editable package if you plan to extend the modules:
   ```bash
   pip install -e src
   ```

Key environment variables recognised by the scripts:

| Variable | Purpose | Default |
| --- | --- | --- |
| `HSI_DATA_DIR` | Root of the ARAD-1K (or compatible) dataset. | `./data/ARAD_1K` |
| `HSI_LOG_DIR` | Folder where logs and hydra outputs are written. | `./artifacts/logs` |
| `HSI_CKPT_DIR` | Folder for checkpoints. | `./artifacts/checkpoints` |
| `PYTORCH_CUDA_ALLOC_CONF` | Recommended CUDA allocator tweak. | `expandable_segments:True,max_split_size_mb:256` |
| `OMP_NUM_THREADS` | Number of intra-op CPU threads for dataloader workers. | `2` |

## Running training

Launch the Sinkhorn-GAN trainer with:

```bash
cd "CSWIN v2"
python src/hsi_model/training_script_fixed.py --config-name config
```

The script uses [Hydra](https://hydra.cc/) for configuration. Override parameters via CLI flags, e.g. `python ... optimizer.generator_lr=1e-4 data_dir=/datasets/arad1k`.

For the memory-optimized pipeline, run:

```bash
python src/hsi_model/train_optimized.py --config-name config
```

### Distributed training

Both drivers support multi-GPU setups via PyTorch Distributed:

```bash
python -m torch.distributed.run --nproc_per_node=4 src/hsi_model/training_script_fixed.py --config-name config
```

Hydra automatically expands log and checkpoint directories per-rank. Ensure `NCCL_P2P_DISABLE=1` on multi-node clusters if you observe NCCL timeouts.

### Checkpoints and logs

- Checkpoints land under `${HSI_CKPT_DIR}` with rolling retention controlled by `checkpoint_keep`.
- Hydra logs and the custom `MetricsLogger` go to `${HSI_LOG_DIR}`.
- Validation metrics follow the MST++ centre-crop protocol (see `DEFAULT_STRIDE`, `DEFAULT_PATCH_SIZE`, etc. in [`src/hsi_model/constants.py`](src/hsi_model/constants.py)).

## Project structure

```
src/
├─ configs/config.yaml          # Shared defaults
└─ hsi_model/
   ├─ constants.py              # Centralised hyperparameters and dataset metadata
   ├─ training_script_fixed.py  # Sinkhorn-GAN trainer (production)
   ├─ train_optimized.py        # Memory-optimized MST++ trainer
   ├─ models/                   # NoiseRobustCSWinModel & loss definitions
   ├─ utils/                    # Logging, metrics, dataloader helpers
   └─ ...                       # Additional support modules
```

## Troubleshooting

- **Padding errors on tiny feature maps** — the Sinkhorn trainer ships with symmetric reflect padding to avoid the `padding (...) at dimension 3` crash when window size exceeds feature map size.
- **Data loader stalls** — set `MST_MEMORY_MODE=lazy` and `MST_LAZY_CACHE_SIZE=3` to reduce I/O pressure, as recommended in the optimized trainer docstring.
- **Hydra output clutter** — add `HYDRA_FULL_ERROR=1` to surface full stack traces or set `hydra.output_subdir=null` (already configured) to keep logs inside `${HSI_LOG_DIR}`.

## Related tools

Pair these trainers with:

- [`../HSIFUSION&SHARP`](../HSIFUSION&SHARP/README.md) for transformer baselines (HSIFusionNet and SHARP) that share dataset utilities.
- [`../mswr_v2`](../mswr_v2/README.md) for a CNN-based baseline with SAM loss.
- [`../hsi_viz_suite`](../hsi_viz_suite/README.md) to visualise reconstructions, error maps, and spectral curves.

## License

This subproject, along with the rest of the toolkit, is distributed under the [MIT License](../LICENSE). Contributions and redistributions must comply with its terms.

## Architecture Overview

- Generator (NoiseRobustCSWinGenerator)
  - U‑Net backbone with two encoder stages, a transformer bottleneck, and symmetric decoder.
  - Dual attention per block: spectral attention and CSWin spatial attention; adaptive GroupNorm everywhere.
  - Noise‑aware gating block; optional output activations (none/sigmoid/tanh/delayed sigmoid) and safe clamping during warm‑up.
  - See `src/hsi_model/models/generator_v3.py`.

- Discriminator (Spectral Normalized Transformer Discriminator)
  - Spectral normalization on all layers; GELU activations.
  - Spectral self‑attention with temperature scaling and NaN‑safe logits; progressive downsampling.
  - Input concatenates RGB and HSI channels; outputs spatial feature maps (no global pooling).
  - See `src/hsi_model/models/discriminator_v2.py`.

## Configuration (Hydra: src/configs/config.yaml)

- Data/Runtime: `data_dir`, `log_dir`, `checkpoint_dir`, `batch_size`, `val_batch_size`, `patch_size`, `stride`, `epochs`, `iterations_per_epoch`, `num_workers`, `memory_mode`, `mixed_precision`.
- Optimizer/Scheduler: `generator_lr`, `discriminator_lr`, `warmup_steps`, `gradient_accumulation_steps`.
- Adversarial: `n_critic`, `use_r1_regularization`, `r1_gamma`.
- Sinkhorn: `sinkhorn_epsilon`, `sinkhorn_iters`, `sinkhorn_flatten_spatial`.
- Loss weights: `lambda_rec`, `lambda_perceptual`, `lambda_adversarial`, `lambda_sam`.
- Checkpointing: `checkpoint_keep`.

Example overrides:

```bash
python src/hsi_model/training_script_fixed.py \
  optimizer.generator_lr=1e-4 optimizer.discriminator_lr=5e-5 \
  data_dir=/datasets/ARAD_1K batch_size=16 memory_mode=standard
```

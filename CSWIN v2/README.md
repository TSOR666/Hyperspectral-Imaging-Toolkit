# CSWIN v2

CSWIN v2 reconstructs a 31-band hyperspectral cube from an RGB image. The
active model is a hierarchical U-Net/Transformer hybrid with spectral
self-attention, bounded local/global spatial attention, and learned
PixelShuffle sampling.

## Active Entry Point

Use the generator-only trainer:

```bash
python src/hsi_model/train_generator.py \
  --config-name config \
  data_dir=/path/to/ARAD_1K
```

The older Sinkhorn-GAN trainers remain available for legacy experiments:

```bash
python src/hsi_model/training_script_fixed.py --config-name config
python src/hsi_model/train_optimized.py --config-name config
```

Their discriminator, Sinkhorn, R1, and gradient-accumulation settings are not
used by `train_generator.py`.

## Environment

Python 3.10-3.12 is supported.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Useful environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `HSI_DATA_DIR` | ARAD-1K/MST++ dataset root | `./data/ARAD_1K` |
| `HSI_LOG_DIR` | Training logs | `./artifacts/logs` |
| `HSI_CKPT_DIR` | Checkpoints | `./artifacts/checkpoints` |

The default `.venv` in this checkout may be incomplete. Recreate it if imports
or `uv run` fail before training starts.

## Active Recipe

The defaults in `src/configs/config.yaml` use:

- RGB input `(B, 3, H, W)` and HSI output `(B, 31, H, W)`.
- Adam with learning rate `4e-4` and a 300k-step cosine decay.
- Annealed pure-MRAE loss (`objective=mrae_annealed`, no L1 term): the
  denominator floor starts stable, then decays to exact MST++/leaderboard MRAE.
- BF16 on Ampere-or-newer CUDA devices, FP16 on older Tensor Core GPUs.
- EMA weights for validation and best-checkpoint export.
- Local 7x7 spatial attention at high resolution and bounded global attention
  at low resolution.
- Deployment-matched 128x128 tiled validation with FP32 overlap blending and
  the fixed centered 226x256 ARAD-1K scoring window.
- Explicit exclusion of the known-corrupt `ARAD_1K_0314` scene, while other
  missing or corrupt split entries fail dataset initialization.
- `[0,1]` validation clamping to match NTIRE inference/export, with unclamped
  `raw_mrae` and `out_of_range_fraction` logged for diagnosis.

Example overrides:

```bash
python src/hsi_model/train_generator.py \
  --config-name config \
  data_dir=/datasets/ARAD_1K \
  batch_size=16 \
  generator_lr=1e-4 \
  objective=mrae_annealed \
  memory_mode=standard
```

Start this objective from a fresh checkpoint. Use `objective=mrae
mrae_epsilon=1e-8` only to reproduce exact MST++ loss behavior from step 0. For
final training, enable and tune `progressive_stages` in the config for the
128 -> 256 -> 512 patch schedule.

## Controlled Ablations

The audit findings that can change reconstruction quality remain opt-in:

```bash
# Annealed pure-MRAE objective
python src/hsi_model/train_generator.py --config-name ablation_stable_mrae

# Pre-compressed decoder1 and two-block full-resolution decoder
python src/hsi_model/train_generator.py --config-name ablation_decoder_lite

# Combined annealed-MRAE and decoder experiment
python src/hsi_model/train_generator.py --config-name ablation_stable_lite
```

These configurations use separate log/checkpoint directories and should start
from random initialization. The active config now uses the annealed-MRAE loss;
the lite ablations additionally change decoder capacity.

## GPU Preflight

The preflight gate defaults to the active generator trainer:

```bash
python gpu_preflight_train.py -- \
  --config-name config \
  data_dir=/datasets/ARAD_1K
```

It checks CUDA visibility, free memory, data paths, model allocation, finite
forward and training steps, AMP, and metrics before launching training.

Legacy trainer selection remains explicit:

```bash
python gpu_preflight_train.py --trainer sinkhorn -- --config-name config
python gpu_preflight_train.py --trainer optimized -- --config-name config
```

## Distributed Training

```bash
python -m torch.distributed.run --nproc_per_node=4 \
  src/hsi_model/train_generator.py \
  --config-name config \
  data_dir=/datasets/ARAD_1K
```

## Inference

Load generator-only or legacy checkpoints through
`hsi_model.utils.inference.load_generator`. For full NTIRE/ARAD evaluation:

```bash
python cswin_test_ntire.py \
  --model_path /path/to/best_model.pth \
  --data_root /path/to/ARAD_1K \
  --output_dir ./cswin_test_results
```

Patch inference uses overlap blending and inference mode. Add
`--ensemble_mode d4` for the eight-way geometric self-ensemble.
`--amp_dtype auto` selects BF16 on Ampere-or-newer GPUs and FP16 on older
Tensor Core GPUs; use `--amp_dtype fp32` for full-precision inference. Tile
outputs are streamed directly into the FP32 overlap accumulator, so retained
tile memory is bounded by the configured patch batch size.

## Verification

```bash
.\.venv-audit\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv-audit\Scripts\python.exe smoke_run.py
.\.venv-audit\Scripts\python.exe smoke_infer.py
```

## Memory Guidance

- Use `memory_mode=standard` for maximum loader throughput.
- Use `memory_mode=float16` to reduce resident scene memory.
- Use `memory_mode=lazy` and tune `lazy_cache_size` when host RAM is limited.
- Reduce stage batch sizes before changing architecture width.

## Project Map

```text
src/configs/config.yaml
src/hsi_model/train_generator.py
src/hsi_model/models/generator_v3.py
src/hsi_model/models/attention.py
src/hsi_model/models/losses_consolidated.py
src/hsi_model/utils/data/
src/hsi_model/utils/inference.py
src/hsi_model/utils/patch_inference.py
```

See `MODEL_OPTIMIZATION_REPORT.md` for the bottleneck audit and benchmark
history.

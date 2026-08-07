# CSWIN v2

CSWIN v2 reconstructs a 31-band hyperspectral cube from an RGB image. The
active model is a generator-only hierarchical U-Net/Transformer hybrid
(~11.4M trainable parameters at `base_channels: 48`) with spectral
self-attention, learned PixelShuffle sampling, and validated local/global
spatial attention.

## Active Entry Point

Use the generator-only trainer:

```bash
python src/hsi_model/train_generator.py \
  --config-name config \
  data_dir=/path/to/ARAD_1K
```

For a fresh recovery run with all validated architecture values pinned, use:

```bash
python src/hsi_model/train_generator.py \
  --config-name sota_cascade \
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

The defaults in `src/configs/config.yaml` retain the checkpoint-compatible
local/global baseline. The fresh-run recipe in `src/configs/sota_cascade.yaml`
pins the same attention architecture, balances the residual-heavy transformer
blocks, restores a normally initialized output projection, enables a shallow
RGB-to-HSI prior, and writes to isolated recovery directories. The
historical filename is retained for command compatibility; it no longer
enables the failed three-pass cascade bundle. The shared training protocol uses:

- RGB input `(B, 3, H, W)` and HSI output `(B, 31, H, W)`.
- Adam with learning rate `4e-4` and the benchmark-aligned MST++ 300k-step
  cosine decay. A shorter June diagnostic reached MRAE 0.2737 / PSNR 29.56,
  but it is architecture evidence rather than a replacement training horizon.
- Stabilized pure-MRAE loss (`objective=mrae_annealed`, no L1 term): the
  denominator floor decays from `1e-2` to `1e-3` during the first 50k
  updates. Exact leaderboard MRAE (`1e-8`) is still used for
  validation/checkpoint selection; move to an exact-objective fine-tune only
  after the stable-floor phase converges.
- Gradient norms and the configured clipping threshold are logged. The fresh
  recipe and checkpoint fine-tunes retain the historical `1.0` clip used by
  the trainer's successful control run.
- BF16 on Ampere-or-newer CUDA devices, FP16 on older Tensor Core GPUs.
- EMA weights for validation and best-checkpoint export.
- A `0.1` outer SSTB residual scale keeps fresh activations bounded while the
  normally initialized output projection preserves gradients into the body.
- A first-update guard that aborts if training loss exceeds `10`, catching a
  pathological initialization before it consumes a long accelerator run.
- A named training contract verifies the full 300k stage, stable objective,
  disabled early stopping, and critical architecture values before dataset
  loading. Every run also writes the resolved Hydra configuration and a
  fingerprint to `resolved_config.yaml` in its log directory.
- No post-convolution or S-MSA-output GroupNorm and no clean-input denoising
  residual in `sota_cascade`; all three are legacy checkpoint-compatibility
  paths that limit brightness reconstruction in a fresh model.
- Deployment-matched local/global spatial attention in both recommended
  configurations.
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

Start this objective from a fresh checkpoint. To salvage the July 27
`sota_radiometric_fresh` run, use `finetune_radiometric_exact_mrae` from its
115k `best_model.pth`; that recipe preserves the old graph and restarts with a
low-LR exact-objective schedule.

## Controlled Ablations

The audit findings that can change reconstruction quality remain opt-in:

```bash
# Annealed pure-MRAE objective
python src/hsi_model/train_generator.py --config-name ablation_stable_mrae

# Pre-compressed decoder1 and two-block full-resolution decoder
python src/hsi_model/train_generator.py --config-name ablation_decoder_lite

# Combined annealed-MRAE and decoder experiment
python src/hsi_model/train_generator.py --config-name ablation_stable_lite

# Fresh residual-balanced recovery with the full MST++ 300k cosine horizon
python src/hsi_model/train_generator.py --config-name sota_cascade

# Isolate the three CSWin recovery levers one at a time
python src/hsi_model/train_generator.py --config-name ablation_axial      # axial attention only
python src/hsi_model/train_generator.py --config-name ablation_smsa_off  # smsa_output_norm=false only
python src/hsi_model/train_generator.py --config-name ablation_cascade   # cascade_stages only

# Start a new low-LR phase from a saturated 128x128 checkpoint
python src/hsi_model/train_generator.py \
  --config-name finetune_128_polish_annealed \
  finetune_checkpoint=/path/to/best_model.pth

# Recommended for the 2026-07-25 stable-init run: load its 164k best checkpoint
# (MRAE 0.5233), then anneal the training floor from 1e-3 to exact MRAE.
python src/hsi_model/train_generator.py \
  --config-name finetune_128_exact_mrae \
  finetune_checkpoint=/path/to/sota_recovery_stable_init/best_model.pth

# Recommended for the 2026-07-27 radiometric run: use its 115k best checkpoint,
# not the degraded latest checkpoint (MRAE 0.5771 at 215k).
python src/hsi_model/train_generator.py \
  --config-name finetune_radiometric_exact_mrae \
  finetune_checkpoint=/path/to/sota_radiometric_fresh/best_model.pth

# Experimental: 256/512 fine-tuning. Validate with a patch-size sweep first;
# the 2026-06-25 run worsened 128-tile validation MRAE after the 256 switch.
python src/hsi_model/train_generator.py \
  --config-name finetune_progressive_annealed \
  finetune_checkpoint=/path/to/best_model.pth
```

> Two July runs must not be resumed: the 2026-07-21 true-CSWin/cascade-3 bundle
> plateaued near 0.63 MRAE, and the 2026-07-23 local/global run started at
> train MRAE 93.5 before plateauing near 0.66 MRAE / 19.1 dB through 37k
> iterations. The latter exposed the oversized random output projection, not
> frozen parameters. The active recipe controls that amplification at the SSTB
> residuals, preserves full output-head gradients, adds a direct RGB spectral
> prior, and uses new output directories. Keep alternative attention modes and
> multi-pass cascades as controlled one-lever ablations.
>
> The 2026-07-25 stable-initialization run is different: it learned normally
> (0.769 MRAE at 1k -> 0.523 at 164k), then overfit while the stabilized
> training objective continued to fall. Stop that run and keep its 164k
> `best_model.pth`; use `finetune_128_exact_mrae` for the next phase.
>
> The 2026-07-27 radiometric run reached its best MRAE 0.524619 at 115k, then
> regressed to 0.577060 at 215k while PSNR continued rising. Stop it, retain the
> 115k `best_model.pth`, and use `finetune_radiometric_exact_mrae` if salvaging
> that checkpoint. The fresh `sota_cascade` recipe addresses the underlying
> body-gradient starvation while preserving the benchmark training horizon.
>
> The 2026-08-03 residual-balanced run in `train_generator 17.log` correctly
> resolved the 300k MST++ horizon; LR `3.58e-4` at 64k was therefore expected.
> Its actual recipe mismatches were annealing the training denominator to
> `1e-8` at 50k and enabling early stopping after epoch 50. It plateaued at
> MRAE 0.573183 and stopped at 64k, only 21% through the intended optimization.
> The active recipe holds a `1e-3` floor, disables early stopping, and rejects
> a shortened horizon or either unstable override before training.

### Attention-mode and recovery levers

| Key | Values | Default | Notes |
| --- | --- | --- | --- |
| `cswin_attention_mode` | `local_global`, `cswin`, `axial` | `local_global` | `cswin` and `axial` remain diagnostic modes; neither is recommended for the production run. |
| `cascade_stages` | int | `1` | Values above one repeat the generator and materially increase activation memory; validate them as ablations. |
| `smsa_output_norm` | bool | `true` (`false` in `sota_cascade`) | Legacy checkpoint path; disabling it preserves low-frequency S-MSA corrections in a fresh model. |
| `sstb_outer_residual_scale` | float | `1.0` (`0.1` in `sota_cascade`) | Scales the gated SST branch before the outer residual. Unit scale preserves historical checkpoints; 0.1 stabilizes fresh deep stacks. |
| `output_head_init_scale` | float or null | `0.01` (`1.0` in `sota_cascade`) | Multiplies the fresh final projection initialization; checkpoint weights overwrite it on load. |
| `use_spectral_input_skip` | bool | `false` (`true` in `sota_cascade`) | Adds a shallow radiometric RGB-to-HSI path; changing it requires a fresh checkpoint. |
| `use_feature_norm` | bool | `true` (`false` in `sota_cascade`) | Legacy GroupNorm after embedding/sampling convolutions; disable only for a fresh radiometric run. |
| `use_input_denoising` | bool | `true` (`false` in `sota_cascade`) | Legacy learned RGB perturbation; disable for clean ARAD input. |

When a key is absent from the base `config.yaml`, set it in an experiment YAML
or append it on the CLI with Hydra's `+key=value` form.

These configurations use separate log/checkpoint directories and should start
from random initialization except the fine-tune recipes. Use
`finetune_checkpoint` to load model weights into a fresh optimizer/stage
schedule. Reserve `resume_checkpoint` for continuing an interrupted run made
with the same config; the two options are mutually exclusive. The lite
ablations additionally change decoder capacity.

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

Inference also accepts a bare `torch.save(model.state_dict(), path)` file. For
an older known architecture, provide its matching JSON/YAML config explicitly:

```bash
python cswin_test_ntire.py \
  --weights_path /path/to/legacy_generator_weights.pth \
  --architecture_config /path/to/legacy_architecture.yaml \
  --data_root /path/to/ARAD_1K
```

To remove optimizer, discriminator, and training-only data from a checkpoint,
convert it to a standalone generator artifact first. The default converted
file embeds the architecture config and can be passed directly to the tester:

```bash
python convert_cswin_checkpoint.py \
  --checkpoint /path/to/latest_checkpoint.pth \
  --output /path/to/cswin_generator_weights.pth

python cswin_test_ntire.py \
  --weights_path /path/to/cswin_generator_weights.pth \
  --data_root /path/to/ARAD_1K
```

If the source checkpoint has no config, add
`--architecture_config /path/to/known_architecture.yaml` to the conversion
command. Use `--raw_state_dict` only when a pure tensor mapping is required;
that output must be loaded with `--architecture_config`.

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

See the config headers in `src/configs/*.yaml` and the probe scripts under
`probes/` for the bottleneck-audit and benchmark history.

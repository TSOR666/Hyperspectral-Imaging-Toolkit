# 1. Inferred Task and Model Family

CSWIN v2 performs RGB-to-hyperspectral reconstruction:

- Input: `(B, 3, H, W)` RGB.
- Output: `(B, 31, H, W)` spectral reflectance.
- Active trainer: `src/hsi_model/train_generator.py`.
- Model: hierarchical U-Net/Transformer hybrid.
- Main quality metrics: MRAE, PSNR, SSIM, SAM, RMSE.
- Target hardware: CUDA training/inference; patch inference supports limited VRAM.

The generator combines CBAM channel gating, full-resolution spectral MSA,
GDFN/SGFN blocks, bounded local/global spatial attention, skip connections,
and PixelUnshuffle/PixelShuffle sampling. The configured objective is stabilized
pure MRAE, with exact MRAE retained as the primary validation metric.

# 2. Critical Paths & Profiling Plan

Training:

`MST/HF dataset -> paired crop/augmentation -> DataLoader -> generator ->
MRAE loss -> AMP backward -> Adam -> cosine schedule -> EMA ->
center-crop validation -> atomic checkpoint`

Inference:

`checkpoint -> strict generator load -> patch extraction -> batched inference ->
FP32 overlap blending -> optional D4 ensemble -> clamp/export/metrics`

Audit execution:

- Inspected model, attention, losses, datasets, training, checkpoint, metrics,
  patch inference, NTIRE testing, and configuration.
- Ran the complete test suite before and after changes.
- Ran CPU train and inference smoke tests.
- Measured parameters, forward latency, attention latency, and validation
  metric overhead.
- Added synthetic probes for metric-domain mismatch, uneven validation batches,
  distributed evaluation sharding, and generator preflight training.

# 3. Bottleneck Summary

| severity | category | file:line | bottleneck | impact | proposed fix |
|---|---|---|---|---|---|
| HIGH | Quality/Inference | `train_generator.py:302`, `cswin_test_ntire.py:436` | Training selected checkpoints on raw linear outputs while NTIRE scoring clamped to `[0,1]` | Best checkpoint could differ from best deployed checkpoint | Fixed: clamp validation metrics; log raw MRAE and out-of-range fraction |
| HIGH | Training | `train_generator.py:259` | Standard `DistributedSampler` padded validation shards with duplicate samples | Biased DDP checkpoint selection when validation size was not divisible by world size | Fixed: non-padding `DistributedEvalSampler` |
| HIGH | Training | `train_generator.py:301-353` | Validation averaged batch means, not sample sums | Partial final batches were overweighted | Fixed: sample-weighted local and distributed reductions |
| HIGH | Training/Deployment | `gpu_preflight_train.py:51-52` | Preflight defaulted to a legacy Sinkhorn trainer and could not launch the active trainer | Documented production path lacked a matching GPU gate | Fixed: generator mode is the default with generator loss/AMP checks |
| BLOCKER | Inference | `utils/inference.py:90-91` | Partial checkpoint loading previously occurred under strict mode | Random parameters could reach evaluation | Verified fixed: strict loading raises |
| HIGH | Training | `train_generator.py:320-334` | Validation failures previously could become valid-looking zero metrics | Corrupt checkpoint selection | Verified fixed: fail-fast, distributed-aware rejection |
| HIGH | Architecture/Memory | `models/attention.py:477`, `:550` | Legacy axial attention scaled quadratically along full image axes | High activation memory and latency | Verified fixed: bounded local windows and low-resolution global attention |
| HIGH | Data/Memory | `utils/data/mst_dataset.py:246-329` | Resident-only scene loading duplicated large arrays across workers | Host-RAM pressure and slow startup | Verified fixed: standard, float16, and lazy modes |
| RESOLVED | Training | `config.yaml`, `train_generator.py:878` | The active recipe previously used AdamW decay unlike MST++ | Could over-regularize sensitive scale parameters | Switched the active MRAE recipe to Adam with zero weight decay |
| MEDIUM | Speed/Maintenance | `train_generator.py:297`, `utils/patch_inference.py:246` | Deprecated `torch.cuda.amp` API remains | Warnings and future compatibility risk | Migrate through a version-compatible AMP helper |
| BLOCKER | Environment | `.venv` | Default environment contains incomplete Torch/NumPy package metadata | `uv run` cannot repair or launch reliably | Recreate `.venv`; `.venv-audit` is healthy |

# 4. Detailed Findings

## 4.1 Validation now matches deployed output semantics

Evidence:

- The NTIRE test path clamps predictions before metrics and export.
- The active training validator previously passed raw linear outputs directly
  to center-crop metrics.
- Synthetic probe with 8.3% out-of-range values:
  raw MRAE `0.6533`, deployed/clamped MRAE `0.5025`.

Fix:

- `validation_clamp_output: true`.
- `validation_report_raw_mrae: true`.
- Validation reports `mrae` in the deployed domain plus `raw_mrae` and
  `out_of_range_fraction`.

Tradeoff:

- Historical raw-validation MRAE is no longer the checkpoint-selection metric.
  It remains logged for comparison.

## 4.2 Distributed validation no longer duplicates or misweights samples

Evidence:

- PyTorch `DistributedSampler(..., drop_last=False)` pads to equal shard sizes.
- The previous reducer summed batch means and divided by batch count.
- For sample losses `[1, 1, 3]` with batch size 2, the old result was `2.0`;
  the correct sample mean is `5/3`.

Fix:

- `DistributedEvalSampler` shards `rank, rank + world_size, ...` without
  padding.
- Losses and metrics are multiplied by batch size, all-reduced as sums, and
  divided by total samples.

Impact:

- Exact validation-set coverage and unbiased checkpoint selection.

## 4.3 Active GPU preflight and instructions were inconsistent

Evidence:

- Configuration and active development target `train_generator.py`.
- README, Quick Start, and preflight defaulted to legacy Sinkhorn-GAN training.

Fix:

- Generator-only preflight is now the default.
- It runs the active pure-MRAE criterion and Adam step.
- Legacy `sinkhorn` and `optimized` modes remain explicit.
- README and Quick Start now document the active path and actual defaults.

## 4.4 Architecture and memory behavior

Verified current implementation:

- Spectral MSA uses channel-sized affinity, linear in spatial token count.
- Spatial attention uses local 2-D windows at high resolution and global
  attention only below `cswin_global_tokens`.
- Activation checkpointing is token-thresholded.
- Lazy MST loading uses bounded caches and file-backed HSI access.
- Patch stitching accumulates in FP32 and does not flush the CUDA allocator per
  tile batch.

Remaining architecture risk:

- High-resolution encoder/decoder blocks still dominate compute.
- The 31-band output is modeled directly rather than through a low-rank
  spectral basis.

## 4.5 Optimizer

The active pure-MRAE recipe uses Adam with zero weight decay, matching the
official MST++ training optimizer and avoiding decay on biases, normalization
scales, and learned attention temperatures.

Why not enabled by default in this audit:

- Existing checkpoints store a one-group optimizer state. Changing group layout
  would break exact resume unless optimizer-state migration is implemented and
  tested. This should be a controlled training ablation, not a silent default.

# 5. Patches Implemented

This audit added:

- Deployment-aligned validation clamping with raw diagnostics.
- Non-padding distributed evaluation sampler.
- Sample-weighted validation reductions.
- Generator-only GPU preflight mode and default.
- Correct active-trainer README and Quick Start instructions.
- Explicit Hydra compatibility version to remove startup warnings.
- Four focused regression tests.

Files changed:

- `README.md`
- `QUICK_START.md`
- `gpu_preflight_train.py`
- `src/configs/config.yaml`
- `src/hsi_model/train_generator.py`
- `src/hsi_model/train_optimized.py`
- `src/hsi_model/training_script_fixed.py`
- `src/hsi_model/utils/data/__init__.py`
- `src/hsi_model/utils/data/loaders.py`
- `src/hsi_model/utils/data/transforms.py`
- `tests/test_datasets.py`
- `tests/test_gpu_preflight_train.py`
- `tests/test_metrics.py`
- `tests/test_runtime_optimization.py`

# 6. Tests Added + How to Run

Added:

- Deployment-clamped versus raw center-crop MRAE test.
- Generator-only preflight optimizer-step test.
- Distributed evaluation no-duplication test.
- Uneven validation batch sample-weighting test.

Run:

```powershell
.\.venv-audit\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv-audit\Scripts\python.exe smoke_run.py
.\.venv-audit\Scripts\python.exe smoke_train.py
.\.venv-audit\Scripts\python.exe smoke_infer.py
```

Result:

- `136 passed, 1 skipped`.
- Train, combined, and inference smoke tests passed.

# 7. Benchmark Results

Audit hardware: CPU only, Torch `2.11.0+cpu`, up to 8 CPU threads.

| measurement | before | after/current |
|---|---:|---:|
| Full tests | 132 passed, 1 skipped | 136 passed, 1 skipped |
| Validation domain | raw linear output | deployed `[0,1]` plus raw diagnostics |
| Synthetic MRAE | 0.6533 raw | 0.5025 deployed; raw retained as 0.6533 |
| Uneven batch loss `[1,1,3]` | 2.0000 | 1.6667 |
| DDP validation padding | duplicate indices possible | every index exactly once |
| Metric latency | 62.1 ms raw | 60.8 ms clamp + raw diagnostic |
| Generator preflight default | legacy Sinkhorn | active generator-only |

Current model measurements:

- Parameters: `7,021,690`.
- 64x64 generator forward median: `233.3 ms`.
- 128x128 local/global attention block median: `20.18 ms`.
- `smoke_run.py` train section: `0.596 s`.
- `smoke_infer.py` maximum reconstruction difference: `0`.

CUDA peak memory, GPU throughput, BF16/FP16 speed, and compile performance
could not be measured because no CUDA device was available.

# 8. Optimization Roadmap

Immediate low risk:

1. Recreate the broken default `.venv`.
2. Replace deprecated AMP calls with a compatibility helper.
3. Add validation data-loader throughput and out-of-range trend logging.
4. Keep `raw_mrae` and `out_of_range_fraction` in experiment dashboards.

Medium risk:

1. Implement optimizer-state migration, then ablate no-decay parameter groups.
2. Benchmark `torch.compile` with fixed progressive-stage shapes on CUDA.
3. Cache static attention indices by shape/device where profiling confirms
   allocation overhead.

High risk/high reward:

1. Add a learned low-rank spectral basis and reconstruct 31 bands from compact
   coefficients.
2. Distill into a lighter CNN/Restormer student for deployment.
3. Retire legacy Sinkhorn training unless it shows reproducible quality gains
   over generator-only training.

# 9. Final Verdict

**FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED**

The active generator is mathematically appropriate for spectral reconstruction,
compact for its model family, numerically guarded, and strongly tested.
Checkpoint loading, validation failure handling, patch inference, data loading,
and attention scaling are in good shape.

This audit removed two checkpoint-selection biases and aligned training,
preflight, documentation, and deployed metric semantics. Remaining risk is
primarily empirical GPU performance, optimizer-group quality ablation, and the
broken default environment rather than a known model-correctness blocker.

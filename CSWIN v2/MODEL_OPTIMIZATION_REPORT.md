# 1. Inferred Task and Model Family

The repository performs RGB-to-hyperspectral reconstruction: input tensors are
`(B, 3, H, W)` RGB images and outputs are `(B, 31, H, W)` spectral cubes.

The active configuration uses `train_generator.py`, not the older GAN trainers.
Its generator is a 7.02M-trainable-parameter hierarchical U-Net/Transformer
hybrid with:

- CBAM-style channel gates.
- Full-resolution feature-channel spectral MSA.
- Bounded 2-D local windows with an offset grid and low-resolution global attention.
- GDFN/SGFN feed-forward blocks.
- PixelUnshuffle/PixelShuffle sampling.
- A linear 31-band output head.

The intended training target is GPU, with bf16 selected on Ampere-or-newer
devices. The configured quality objective is L1 plus a stabilized MRAE term;
validation ranks checkpoints by center-crop MRAE.

# 2. Critical Paths & Profiling Plan

Critical call graph:

`MST/HF dataset -> paired crop/augmentation -> DataLoader -> mst_to_gan_batch
-> NoiseRobustCSWinGenerator -> L1PlusMRAELoss -> AMP backward -> AdamW
-> cosine scheduler -> EMA -> center-crop metrics -> atomic checkpoint`

Inference path:

`checkpoint -> load_generator -> direct or overlapping patch inference
-> weighted stitching -> optional D4 self-ensemble -> metrics/export`

The audit ran:

- Full pytest baseline and post-patch suite.
- CPU train and inference smoke runs.
- Windows spawn-worker probe.
- Incomplete-checkpoint probe.
- All-failed validation probe.
- Tiled-inference cache-flush probe.
- Parameter and optimizer-state accounting.
- Stage-level forward timing.
- Attention shape and scaling probes.
- Legacy Sinkhorn microbenchmarks.

# 3. Bottleneck Summary

| severity | category | file:line | bottleneck | impact | proposed fix |
|---|---|---|---|---|---|
| BLOCKER | Inference | `utils/inference.py:90` | `strict=True` silently fell back to partial loading | Incomplete checkpoints could run with random parameters | Fixed: strict loading now raises |
| HIGH | Training | `train_generator.py:280-326` | Validation swallowed batch failures and could return `gen_loss=0` | Corrupt checkpoint selection and false early-stopping signals | Fixed: distributed-safe fail-fast validation |
| HIGH | Data | `utils/data/loaders.py:118` | Lambdas/nested worker callbacks were not spawn-picklable | `num_workers>0` failed on Windows/macOS spawn | Fixed: module-level `partial` callback |
| HIGH | Training | `train_optimized.py:399` | DDP sampler seed differed by rank | Rank shards could overlap instead of partitioning one permutation | Fixed: shared sampler seed |
| MEDIUM | Inference | `utils/patch_inference.py:241` | `torch.cuda.empty_cache()` ran after every tile batch | Allocator churn and synchronization in the hot path | Fixed: removed; inference mode enabled |
| HIGH | Architecture | `models/attention.py` | Legacy `split_size` rows/columns were folded into batch, not stripe tokens | Axial cost stayed quadratic along the full image axis | Fixed: scalable 2-D local/global mode; legacy axial mode retained for checkpoints |
| HIGH | Memory/Data | `utils/data/mst_dataset.py` | `memory_mode` was stored but ignored; all scenes loaded into RAM | Large host-RAM residency, duplicated worker address spaces, slow startup | Fixed: real float16 and file-backed lazy modes with bounded per-worker caches |
| MEDIUM | Quality | `train_generator.py:861` | AdamW weight decay applies to norms, biases, and attention temperatures | Can over-regularize sensitive scale parameters | Add no-decay parameter groups |
| MEDIUM | Quality/Inference | `config.yaml:150`, `train_generator.py:296` | Linear outputs are scored without optional physical-range projection | Checkpoint ranking may differ from deployed `[0,1]` reconstruction metrics | Add an explicit validation-clamp protocol and report both |
| MEDIUM | Speed | `models/losses_consolidated.py:289-390` | Legacy GAN computes three iterative OT solves per sample | Sinkhorn dominated CPU loss time and scales poorly | Keep generator-only path; otherwise vectorize/use GeomLoss or lower point/iteration caps |
| HIGH | Environment | `.venv/Lib/site-packages` | Default `.venv` contains incomplete namespace-only Torch/NumPy installs | Tests fail during import | Rebuild `.venv`; `.venv-audit` is currently healthy |

# 4. Detailed Findings

## 4.1 Scalable local/global attention implemented

The active `local_global` mode now applies true 2-D `split_size x split_size`
window attention. Its second branch uses an offset, non-wrapping window grid to
mix across local boundaries. Feature maps at or below
`cswin_global_tokens` use full spatial attention, preserving global context only
where its quadratic cost is bounded.

The old axial implementation remains the default when the new config key is
absent, so checkpoints saved before this change rebuild with their original
parameter layout. New training uses `local_global`.

On CPU, a 128x128 attention block improved from 48.28 ms to 33.07 ms
(31.5%). A 64x64 full generator forward improved from 298.51 ms to 292.43 ms.
The new generator has 7,021,690 parameters versus 7,218,202 in axial mode
because long-axis bias tables are no longer allocated.

## 4.2 Real lazy/cache-backed MST loading implemented

`memory_mode` now accepts only `standard`, `float16`, or `lazy`. Float16 mode
stores resident scenes at half precision and promotes returned samples to
float32. Lazy mode indexes scene paths and shapes, reads only requested HSI
hyperslabs, and maintains bounded per-worker RGB and open-HDF5 LRU caches.
Open handles are stripped during pickling for spawn-worker safety.

On eight synthetic 256x256 scenes, initialization fell from 0.413 s to 0.038 s
and resident NumPy storage fell from 68 MiB to zero. Thirty-two random patches
took 0.370 s in lazy mode versus 0.047 s resident. Therefore `standard` remains
the throughput-oriented default; `lazy` is an explicit memory-first option.

## 4.3 Validation previously converted failures into valid-looking output

Evidence: before the patch, two systematic model exceptions returned
`{"gen_loss": 0.0}`. The exception handler continued, and the denominator was
forced to at least one.

Fix: validation records the failure, synchronizes a failure flag across ranks,
and raises before metric reduction/checkpoint selection.

Impact: correctness and observability improve; a corrupt sample now stops a run
instead of silently changing model-selection behavior.

## 4.4 Checkpoint loading was not strict

Evidence: an intentionally incomplete state dict loaded successfully with
`strict=True` because the code retried `strict=False`.

Fix: strict mode now preserves PyTorch semantics. Non-strict mode remains an
explicit opt-in, logs missing/unexpected keys, and rejects zero-key matches.

Impact: prevents evaluation or deployment of partially random models.

## 4.5 Worker spawning and DDP sharding

Evidence: pickling the old active-trainer callback raised
`AttributeError: Can't pickle local object ... <lambda>`. A real Windows spawn
DataLoader now returns a batch successfully.

The optimized legacy trainer also seeded `DistributedSampler` with a
rank-specific seed. All ranks must use one shared permutation seed and let the
sampler select rank-strided shards.

Fix: all trainer callbacks now use a spawn-safe `functools.partial`; the legacy
optimized sampler now uses the common seed while augmentation workers retain
rank-specific seeds.

## 4.6 Output and objective alignment

The active loss uses L1 plus MRAE with denominator floor `1e-2`, while reported
MRAE uses `1e-8`. This is a deliberate stability/metric tradeoff, but training
loss and leaderboard metric are not identical.

The linear output head is not projected to `[0,1]` for validation. If ARAD
targets are guaranteed in `[0,1]`, clipping weakly improves per-element L1,
MRAE, MSE, and PSNR, but may change SAM/SSIM and benchmark comparability.
Therefore the protocol should be explicit and both raw/clamped metrics should
be recorded before changing best-checkpoint selection.

## 4.7 Optimizer grouping

All generator parameters are passed to one AdamW group. Biases, GroupNorm and
LayerNorm scales, and learned attention temperatures receive `weight_decay=0.01`.

Minimal fix: no decay for 1-D parameters, biases, normalization parameters, and
temperature scalars.

Tradeoff: this is likely better conditioned, but quality benefit requires an
ablation because existing checkpoints were trained with uniform decay.

# 5. Patches Implemented

Changed files:

- `src/hsi_model/utils/data/loaders.py`
- `src/hsi_model/utils/data/__init__.py`
- `src/hsi_model/utils/data/mst_dataset.py`
- `src/hsi_model/models/attention.py`
- `src/configs/config.yaml`
- `src/hsi_model/train_generator.py`
- `src/hsi_model/training_script_fixed.py`
- `src/hsi_model/train_optimized.py`
- `src/hsi_model/utils/inference.py`
- `src/hsi_model/utils/patch_inference.py`
- `tests/test_attention.py`
- `tests/test_datasets.py`
- `tests/test_runtime_optimization.py`

Implemented changes:

- Spawn-safe DataLoader worker initializer factory.
- Fail-fast, distributed-aware validation error propagation.
- Honest strict checkpoint loading.
- Shared DDP sampler seed in the optimized legacy trainer.
- `torch.inference_mode()` in validation and patch inference.
- Removal of per-tile-batch CUDA cache flushing.
- Scalable 2-D local/global attention with legacy axial compatibility.
- Functional resident-float16 and lazy/cache-backed MST dataset modes.
- Eight focused regression tests.

# 6. Tests Added + How to Run

Added:

- Worker callback pickle test.
- All-failed validation rejection test.
- Strict incomplete-checkpoint rejection test.
- No per-batch CUDA cache flush test.
- Bounded local-window and low-resolution-global attention tests.
- Lazy initialization, exact patch equivalence, cache bound, and pickle tests.
- Float16 resident-storage and float32 sample-output test.

Run:

```powershell
.\.venv-audit\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv-audit\Scripts\python.exe smoke_run.py
.\.venv-audit\Scripts\python.exe smoke_infer.py
```

Result: `132 passed, 1 skipped`. Both smoke runs pass. A separate real
Windows-spawn DataLoader probe also passed.

# 7. Benchmark Results

Hardware available for this audit: CPU only, Torch 2.11.0+cpu, 8 threads.

| measurement | before | after |
|---|---:|---:|
| Full tests | 124 passed, 1 skipped | 132 passed, 1 skipped |
| Spawn callback | pickle failure | real spawn batch succeeds |
| All-failed validation | returned `gen_loss=0.0` | raises RuntimeError |
| Incomplete strict checkpoint | loaded partially | raises RuntimeError |
| CUDA cache flushes for 5 tile batches | 5 calls | 0 calls |
| CPU smoke train section | 0.836 s | 0.870 s |
| Smoke inference max difference | 0 | 0 |

Model/compute measurements:

- Parameters: 7,021,690 in local/global mode; 7,218,202 in axial mode.
- Local/global 128x128 block: 33.07 ms versus 48.28 ms axial.
- Local/global generator 64x64: 292.43 ms versus 298.51 ms axial.
- Lazy initialization on eight synthetic 256x256 scenes: 0.038 s versus 0.413 s.
- Lazy resident NumPy arrays: 0 MiB versus 68 MiB standard.
- Thirty-two lazy random patches: 0.370 s versus 0.047 s standard.
- At 64x64, encoder1 + decoder2 consumed about 224 ms of a 449 ms staged
  forward, confirming high-resolution blocks dominate.
- Legacy 1024-point symmetric Sinkhorn forward/backward: approximately 369 ms.
- CUDA peak memory and realistic GPU throughput could not be measured because
  no CUDA device was available.

# 8. Optimization Roadmap

## Immediate low risk

1. Rebuild the broken default `.venv` and pin a known-good Torch build.
2. Add no-decay AdamW parameter groups and run a short controlled ablation.
3. Log raw and `[0,1]`-clamped validation metrics side by side.
4. Add dataloader throughput logging and configurable prefetch factor.
5. Update README entry points so generator-only training is clearly primary.

## Medium risk

1. Cache long-axis relative-index tensors by `(length, device)` to reduce small
   repeated allocations.
2. Profile `torch.compile` on a CUDA host; dynamic padding and checkpointing
   may require fixed stage shapes for reliable gains.

## High risk/high reward

1. Add a low-rank spectral basis/output head to exploit 31-band correlation.
2. Distill the current model into a CNN/Restormer-style student for deployment.
3. Retire Sinkhorn-GAN training unless it demonstrates a reproducible spectral
   quality gain over the cheaper generator-only objective.

# 9. Final Verdict

**FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED**

The active generator is mathematically plausible, compact, numerically guarded,
and covered by a strong test suite. The patched runtime no longer silently
accepts incomplete checkpoints or failed validation, and worker/inference paths
are safer.

The two highest-priority code bottlenecks are now implemented. Remaining
deployment risk is primarily empirical: no CUDA peak-memory or end-to-end
throughput measurement was available, and the local/global attention change
still needs a controlled quality ablation on the real training set. Validation
projection and optimizer no-decay protocols also remain unresolved.

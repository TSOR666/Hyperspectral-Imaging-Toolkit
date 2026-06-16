# MSWR v2 Bottleneck & Optimization Audit - Round 5

Audit date: 2026-06-16
Environment: Windows / Python 3.11 via `CSWIN v2/.venv-audit`; CPU-only PyTorch
in this workspace, so CUDA memory, SDPA backend, and bf16/fp16 quality deltas
were not measured here.

## 1. Inferred Task and Model Family

- Task: RGB-to-hyperspectral reconstruction, `B x 3 x H x W` -> `B x 31 x H x W`.
- Model family: U-shaped CNN/attention restoration network with window attention,
  landmark/global attention, optional spectral attention, and per-stage CNN
  wavelet DWT/IDWT branches.
- Training recipe: ARAD/MST++-style patch training, MRAE-first objective,
  AdamW/cosine, AMP, channels-last, EMA, validation on raw and EMA sources.
- Inference target: full-image or tiled HSI reconstruction with optional TTA.

## 2. Critical Paths & Profiling Plan

Critical path:
`dataloader.py` -> patch/full-image tensors -> `IntegratedMSWRNet.forward` ->
loss (`Loss_MRAE` or `EnhancedMSWRLoss`) -> AMP backward -> AdamW/EMA ->
validation/checkpoint -> `mswr_test_ntire.py` or `mswr_inference.py`.

Profiling and audit plan executed:
- Re-ran round-4 regression tests and CPU smoke train/infer.
- Inspected remaining high-risk zones: host caching, tiled inference, CLI precision
  plumbing, landmark attention, wavelet low-frequency processing.
- Patched confirmed implementation bottlenecks in inference.
- Added regression tests and measured a tiled-inference microbenchmark.

## 3. Bottleneck Summary

| severity | category | file:line | bottleneck | impact | proposed fix |
|---|---|---|---|---|---|
| MED, fixed | Speed/Inference | `mswr_inference.py:101,792-807` | `InferenceConfig.batch_size` existed, but tiled inference processed one tile per model call | One launch and one CPU transfer per tile; low GPU occupancy on large images | Batch tiles with `batch_size`, share model execution helper, postprocess batched BCHW outputs |
| MED, fixed | Numerical/Inference | `mswr_inference.py:104-107,1109-1115` | CLI default disabled AMP despite config default `use_amp=True`; CLI also could not set `amp_dtype` | Command-line inference could silently run a different precision path from training/validation | Make CLI AMP default match config, add `--no_amp`, add `--amp_dtype auto|fp16|bf16` |
| MED, addressed (§10) | Memory/Data | `dataloader.py` | Train and validation samples are eagerly cached as float32 | ARAD-sized runs can spend tens of GB of host RAM before training starts | **DONE:** opt-in `cache_dtype=float16` (default float32; `__getitem__` upcasts) — ~28.1 GB → ~14.1 GB, parity-exact |
| MED, addressed (§10) | Architecture/Quality | `model/mswr_net_v212.py` | Default `learned` landmarks are static K/V vectors, not content-pooled spatial landmarks | Weak global spatial mixing; can cap quality despite extra attention cost | **ENABLED:** ablation config `ablation_landmark_adaptive.yaml` + regression test proving `adaptive` gives global mixing vs `learned`; default unchanged pending retrain |
| MED, addressed (§10) | Architecture/Quality | `model/mswr_net_v212.py` | Wavelet branch runs attention/FFN on LL while high-frequency bands are only gated | Detail bands may be under-modeled; likely retraining-sensitive | **ENABLED:** opt-in `wavelet_detail_processing` (zero-init depthwise residual, +0.17% params, checkpoint-safe) + `ablation_wavelet_detail.yaml`; default off pending retrain |

## 4. Detailed Findings

### Fixed: tiled inference ignored `batch_size`

Evidence: `InferenceConfig.batch_size` is defined at `mswr_inference.py:101`, and
the tiled path now batches tiles in `MSWRInference._process_tiled` at
`mswr_inference.py:792-807`. Before this patch, the same function looped over
every tile individually and called `postprocess()` per tile.

Why it matters: large inputs split into many tiles. Processing them one by one
creates unnecessary model calls, kernel launches, and device-to-host transfers.

Minimal fix: batch equal-shaped tiles with `torch.cat`, call the model once per
tile batch, and convert the BCHW output to HWC samples in one postprocess step.

Stronger alternative: on CUDA, keep tile outputs on device and merge/blend on
device, with one final host transfer. That is a larger change because current
merge logic is NumPy-based.

Expected impact: no quality change; lower tiled inference overhead. CPU tiny
model benchmark improved from 162.21 ms to 105.88 ms for the tested tiled case.

Tradeoffs: larger `batch_size` increases activation memory. Default remains 1,
so old memory behavior is preserved unless users opt in.

### Fixed: CLI precision path did not match config

Evidence: `InferenceConfig.use_amp=True` and `amp_dtype='auto'` live at
`mswr_inference.py:104-107`. The CLI now exposes matching defaults and controls
at `mswr_inference.py:1109-1115`.

Why it matters: the command-line path is the user-facing production path. If it
silently disables AMP or hides `amp_dtype`, checkpoint scoring and deployment
precision can diverge from training validation.

Minimal fix: default CLI AMP to enabled, add `--no_amp`, and pass `--amp_dtype`
into `InferenceConfig`.

Stronger alternative: share a small precision-config parser between training,
NTIRE test, and inference to prevent future drift.

Expected impact: command-line CUDA inference now matches the dataclass/training
precision intent. CPU behavior is unchanged because autocast is CUDA-gated.

Tradeoffs: users relying on implicit no-AMP CLI behavior should now pass
`--no_amp` explicitly.

### Open: host RAM cache remains large

Evidence: `TrainDataset` appends full RGB/HSI arrays at `dataloader.py:202-234`;
`ValidDataset` stores full tensors at `dataloader.py:386-387`.

Why it matters: ARAD-scale float32 RGB+HSI caching can consume a large host RAM
budget before the first optimizer step. This is not a correctness bug, but it
can block training on ordinary workstations.

Minimal fix: add an opt-in `cache_dtype=float16` mode and cast patches back to
float32 in `__getitem__`.

Stronger alternative: memory-map HSI cubes or add lazy/on-demand sample loading
with an LRU scene cache.

Expected impact: fp16 cache roughly halves host RAM with tiny loader overhead.

Tradeoffs: fp16 host cache should be validated against MRAE on real ARAD data
before becoming default.

### Open: static learned landmarks are weak global mixing

Evidence: `OptimizedLandmarkAttention2D.forward` expands learned landmarks at
`model/mswr_net_v212.py:876-877`; content-dependent landmarks are only used in
the `adaptive` branch at `model/mswr_net_v212.py:878-887`.

Why it matters: default learned landmarks behave like a learned dictionary for
K/V, not a pooled summary of the current image. That weakens the intended global
spatial attention inductive bias.

Minimal fix: run a controlled retraining ablation of `learned`, `uniform`, and
`adaptive` landmark pooling under the same recipe.

Stronger alternative: replace static landmarks with content-dependent low-rank
pooling or window-to-global token pooling.

Expected impact: likely quality improvement if global spatial mixing is limiting
MRAE, but this requires retraining evidence.

Tradeoffs: adaptive pooling costs extra compute and may change regularization.

## 5. Patches Implemented

Files changed:
- `mswr_inference.py`
- `tests/test_inference.py`

Summary:
- Added `MSWRInference._run_model()` to centralize AMP/TTA model execution.
- Added `MSWRInference.postprocess_batch()` for BCHW -> HWC batched outputs.
- Changed `_process_tiled()` to process tiles in batches using
  `InferenceConfig.batch_size`.
- Changed CLI AMP behavior to match `InferenceConfig`: default on, `--no_amp`
  opt-out, `--amp_dtype` exposed and passed through.
- Added tests for batched tile parity, reduced model calls, batched postprocess,
  and CLI AMP/amp_dtype plumbing.

## 6. Tests Added + How to Run

Added to `tests/test_inference.py:215-335`:
- `test_tiled_inference_uses_configured_batch_size`
- `test_postprocess_batch_returns_hwc_float32_samples`
- `test_main_cli_defaults_amp_and_forwards_amp_dtype`
- `test_main_cli_can_disable_amp`

Commands run:
```powershell
cd mswr_v2
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' -m py_compile mswr_inference.py tests\test_inference.py
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' -m pytest -q tests\test_inference.py
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' -m pytest -q
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' smoke_infer.py --device cpu
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' smoke_train.py --steps 3 --device cpu
```

## 7. Benchmark Results

| measurement | before / emulated old path | after / patched path |
|---|---:|---:|
| Full MSWR pytest | 202 passed, 2 skipped (round 4) | 206 passed, 2 skipped, 1 warning |
| `tests/test_inference.py` | 15 tests before patch | 19 passed |
| CPU smoke infer | passed | passed; 64/128/256 outputs OK |
| CPU smoke train | passed | passed; 3 steps, avg 277.7 ms/step |
| Default base params | unchanged | 2,597,500 |
| Canonical dual+spectral base params | unchanged | 3,035,684 |
| Tiny model tiled 144x144, tile 64, batch 1 | 9 calls/pass, avg 162.21 ms | - |
| Tiny model tiled 144x144, tile 64, batch 4 | - | 3 calls/pass, avg 105.88 ms |
| Fake model tiled 512x512, tile 128, batch 1 | 25 calls/pass, avg 122.86 ms | - |
| Fake model tiled 512x512, tile 128, batch 4 | - | 7 calls/pass, avg 126.62 ms |

Notes:
- Fake-model timing is dominated by preprocessing/postprocess/NumPy merge, so
  batching reduces model calls but not wall time in that synthetic case.
- No CUDA was available; peak GPU memory, CUDA throughput, and SDPA backend were
  not measured.

## 8. Optimization Roadmap

Immediate low-risk:
- Add optional fp16 host caching for `TrainDataset` and validate patch parity on
  a few real ARAD samples.
- Add a CUDA benchmark script for tiled inference with `batch_size` sweep and
  peak memory recording.

Medium-risk architecture:
- Landmark pooling ablation: `learned` vs `uniform` vs `adaptive`.
- Wavelet detail processing ablation: add lightweight detail-band processing or
  convert wavelet branch into a high-frequency residual.

High-risk/high-reward:
- Device-side tiled merge and overlap blending for large inference workloads.
- Revisit spectral attention fp32-only softmax under bf16 with an ARAD MRAE A/B
  test.

## 9. Final Verdict

FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED.

The implementation is stable enough for RGB-to-HSI experimentation and the
round-5 patch removes a concrete tiled-inference inefficiency without changing
model outputs. The remaining limits are mostly host-memory strategy and
architecture choices that require real ARAD retraining or CUDA profiling before
they should be changed by default.

## 10. Round-5 Follow-up — open items addressed (2026-06-16)

All three open items are now actionable. The two architecture items are
retraining-sensitive, so they are delivered as opt-in / ablation-ready levers
with the default behavior unchanged (no blind benchmark-affecting flip), each
guarded by a regression test. Suite: **206 → 214 passed, 2 skipped** (+8 tests,
`tests/test_audit5_fixes.py`). No parameters added to the default model.

**A — fp16 host cache (DONE, default-safe).** `TrainDataset`/`ValidDataset`
gained `cache_dtype` (`'float32'` default, `'float16'` opt-in), plumbed through
`TrainingConfig` and a new `--cache_dtype` CLI flag. `__getitem__` upcasts each
sample to float32, so training/validation precision is unchanged. Verified
parity-exact (fp16 output == float32 quantized to fp16). Footprint: ARAD 900-scene
train cache ~28.1 GB → ~14.1 GB. Files: `dataloader.py`,
`train_mswr_v212_logging.py`.

**B — landmark pooling (ENABLED for ablation).** `landmark_pooling` was already
plumbed; added `configs/experiments/ablation_landmark_adaptive.yaml` (canonical
recipe, only `landmark_pooling: adaptive` differs) and a regression test that
demonstrates the gap concretely: a single-pixel input perturbation changes only
~1 output position under `learned` (static dictionary) but ≥50% of positions
under `adaptive` (content-pooled → genuine global mixing). The default stays
`learned` pending a head-to-head retraining result.

**C — wavelet detail-band processing (ENABLED, opt-in, checkpoint-safe).** Added
`WaveletDetailBlock` (a per-channel depthwise 3×3 residual on the LH/HL/HH detail
bands) gated by a new `wavelet_detail_processing` config flag (default `False`),
plumbed through the trainer and `--wavelet_detail_processing`. The block is
zero-initialized → exact identity at start, so enabling it does not perturb a
fresh model or destabilize training, and existing checkpoints load cleanly
(`strict=False` missing keys are only the zero-init detail blocks). Cost: +4,480
params (+0.17% on base). Ablation config `configs/experiments/ablation_wavelet_detail.yaml`.
The default keeps the original LL-only attention path until an ARAD retrain
confirms the benefit. Files: `model/mswr_net_v212.py`,
`train_mswr_v212_logging.py`.

Still genuinely deferred (need CUDA / real ARAD data, cannot be done in this
CPU-only workspace): measuring the bf16-vs-fp16 MRAE delta, peak GPU memory, and
the actual MRAE impact of the landmark/wavelet ablations — i.e., deciding
whether B/C should become the new defaults.

# MSWR v2 Bottleneck & Optimization Audit — Round 4

Audit date: 2026-06-16
Branch: `Performance-optimization-`
Environment: Windows 11, Python 3.11.14, PyTorch 2.11.0+cpu (no CUDA available;
CUDA-only behavior reasoned from code, not benchmarked).
Method: 9-dimension parallel static audit with adversarial verification
(31 agents, 47 raw findings → 20 confirmed, 2 refuted/downgraded), followed by
patching, regression tests, and CPU microbenchmarks run by the orchestrator.

Supersedes nothing in `AUDIT_REPORT_2026-06-12.md`; it extends it with the
uncommitted working-tree changes and the previously-open items.

## 1. Inferred Task and Model Family

- Task: RGB→hyperspectral reconstruction, `B×3×H×W` → `B×31×H×W` reflectance.
- Model: U-shaped CNN/Transformer hybrid (`IntegratedMSWRNet`, 3,035,684 params),
  per-stage CNN wavelet branch (DWT → attend LL → gate high-freq → IDWT) wrapping
  dual spatial attention (window + landmark) + MST++-style spectral attention.
- Recipe: batch 20, 128² patches, 1000 steps/epoch, AdamW + cosine, AMP (bf16
  'auto'), channels-last, EMA; MRAE-only loss for MST++ comparability.
- Known: overfits ~epoch 70 on ARAD's 900 scenes (not capacity-bound).

## 2. Critical Paths
`dataloader (RAM-preloaded, geometric aug) → IntegratedMSWRNet.forward → loss
(MRAE or EnhancedMSWRLoss) → AMP backward → AdamW/cosine/EMA → validate
(raw+EMA, source selection) → checkpoint → inference (full/tiled) / NTIRE eval`

## 3. Bottleneck Summary (this round)

| severity | category | file:line | bottleneck | status | action |
|---|---|---|---|---|---|
| HIGH | Quality/Inference | `mswr_test_ntire.py:866` | NTIRE engine reported only CLAMPED MRAE; no MST++-comparable unclamped MRAE (the quantity the trainer selects on) | prior-fix-incomplete | **PATCHED (P1)** |
| MED | Numerical/Inference | `mswr_test_ntire.py:827` | test/inference autocast = bare fp16; trainer validates in bf16 → different numeric regime than `best_mrae` | new | **PATCHED (P3)** |
| MED | Speed | `mswr_test_ntire.py:170,269` | SSIM + per-band metrics loop over 31 bands one at a time | open-from-prior | **PATCHED (P4)** |
| MED | Training | `train_mswr_v212_logging.py:875` | weight decay still hits layer-scale `gamma`/`gamma2`, spectral `rescale`, `landmarks`, `relative_position_bias_table` | open-from-prior | **PATCHED (P2)** |
| MED | Architecture | `mswr_net_v212.py:832,859` | default `learned` landmark branch is a static per-pixel dictionary, not global mixing | open-from-prior | roadmap |
| MED | Architecture | `mswr_net_v212.py:1244-1308` | wavelet encoder stages run attention/FFN only on the half-res LL subband; detail bands only gated | new | roadmap |
| MED | Quality | `configs/experiments/sota_mrae_ssim.yaml:40` | aux losses (esp. unwarmed L1 0.20) break MST++ comparability without fixing overfitting | new | **PARTLY PATCHED (P8)** + labeled |
| MED | Memory | `dataloader.py:202` | full train+val set eagerly cached float32 (~27 GB) | new | roadmap (opt-in fp16) |
| MED | Speed/Inference | `mswr_inference.py:749` | tiled inference one tile at a time + per-tile D2H sync | open-from-prior | roadmap (not on NTIRE scoring path; 4 caveats) |
| LOW | Memory | `mswr_net_v212.py:236-263` | inverse-DWT filter cache unbounded | new | **PATCHED (P5)** |
| LOW | Speed | `mswr_net_v212.py:1750-1856` | PerformanceMonitor runs every forward; output consumed only at startup | new | **PATCHED (P6)** |
| LOW | Speed | `mswr_net_v212.py:964-978` | SpectralMSA2D forces fp32 softmax/normalize, no SDPA | new | roadmap (quality-sensitive; needs A/B) |
| LOW | Training | `train_mswr_v212_logging.py:2076` | `selection_source` silently ignored unless `ema_eval_mode='both'` | regression-in-diff | **PATCHED (P7)** |
| LOW | Quality | `configs/.../sota_mrae_ssim.yaml:36` | added dropout/drop_path masks overfit root cause | new | roadmap |
| LOW | Training | `configs/train.yaml:15` | `end_epoch` 150→220 keeps LR elevated past overfit onset | open-from-prior | roadmap (tuning) |

Refuted by adversarial verification: channels-last "dropped by rearrange" (INVALID
— memory_format propagates through einops/SDPA); lightweight-checkpoint "loses EMA"
(downgraded LOW — `selected_state_dict` carries the chosen weights).

## 4. Patches Implemented

All patches are behavior-preserving for the canonical MRAE-only benchmark config
and add no parameters (3,035,684 unchanged).

**P1 — NTIRE unclamped MRAE (`mswr_test_ntire.py`).** `MetricsCalculator` now
tracks `mrae_unclamped` (raw output) alongside the clamped `mrae`; `update()`
takes `pred_unclamped`; `test_single_image` captures the pre-clamp border-cropped
prediction; the print/baseline-compare sites surface `MRAE*` as the
MST++/NTIRE-comparable headline. This closes the gap where checkpoints were
*selected* on unclamped MRAE but *reported* on clamped MRAE.

**P2 — weight-decay exemptions (`train_mswr_v212_logging.py:create_optimizer`).**
No-decay group now covers any param with `ndim<=1` plus leaf names
`{gamma, gamma2, rescale, landmarks, relative_position_bias_table}`, regardless of
the `attn` substring. 20 tensors / 42,280 elements moved decayed→no-decay.

**P3 — eval/inference AMP dtype (`mswr_test_ntire.py`, `mswr_inference.py`).**
Added `amp_dtype='auto'` + a resolver mirroring the trainer (bf16 on Ampere+);
the NTIRE autocast is now CUDA-gated. Scoring matches the precision `best_mrae`
was measured in.

**P4 — vectorized NTIRE metrics (`mswr_test_ntire.py`).** SSIM runs one
multi-channel `avg_pool2d` (mathematically identical to the per-band loop);
per-band MRAE/RMSE/PSNR computed in vectorized form (PSNR via per-image-per-band
MSE→log→batch-mean to preserve its non-linearity). Removes ~31 kernel launches and
~93 `.item()` host syncs per image (GPU-side win; CPU compute-neutral). Parity
guaranteed by an equivalence test.

**P5 — bounded inverse-DWT cache (`mswr_net_v212.py`).** Mirrors the forward
transform's `_cache_size_limit=16` + LRU eviction.

**P6 — PerformanceMonitor off in hot loop (`train_mswr_v212_logging.py`).** The
model's `performance_monitoring` is now driven by `--profile_model` (its only
consumer) instead of `memory_monitoring`, removing ~5.6% per-forward CPU overhead
during normal training. Per-epoch CUDA peak logging (memory_monitoring) is unchanged.

**P7 — selection_source warning (`train_mswr_v212_logging.py`).** Logs a startup
warning when `selection_source != 'auto'` while `ema_eval_mode != 'both'` (the
only mode where it can be honored). Non-fatal.

**P8 — MRAE warmup + sota labeling (`train_mswr_v212_logging.py`,
`sota_mrae_ssim.yaml`).** The primary MRAE term is no longer warmup-scaled (it
matches L1 as a base signal); only SSIM/SAM/gradient are warmed. Fixes the
epoch-0 inversion where L1 (0.20) outweighed effective MRAE (1.0×0.1). The sota
config header is labeled NOT MST++-comparable.

## 5. Tests Added

`tests/test_audit4_fixes.py` — 15 cases:
- P1: unclamped MRAE distinct from clamped; defaults to clamped when omitted.
- P2: scale/bias-table params land in no-decay group; all 1-D params exempt.
- P3: dtype resolver (`fp16`/`auto`/`bf16`/invalid) for both engines; configs expose `amp_dtype`.
- P4: vectorized SSIM and per-band metrics equal the old per-band loops (oracle).
- P5: inverse-DWT cache bounded at 16 across 39 distinct keys.
- P6: PerformanceMonitor populated iff the config flag is set.
- P7: `_select_validation_source` auto-picks lower MRAE; explicit honored; fallback works.
- P8: MRAE term unscaled at epoch 0; SSIM term still warmup-scaled.

Run:
```powershell
cd mswr_v2
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' -m pytest -q
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' smoke_train.py --steps 3 --device cpu
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' smoke_infer.py --device cpu
```

## 6. Benchmark Results

| measurement | before | after |
|---|---:|---:|
| Full pytest | 187 passed, 2 skipped | 202 passed, 2 skipped |
| Canonical base params (dual+spectral) | 3,035,684 | 3,035,684 |
| 128² b1 CPU forward, monitor ON | 145.0 ms | — |
| 128² b1 CPU forward, monitor OFF (new training default) | — | 136.9 ms (−5.6%) |
| Params weight-decayed that should not be | 20 tensors / 42,280 | 0 |
| NTIRE SSIM 31-band 256² (CPU) | 214.3 ms (loop) | 213.0 ms (vectorized; GPU win is in removed launches/syncs) |
| smoke_train 3 steps CPU | — | PASSED, ~159 ms/step |
| smoke_infer 128² CPU | — | PASSED, 58.7 ms |

Not measurable here (no CUDA): peak GPU memory, SDPA backend, bf16 vs fp16
benchmark MRAE delta, GPU host-sync savings from P4.

## 7. Optimization Roadmap (not patched this round)

Immediate, low-risk:
1. On the target GPU, re-run the canonical config and record latency, peak
   allocated memory, selected SDPA backend, and raw/EMA MRAE. (still pending)
2. Opt-in fp16 RAM caching for `TrainDataset` (parity-safe: train patches already
   cast to fp32 in `__getitem__`); halves the ~27 GB host footprint.

Medium-risk architecture (require retraining to validate — do not change blind):
1. Landmark branch: ablate `learned` vs `uniform` vs `adaptive`, or replace
   static landmarks with content-dependent low-rank spatial pooling.
2. Wavelet stages: the detail bands (LH/HL/HH) are only gated, never processed;
   either process them with a lightweight depthwise op or make the wavelet an
   auxiliary high-frequency *residual* so the main transformer keeps full-res
   tokens (MST++/Restormer pattern). This is a strong candidate for the quality
   ceiling, since attention currently runs on half-resolution features.
3. Batched tiled inference (mswr_inference._process_tiled): group uniform tiles
   into sub-batches, keep on-device, single D2H after merge. Not on the ARAD
   scoring path (images <1024px never tile), so low realized impact + 4 caveats
   (ensemble path, postprocess clamp, memory budget, shape grouping).

Higher-risk / quality-sensitive:
1. SpectralMSA2D: the forced fp32 normalize+softmax defeats AMP on the spectral
   branch; only relax under bf16 with an A/B MRAE check (channel-cosine attention
   relies on fp32 mantissa precision).

## 8. Final Verdict

**FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED.**

The model is mathematically sound for RGB→HSI and the recipe is now more
internally consistent: benchmark reporting (P1), scoring precision (P3), and
training regularization of scale/bias params (P2) all align with the MST++
protocol the project targets, and the loss objective no longer transiently
inverts (P8). The two remaining substantive limitations are architectural and
deliberately left for a retraining cycle: attention runs on half-resolution
(LL-only) wavelet features and the default landmark branch is not global. GPU
deployment readiness still depends on measuring peak memory, SDPA backend, and
real ARAD-1K MRAE on target hardware.

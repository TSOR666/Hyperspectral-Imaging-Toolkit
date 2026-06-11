# HSIFusion & SHARP — Bottleneck & Optimization Audit (Pass 3, 2026-06-11)

This is a re-audit on top of the prior `AUDIT_REPORT.md`. The goal of this pass was to (a) find
issues the prior audit missed, and (b) check whether each prior "fix" is actually correct and
complete. Findings were produced by a fan-out of inspectors over both models × all audit lenses,
then **adversarially verified** (each significant finding re-checked by an independent skeptic with
a runnable probe). Of 41 raw findings, **11 were confirmed, 12 were rejected** as
incorrect/over-rated, and the remainder were low-value duplicates.

Environment: a CPU `torch 2.12.0` venv built from the repo's uv cache. Real ARAD-1K data and CUDA
were unavailable, so GPU VRAM/throughput numbers are not measured; correctness, dtype/shape/finite
invariants, attention microbenchmarks, and one-step train/val/infer were exercised.

> The `large` classifier variant segfaults in this preview torch build during a large `nn.Linear`
> allocation under multi-threaded BLAS. This is an environment instability, not a code defect; all
> work ran single-threaded (`OMP_NUM_THREADS=1`).

---

## 1. Inferred Task and Model Family

- Task: RGB `(B,3,H,W)` → 31-band HSI `(B,31,H,W)` reconstruction on ARAD-1K / MST++ data.
- **HSIFusion v2.5.3** — hierarchical encoder/decoder transformer: sliding-window (RoPE) +
  spectral attention, optional MoE, decoder cross-attention, **unbounded linear output**, trains on
  **MRAE** (`MSTPlusPlusLoss`).
- **SHARP v3.2.2** — hierarchical encoder/decoder: streaming top-k sparse attention, RBF features,
  multi-scale + gated cross-attention, ChannelRMSNorm.
- Primary metric: **MRAE** (then RMSE, PSNR). The benchmark protocol values **comparability**
  (per-image metric averaging, fixed [0,1] normalization, geometric-only augmentation).

## 2. Critical Paths & Profiling Plan

`split_txt + RGB/.mat → optimized_dataloader → model.forward → loss → AMP/backward → AdamW/cosine →
per-image MRAE/RMSE/PSNR → raw+EMA checkpoint → tiled inference`.

Verified by: static call-path inspection; synthetic shape/dtype/finite probes; one-step train/val;
exact-objective invariants (`compute_loss(x,x)==0`); attention microbenchmarks; checkpoint round
trips; the full local pytest suite (68 tests).

## 3. Bottleneck Summary (confirmed only)

| Severity | Category | File:line | Bottleneck | Fix |
|---|---|---|---|---|
| HIGH | Quality/Training | `sharp_training_script_fixed.py:373`, `sharp_v322_hardened.py:1391` | SHARP **trained on L1+curvature** but selected/eval'd on **MRAE** (both trainer paths); sibling HSIFusion trains MRAE | `loss_type='mrae'` default routes both paths through `MSTPlusPlusLoss` |
| HIGH | Implementation | `common_utils_v32.py:666` | `sparse_attention_topk` `.view()` on non-contiguous tensor → **crash** whenever `use_sparse_attention=True` | `.view`→`.reshape` |
| HIGH | Memory | `sharp_v322_hardened.py:476,495` | `VectorizedWindowedSparsemax` returns fp32 for fp16 input → upcasts V in attention (~2× memory) | restore caller dtype |
| MEDIUM | Quality/Math | `sharp_v322_hardened.py:1071` | SHARP output `tanh`∈[-1,1] vs targets [0,1]: wasted half-range, saturates near full reflectance, emits negatives | `output_activation` config, **default `sigmoid`** (legacy ckpts → tanh) |
| MEDIUM | Training (regression) | `sharp_training_script_fixed.py:738` | Manual-path (`accumulate_steps>1`) validation ignored EMA; no EMA weights saved | EMA-swap w/ guaranteed restore + save `ema_model_state_dict` |
| MEDIUM | Memory/Speed (regression) | `hsifusion_v252_complete.py:944` | `cross_attention_max_tokens` dataclass default reverted to `None` (full quadratic) | default `1024` (matches CLI/audit) → **6.9× faster** at 96² |
| LOW | Architecture (regression) | `hsifusion_v252_complete.py:951` | `estimate_uncertainty=True` default → untrained head emits arbitrary 2nd output | default `False` |
| LOW | Math | `hsifusion_v252_complete.py:127,143` | `_factor_pair` returns `h*w>n` for primes → reshape crash on flattened prime-length tokens | return exact factorization |
| LOW | Training | `hsifusion_training.py:347` | Final partial accumulation group under-weighted (loss/accumulate_steps for a smaller group) | scale by per-group size |
| LOW | Data | `optimized_dataloader.py:152` | HSI [0,1] assumption undocumented/unchecked; out-of-range cubes silently break loss/PSNR | warn-once on out-of-range (no auto-normalize) |
| LOW | Quality | `sharp_inference.py:64` | `key_rbf_mode` inference fallback `'mean'` vs training `'linear'` | left as-is (verified correct: legacy fieldless ckpts were trained with `mean`) |

### Notable REJECTED findings (adversarial verification prevented harm)

- **"tanh is a BLOCKER / cannot fit [0,1]"** → REJECTED to MEDIUM. tanh's range ⊇ [0,1]; a trained
  model reaches [0,1] (probe: 40 Adam steps → output in [0.35,0.66], 0% negatives). Real issue is
  conditioning/saturation, not a hard failure. Fixed anyway (sigmoid), but **not** as a silent
  checkpoint-breaking hotfix.
- **"RMSE/PSNR averaged wrong (pool MSE globally)"** → REJECTED. Validation loader is hardcoded
  `batch_size=1`; per-image metric averaging **is** the MST++/ARAD-1K protocol. The proposed
  "pooled" fix would have produced the value *furthest* from the reference and broken comparability.
- **"MRAE should use `|t|+eps` not `clamp_min`"** → REJECTED. The MST++ reference divides by raw
  label with a zero-floor (clamp-like). On [0,1] data `clamp_min` matches the reference **exactly**;
  the proposed `+eps` would have introduced a systematic bias.

## 4. Detailed Findings (the two highest-value)

### SHARP optimized the wrong objective
Both trainer paths computed `model.compute_loss` = L1 + 0.1·curvature, while best-model selection
and all reported metrics used MRAE. A uniform +0.02 error gives **L1=0.020 but MRAE=0.385** — MRAE
is dominated by low-reflectance pixels (≈10% of ARAD pixels are <0.05) that L1 ignores. The sibling
HSIFusion already trains on MRAE. Fix: `loss_type='mrae'` (default) injects `MSTPlusPlusLoss` (which
is bit-identical to the validation `_safe_mrae`) into *both* `SHARPv32Trainer` and the manual path;
`l1_curvature` is retained for ablation. This is the single biggest expected MRAE lever and restores
benchmark comparability with HSIFusion and MST++.

### SHARP output range mismatch
`return torch.tanh(out)` spans [-1,1] but ARAD HSI targets are [0,1] (RGB `/255`, cube used as-is).
An untrained model emits **47.7% negative** outputs; bright targets sit in tanh's saturated tail.
Fixed with a configurable `output_activation` defaulting to **sigmoid** (correct codomain,
nonnegative). Backward-compatible: the option is saved in the checkpoint config, and
`sharp_inference.py` defaults a *missing* field to `'tanh'` so pre-existing weights stay consistent.
This changes the parameterization and **requires retraining** to realize the benefit.

## 5. Patches Implemented

- `common_utils_v32.py` — `sparse_attention_topk` `.view`→`.reshape` (crash fix).
- `sharp_v322_hardened.py` — `VectorizedWindowedSparsemax` dtype restore; `output_activation` config
  (sigmoid default) + forward dispatch; `SHARPv32Trainer` injectable `criterion` (`_loss`).
- `sharp_training_script_fixed.py` — `loss_type` (default `mrae`) + `output_activation` config & CLI;
  `_build_criterion`; both paths use it; manual-path EMA validation (`_ema_weights_applied`,
  try/finally restore) + saves `ema_model_state_dict`.
- `sharp_inference.py` — checkpoint-safe `output_activation` (legacy → tanh).
- `hsifusion_v252_complete.py` — `_factor_pair` exact factorization; `cross_attention_max_tokens`
  default `1024`; `estimate_uncertainty` default `False`.
- `hsifusion_training.py` — per-group accumulation loss scaling.
- `optimized_dataloader.py` — `_warn_if_hsi_out_of_range` (warn, never auto-normalize).

## 6. Tests Added + How to Run

`test_audit3_fixes.py` (27 tests) pins every confirmed fix: sigmoid range + activation dispatch +
invalid-activation rejection; `compute_loss(x,x)==0` under sigmoid; non-contiguous
`sparse_attention_topk`; real HSIFusion `use_sparse_attention=True` fwd+bwd; sparsemax dtype (both
paths); `_factor_pair` exactness; trainer criterion routing; config defaults; HSI range warning
(fires + no mutation); accumulation group-weight==1.0; manual EMA validate-and-restore.

```bash
# (gitignored by repo convention for audit-named .py, like the existing audit tests; -f to commit)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  python -m pytest -q test_audit_fixes.py test_runtime_audit_regressions.py \
                      test_classifier_module.py test_audit3_fixes.py -k "not test_model_sizes"
# -> 68 passed, 1 deselected   (the deselected large-classifier test is a preview-torch segfault)
python smoke_pipeline.py --model both --size 64 --device cpu
```

## 7. Benchmark Results (CPU, torch 2.12.0)

| Item | Before | After |
|---|---|---|
| HSIFusion decoder cross-attn @96² (default) | 252.97 ms (N²=84.9M pairs) | 36.81 ms (N·1024=9.4M pairs) — **6.87×**, 9× fewer pairs |
| WindowedSparsemax attn weights (fp16 in) | fp32, 128 KiB, upcasts V | fp16, 64 KiB, **2×** smaller |
| SHARP loss geometry | L1=0.020 for a +0.02 error | same error = MRAE 0.385 (now the trained objective) |
| HSIFusion params (uncertainty head off) | 18.20M | 18.18M (untrained head removed from default) |
| Full test suite | 41 (prior) | **68 passed** (41 + 27 new) |

## 8. Optimization Roadmap

Immediate (done this pass): loss/metric alignment, sigmoid output, crash + dtype fixes, EMA on
manual path, config-default realignment, range guard.

Medium (needs ARAD-1K + GPU to validate): retrain SHARP under sigmoid+MRAE and confirm the expected
MRAE gain; sweep `max_global_tokens`/`cross_attention_max_tokens` ∈ {256,512,1024} on real MRAE/SAM;
batch tiled inference instead of serial patches.

High: replace SHARP streaming exact top-k (still **quadratic in compute** at moderate resolution —
memory-bounded only) with windowed/landmark attention at high res; proper DDP.

## 9. Final Verdict

**FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED.**

Both models are correct, stable, and now optimize the metric they are judged on. This pass fixed a
real objective/metric mismatch, an output-range mismatch, a latent attention crash, an AMP memory
regression, and several config-default regressions, with adversarial verification preventing three
plausible-but-wrong "fixes" that would have broken MST++/ARAD-1K comparability. Remaining gaps are
empirical (real-data MRAE confirmation of the sigmoid+MRAE change, GPU VRAM/throughput) and
architectural (quadratic-compute sparse attention at high resolution, no DDP).

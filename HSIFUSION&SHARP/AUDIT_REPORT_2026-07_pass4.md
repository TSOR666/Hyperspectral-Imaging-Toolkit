# HSIFusion & SHARP — Bottleneck & Optimization Audit (Pass 4, 2026-07-10)

Re-audit on top of `AUDIT_REPORT.md` (Pass 1) and `AUDIT_REPORT_2026-06_pass3.md` (Pass 3). Goals:
(a) find what the prior passes missed, and (b) independently re-verify that each prior "fix" is
actually correct **and complete**. Findings were produced by a fan-out of 10 inspectors (both models
× all audit lenses + a dedicated prior-fix verifier), each finding then **adversarially verified**
with a runnable CPU probe, then a completeness critic. Of **36 raw findings, 32 survived** adversarial
verification and **2 were rejected** (both correctly — the rejections prevented a bogus "channels_last
is defeated" change and a non-actionable `.item()` micro-opt). Every fix below was additionally
re-reproduced by hand before patching.

**Environment:** CPU `torch 2.11.0` venv (`CSWIN v2/.venv-audit`). Real ARAD-1K data and CUDA were
unavailable, so GPU VRAM/throughput are not measured; correctness, dtype/shape/finite invariants,
attention/tiling microbenchmarks, and one-step train/val/infer were exercised. Baseline suite: 73
tests passing; after this pass: **88 passing** (73 + 15 new).

---

## 1. Inferred Task and Model Family

- Task: RGB `(B,3,H,W)` → 31-band HSI `(B,31,H,W)` reconstruction on ARAD-1K / MST++ data, targets in
  `[0,1]`. Primary metric **MRAE** (then RMSE, PSNR), per-image averaging (batch_size=1 val) for
  MST++/ARAD comparability.
- **HSIFusion v2.5.3** — hierarchical enc/dec transformer: sliding-window (RoPE) + multi-scale spectral
  attention, optional MoE, decoder cross-attention, unbounded linear output, trains on MRAE.
- **SHARP v3.2.2** — hierarchical enc/dec: streaming top-k / local+landmark sparse attention, RBF
  features, multi-scale + gated cross-attention, ChannelRMSNorm, sigmoid output, EMA, DDP.
- Shared: `common_utils_v32.py` (sdpa, window ops, RoPE, MoE, sparsemax, RBF, wavelets),
  `optimized_dataloader.py` (MST++ patching, MRAE loss, samplers).

## 2. Critical Paths & Profiling Plan

`split_txt + RGB/.mat → optimized_dataloader → model.forward → loss → AMP/backward → AdamW/cosine →
per-image MRAE/RMSE/PSNR → raw+EMA checkpoint → tiled inference`.

Verified by: static call-path inspection; synthetic shape/dtype/finite probes; one-step train/val;
GradScaler state-machine reproduction; RoPE orthogonality/Toeplitz probes; sparsemax vs a canonical
reference; spectral-attention softmax-axis spy; tiling coverage maps; pickling round-trips;
checkpoint round-trips; the full local pytest suite.

## 3. Bottleneck Summary (confirmed, most-severe first)

| Severity | Category | File:line | Bottleneck | Fix |
|---|---|---|---|---|
| **BLOCKER** | Training | `hsifusion_training.py:363` (+ `sharp_v322_hardened.py:1630`, `sharp_training_script_fixed.py:853`) | Non-finite-grad skip runs `unscale_` but never `update()`; the **next** step's `unscale_` raises → whole AMP run crashes on the first fp16 overflow the guard exists to survive | call `scaler.update()` in the skip branch of all 3 trainers |
| **HIGH** | Numerical/Quality | `common_utils_v32.py:465` | **RoPE is broken**: cos/sin use half-layout `cat((freqs,freqs))` but `apply_rotary_emb` uses interleaved pairing → not a rotation, no relative-position invariance (norm change 0.53, Toeplitz spread 4.7). Sole positional signal of HSIFusion sliding-window attention | `emb = freqs.repeat_interleave(2, -1)` (norm change → 5e-7, spread → 2e-6) |
| **HIGH** | Numerical | `common_utils_v32.py:754` | **Sparsemax** derives τ from `max(masked cumsum)` = the post-max-subtraction sentinel `0`, not `cumsum[k_z-1]` → outputs don't sum to 1 (166/200 simplex violations) | gather `cumsum` at index `k_z-1` (0/300 violations) |
| **HIGH** | Implementation | `optimized_dataloader.py:199` | Lazy mode stores an `@lru_cache` closure on the dataset → unpicklable → DataLoader workers crash at startup under **spawn** (Windows/macOS) | `__getstate__/__setstate__` drop + rebuild the cache |
| **HIGH** | Inference | `sharp_inference.py:201` | Patch tiling makes **negative-start slices** for any image with a side `< patch_size` → most pixels stay 0 (silent garbage); `dim <= overlap` → 0 tiles (all-zero) | clamp starts `max(0,…)`, `range(0, max(1, dim-overlap), …)` |
| **HIGH** | Inference | `hsifusion_v252_complete.py:1332` | `from_pretrained` splats the saved **training** dataclass into `LightningProConfig(**cfg)` → `TypeError` on **every** real HSIFusion checkpoint | detect `model_size` → rebuild via factory; prefix-strip; `.eval()` |
| **MEDIUM** | Quality | `hsifusion_v252_complete.py:725` | **Spectral attention softmaxes over the degenerate `bands_per_group` axis (2–9), never the 31 bands**; the 31×31 correlation is never the attended map → no real cross-band modeling (critic's #1 MRAE lever) | attend over 31 bands; 31×31 correlation as additive bias |
| **MEDIUM** | Architecture | `hsifusion_v252_complete.py:1184` | Decoder cross-attention has **no residual** → bottleneck features survive only as query logits; ~10× weaker grad to the trunk | add `x_flat = x_flat + cross_attn(...)` |
| **MEDIUM** | Speed/Memory | `sharp_v322_hardened.py:1025` | Model-level `max_global_tokens` default `None` → dense global + cross-attention uncapped `O(N²)` for direct/inference builds (**4.8×** slower fwd @128² here; 16× fewer score pairs) | default `1024` (matches trainer) |
| **MEDIUM** | Speed | `sharp_training_script_fixed.py:850` | Manual accumulation never uses DDP `no_sync()` → **K** gradient all-reduces per optimizer step | `no_sync()` on non-stepping micro-batches |
| **MEDIUM** | Speed | `hsifusion_training.py:176` (×3 trainers) | `_has_finite_gradients` does one `.all()` bool-sync **per parameter** (hundreds/step), duplicating GradScaler's own check | one fused `torch.stack(...).all()` reduction |
| **MEDIUM** | Numerical | `hsifusion_training.py:433` | Validation runs under fp16 autocast → reported/selected MRAE computed on fp16 preds (non-comparable to fp32 protocol) | validate in fp32 (autocast off + upcast) |
| **MEDIUM** | Speed | `optimized_dataloader.py:562` | `persistent_workers` gated to lazy only → preloaded dataset re-pickled to every worker each epoch under spawn | `persistent_workers = num_workers > 0` |
| **MEDIUM** | Implementation | `sharp_training_script_fixed.py:576` | Manual path silently ignores `ema_update_every` (differs from built-in path for identical HPs) | honor it (+ decay compensation) |
| **LOW** | Numerical | `sharp_v322_hardened.py:1706` | `evaluate` hardcodes MRAE floor `1e-6`, ignoring `min_mrae_denom` → breaks Pass-3's "objective == metric" when customized | thread `mrae_eps=config.min_mrae_denom` |
| **LOW** | Quality | `sharp_v322_hardened.py:1562` | EMA throttling keeps decay un-scaled → `ema_update_every>1` makes the EMA window N× too long | `decay ** ema_update_every` (both paths) |
| **LOW** | Speed | `optimized_dataloader.py:493` | Lazy `cache_size=4` → <1% hit rate under shuffled patches (re-decodes a full cube per patch) | default 32 + document RAM tradeoff |
| **LOW** | Inference | `hsifusion_v252_complete.py:1344` | `str.replace('model.','')` strips the substring anywhere; model returned in train mode | prefix-strip + `.eval()` (folded into the fix above) |

Additional confirmed LOW findings (not separately patched — see Roadmap): dead `spectral_basis`
buffer + unused `RMSNorm` class in SHARP; `merge_sliding_windows` (a **dead** helper in
`common_utils`, superseded by `merge_sliding_windows_fixed`) crashes for `B>1`; `sparse_attention_topk`
materializes a dense L×L matrix; MoE `overflow_fraction` and spectral-failure-rate added to the loss
as zero-gradient constants; MEAN `key_rbf_mode` is rank-1 (masked at real resolution; the trainer
already defaults to `linear`); exact streaming top-k is O(N²)-compute at moderate resolution; serial
(un-batched) tiled inference.

## 4. Detailed Findings (highest value)

### GradScaler corruption made AMP training un-crash-survivable (BLOCKER)
On an optimizer-step batch, all three trainers call `scaler.unscale_()`, then on a non-finite gradient
they `zero_grad(); continue`/`return` **without** `scaler.update()`. GradScaler's per-optimizer state
stays in the "unscaled" stage, so the next step's `unscale_` raises
`"unscale_() has already been called on this optimizer since the last update()."` Reproduced with an
enabled CPU scaler: skip → next `unscale_` **raises**; adding `update()` → clean. The very guard meant
to survive the first fp16 overflow instead guaranteed a crash on it. (On CPU the trainers self-disable
AMP, so this only bites on CUDA — which is the deployment target.)

### HSIFusion's three headline inductive biases were all independently broken (the MRAE plateau)
The completeness critic's key synthesis: HSIFusion (the MRAE-trained model) had **all three** of its
advertised priors defeated at once, which plausibly explains the "architectural MRAE ceiling" recorded
in project memory far better than model width does:
1. **RoPE** was not a rotation and carried no relative-position structure (norm change 0.53, Toeplitz
   spread 4.7 vs ~0). Fixed by matching the cos/sin layout to the interleaved apply.
2. **Spectral attention** softmaxed over the 2–9-element `bands_per_group` axis; the 31×31 spectral
   correlation was buried inside the QK product and never the attended distribution. Rewritten so the
   31 bands are the attended tokens with the 31×31 matrix as an additive relative bias (softmax shape
   now `(B, HW, 31, 31)`; no parameter-shape change → checkpoints still load, behavior changes →
   retrain to benefit).
3. **Decoder cross-attention** had no residual, so the upsampled bottleneck survived only as query
   logits (constant-skip probe: output independent of query; ~10× weaker grad to the trunk). Added the
   standard residual.

### Sparse-attention numerics: sparsemax off the simplex
`Sparsemax` computed τ from `max` over a masked cumsum, but after the `x -= x.max()` stabilization the
descending cumsum starts at 0 and is otherwise ≤0, so `max` returned the sentinel 0 instead of
`cumsum[k_z-1]`. Outputs summed to <1 (example `[3,2.9,2.8,…]` → sum 0.70 vs 1.0). SHARP renormalizes
rows so totals are patched, but the **relative** weights were wrong. Fixed by gathering the cumsum at
the true threshold index; now matches a canonical reference on 300/300 random inputs.

### Inference correctness: tiling and checkpoint loading
`_predict_patches` produced negative-start slices whenever a side was `< patch_size` (e.g. a 482-row
frame with `--patch_size 500`, or any panoramic strip), leaving most pixels 0 after weight
normalization — silent garbage. `from_pretrained` raised `TypeError` on every checkpoint the trainer
actually writes (it saves the training dataclass, not `LightningProConfig`). Both fixed and covered by
end-to-end tests.

## 5. Patches Implemented

- `common_utils_v32.py` — RoPE cos/sin `repeat_interleave` (orthogonal rotation restored); Sparsemax τ
  via direct gather (simplex restored).
- `hsifusion_v252_complete.py` — spectral attention attends over the 31 bands with the 31×31 correlation
  as a relative bias; decoder cross-attention residual; `from_pretrained` rebuilds via factory for
  training-config checkpoints, prefix-strips `model.`/`_orig_mod.`, returns `.eval()`;
  `import dataclasses`.
- `sharp_v322_hardened.py` — `SHARPv32Trainer.train_step` scaler-state reset + fused finite-grad check;
  `update_ema` decay compensation; `evaluate(mrae_eps=…)` + fp32 metrics; `SHARPv32Config.max_global_tokens`
  default `1024`.
- `sharp_training_script_fixed.py` — manual `_train_step` scaler-state reset, DDP `no_sync()` on
  accumulation, `ema_update_every` honored; fused finite-grad check; `_update_ema` decay compensation;
  `_validate` passes `mrae_eps=min_mrae_denom`.
- `hsifusion_training.py` — scaler-state reset in the skip branch; fused finite-grad check; fp32
  validation.
- `optimized_dataloader.py` — `__getstate__/__setstate__` for the lazy cache (spawn-safe);
  `persistent_workers = num_workers > 0`; lazy `cache_size` default 32 + doc.
- `sharp_inference.py` — tiling clamps starts ≥0 and always emits ≥1 tile per axis.

## 6. Tests Added + How to Run

`test_audit_pass4.py` — 15 tests pinning each fix: GradScaler skip→step (mechanism) + all-3-trainers
reset (static); RoPE orthogonality + Toeplitz; sparsemax simplex + reference match + non-default dim;
lazy-dataset pickle round-trip; tiling coverage + end-to-end small-image no-zeros; `from_pretrained`
trainer-checkpoint round-trip (+eval mode + finite forward); spectral softmax over 31 bands; cross-attn
residual; `max_global_tokens` default bounded; `evaluate` uses configured eps; EMA decay compensation.

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  python -m pytest -q test_audit_fixes.py test_runtime_audit_regressions.py \
                      test_classifier_module.py test_audit3_fixes.py test_audit_pass4.py \
                      -k "not test_model_sizes"
# -> 88 passed, 1 deselected
python smoke_pipeline.py --model both --size 64 --device cpu
```

## 7. Benchmark Results (CPU, torch 2.11.0, single-thread)

| Item | Before | After |
|---|---|---|
| SHARP forward @128×128, dense global-attention default | 20,019 ms (`max_global_tokens=None`) | 4,201 ms (`=1024`) — **4.8×**, 16× fewer score pairs @128² |
| RoPE: per-token norm change (must be ~0) | 0.531 | 4.8e-7 |
| RoPE: relative-position Toeplitz spread (must be ~0) | 4.72 | 1.9e-6 |
| Sparsemax simplex violations / reference mismatches | 166/200 · 166/200 | 0/300 · 0/300 |
| Spectral-attention softmax axis | 2–9 (bands_per_group) | 31 (spectral bands) |
| `_has_finite_gradients` host-device syncs / step | O(#params) (538 for base) | 1 |
| Full test suite | 73 | **88** |
| SHARP params (tiny / base) | 7.30M / 38.63M | unchanged |

GPU VRAM/throughput and real-ARAD MRAE were not measured (no CUDA/data). The quality fixes (RoPE,
spectral axis, cross-attn residual, sparsemax) change model behavior and **require retraining** to
realize the MRAE benefit.

## 8. Optimization Roadmap

- **Immediate (done):** GradScaler crash, RoPE, sparsemax, tiling, `from_pretrained`, spectral axis,
  cross-attn residual, `max_global_tokens`, fp16-val, DDP `no_sync`, EMA throttle/eps, lazy pickling.
- **Validate on ARAD-1K + GPU (highest value):** retrain HSIFusion with all three inductive-bias fixes
  and confirm the MRAE plateau moves; retrain SHARP and confirm the sparsemax/eps fixes. Sweep
  `max_global_tokens ∈ {256,512,1024}`.
- **Medium:** batch tiled inference instead of serial; audit `MSTPlusPlusLoss` reduction vs the three
  denominators (train eps / min_mrae_denom / eval floor); probe `sdpa_unified`, `WaveletTransform`
  perfect-reconstruction, and MoE token conservation (critic's remaining blind spots).
- **High / high-reward:** replace SHARP exact streaming top-k (O(N²) compute at moderate resolution)
  with true windowed/landmark attention at high res; delete the dead `spectral_basis` buffer +
  `RMSNorm` class + `merge_sliding_windows` helper.

## 9. Final Verdict

**FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED.**

This pass fixed a training-crashing GradScaler BLOCKER, three broken HSIFusion inductive biases that
plausibly explain its MRAE ceiling, a simplex-violating sparsemax, two inference-path failures
(`from_pretrained`, small-image tiling), a spawn-fatal lazy dataloader, and a 4.8× dense-attention
default regression — with adversarial verification preventing two plausible-but-wrong changes. Both
models train, validate, checkpoint, and infer end-to-end (88 tests green). Remaining gaps are empirical
(real-ARAD MRAE confirmation of the quality fixes, GPU VRAM/throughput) and architectural (O(N²)
sparse-attention compute at high resolution, several unaudited shared primitives).

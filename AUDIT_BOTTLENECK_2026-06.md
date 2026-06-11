# CSWIN v2 + mswr_v2 — Bottleneck & Optimization Audit (2026-06-11)

Audit machine: CPU-only (torch 2.11.0+cpu, 8 threads, Windows 11). GPU-only effects
(flash-kernel eligibility, sync stalls, AMP) are reasoned from code + PyTorch semantics
and marked; everything else was probe-verified (`_audit_probes\`).

Scope: full audit of model architecture, math, training loops, data pipelines,
inference/eval tools across both projects. Prior audits (`CSWIN v2\AUDIT_REPORT_V2.md`,
`mswr_v2\AUDIT_REPORT.md`) were regression-checked first: **all 11 prior fixes intact.**

## Verdict

**FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED** (both projects). Correctness of the shipped
training/eval paths is now good after this patch set; the remaining items are
quality/efficiency opportunities (roadmap below), none blocking.

## Fixed in this audit (all covered by new regression tests)

### mswr_v2
| Sev | Issue | Fix |
|---|---|---|
| HIGH | `OptimizedWindowAttention2D`: symmetric pad + top-left crop shifted the attention branch by pad//2 px vs its residual whenever inner padding triggered — **including the deepest stage at default 128×128 training crops** (4×4 LL → 8×8, shift (2,2), probe-verified exact) | pad bottom/right only (`mswr_net_v212.py`) |
| HIGH | Inverse DWT used zero-pad `conv_transpose` against a circular-pad analysis: **db2 roundtrip border error 67%** (db3 120%), interior exact — all 4 shipped configs train db2 | circular-pad **adjoint fold** in `OptimizedCNNInverseWaveletTransform.forward`; roundtrip now 3.6e-7 everywhere |
| HIGH | `mswr_inference.py` loaded **zero weights** from trainer full checkpoints (EMA wrapper `{'decay','ema_state'}` never unwrapped; `strict=False` swallowed it) → garbage `.mat` outputs | container normalization ported from `mswr_test_ntire.py` + hard-fail when >25% keys missing |
| HIGH | TTA `ensemble_mode=full` branch 7 inverse was flip∘rot⁻¹ instead of rot⁻¹-then-flip → averaged a **rot180-misaligned** prediction | corrected inverse lambda |
| HIGH (comparability) | Validation clamps pred to [0,1] before MRAE — MST++ does NOT → flattering val numbers | `mrae_unclamped` now tracked/synced alongside (selection metric unchanged) |
| MED | `self.apply(_init_weights)` wiped the wavelet-gate soft-identity init (gates started random) | gates re-initialized after `apply()` |
| MED | SDPA bias `.expand(B·nW,…).to(dtype)` materialized the full per-window bias under autocast | broadcastable `(1,H,N,N)` mask, cast before unsqueeze |
| MED | OOM handler `del locals()[v]` is a CPython no-op — graph never freed, OOM recovery cascaded | plain rebinding to `None` |
| MED | Full-tensor `isfinite(output)` + ~4 host syncs every step | output check gated to every 100 iters (loss guard still per-step); CPU step time 149.8→106.0 ms in smoke |
| MED | Stale accumulated grads crossed epoch boundaries when `batches % accum != 0` | `zero_grad` at epoch start |
| MED | EMA cold start: first EMA validation was ~37% epoch-0 weights (`0.999^1000`) | shadow re-snapshot at first update after `ema_start_epoch` (resume-safe via flag) |
| MED | Unknown YAML keys silently dropped; `drop_path`/`dropout`/`attention_dropout` (the comparability-safe anti-overfit levers) unreachable from configs | loud ignored-keys warning + all three plumbed (CLI + YAML → model config) |
| LOW | `PerformanceMonitor.reset()` cleared CUDA peak stats every forward → trainer's epoch peak-memory log was meaningless | peak-stats reset removed |
| LOW | `torch.compile(mode='reduce-overhead')` unconditional in inference (recompiles per TTA/tile shape; failures surface outside the try) | opt-in `use_compile`, `dynamic=True` |
| LOW | TTA silently skipped in the tiled path | ensemble routed in `_process_tiled` |
| LOW | test filter double-counted `*_contrib_frac` keys (the one red test) | filter fixed; suite fully green |

### CSWIN v2
| Sev | Issue | Fix |
|---|---|---|
| HIGH | fp16 path: manual non-finite-grad skip bypassed `scaler.update()` → next iteration's `unscale_()` **raises and crashes the run** (and loss scale never shrank) | `scaler.update()` in the skip branch |
| HIGH | NaN-poisoned state → infinite non-advancing loop (skip paths never advanced `iteration`, param-NaN guard gated on the frozen counter) | consecutive-skip counter + param check + `FloatingPointError` after threshold |
| HIGH | Patch-inference blend mask exactly 0 at tile edges → **zeroed 1-px ring** in every stitched cube; +1e-8 normalizer biased single-tile borders | profile floored at 1e-2 + `clamp_min` normalization; stitched identity now exact (0.000000 everywhere, probe) |
| HIGH | `eval_mrae_variants.py` scored **random weights** for generator-only checkpoints (`strict=False`, 0 keys matched) | bare-checkpoint key prefixing + hard-fail on 0-key match |
| MED | fp16 stitch accumulation (~1e-3 relative noise into MRAE) | fp32 accumulation in `_stitch_patches` |
| MED | `DistributedSampler(seed=seed+rank*1000)` violated the identical-seed contract: ~35% duplicated draws/epoch at world=4 (probe) | shared seed |
| MED | DDP validation never all-reduced → best-ckpt/early-stop decided on rank 0's shard | SUM all-reduce of loss/metrics/batches |
| MED | `DDP(device_ids=[rank])` used GLOBAL rank (multi-node crash) + `find_unused_parameters=True` overhead | local rank + configurable flag default False |
| MED | EMA shadow cloned from random init before resume; comment claimed otherwise | `GeneratorEMA.reinit_from()` on EMA-less checkpoints |
| MED | Progressive stages re-loaded the ~30 GB in-RAM scene set (+ discarded val copy) per stage | `MST_TrainDataset.set_patch_geometry()` mutation path |
| MED | 3 host syncs/step (isfinite(loss), isfinite(grad_norm), loss.item()) | 1 sync/step (grad-norm check subsumes loss NaN; device-side loss accumulation) |
| MED | Long-axis bias rows beyond ±~132 never trained at patch 128 but indexed at full-res eval (random trunc_normal noise) | zero-init (fresh runs only) |
| LOW | objective log default 'mrae' vs code default 'l1'; unknown objective silently → MRAE; stale docstring; `checkpoint_keep` unread | aligned, hard `ValueError`, docstring + config comparability note, plumbed |

**Retraining note:** the mswr pad-alignment + IDWT-adjoint fixes change the model function
(for the better). Existing mswr checkpoints were trained against the shifted/corrupted
behavior; expect a fresh run to be needed for best results (same situation as the prior
B-1 wavelet fix). CSWIN zero-init bias affects fresh runs only.

## Key open items (roadmap; see chat report / agent analyses for full detail)

1. **[CSWIN, decisive for 256/512 stages]** Learnable additive bias as SDPA `attn_mask`
   excludes the flash kernel on CUDA; if the mem-efficient backend also rejects it
   (version/alignment dependent), the math kernel materializes ~2 GB/block at 482×512.
   Run `_audit_probes\cswin-arch-8.py` **on the training GPU** to settle the backend;
   options: drop additive bias (LePE already present) or FlexAttention (torch≥2.5).
2. **[CSWIN]** `split_sizes: [1,1,1]` — split_size=7 adds +8.2% padded tokens at patch
   128 for zero modeling effect (attention is per-row/per-column regardless). Fresh run.
3. **[CSWIN]** decoder1 runs pre-compression at C=192: 29.7% of params, ~35% of stage
   compute; mirroring decoder2 (compress first) frees ~1.07M params / ~6.4 GMAC.
4. **[mswr]** `landmark_pooling='learned'` (default) is a per-pixel MLP over a static
   dictionary — no global spatial mixing at all; `'adaptive'` is the candidate ablation.
5. **[mswr overfit-at-ep70]** within MST++ comparability: sweep the now-plumbed
   `drop_path` 0.1→0.2–0.3; EMA decay 0.999→{0.9995, 0.9999}; exempt layer-scale
   gammas/landmarks/rescale from weight decay; baseline overfits in a high-LR/zero-wd/
   no-EMA regime (LR still 88% of peak at ep70) — Run A's levers are the right ones.
6. **[both]** Benchmark-protocol hygiene vs MST++: report unclamped MRAE (now logged in
   mswr); CSWIN test script still clamps pred and tiles instead of full-image inference
   (`cswin_test_ntire.py`); `.mat` outputs are v5 (scipy) while ARAD tooling expects
   v7.3 (`utils.save_matv73` exists, unused); `mswr_test_ntire` retains ~3 GB of
   device-resident predictions on the 50-scene split.
7. **[CSWIN]** torch.compile the generator (8.2% of CPU self-time is layout copies;
   ~20 rearrange/permute round-trips per SSTB); AdamW norm/bias decay exemption.

## Tests / how to run

New: `mswr_v2\tests\test_audit3_fixes.py` (9 tests), `CSWIN v2\tests\test_train_generator_audit3.py` (6 tests).

```powershell
cd mswr_v2;    & '..\CSWIN v2\.venv-audit\Scripts\python.exe' -m pytest -q --basetemp=.tmp_pytest   # 146 passed, 17 skipped
cd 'CSWIN v2'; & '.\.venv-audit\Scripts\python.exe' -m pytest -q --basetemp=.tmp_pytest             # 124 passed, 1 skipped
```

Probes (evidence + reproduction): `_audit_probes\*.py`, run with the `.venv-audit` python.

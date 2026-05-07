# CSWIN v2 — Independent Re-Audit (v2)

Date: 2026-05-07
Auditor mode: independent re-audit (the prior `AUDIT_REPORT.md` was set aside).
Hardware: CPU only (Windows 11, Torch 2.11.0+cpu, 8 threads).

## 1. Critical paths & inspection plan

Entrypoints inspected:

- Training: [src/hsi_model/training_script_fixed.py](src/hsi_model/training_script_fixed.py) (primary), [src/hsi_model/train_optimized.py](src/hsi_model/train_optimized.py) (secondary).
- Inference: [smoke_infer.py](smoke_infer.py), [smoke_run.py](smoke_run.py), [src/hsi_model/utils/patch_inference.py](src/hsi_model/utils/patch_inference.py).
- Model: [model.py](src/hsi_model/models/model.py) → [generator_v3.py](src/hsi_model/models/generator_v3.py) + [discriminator_v2.py](src/hsi_model/models/discriminator_v2.py).
- Attention: [attention.py](src/hsi_model/models/attention.py) (`CSWinAttentionBlock`, `EfficientSpectralAttention`).
- Losses: [losses_consolidated.py](src/hsi_model/models/losses_consolidated.py) (`NoiseRobustLoss`, `SinkhornDivergence`, `SinkhornLoss`, `SAMLoss`, `ImprovedPerceptualLoss`, `ComputeSinkhornDiscriminatorLoss`).
- Data: [loaders.py](src/hsi_model/utils/data/loaders.py) → [mst_dataset.py](src/hsi_model/utils/data/mst_dataset.py) / [hf_arad_dataset.py](src/hsi_model/utils/data/hf_arad_dataset.py); [transforms.py](src/hsi_model/utils/data/transforms.py).
- Metrics: [metrics.py](src/hsi_model/utils/metrics.py).
- Utilities: [checkpoint.py](src/hsi_model/utils/checkpoint.py), [training_setup.py](src/hsi_model/utils/training_setup.py).

Behaviour was verified by writing `probes/probe_audit_v2.py` (independent reproductions of each suspected issue) and running them against the current tree before changing any code.

## 2. Executive summary

Verdict: **Partial** before this patch set. The repo trains and the prior audit closed several real bugs, but the independent re-audit found four additional problems that affect correctness or silent failure modes:

| # | Severity | Title | Status |
|---|----------|-------|--------|
| 1 | BLOCKER | `NoiseRobustLoss` returns a graph-disconnected `torch.tensor(1.0, requires_grad=True)` on NaN input — `backward()` is a silent no-op, the trainer's `isfinite(gen_loss)` guard cannot detect it. | **Fixed** |
| 2 | BLOCKER | Legacy `SinkhornLoss` log-space update has wrong math (`log_a = -ε · LSE(...)`) — returns 0.0023 against a reference 5.62 (3 orders of magnitude off); exported and importable. | **Fixed** |
| 3 | HIGH | `SinkhornDivergence` self-OT terms use `OT(X, X.detach())` — only half the entropic-debiasing gradient flows; divergence is biased. | **Fixed** |
| 4 | HIGH | `CSWinAttentionBlock` "relative position" bias along the long axis is **window-cyclic**, not relative — diagonals of the bias matrix have non-zero standard deviation where a true relative-position bias has std=0. | Documented (semantics-affecting; not patched) |
| 5 | MED | Generator iteration counter drifts: trainer calls `set_iteration(iteration)` *and* the forward auto-increments — counter advances by `accumulation_steps × n_replicas` per optimizer step. | **Fixed** (trainer-managed mode) |
| 6 | LOW | `compute_sam_value` uses `acos(clamp(., -1+ε, 1-ε))` which floors identical spectra at ~`sqrt(2ε)` rad. Inconsistent with `SAMLoss` (which the prior audit fixed to `atan2`). | **Fixed** |

After the patch set: **80 tests pass, 2 skipped** (was 70+2 before patches; +10 new tests). `smoke_run.py` exits cleanly on CPU.

## 3. Risk register with evidence

### BLOCKER #1 — NoiseRobustLoss NaN fallback is graph-disconnected

- **Evidence**: [losses_consolidated.py:794](src/hsi_model/models/losses_consolidated.py#L794) and `:839` returned `torch.tensor(1.0, device=pred.device, requires_grad=True)`. The probe verified post-`backward()`: `total.grad_fn is None`, `pred.grad is None`.
- **Why it matters**: When a single NaN slips through the generator forward (a routine event during early adversarial training), the loss is a finite `1.0`, so `if not torch.isfinite(gen_loss)` in [training_script_fixed.py:576](src/hsi_model/training_script_fixed.py#L576) doesn't trip. `gen_loss.backward()` runs successfully but produces no gradients. The optimizer step is a silent no-op and the LR/scaler accounting remains as if a real step happened. No log warning is emitted past the initial `error` line.
- **Fix**: Anchor the fallback to `nan_to_num(pred).sum() * 0 + 1.0`. Graph stays intact; `pred.grad` becomes a real (zero) tensor; sanitize first so `NaN * 0 = NaN` doesn't propagate. The probe now reports `grad_fn = AddBackward0` and `pred.grad` is a tensor of zeros.
- **Tradeoff**: One extra `nan_to_num + sum` per fallback (~microseconds). The optimizer step is still effectively a no-op, but observably so via grad norms.
- **Test**: `tests/test_audit_v2_fixes.py::test_noise_robust_loss_nan_input_keeps_graph_connected`.

### BLOCKER #2 — Legacy SinkhornLoss log-space math

- **Evidence**: [losses_consolidated.py](src/hsi_model/models/losses_consolidated.py) (pre-fix) computed `log_a = -ε · logsumexp(-C/ε + log_b)`, i.e. dual-potential scaling mistaken for log-probability scaling. Probe: `SinkhornLoss(real, real+1.0) = 0.0023` against `Reference Sinkhorn(real, real+1.0) = 5.62` for the same data. `SinkhornLoss(real, real)` returned `~2e-5` so the bug was masked at the diagonal.
- **Why it matters**: Although the active training scripts use `SinkhornDivergence`, `SinkhornLoss` is exported through `hsi_model.models.__init__` and importable as a public API. Anyone who imports it gets garbage OT costs.
- **Fix**: Rewritten as the standard log-domain Sinkhorn for uniform marginals: `log_u = log(a) - LSE(log_K + log_v)`, `log_v = log(b) - LSE(log_K + log_u)`. Probe post-fix: `(real, real+1) = 3.998`, `(real, real) = 2e-5`.
- **Test**: `test_sinkhorn_loss_legacy_returns_zero_on_identical_clouds`, `test_sinkhorn_loss_legacy_recovers_correct_magnitude_on_shift`.

### HIGH #3 — Sinkhorn divergence detach asymmetry

- **Evidence**: `SinkhornDivergence.forward` computed `ot_xx = self._sinkhorn_cost(X, X.detach())`. Probe measured `||∂OT(X, X.detach())/∂X|| ≈ 7.8e-3` on a 64-point 1-D cloud — a true Sinkhorn divergence at the diagonal must have ~zero gradient (Envelope theorem on `OT(P,P)`).
- **Why it matters**: The asymmetric detach only propagates *half* of the entropic-debiasing gradient through `X` (it differentiates one of the two `X` arguments). The result is a partially debiased divergence: empirically the value at the diagonal is correct (the probe shows `S_eps(X, X) = 0`), but the gradient is biased away from zero, which subtly misdirects the generator/discriminator updates near convergence.
- **Fix**: Compute `ot_xx = self._sinkhorn_cost(X, X)` symmetrically (no detach). Cost: the backward graph for the self-OT terms now flows through both `X` arguments — roughly 1.5–2× more autograd memory in the OT backward. Bounded by the existing `max_points` cap (default 1024).
- **Test**: `test_sinkhorn_divergence_at_diagonal_has_small_gradient` (asserts `||grad|| < 1e-3` at the diagonal vs the pre-fix 7.8e-3).

### HIGH #4 — CSWin "relative position" bias is window-cyclic (NOT patched)

- **Evidence**: Probe with deterministic table values `bias_table[i, j, h] = 100*i + 10*j + h`, split_size=4, 16-token long axis:
  ```
  expanded[0] first row : [330, 320, 310, 300, 330, 320, 310, 300, ...]   # cycles every s=4
  expanded[0] diagonal  : [330, 330, ...]                                 # ok
  Per-diagonal std       : 0, 0, 0, 21.9, 20.6, 15.1, 0, 20.0, ...        # mostly non-zero
  ```
- **Why it matters**: A true relative-position bias along W has the form `bias[i, j] = f(i - j)` and therefore is constant along each diagonal of the (W, W) attention matrix — std = 0 on every offset. The current `_expand_bias` tiles a `(s, s)` intra-window bias across W, so token 0 attends to tokens `0, s, 2s, …` with the **same** bias value. The model can still train (the bias is learned), but it learns a window-cyclic position prior, not a translation-invariant relative-position prior. This is inconsistent with the docstring's claim of being a relative-position bias.
- **Why it is NOT patched**: Replacing with a true `(2W-1)`-entry table changes the parameterization and likely changes training dynamics; downstream checkpoints would not be loadable. Recommended as a separate, opt-in change with a config flag (e.g. `cswin_bias_mode: 'window_cyclic' | 'relative_long_axis'`) and a fresh training run.
- **Recommended next step**: Add `cswin_bias_mode` flag and a unit test asserting per-diagonal std = 0 in `relative_long_axis` mode.

### MED #5 — Generator iteration counter drifts under set_iteration

- **Evidence**: Probe before fix:
  ```
  Initial _iteration_count: 0
  After step 1 (2 forwards): _iteration_count = 2
  After step 2 (2 forwards): _iteration_count = 4
  ```
  In [training_script_fixed.py:511](src/hsi_model/training_script_fixed.py#L511) the trainer calls `generator.set_iteration(iteration)` once per loop iteration, and forward then *also* increments `_iteration_count`. `iteration` is the loop counter, not the optimizer-step counter, so the counter drifts by `accumulation_steps` per optimizer step.
- **Why it matters**: `delayed_sigmoid` and `clamp_after_iters` thresholds key off the generator's internal counter. With `accumulation_steps=2`, sigmoid switches on at half the intended optimizer-step count. With DDP the drift compounds across replicas.
- **Fix**: Added `_iteration_externally_managed` flag in [generator_v3.py](src/hsi_model/models/generator_v3.py). Calling `set_iteration` flips it, and forward then skips the auto-increment. Legacy callers that never call `set_iteration` still get auto-increment.
- **Test**: `test_generator_counter_externally_managed_does_not_drift` asserts `[0, 0, 1, 1, 2, 2]` for an `accumulation_steps=2` simulation; pre-fix the same loop would have produced `[1, 2, 3, 4, 5, 6]`.

### LOW #6 — SAM eval inconsistent with SAM loss

- **Evidence**: [metrics.py](src/hsi_model/utils/metrics.py) `compute_sam_value` used `acos(clamp(., -1+ε, 1-ε))` with `ε = 1e-8`. `acos(1 − 1e-8) ≈ 1.4e-4 rad ≈ 0.008°` — small floor, but not zero. The earlier audit fixed `SAMLoss` (training-time loss) to use `atan2(||u − (u·v)v||, u·v)`; eval did not get the same fix.
- **Why it matters**: Reported SAM is biased upward by ~0.008° for perfect spectra, and inconsistent between training-time loss and eval-time metric.
- **Fix**: Switched `compute_sam_value` to the same `atan2` formulation. Probe confirms exact zero on identical inputs.
- **Test**: `test_compute_sam_value_zero_on_identical_spectra`.

## 4. Issues observed but not patched (out of scope or ambiguous)

- **R1 missing 0.5 factor**: [training_script_fixed.py:333](src/hsi_model/training_script_fixed.py#L333) computes `(grad.pow(2).sum([1,2,3])).mean() * gamma`; standard R1 (Mescheder 2018) is `(γ/2) · E[‖∇D(x)‖²]`. Effectively 2× stronger penalty. Configurable via `r1_gamma`, so practically benign — flag only.
- **R1 conditioning on HSI only**: applies penalty only to the HSI argument of `D(rgb, hsi)`. For paired conditional GANs the standard form penalizes both inputs. Design choice; flag only.
- **MRAE near-zero denominator**: [metrics.py compute_mrae](src/hsi_model/utils/metrics.py) uses `target + ε` (additive ε on a possibly negative number). For non-negative HSI in [0, 1] this is fine, but a dark-pixel near zero will inflate the metric. Common variant uses `|target| + ε`. Low impact for ARAD-1K.
- **`EfficientSpectralAttention` naming**: attends over per-head channel dim of mean-pooled features, not over true spectral bands. Not a bug but the name is misleading. Documentation issue.
- **CSWin stripe attention scope**: each row in a stripe attends only over the long axis independently of other rows in the stripe; the standard CSWin attends over the full `s × W` stripe. Different by design; OK.
- **DataLoader `worker_init_fn=lambda` on Windows**: lambdas do not pickle for spawn-mode multiprocessing. Linux fork is fine. If Windows DDP is needed, refactor to a module-level callable.
- **`sinkhorn_loss_clip` and `clamp(divergence, 0, 1e4)`**: the Sinkhorn divergence can be slightly negative due to entropic-bias residuals before convergence; clamping to ≥ 0 is a defensible numerical guard but discards information.

## 5. Patches implemented

All in this PR / commit. Files changed:

- [src/hsi_model/models/losses_consolidated.py](src/hsi_model/models/losses_consolidated.py)
  - `NoiseRobustLoss.forward`: NaN fallback now `nan_to_num(pred).sum()*0 + 1` (graph-connected).
  - `SinkhornDivergence.forward`: removed `.detach()` from self-OT terms.
  - `SinkhornLoss.forward`: rewritten as standard log-domain Sinkhorn for uniform marginals.
  - Added `import math`.
- [src/hsi_model/models/generator_v3.py](src/hsi_model/models/generator_v3.py)
  - Added `_iteration_externally_managed` flag; `set_iteration` flips it; forward skips auto-increment when flagged.
- [src/hsi_model/utils/metrics.py](src/hsi_model/utils/metrics.py)
  - `compute_sam_value`: `acos(clamp)` → `atan2(||orth||, cos)` form.

## 6. Tests added & how to run

New file: [tests/test_audit_v2_fixes.py](tests/test_audit_v2_fixes.py) — 10 tests covering each patch.

Run from `CSWIN v2/`:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& '.\.venv-audit\Scripts\python.exe' -m pytest -q
```

Result on this commit: `80 passed, 2 skipped`.

Reproducing the original probes (pre-fix vs post-fix evidence):

```powershell
& '.\.venv-audit\Scripts\python.exe' '.\probes\probe_audit_v2.py'
```

Smoke run (CPU):

```powershell
& '.\.venv-audit\Scripts\python.exe' '.\smoke_run.py'
# expected: smoke_run_ok device=cpu train_seconds=~0.4 disc_loss=-0.09 gen_loss=~1.20
```

## 7. Benchmark note (CPU, 8 threads, Torch 2.11.0)

Captured by `probes/bench_audit_v2.py` (5-iteration average):

| Component | Post-fix ms/iter | Notes |
|-----------|------------------|-------|
| Generator forward+backward (1×3×64×64, base=32) | 311 ms (auto-inc) / 304 ms (set_iteration) | The counter-drift fix has no measurable cost. |
| `SinkhornDivergence(X, Y)` 1024×1 cloud | 203 ms | ~1.5–2× the pre-fix detach form (full backward graphs for the self-OT terms). |
| Legacy `SinkhornLoss` 8×4 / 200 iters | 18 ms | Math fixed; cost roughly unchanged. |
| `NoiseRobustLoss` healthy fwd+bwd | 36 ms | |
| `NoiseRobustLoss` NaN-fallback | 0.7 ms | Was a silent no-op; now a graph-connected zero-grad step. |

### Top 3 bottlenecks (CPU)

1. **CSWin stripe attention** dominates the generator forward/backward. The current implementation falls back to `F.scaled_dot_product_attention` when available, which on CPU is BLAS-bound. For tiles of size `(s × W)` the attention is `O(W²)` per stripe-row. Suggested mitigations: (a) lower `split_size` for short stripes during training; (b) consider windowed (intra-stripe) attention for very large W; (c) on GPU, FlashAttention's CSWin-friendly variants further reduce memory pressure.
2. **`SinkhornDivergence` symmetric self-OT** is now the second largest cost in the loss step. With 1024-point clouds and 50 iterations it is ~200 ms/iter on CPU. If correctness is acceptable, lower `sinkhorn_iters` (e.g., 25) and `sinkhorn_max_points` (e.g., 512) for early-warmup epochs.
3. **`isfinite` checks scattered through the model**. The `NaNSafeAttention` wrapper already gates checks behind `_check_freq=100` in training; many of the discriminator's per-block checks still fire on every forward and force a CPU sync on GPU runs. Consider promoting `check_freq` to a global config and applying it consistently across discriminator and generator paths.

## 8. Residual risks

- **Real ARAD/MST data not available** in this environment: the data-pipeline shape/orientation checks reproduce the prior audit's findings (transpose `[0,2,1]` after h5py read is correct for `.mat v7.3` cubes), but actual scene-level alignment between RGB and HSI is unverified.
- **GPU not available**: bf16/fp16 stability of the SDPA + symmetric Sinkhorn path remains unverified. Recommended: rerun the test suite on a CUDA host with `mixed_precision=True`.
- **CSWin bias mathematical defect** is documented but not patched (HIGH #4). If you intend to publish numbers as "CSWin", this is the most important remaining item.
- **ERGAS metric still missing** from `utils/metrics.py` — flagged by the prior audit; not added in this pass.
- **`worker_init_fn=lambda`** breaks DataLoader on Windows spawn-mode multiprocessing. Linux fork is unaffected.

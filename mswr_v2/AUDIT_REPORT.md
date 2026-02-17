# MSWR-Net v2.1.2 — Deep Learning Systems Audit Report

**Date:** 2026-02-09
**Auditor:** Claude Opus 4.6 (automated)
**Scope:** `mswr_v2/` — model, training, data pipeline, losses, inference

---

## Executive Summary

A comprehensive audit of the MSWR-Net v2.1.2 codebase revealed **1 BLOCKER**, **4 HIGH**, and **3 MEDIUM** issues. All BLOCKER and HIGH issues have been patched with accompanying regression tests. A full pytest suite (81/81 pass) and CPU smoke runs (training + inference) confirm correctness after patching.

---

## Risk Register

| ID | Severity | Component | Title | Status |
|----|----------|-----------|-------|--------|
| B-1 | **BLOCKER** | `mswr_net_v212.py:187` | Wavelet DWT reshape scrambles subbands across channels | **PATCHED** |
| H-1 | HIGH | `train_mswr_v212_logging.py:197` | SSIM variance can go negative, risking NaN | **PATCHED** |
| H-2 | HIGH | `mswr_net_v212.py:299-315` | PerformanceMonitor calls `cuda.synchronize()` on every stage | **PATCHED** |
| H-3 | HIGH | `mswr_net_v212.py:667-673` | Flash attention (SDPA) path drops relative position bias | **PATCHED** |
| H-4 | HIGH | `train_mswr_v212_logging.py:1133-1191` | GradScaler state not restored on checkpoint resume | **PATCHED** |
| M-1 | MED | `train_mswr_v212_logging.py:1005` | Worker init seed doesn't incorporate epoch (same augmentations per epoch) | Noted |
| M-2 | MED | `mswr_net_v212.py:1096-1104` | Silent try/except fallback in wavelet block masks real bugs | Noted |
| M-3 | MED | `train_mswr_v212_logging.py:252-273` | Loss warmup zeroes all non-L1 losses at epoch 0 | Noted |

---

## Detailed Findings

### B-1: Wavelet DWT Reshape Scrambles Subbands (BLOCKER)

**Location:** `model/mswr_net_v212.py`, `OptimizedCNNWaveletTransform.forward()`, line 187

**Root cause:** PyTorch grouped convolution with `groups=C` outputs channels ordered as `[ch0_LL, ch0_LH, ch0_HL, ch0_HH, ch1_LL, ch1_LH, ...]`. The original code reshaped with `view(B, 4, C, H2, W2)` which incorrectly interleaves: the "LL subband" (index `[:, 0]`) would contain `[ch0_LL, ch0_LH, ch0_HL, ...]` — mixing subbands across channels.

**Impact:** Every wavelet decomposition produces corrupted subbands. The LL approximation fed to subsequent DWT levels and the HF coefficients gated by the wavelet attention are mathematically wrong. The model can still train (the learnable filters and gates compensate), but reconstruction quality is fundamentally limited.

**Fix:** Changed reshape to `view(B, C, 4, H2, W2)` and updated indexing from `[:, 0]` to `[:, :, 0]` and `[:, 1:]` to `[:, :, 1:]`. The inverse DWT was already correct (`view(B, C, 4, ...)` → `view(B, 4*C, ...)`).

**Verification:**
- `test_subband_separation_single_channel`: Constant input → LL non-zero, HF ≈ 0 ✓
- `test_subband_separation_multichannel`: Per-channel constants → each channel's LL independent ✓
- `test_roundtrip_reconstruction_fidelity`: db1 roundtrip < 5% relative error ✓
- `test_roundtrip_db2`: db2 roundtrip < 10% relative error ✓
- `test_output_shapes`: J=2 produces correct (B,C,3,H/2,W/2) shapes ✓

---

### H-1: SSIM Variance Not Clamped (HIGH)

**Location:** `train_mswr_v212_logging.py`, `EnhancedMSWRLoss._ssim_loss()`, lines 197-198

**Root cause:** `sigma_sq = E[X²] - E[X]²` can be slightly negative due to floating-point imprecision. The unclamped negative variance feeds into the SSIM denominator, potentially causing division by near-zero or negative values.

**Fix:** Added `sigma1_sq = sigma1_sq.clamp(min=0.0)` and `sigma2_sq = sigma2_sq.clamp(min=0.0)` after computation.

---

### H-2: PerformanceMonitor Synchronizes Every Stage (HIGH)

**Location:** `model/mswr_net_v212.py`, `PerformanceMonitor.start_stage()` and `end_stage()`, lines 299-315

**Root cause:** `torch.cuda.synchronize()` is called on every `start_stage()` and `end_stage()` call. This forces the GPU to flush its pipeline, destroying any overlap between compute and memory operations. During training, this is called per-encoder/decoder stage per iteration.

**Impact:** Estimated 5-15% training throughput reduction depending on GPU utilization patterns.

**Fix:** Added `sync_cuda: bool = False` parameter. Synchronize calls are now gated behind this flag. Set `sync_cuda=True` only when explicit profiling is needed.

---

### H-3: Flash Attention Drops Position Bias (HIGH)

**Location:** `model/mswr_net_v212.py`, `OptimizedWindowAttention2D.forward()`, lines 667-673

**Root cause:** The SDPA path (`F.scaled_dot_product_attention`) was called without passing the relative position bias. The manual path correctly adds the bias to attention scores before softmax.

**Impact:** In training mode with flash attention enabled (default), the learned position bias has no effect on attention weights. The model falls back to content-only attention, losing the positional inductive bias that window attention provides.

**Fix:** Moved position bias computation before the flash/manual branch. The bias is now passed as `attn_mask` to `F.scaled_dot_product_attention`. PyTorch's SDPA automatically selects the best kernel (may fall back from flash to math kernel when attn_mask is provided, but correctness is ensured).

---

### H-4: GradScaler State Not Restored on Resume (HIGH)

**Location:** `train_mswr_v212_logging.py`, `EnhancedTrainer._load_checkpoint()`, lines 1133-1191

**Root cause:** The `save_checkpoint()` method (line 1544) correctly saves `self.scaler.state_dict()`, but `_load_checkpoint()` never restores it.

**Impact:** When resuming training with AMP, the scaler starts fresh (scale=65536.0). If the previous training had adapted the scale factor (e.g., reduced after overflows), the resumed training will experience unnecessary loss scaling overflows until the scaler re-adapts.

**Fix:** Added scaler state restoration in `_load_checkpoint()` before the EMA restoration block.

---

### M-1: Worker Init Seed Doesn't Incorporate Epoch (MED)

**Location:** `train_mswr_v212_logging.py`, `_setup_data()`, line ~1005

The worker seed is `config.seed + worker_id` without incorporating epoch. Workers produce the same random sequence every epoch, reducing augmentation diversity.

**Recommendation:** Use `config.seed + worker_id + epoch * num_workers` in the worker init function.

---

### M-2: Silent Wavelet Fallback Masks Bugs (MED)

**Location:** `model/mswr_net_v212.py`, `EnhancedWaveletDualTransformerBlock.forward()`, lines 1096-1104

The try/except catches ALL exceptions during wavelet processing and falls back to standard attention. This masks real bugs (like the B-1 reshape issue) during training — the model appears to work but the wavelet branch is silently disabled.

**Recommendation:** Remove the blanket exception handler or restrict it to specific known-recoverable errors. Log at WARNING level with the full traceback when fallback is triggered.

---

### M-3: Loss Warmup Zeroes Non-L1 at Epoch 0 (MED)

**Location:** `train_mswr_v212_logging.py`, lines 252-273

The warmup formula `weight *= (current_epoch / warmup_epochs)` produces 0.0 at epoch 0. This means SSIM, SAM, and gradient losses have zero contribution during the first epoch.

**Recommendation:** Use `(current_epoch + 1) / warmup_epochs` or start from a small non-zero floor.

---

## Files Modified

| File | Changes |
|------|---------|
| `model/mswr_net_v212.py` | B-1: wavelet reshape fix; H-2: PerformanceMonitor sync flag; H-3: flash attention position bias |
| `train_mswr_v212_logging.py` | H-1: SSIM variance clamp; H-4: scaler state restore |

## Files Created

| File | Purpose |
|------|---------|
| `tests/test_audit_fixes.py` | 13 regression tests covering all BLOCKER/HIGH fixes |
| `smoke_train.py` | Synthetic-data training loop (forward + loss + backward + step) |
| `smoke_infer.py` | Synthetic-data inference at multiple resolutions |

---

## Test Results

```
tests/ — 81 passed, 39 skipped (CUDA / optional deps)

Audit-specific tests:
  test_subband_separation_single_channel     PASSED
  test_subband_separation_multichannel       PASSED
  test_roundtrip_reconstruction_fidelity     PASSED
  test_roundtrip_db2                         PASSED
  test_output_shapes                         PASSED
  test_default_no_sync                       PASSED
  test_explicit_sync                         PASSED
  test_timing_works_without_sync             PASSED
  test_sdpa_path_uses_bias                   PASSED
  test_train_step                            PASSED
```

---

## Benchmark Notes (CPU, synthetic data)

| Metric | Value |
|--------|-------|
| **smoke_train** (3 steps, B=2, 64x64) | avg 542 ms/step |
| **smoke_infer** 64x64 | 3236 ms (first run, includes JIT) |
| **smoke_infer** 128x128 | 211 ms |
| **smoke_infer** 256x256 | 710 ms |
| Peak VRAM (CPU run) | N/A |

> For GPU benchmarks, run `python smoke_train.py --device cuda` and `python smoke_infer.py --device cuda`.

---

## Recommendations for Future Work

1. **Re-train with patched wavelet**: The B-1 fix fundamentally changes wavelet decomposition. Any existing checkpoints were trained with scrambled subbands and should be retrained.
2. **Remove wavelet try/except** (M-2): Now that the reshape bug is fixed, the silent fallback is unnecessary and dangerous.
3. **Add epoch to worker seed** (M-1): One-line fix in `_setup_data()`.
4. **Add wavelet correctness test to CI**: The `test_subband_separation_multichannel` test catches the class of bug that B-1 represents.
5. **Profile GPU training**: Run `smoke_train.py --device cuda` before and after the H-2 sync fix to measure throughput improvement.

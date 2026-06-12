# MSWR v2 Bottleneck and Optimization Audit

Audit date: 2026-06-12

Environment: Windows 11, Python 3.11.14, PyTorch 2.11.0+cpu. CUDA-only
behavior is identified from code but not benchmarked.

## 1. Inferred Task and Model Family

- Intended task: RGB-to-hyperspectral reconstruction.
- Input: `B x 3 x H x W`, RGB normalized with the MST++ per-image min/max recipe.
- Output: `B x 31 x H x W`, hyperspectral reflectance cube.
- Model family: U-shaped CNN/Transformer hybrid with DWT/IDWT branches.
- Spatial modeling: local window attention plus landmark attention.
- Spectral modeling: optional MST++-style channel attention, enabled by the canonical config.
- Primary metrics: MRAE, RMSE, PSNR, SAM, and optional SSIM.
- Canonical training shape: batch 20, `128 x 128` patches, 1000 logical steps/epoch.
- Hardware target: CUDA GPU with AMP, channels-last tensors, EMA, and optional checkpointing.

## 2. Critical Paths & Profiling Plan

Call graph:

`dataloader.py` -> RGB/HSI preprocessing -> `IntegratedMSWRNet.forward` ->
wavelet attention blocks -> output projection -> MRAE/enhanced loss -> backward ->
optimizer/scheduler/EMA -> validation -> checkpoint -> inference or NTIRE export.

Critical symbols:

- Data: `TrainDataset`, `ValidDataset` in `dataloader.py`.
- Model: `IntegratedMSWRNet.forward` in `model/mswr_net_v212.py`.
- Blocks: `EnhancedWaveletDualTransformerBlock`,
  `EnhancedDualAttention2D`, `SpectralMSA2D`.
- Training: `EnhancedTrainer.train_epoch`, `validate`, `_load_checkpoint`.
- Inference: `MSWRInference._process_full`, `_process_tiled`.
- Evaluation: `NTIRETestEngine.test_single_image`, `MetricsCalculator`.

Executed plan:

1. Run the complete test suite and synthetic train/inference smoke tests.
2. Probe attention modes, landmark spatial influence, checkpoint calls, wavelet fidelity,
   channel-width validation, and accumulation boundaries.
3. Profile canonical base inference with `torch.profiler`.
4. Patch confirmed correctness and high-impact implementation defects.
5. Re-run tests and repeat microbenchmarks.

## 3. Bottleneck Summary

| severity | category | file:line | bottleneck | impact | proposed fix |
|---|---|---|---|---|---|
| HIGH, fixed | Implementation | `model/mswr_net_v212.py:988` | `attention_type` was a no-op; all four choices built and ran the same branches | Misleading ablations, wasted parameters/compute | Construct and execute only the selected branches; define `hybrid` as dual plus spectral |
| HIGH, fixed | Training | `train_mswr_v212_logging.py:1678` | Partial gradient-accumulation groups were discarded | Up to `accum-1` batches lost per epoch; zero updates if epoch batches `< accum` | Scale by actual group size and step on the final batch |
| HIGH, fixed | Quality/Training | `train_mswr_v212_logging.py:2081` | Best checkpoints used clamped MRAE while MST++ uses raw predictions | Benchmark mismatch and potentially wrong model selection | Select/early-stop on raw MRAE; retain clamped metrics for reporting |
| HIGH, fixed | Implementation | `train_mswr_v212_logging.py:1546` | Resume allowed severely mismatched checkpoints with `strict=False` | Training could continue from mostly random weights | Reject loads with more than 25% missing model keys |
| MEDIUM, fixed | Architecture | `model/mswr_net_v212.py:1199` | Wavelet gate initialized at `sigmoid(1)=0.731` | About 23.2% relative feature reconstruction error before learned processing | Initialize final gate projection to zero weights and bias 4.0 |
| MEDIUM, fixed | Speed | `model/mswr_net_v212.py:1210` | Internal checkpoints ran even when `use_checkpoint=False` | Hidden recomputation and false configuration semantics | Gate all checkpoint calls on `use_checkpoint` and avoid nested block/FFN checkpointing |
| MEDIUM, fixed | Implementation | `model/mswr_net_v212.py:482` | Only base width was validated against `num_heads` | Invalid custom expansion failed deep in `einops` | Validate every encoder width and decoder inversion at config creation |
| HIGH, open | Architecture/Quality | `model/mswr_net_v212.py:860` | Default learned landmarks are static K/V vectors, not pooled image landmarks | The "global" branch performs no spatial mixing | Ablate `uniform` and `adaptive`; redesign content-dependent low-rank pooling |
| MEDIUM, open | Speed | `mswr_test_ntire.py:176,273` | SSIM and per-band metrics loop over all 31 bands | Many small kernels and host synchronizations during evaluation | Vectorize reductions over the channel dimension |
| MEDIUM, open | Speed/Inference | `mswr_inference.py:739` | Tiled inference processes one tile at a time | Low GPU occupancy and repeated launch overhead | Batch equal-sized tiles and stream merge accumulation |
| MEDIUM, open | Training | `train_mswr_v212_logging.py:859` | Weight decay exemption only matches names containing `norm` or `bias` | Layer scales, spectral temperature, and positional bias are decayed | Add explicit no-decay groups for `gamma*`, `rescale`, and positional bias |
| GPU check | Memory/Speed | `model/mswr_net_v212.py:769,892` | SDPA backend eligibility is hardware/dtype dependent | Manual fallback can materialize large attention matrices at full resolution | Profile on target GPU; record selected SDPA backend and peak memory |

## 4. Detailed Findings

### 4.1 Attention mode configuration was non-functional

Evidence:

- Before the patch, `EnhancedDualAttention2D` always instantiated and executed both
  spatial branches.
- Seeded probes produced identical output statistics and exactly 184,072 parameters
  for `window`, `landmark`, `dual`, and `hybrid`.

Why it matters:

- Reported attention ablations did not change the model.
- Single-branch deployment modes paid dual-branch memory and compute.

Implemented minimal fix:

- `window`: local window branch only.
- `landmark`: landmark branch only.
- `dual`: both branches with configured fusion.
- `hybrid`: dual spatial branches plus spectral attention.

Stronger alternative:

- Replace the enum with explicit branch booleans and a typed fusion configuration.

Impact/tradeoff:

- Tiny-model parameters now range from 145,864 (`window`) to 211,092 (`hybrid`).
- Non-dual checkpoints labeled with the old modes represented different behavior and
  should be treated as legacy dual models.

### 4.2 Partial accumulation groups were dropped

Evidence:

- The optimizer stepped only when `(i + 1) % accumulation_steps == 0`.
- For 5 batches with accumulation 4, batch 5 was discarded.
- For 3 batches with accumulation 4, no optimizer step occurred.

Implemented fix:

- The last batch is an accumulation boundary.
- Loss scaling uses the actual final group size.
- Scheduler steps per epoch use `ceil(N / accumulation_steps)`.

Tradeoff:

- Runs using non-divisible logical epoch lengths now perform one additional optimizer
  update per epoch, which is the intended behavior.

### 4.3 Checkpoint selection used a different MRAE domain than MST++

Evidence:

- Validation computed both raw and clamped MRAE but `best_mrae` used the clamped value.
- Official MST++ validation evaluates raw output after the 128-pixel border crop.

Implemented fix:

- `selection_mrae` is raw/unclamped MRAE.
- Best raw/EMA weights and minimization early stopping use that value.
- Clamped MRAE, RMSE, PSNR, and SAM remain available for reflectance-domain reporting.

Tradeoff:

- Best-checkpoint names from new runs are benchmark-comparable but may differ from older
  runs selected on clamped MRAE.

### 4.4 Wavelet branch did not start near identity

Evidence:

- The documented soft-identity gate was `sigmoid(1)=0.731`.
- A db2 DWT/IDWT probe with this gate produced 23.17% relative MAE.

Implemented fix:

- Zero final gate weights and bias 4.0, yielding `sigmoid(4)=0.982`.
- The same probe now produces 1.55% relative MAE.

Stronger alternative:

- Parameterize the gate as `1 - alpha * sigmoid(z)` with small learnable `alpha`.

Tradeoff:

- New training runs preserve high-frequency information at initialization.
- Existing checkpoints are unaffected because checkpoint weights overwrite initialization.

### 4.5 Default learned landmarks are not global

Evidence:

- Learned mode expands a static `L x C` parameter tensor as K/V.
- Perturbing one input pixel changed 1/64 output pixels in the standalone branch.
- The same probe changed 64/64 pixels for adaptive and uniform pooling.

Why it matters:

- The branch is a per-pixel query against a static dictionary, not image-wide context.
- It overlaps with channel gating while carrying a "global" name and compute cost.

Minimal improvement:

- Train controlled `learned` vs `uniform` vs `adaptive` ablations.

Stronger alternative:

- Pool image tokens into spatially distributed landmarks with content-dependent weights,
  or use low-resolution global attention.

Tradeoff:

- Adaptive pooling increased tiny-model parameters from 184,072 to 285,512.
- Uniform pooling is cheap but samples fixed positions and may alias.

### 4.6 Runtime profile

Canonical base at `128 x 128`, CPU:

- 101 MKLDNN convolutions consumed 37.1% self CPU time.
- Softmax consumed 15.1%.
- Batched matrix multiplication consumed 6.9%.
- Tensor copies consumed 6.7%.
- GELU consumed 6.4%.

The encoder owns 66.8% of parameters, the decoder blocks 15.9%, and all remaining
projections/downsampling/upsampling modules 17.3%.

Quality-preserving optimization order:

1. Batch tiled inference and vectorize evaluation metrics.
2. Verify CUDA SDPA kernels and channels-last behavior.
3. Consider convolution fusion or `torch.compile` only after GPU profiling.
4. Do not reduce width until landmark and spectral ablations establish the quality ceiling.

## 5. Patches Implemented

Changed files:

- `model/mswr_net_v212.py`
  - Functional attention modes.
  - Full stage-width validation.
  - Near-identity wavelet gate initialization.
  - Checkpoint behavior tied to `use_checkpoint`.
  - Factory overrides no longer collide with hard-coded defaults.
- `train_mswr_v212_logging.py`
  - Correct partial gradient accumulation.
  - Raw-MRAE checkpoint selection.
  - Large checkpoint mismatch rejection.
- `README.md`
  - Attention-mode and validation semantics.
- `tests/test_mswr_v4_regressions.py`
  - Regression coverage for all fixes above.

## 6. Tests Added + How to Run

Added 14 regression cases covering:

- Attention branch construction and output differences.
- Zero checkpoint calls when checkpointing is disabled.
- Factory override plumbing.
- Invalid expanded widths.
- Raw-MRAE source selection.
- Resume mismatch rejection.
- Near-identity wavelet gates.
- Partial accumulation scaling and boundaries.

Run:

```powershell
cd mswr_v2
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' -m pytest -q
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' smoke_train.py --steps 3 --device cpu
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' smoke_infer.py --device cpu
```

Result: 183 passed, 2 skipped. The skipped tests require CUDA.

## 7. Benchmark Results

| measurement | before | after |
|---|---:|---:|
| Full tests | 169 passed, 2 skipped | 183 passed, 2 skipped |
| Checkpoint calls with `use_checkpoint=False` | 6 per tiny train step | 0 |
| db2 gated wavelet relative MAE at initialization | 23.17% | 1.55% |
| Optimizer steps, 5 batches / accumulation 4 | 1, with 1 batch discarded | 2, no batch discarded |
| Optimizer steps, 3 batches / accumulation 4 | 0 | 1 |
| Tiny `window` parameters | 184,072, same as all modes | 145,864 |
| Tiny `landmark` parameters | 184,072, same as all modes | 149,372 |
| Tiny `dual` parameters | 184,072 | 184,072 |
| Tiny `hybrid` parameters | 184,072, same behavior as dual | 211,092 |

Post-patch absolute measurements:

| measurement | result |
|---|---:|
| Canonical base parameters (`dual + spectral`) | 3,035,684 |
| Canonical base `128 x 128` median CPU inference | 245.5 ms |
| Profiler-estimated canonical base compute | 14.94 GFLOPs |
| Tiny `64 x 64`, batch 1 median CPU train step | 73.5 ms |
| Synthetic smoke train, batch 2 `64 x 64` | 304.8 ms/step |

Notes:

- CPU timings used four PyTorch threads for microbenchmarks.
- FLOPs are profiler estimates; `fvcore` is unavailable in the audit environment.
- Peak CUDA memory, AMP throughput, and Flash/efficient SDPA selection were not measurable.
- No ARAD-1K data or trained checkpoint was available, so quality metrics could not be
  benchmarked before/after.

## 8. Optimization Roadmap

Immediate low-risk:

1. Run the canonical config on the target GPU and record latency, peak allocated memory,
   selected SDPA backend, and MRAE for raw and EMA weights.
2. Vectorize SSIM and per-band metric reductions.
3. Batch equal-sized inference tiles.
4. Add explicit optimizer no-decay groups for layer scales, spectral temperature, and
   positional bias.

Medium-risk architecture work:

1. Ablate landmark pooling modes with identical seeds and schedules.
2. Compare `dual + spectral` against `window + spectral`; the learned landmark branch may
   add little image context.
3. Sweep drop path, EMA decay, and weight decay using raw MRAE selection.
4. Profile checkpoint-block placement instead of checkpointing a fixed stage fraction.

High-risk/high-reward:

1. Replace static landmarks with content-dependent low-rank spatial pooling.
2. Factorize high-resolution convolutions in the input/output projections.
3. Evaluate a spectral-first U-Net with global attention only at low resolution.
4. Distill the canonical model into the corrected `window + spectral` deployment variant.

## 9. Final Verdict

**FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED**

The shipped model is mathematically capable of RGB-to-HSI reconstruction and now has
sounder configuration, accumulation, checkpoint, initialization, and model-selection
behavior. Its largest unresolved limitation is that the default landmark branch is not
actually global. Deployment readiness is incomplete until CUDA memory, SDPA kernel
selection, AMP stability, and real ARAD-1K quality are measured on target hardware.

Protocol references:

- Official MST++ dataset preprocessing:
  https://github.com/caiyuanhao1998/MST-plus-plus/blob/master/train_code/hsi_dataset.py
- Official MST++ raw-output validation:
  https://github.com/caiyuanhao1998/MST-plus-plus/blob/master/train_code/train.py

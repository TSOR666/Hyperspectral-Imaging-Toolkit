# 1. Inferred Task and Model Family

CSWIN v2 reconstructs a 31-band hyperspectral cube from RGB input:

- Input: `(B, 3, H, W)`.
- Output: `(B, 31, H, W)`.
- Active trainer: `src/hsi_model/train_generator.py`.
- Model: hierarchical U-Net/Transformer hybrid with spectral MSA, bounded
  local/global CSWin attention, gated FFNs, skip connections, and learned
  PixelUnshuffle/PixelShuffle sampling.
- Primary metric/objective: MRAE; secondary metrics: PSNR, SSIM, SAM, RMSE.
- Target hardware: CUDA training and tiled CUDA inference. This audit host
  exposed CPU-only Torch `2.11.0+cpu`, so CUDA backend and peak-memory claims
  remain explicitly unverified.

# 2. Critical Paths & Profiling Plan

Training:

`MST/HF pair -> paired crop/augmentation -> DataLoader -> generator -> MRAE ->
AMP backward -> Adam -> cosine schedule -> EMA -> tiled validation -> atomic
checkpoint`

Inference:

`strict checkpoint load -> optional EMA -> overlapping tile batches -> FP32
blend -> clamp -> centered 226x256 metrics/export`

Audit execution:

- Regression-tested all prior fixes with the complete suite.
- Ran synthetic training and tiled-inference smoke paths.
- Profiled active-model parameters, 64/128 forward latency, top-level stages,
  CPU operators, metric semantics, MRAE gradients, and full-frame versus tiled
  predictions.
- Added regressions for tiled validation, DDP-wrapper bypass, empty validation,
  bounded OOM retries, fixed ARAD crop geometry, partial EMA, and FP32 stitching.

# 3. Bottleneck Summary

| severity | category | file:line | bottleneck | impact | proposed fix |
|---|---|---|---|---|---|
| HIGH, fixed | Quality/Inference | `train_generator.py:355`, `cswin_test_ntire.py:382` | Validation used full-frame execution while deployment used 128 tiles; token-dependent attention changed the model graph | Biased checkpoint selection; synthetic relative output MAE 2.50% | Validate with the deployment tile size/overlap |
| HIGH, fixed | Training | `train_generator.py:301,340-350` | Uneven DDP validation forwarded through DDP and reduced rank-dependent metric dictionaries | Collective mismatch/deadlock for unequal or empty shards | Bypass DDP for inference and reduce a fixed metric schema |
| HIGH, fixed | Quality/Data | `utils/data/transforms.py:62-71` | Training used fixed 128px borders while the tester used a centered 226x256 ARAD window | Metrics diverged on non-482x512 images; small inputs returned sentinels | Use the same fixed centered window, with aligned full-image fallback |
| HIGH, fixed | Implementation | `utils/inference.py:110-144`, `utils/training_setup.py:545` | Partial EMA state could produce mixed or random-shadow weights | Corrupt validation/inference after incomplete legacy checkpoints | Rebase resume EMA on loaded weights; reject partial EMA at inference |
| MEDIUM, fixed | Training | `train_generator.py:488,781-800` | Persistent OOM retried forever without changing geometry | Hung jobs and wasted allocations | Abort after configurable consecutive OOMs |
| MEDIUM, fixed | Quality/Inference | `utils/patch_inference.py:47-53,165` | Inference forced FP16 and rounded FP32 overlap blends back to half | Avoidable precision loss; mismatch with BF16 training policy | Support explicit/auto AMP dtype and return FP32 reconstruction |
| HIGH, open | Math/Training | `losses_consolidated.py:163`, `config.yaml:89` | Pure MRAE with epsilon `1e-8` has extreme dark-pixel gradients | Synthetic gradient reaches `1e8` at zero target and relies on global clipping | Ablate stabilized MRAE or MRAE+L1 before changing benchmark recipe |
| HIGH, open | Architecture/Speed | `models/generator_v3.py:504-509` | Decoder1 processes concatenated features at 4x base width before compression; decoder2 has four full-resolution blocks | Decoder1/decoder2 consumed 26.4%/37.1% of hooked CPU stage time | Compress before decoder1; ablate shallower high-resolution decoder |
| MEDIUM, open | Speed | `models/generator_v3.py:166-176` | Four NHWC LayerNorm permute/copy cycles per transformer block | `aten::copy_` was 9.08% of profiled CPU self-time over 48 LayerNorm calls | Benchmark a fused/channel-first LayerNorm under CUDA/compile |
| MEDIUM, open | Memory/Inference | `utils/patch_inference.py:280-283` | All output tiles are retained before stitching | Tile memory grows with image area; material on multi-megapixel inputs | Stream each output batch into preallocated FP32 accumulators |
| GPU check | Memory/Speed | `models/attention.py:439-445` | Learned additive bias is passed as SDPA `attn_mask` | Flash/memory-efficient backend eligibility is hardware and dtype dependent | Profile selected SDPA backend and peak memory on the training GPU |

# 4. Detailed Findings

## 4.1 Validation now executes the deployed model

Evidence:

- NTIRE inference always used `PatchInference` with 128px tiles.
- The validator previously called the full frame directly.
- `cswin_global_tokens=1024` means a 128px tile uses global attention at the
  32x32 bottleneck, while a full ARAD frame uses local attention there.
- A seeded small-CSWIN probe at 160x176 measured full-versus-tiled relative MAE
  `0.0250` and max absolute difference `0.2127`.

Implemented fix:

- Active config enables tiled validation with the same `128/16` tile/overlap
  defaults as `cswin_test_ntire.py`.

Tradeoff:

- A 240x256 CPU validation probe was 1.62x slower tiled, but its MRAE changed by
  `0.000775`; semantic correctness is worth the validation-only cost.
- Progressive 256/512 training still changes the bottleneck from global to
  local while 128-tile deployment changes it back. A stronger redesign should
  make attention behavior input-size invariant or deploy with the final-stage
  tile size.

## 4.2 Distributed validation collectives are now rank-safe

Evidence:

- `DistributedEvalSampler` intentionally gives non-padding, potentially uneven
  shards.
- DDP forward can broadcast buffers, requiring equal forward counts.
- The old dynamic metric dictionary was empty on a rank with no samples, so
  ranks could issue different all-reduce sequences.

Implemented fix:

- Validation forwards the synchronized bare module, preallocates a fixed metric
  vector, all-reduces sample sums, and rejects globally empty validation.

## 4.3 Metric crop geometry now matches NTIRE

Evidence:

- A fixed `128:-128` crop is protocol-equivalent only at canonical 482x512.
- At 512x512 it produced 256x256, while NTIRE uses centered 226x256.
- Small inputs produced sentinel metrics rather than scoring aligned pixels.

Implemented fix:

- Both paths now use centered 226x256 when available and aligned full-image
  metrics for smaller synthetic/custom inputs.

## 4.4 Checkpoint and inference hardening

Implemented fixes:

- Partial EMA shadows no longer mix EMA and raw parameters during inference.
- Resume initializes every EMA shadow from the strictly loaded model before
  overlaying legacy EMA keys.
- Repeated OOMs now fail after `max_consecutive_ooms` instead of spinning.
- Patch inference supports BF16/FP16 selection and keeps the blended cube FP32.

## 4.5 Remaining mathematical and architectural limits

MRAE:

- Synthetic target/prediction offset `1e-3` produced MRAE gradients of `1e8`,
  `1e6`, `1e4`, `100`, and `2` at target values `0`, `1e-6`, `1e-4`, `1e-2`,
  and `0.5`.
- Minimal change: log target-intensity quantiles and clipped-gradient frequency,
  then ablate `relative_mrae` or `mrae_l1`.
- Stronger alternative: intensity-stratified or uncertainty-aware spectral loss.
- Tradeoff: changing the objective can improve stability but reduces direct
  comparability with MST++ benchmark training.

Decoder:

- Total parameters: `7,021,690`.
- Decoder1: `2,112,240` parameters (30.1%); it runs before 192->96 compression.
- Hooked 128px CPU pass: decoder1 26.4%, decoder2 37.1% of top-level stage time.
- Minimal change: concatenate, compress, then run decoder1 at 96 channels.
- Stronger alternative: two-scale decoder with fewer full-resolution SSTBs and
  a lightweight convolutional refinement head.
- Tradeoff: checkpoint incompatible and requires quality retraining.

# 5. Patches Implemented

Changed production files:

- `src/hsi_model/train_generator.py`
- `src/hsi_model/utils/data/transforms.py`
- `src/hsi_model/utils/inference.py`
- `src/hsi_model/utils/patch_inference.py`
- `src/hsi_model/utils/training_setup.py`
- `src/configs/config.yaml`
- `cswin_test_ntire.py`
- `README.md`

Summary:

- Deployment-matched tiled validation.
- DDP-safe validation reductions and empty-validation rejection.
- Exact ARAD crop alignment.
- Complete EMA loading semantics.
- Bounded OOM retries.
- BF16-aware inference and FP32 overlap output.

# 6. Tests Added + How to Run

Added focused tests for:

- Tiled validation and DDP-wrapper bypass.
- Zero-sample validation and bounded OOM retries.
- Non-canonical and small-image crop behavior.
- Partial EMA resume/inference behavior.
- FP32 stitched output.

Run:

```powershell
.\.venv-audit\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv-audit\Scripts\python.exe smoke_run.py
.\.venv-audit\Scripts\python.exe smoke_infer.py
```

Result: `161 passed, 1 skipped`; both smoke paths passed.

# 7. Benchmark Results

Audit host: CPU-only, 8 Torch threads, Torch `2.11.0+cpu`.

| measurement | before | after/current |
|---|---:|---:|
| Full tests | 152 passed, 1 skipped | 161 passed, 1 skipped |
| Validation execution | full frame | deployment-matched 128/16 tiles |
| Small-CSWIN full/tiled relative output MAE | 2.50% mismatch | same tiled path selected and deployed |
| 240x256 validation latency | 0.287 s full | 0.463 s tiled |
| 240x256 validation MRAE | 9.43032 | 9.43109 |
| Stitch output | FP16 after FP32 blend | FP32; avoids up to `1.33e-5` probe rounding |
| Persistent OOM | unbounded retry | abort after 3 by default |

Current active model:

- Parameters: `7,021,690`.
- 64x64 forward median: `0.2069 s`.
- 128x128 forward median: `0.5606 s`.
- 128px CPU profile: convolution 55.2%, SDPA 10.2%, copies 9.08%.
- CUDA peak memory, CUDA throughput, selected SDPA kernel, BF16 speed, and
  `torch.compile` performance were not measurable on this host.

# 8. Optimization Roadmap

Immediate low risk:

1. Run GPU preflight with the patched tiled validator and record BF16/FP16 peak
   memory, selected SDPA backend, and tile throughput.
2. Log dark-target quantiles, gradient clipping frequency, out-of-range output,
   and raw/deployed MRAE together.
3. Stream patch outputs into the stitch accumulator for large-image deployment.

Medium risk:

1. Ablate stabilized MRAE and MRAE+L1 against pure MRAE.
2. Compress the first decoder skip before transformer processing.
3. Reduce final high-resolution depth from four to two blocks and add a cheap
   convolutional refinement block.
4. Benchmark a channel-first/fused LayerNorm and `torch.compile` on fixed stage
   shapes.

High risk/high reward:

1. Replace the token-threshold mode switch with input-size-invariant bounded
   global context.
2. Predict a learned low-rank spectral basis rather than 31 unconstrained bands.
3. Distill the model into a lighter restoration backbone for deployment.

# 9. Final Verdict

**FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED**

The active generator is mathematically suitable for RGB-to-HSI reconstruction,
compact for its family, and now aligns checkpoint selection with deployed
inference. Validation, crop geometry, EMA behavior, OOM handling, and output
precision are materially safer.

The main remaining risks are empirical: pure-MRAE conditioning on dark targets,
decoder-heavy compute, input-size-dependent attention during progressive
training, and unverified CUDA SDPA/memory behavior. Those require GPU profiling
and controlled quality ablations rather than silent production defaults.

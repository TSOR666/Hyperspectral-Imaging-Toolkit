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
  bounded OOM retries, fixed ARAD crop geometry, partial EMA, streamed
  stitching, and controlled loss/decoder ablations.

# 3. Bottleneck Summary

| severity | category | file:line | bottleneck | impact | proposed fix |
|---|---|---|---|---|---|
| HIGH, fixed | Quality/Inference | `train_generator.py:355`, `cswin_test_ntire.py:382` | Validation used full-frame execution while deployment used 128 tiles; token-dependent attention changed the model graph | Biased checkpoint selection; synthetic relative output MAE 2.50% | Validate with the deployment tile size/overlap |
| HIGH, fixed | Training | `train_generator.py:301,340-350` | Uneven DDP validation forwarded through DDP and reduced rank-dependent metric dictionaries | Collective mismatch/deadlock for unequal or empty shards | Bypass DDP for inference and reduce a fixed metric schema |
| HIGH, fixed | Quality/Data | `utils/data/transforms.py:62-71` | Training used fixed 128px borders while the tester used a centered 226x256 ARAD window | Metrics diverged on non-482x512 images; small inputs returned sentinels | Use the same fixed centered window, with aligned full-image fallback |
| HIGH, fixed | Implementation | `utils/inference.py:110-144`, `utils/training_setup.py:545` | Partial EMA state could produce mixed or random-shadow weights | Corrupt validation/inference after incomplete legacy checkpoints | Rebase resume EMA on loaded weights; reject partial EMA at inference |
| MEDIUM, fixed | Training | `train_generator.py:488,781-800` | Persistent OOM retried forever without changing geometry | Hung jobs and wasted allocations | Abort after configurable consecutive OOMs |
| MEDIUM, fixed | Quality/Inference | `utils/patch_inference.py:47-53,165` | Inference forced FP16 and rounded FP32 overlap blends back to half | Avoidable precision loss; mismatch with BF16 training policy | Support explicit/auto AMP dtype and return FP32 reconstruction |
| HIGH, experiment-ready | Math/Training | `losses_consolidated.py:163`, `ablation_stable_mrae.yaml` | Pure MRAE with epsilon `1e-8` has extreme dark-pixel gradients | Synthetic gradient reaches `1e8` at zero target and relies on global clipping | Fresh-run stabilized MRAE+L1 ablation added; dataset quality decision remains open |
| HIGH, experiment-ready | Architecture/Speed | `models/generator_v3.py:504-517`, `ablation_decoder_lite.yaml` | Decoder1 processes concatenated features at 4x base width before compression; decoder2 has four full-resolution blocks | Opt-in variant is 24.4% smaller and 1.45x faster at 128px on CPU | Fresh-run lightweight decoder ablation added; checkpoint-incompatible quality decision remains open |
| MEDIUM, GPU check | Speed | `models/generator_v3.py:166-176` | Four NHWC LayerNorm permute/copy cycles per transformer block | `aten::copy_` was 9.08% of profiled CPU self-time over 48 LayerNorm calls | Manual channel-first CPU form was 3.7-4.9x slower; benchmark fused/compiled CUDA kernels |
| MEDIUM, fixed | Memory/Inference | `utils/patch_inference.py:280-333` | All output tiles were retained before stitching | A 2048px image retained about 350 MiB of FP16 output tiles | Stream each output batch into preallocated FP32 accumulators |
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
- Patch inference now extracts one input batch and blends one output batch at a
  time instead of retaining every output tile.

## 4.5 Controlled ablations and remaining hardware checks

MRAE:

- Synthetic target/prediction offset `1e-3` produced MRAE gradients of `1e8`,
  `1e6`, `1e4`, `100`, and `2` at target values `0`, `1e-6`, `1e-4`, `1e-2`,
  and `0.5`.
- `ablation_stable_mrae.yaml` now runs `mrae_l1` with denominator epsilon
  `1e-2` and L1 weight `0.3`.
- `ablation_stable_lite.yaml` combines that objective with the lightweight
  decoder.
- Stronger alternative: intensity-stratified or uncertainty-aware spectral loss.
- Tradeoff: changing the objective can improve stability but reduces direct
  comparability with MST++ benchmark training.

Decoder:

- Total parameters: `7,021,690`.
- Decoder1: `2,112,240` parameters (30.1%); it runs before 192->96 compression.
- Hooked 128px CPU pass: decoder1 26.4%, decoder2 37.1% of top-level stage time.
- `decoder1_compress_first=true` now concatenates and compresses before running
  decoder1 at 96 channels.
- `ablation_decoder_lite.yaml` also reduces decoder2 depth from four blocks to
  two.
- Measured parameters fall from `7,021,690` to `5,309,570` (24.4%); 128px CPU
  median latency falls from `0.5594 s` to `0.3846 s` (1.45x).
- Stronger alternative: two-scale decoder with fewer full-resolution SSTBs and
  a lightweight convolutional refinement head.
- Tradeoff: checkpoint incompatible and requires quality retraining.

LayerNorm:

- A manual channel-first formulation was tested on CPU and was 3.7-4.9x slower
  than native NHWC `nn.LayerNorm`, so it was not adopted.
- CUDA fused/compiled performance and the selected SDPA backend remain
  unverified because this host exposes CPU-only Torch.

# 5. Patches Implemented

Changed production files:

- `src/hsi_model/train_generator.py`
- `src/hsi_model/utils/data/transforms.py`
- `src/hsi_model/utils/inference.py`
- `src/hsi_model/utils/patch_inference.py`
- `src/hsi_model/utils/training_setup.py`
- `src/hsi_model/models/generator_v3.py`
- `src/configs/config.yaml`
- `src/configs/ablation_stable_mrae.yaml`
- `src/configs/ablation_decoder_lite.yaml`
- `src/configs/ablation_stable_lite.yaml`
- `cswin_test_ntire.py`
- `README.md`

Summary:

- Deployment-matched tiled validation.
- DDP-safe validation reductions and empty-validation rejection.
- Exact ARAD crop alignment.
- Complete EMA loading semantics.
- Bounded OOM retries.
- BF16-aware inference and FP32 overlap output.
- Batch-bounded streamed patch extraction and stitching.
- Opt-in stable-loss and lightweight-decoder experiment configurations.

# 6. Tests Added + How to Run

Added focused tests for:

- Tiled validation and DDP-wrapper bypass.
- Zero-sample validation and bounded OOM retries.
- Non-canonical and small-image crop behavior.
- Partial EMA resume/inference behavior.
- FP32 streamed output, exact legacy stitching equivalence, and bounded tile
  batches.
- Hydra composition and finite backward pass for all new ablation configs.

Run:

```powershell
.\.venv-audit\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv-audit\Scripts\python.exe smoke_run.py
.\.venv-audit\Scripts\python.exe smoke_infer.py
```

Result: `168 passed, 1 skipped`; both smoke paths passed.

# 7. Benchmark Results

Audit host: CPU-only, 8 Torch threads, Torch `2.11.0+cpu`.

| measurement | before | after/current |
|---|---:|---:|
| Full tests | 152 passed, 1 skipped | 168 passed, 1 skipped |
| Validation execution | full frame | deployment-matched 128/16 tiles |
| Small-CSWIN full/tiled relative output MAE | 2.50% mismatch | same tiled path selected and deployed |
| 240x256 validation latency | 0.287 s full | 0.463 s tiled |
| 240x256 validation MRAE | 9.43032 | 9.43109 |
| Stitch output | FP16 after FP32 blend | FP32; avoids up to `1.33e-5` probe rounding |
| Retained 2048px FP16 output tiles, batch 4 | 349.72 MiB | 3.88 MiB (90.25x lower) |
| Decoder ablation parameters | 7.02M | 5.31M (24.4% lower) |
| Decoder ablation 128px CPU median | 0.5594 s | 0.3846 s (1.45x faster) |
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
3. Run the three supplied ablation configs with identical seeds and validation
   cadence.

Medium risk:

1. Promote the stable objective only if MRAE, SAM, dark-target error, and
   clipping frequency improve consistently.
2. Promote the lightweight decoder only if its 1.45x CPU speedup survives CUDA
   profiling without a material reconstruction-quality loss.
3. Add a cheap convolutional refinement block if the two-block decoder loses
   fine spectral/spatial detail.
4. Benchmark fused LayerNorm and `torch.compile` on fixed stage
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

The inference-memory issue is closed, and the two quality-sensitive findings
now have isolated, tested experiment configurations. The main remaining risks
are empirical: whether stabilized MRAE and the lighter decoder preserve or
improve dataset quality, input-size-dependent attention during progressive
training, and unverified CUDA SDPA/memory behavior. Those require GPU profiling
and controlled training results rather than silent production defaults.

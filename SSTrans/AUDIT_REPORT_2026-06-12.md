# SSTrans Bottleneck and Optimization Audit

Date: 2026-06-12

Reference: `Spectral_Spatial_Transformer_for_hyperspectral_recovery_from_RGB_image.pdf`

Architecture lineage check:
`https://github.com/caiyuanhao1998/MST-plus-plus/blob/master/train_code/architecture/MST_Plus_Plus.py`

## 1. Inferred Task and Model Family

- Task: direct RGB-to-hyperspectral reconstruction on ARAD-1K.
- Input: normalized RGB tensors in `B x 3 x H x W`.
- Output: reconstructed HSI tensors in `B x 31 x H x W`, covering 400-700 nm.
- Model: U-shaped hybrid Transformer with spectral channel attention, CSWin
  spatial attention, a CAT bottleneck, pixel unshuffle/shuffle, skip fusion,
  and convolutional embedding/output layers.
- Primary quality metrics: MRAE, RMSE, PSNR, and SAM.
- Reported paper result: the no-spectral-RPE ablation reaches MRAE 0.1468,
  RMSE 0.0248, PSNR 33.33, and SAM 0.0774 on ARAD-origin.
- Intended hardware: single NVIDIA A100 80 GB for training; CUDA GPU for
  practical inference.

The legacy-compatible `ablation_no_rpe` model has 18,439,699 parameters. The
recommended fresh-training model has 18,439,547 parameters (70.34 MiB in
fp32). The CAT bottleneck owns most parameters, but the highest
activation-memory risk is the high-resolution CSWin attention path.

## 2. Critical Paths & Profiling Plan

Critical call graph:

`ARAD1KDataset.__getitem__`
-> RGB normalization and paired crop/augmentation
-> `SSTransformer.forward`
-> encoder SST layers
-> CAT bottleneck
-> decoder SST layers
-> L1 reconstruction loss
-> AMP/backward/Adam/cosine scheduler
-> `evaluate_loader`
-> raw spectral metrics
-> checkpoint or NTIRE cube export

Audit procedure:

1. Compare the implementation and training configuration with the supplied
   paper.
2. Run the complete existing test suite.
3. Record parameter count, CPU inference timing, profiler allocation sites,
   and ARAD-shaped data-loader timing.
4. Replace only mathematically equivalent attention kernels.
5. Preserve existing checkpoint keys and verify full-model outputs.
6. Add regression tests for attention math, HDF5 axis handling, the published
   L1 objective, one-step training, and tiny-batch optimization.

The environment provides PyTorch 2.12.0 CPU only. CUDA kernel selection,
actual GPU peak memory, AMP throughput, and real ARAD quality could not be
measured locally.

## 3. Bottleneck Summary

| severity | category | file:line | bottleneck | impact | proposed fix |
|---|---|---|---|---|---|
| HIGH, fixed | Memory/Speed | `src/hsiformer/attention.py:13` | CSWin explicitly materialized quadratic score and softmax tensors | Up to 512 MiB per fp32 score tensor at the padded 512 stage, before gradients | Use CUDA SDPA while retaining the original CPU path |
| HIGH, fixed | Quality/Training | `src/hsiformer/training.py:40`, `configs/train_arad1k.json:29` | Retraining default used MRAE although the paper reports L1 training | Failed to reproduce the demonstrated beyond-SOTA recipe | Restore L1 weight 1.0 and MRAE weight 0.0 |
| HIGH, fixed | Data/Memory/Speed | `src/hsiformer/data.py:173`, `src/hsiformer/data.py:386` | Every 128 crop loaded the complete 31x482x512 cube | About 15x excess HDF5 payload and temporary host memory | Select the crop first and slice the HDF5 dataset directly |
| HIGH, fixed for retraining | Architecture/Quality | `src/hsiformer/model.py:26` | `dim // head` was passed as a head count, producing 16 spectral heads at every default stage | Departed from the originating constant-head-width design and made early heads only 2 channels wide | Add a stage-head mode with 2/4/8/16 heads and constant 16-channel width |
| MEDIUM, fixed | Data/Speed | `src/hsiformer/data.py:434` | YCrCb was decoded and converted when unused | Repeated CPU work on every train and inference sample | Load RGB only unless `include_ycrcb=True` |
| MEDIUM, fixed | Training/Speed | `src/hsiformer/training.py:207` | `loss.item()` synchronized CUDA every iteration | Avoidable host-device barrier at every training step | Accumulate a detached device scalar and synchronize only when logging |
| MEDIUM, fixed | Inference/Memory | `src/hsiformer/cli.py:106`, `src/hsiformer/cli.py:173` | Full trainer payload was loaded and retained on GPU | Raw checkpoints waste about 70 MiB; Adam trainer payloads can waste about 211 MiB | Load on CPU, copy model, delete payload, then move model to GPU |
| MEDIUM, fixed | Quality/Training | `src/hsiformer/data.py:172` | Full-resolution rectangular training disabled augmentation | Diverged from the paper's flip/rotation recipe during the last 50k steps | Keep safe flips and 0/180-degree rotations |
| MEDIUM, assessed | Architecture/Quality | `src/hsiformer/model.py:260`, `src/hsiformer/model.py:298` | Legacy residual topology differs from paper equations | Neutral branches amplify by 2x per legacy block | Use paper topology for the recommended retraining preset; retain exact-identity branch-delta as an ablation |
| MEDIUM, assessed | Architecture/Quality | `src/hsiformer/presets.py:55` | Paper text says CAT has no positional encoding, while source uses CAT RPE | Paper prose and demonstrated implementation disagree | Retain the valid pre-softmax 2-D CAT bias; remove only spectral RPE |
| MEDIUM, unresolved | Memory/Speed | `src/hsiformer/model.py:434` | Activation checkpointing covers CAT blocks, not dominant high-resolution SST blocks | Training activations remain expensive at 256/512 stages | Add an opt-in whole-SST checkpoint mode after CUDA measurement |
| LOW, candidate implemented | Speed/Quality | `src/hsiformer/model.py:720`, `src/hsiformer/attention.py:499` | Rectangular ARAD frames were padded to square for cross-attention | 482x512 became 512x512, adding 6.22% pixels | Opt-in native rectangular stripe pairing lowers outer padding to 496x512; retraining validation required |

## 4. Detailed Findings

### 4.1 Quadratic CSWin attention

Evidence:

- `LePEAttentionCross` and `CSWinCrossAttention` previously formed
  `Q @ K^T`, softmax, and `attention @ V` explicitly.
- At 512x512 with split size 1, one branch contains 134,217,728 score
  elements: 512 MiB in fp32 or 256 MiB in fp16.
- The same estimate is 256 MiB fp32 at the half-resolution stage and
  128 MiB fp32 at the quarter-resolution stage.

Minimal fix implemented:

- `_scaled_cosine_attention` dispatches CUDA tensors to
  `torch.nn.functional.scaled_dot_product_attention`.
- Per-head learned temperature is preserved by scaling normalized queries.
- Relative bias, dropout behavior, LePE, projection order, and all parameters
  remain unchanged.
- CPU keeps the original explicit math because SDPA was slower in this CPU
  environment.

Verification:

- Default full-model output is bit-exact on CPU.
- CUDA SDPA equivalence has a dedicated test and is skipped when CUDA is absent.

Stronger alternative:

- Benchmark Flash, memory-efficient, and math SDPA backends on the target A100.
- Add whole-SST activation checkpointing only if the measured memory reduction
  justifies recomputation.

Tradeoff:

- CUDA floating-point reduction order can differ at approximately 1e-5 scale.
  This does not change weights or attention equations, but real checkpoint
  quality should still be checked on ARAD.

### 4.2 Published loss mismatch

Evidence:

- The supplied paper states that L1 is used during training.
- The repository config previously selected pure MRAE.
- `README.md:53`, `configs/train_arad1k.json:29`, and
  `src/hsiformer/training.py:40` now agree on L1.

Why it matters:

- Loss choice directly changes the trained solution. The previous config could
  not be claimed to reproduce the reported MRAE 0.1468 result.

Minimal fix implemented:

- L1 weight 1.0, MRAE and SAM weights 0.0.
- Existing trained checkpoints and inference behavior are unaffected.

Stronger alternative:

- After reproducing the L1 baseline, evaluate small MRAE or SAM auxiliary
  weights against raw validation MRAE and SAM. Do not replace the baseline
  without a controlled ARAD run.

### 4.3 Full-cube loading for small crops

Evidence:

- The old loader converted the complete HDF5 cube to NumPy before selecting a
  crop.
- A 31x482x512 fp32 cube is 29.18 MiB; a 31x128x128 crop is 1.94 MiB.

Minimal fix implemented:

- Determine the paired crop coordinates before opening the spectral dataset.
- Infer the source axis permutation from the full dataset shape.
- Slice the source HDF5 axes directly and transpose only the cropped array.
- Fall back to the original full read for non-standard dimensional layouts.

Verification:

- All six 3-D axis permutations are tested.
- Paired boundary crops, random crops, oversized full frames, and YCrCb mode
  remain covered.

Tradeoff:

- Compressed HDF5 files may still decompress complete chunks internally, so
  disk speedup depends on chunk layout. Host allocation still drops sharply.

### 4.4 Training and inference synchronization

Evidence:

- Training previously called `.item()` after every loss computation.
- Inference/test CLI loaded the complete checkpoint with
  `map_location=device` and retained the returned payload.

Minimal fixes implemented:

- Loss is accumulated as a detached device scalar and transferred only at
  `log_every` intervals, reducing logging synchronizations from 50 to 1 with
  the default interval.
- Checkpoints load on CPU; payload is deleted before moving the model to CUDA.

Stronger alternative:

- Aggregate validation metrics on-device and transfer them once per loader.
- Save an inference-only weights artifact alongside trainer checkpoints.

### 4.5 Paper and source architecture disagreements

Evidence:

- `Spectral_MSAB` passed `dim // head` to an argument named `num_heads`.
  With the default width this produced 16 heads at all stages:
  `(32,16), (64,16), (128,16), (256,16)`.
- The originating MST++ implementation instead keeps the head width fixed and
  increases head count with stage depth. For SSTrans this gives
  `(32,2), (64,4), (128,8), (256,16)`, all with width 16.
- The paper's SSTB equation adds the original input once after the
  channel-attention/SST branch.
- Legacy source behavior applies residual additions in both `SSTB` and
  `SSTLayer`.
- The paper says CAT attention has no positional encoding, while legacy and
  default no-spectral-RPE presets retain CAT RPE.

Focused probes:

- With all transform parameters zeroed, a stack of 1/2/3/6 legacy blocks
  scales its input by 2/4/8/64.
- Literal paper topology scales the same neutral input by
  1.5/2.25/3.375/11.390625 because the channel gate is still inside the SST
  residual stream.
- The branch-delta form, `x + SST(CA(x)) - CA(x)`, remains exactly identity at
  every tested depth.
- An eight-seed initialization probe reduced parameter-gradient norm from
  0.579 for legacy to 0.405 for paper topology and 0.338 for branch-delta.
- A very short synthetic fit favored legacy amplification, so branch-delta is
  not promoted without a real ARAD-1K quality ablation.

Decision:

- Preserve `legacy_constant` heads and legacy residuals for checkpoint loading.
- Use stage-wise spectral heads and paper residual topology for fresh training.
- Keep `branch_delta` available as the mathematically best-conditioned residual
  ablation, but do not make it the quality default yet.
- Keep CAT RPE. It is a conventional pre-softmax two-dimensional relative
  bias, while the paper's best reported no-RPE result removes spectral RPE.
- Keep learned per-head spectral temperature. After cosine normalization this
  is more expressive than a fixed `1/sqrt(d)` factor and matches the MST++
  implementation family.
- Do not change Q/K orientation: without spectral RPE, independently learned
  Q and K projections make the two conventions equivalent up to
  reparameterization when training from scratch.

### 4.6 Native rectangular spatial path

Problem:

- Cross-shaped cross-attention paired one vertical and one horizontal stripe
  by batch index. The original implementation therefore required equal stripe
  counts and equal token lengths, which only holds for square features.
- The whole model padded both axes to one square side. ARAD's 482x512 input
  consequently became 512x512.

Candidate design:

- Square inputs retain the original cross-attention path exactly.
- For rectangular inputs, perpendicular key/value stripe banks are selected by
  nearest normalized spatial position to match the query stripe count.
- Query and key token lengths remain independent, which is directly supported
  by scaled dot-product attention.
- Cross-attention output windows are explicitly reversed into query spatial
  order rather than flattened as if they were already raster ordered.
- CAT pads once at the bottleneck boundary to its patch multiple and crops
  once after the complete layer.
- Outer padding is independent per axis and needs only the downsampling and
  CSWin stripe multiple. For the default network this is 16 rather than 64.

Safety:

- `rectangular_spatial=False` remains the constructor default.
- Existing presets and checkpoints retain their prior behavior.
- The opt-in `rectangular_candidate` preset requires retraining.
- A full square model with identical weights is bit-exact between the
  recommended and rectangular modes.
- Both rectangular stripe directions pass finite forward and backward tests.

Measured CPU proxy:

| input | square-padded path | rectangular path | speedup |
|---|---:|---:|---:|
| 120x128 | 258.1 ms | 238.2 ms | 1.08x |
| 240x256 | 1787.5 ms | 1450.2 ms | 1.23x |

These timings use a one-stage, width-8 CPU model with one thread. They
demonstrate that the overhead is amortized at representative spatial sizes;
they are not a CUDA or full-width throughput claim. On ARAD dimensions, outer
tensor area drops 3.125% relative to the old padded tensor and stripe-attention
score work drops by approximately 4.64%.

## 5. Patches Implemented

- `src/hsiformer/attention.py`
  - Added CUDA SDPA dispatch for spectral, CSWin self-, and CSWin
    cross-attention.
  - Preserved the legacy post-softmax spectral RPE path.
  - Added native rectangular cross-stripe pairing while retaining the exact
    square path.
- `src/hsiformer/cat.py`
  - Routed CAT attention through the shared cosine-attention kernel.
- `src/hsiformer/data.py`
  - Added direct HDF5 crop slicing across all axis orders.
  - Skipped unused YCrCb conversion.
  - Restored full-resolution safe augmentation.
- `src/hsiformer/training.py`
  - Restored the paper's L1 default.
  - Removed per-iteration logging synchronization.
- `src/hsiformer/cli.py`
  - Prevented trainer payloads from residing on inference GPUs.
- `src/hsiformer/model.py`
  - Added checkpoint-compatible stage-wise spectral head selection.
  - Added an opt-in exact-identity `branch_delta` residual form.
  - Added independent-axis outer padding and one-time CAT bottleneck padding
    for the rectangular candidate.
- `src/hsiformer/presets.py`
  - Added `recommended_retrain`: no spectral RPE, stage heads, CAT RPE, paper
    residual topology, and CAT activation checkpointing.
  - Added opt-in `rectangular_candidate`.
- `configs/train_arad1k.json`
  - Restored L1 training weights and selected `recommended_retrain`.
- `src/hsiformer/resources/train_arad1k.json`
  - Kept packaged config identical to the repository config.
- `README.md`
  - Corrected the documented objective.

Legacy presets retain their parameter shapes, checkpoint keys, and inference
outputs. The new retraining preset intentionally changes spectral temperature
shapes and therefore requires training from scratch.

## 6. Tests Added + How to Run

Added coverage:

- Manual cosine attention versus optimized kernel.
- CUDA SDPA equivalence when a CUDA device is available.
- HDF5 crop extraction for all six axis layouts.
- Published L1 config plumbing.
- Full-resolution augmentation remains enabled.
- Tiny-batch L1 loss decreases.
- Recommended spectral heads are 2/4/8/16 with constant width 16.
- Recommended preset keeps CAT RPE while removing spectral RPE.
- A zero transform branch is exact identity in `branch_delta` mode.
- Native rectangular cross-attention supports both stripe directions and
  finite gradients.
- Square output is bit-exact with rectangular mode enabled.
- Rectangular outer padding remains rectangular.

Run:

```powershell
uv run python -m pytest -p no:cacheprovider --basetemp C:\tmp\sstrans-pytest
uv run python scripts/smoke_model.py --preset recommended_retrain
```

Final local result:

```text
34 passed, 1 skipped
```

The skipped test is CUDA-only.

## 7. Benchmark Results

| measurement | before | after | result |
|---|---:|---:|---|
| ARAD-shaped 128 crop loader median | 36.495 ms | 9.287 ms | 3.93x faster |
| Crop throughput | 27.4 samples/s | 107.7 samples/s | 3.93x higher |
| HSI array requested per crop | 29.184 MiB | 1.938 MiB | 15.06x less |
| RGB decode plus unused YCrCb median | 4.110 ms | 2.964 ms RGB-only | 1.39x faster |
| Fresh-training model parameters | 18,439,699 | 18,439,547 | 152 fewer temperature scalars |
| Fresh-training fp32 parameter bytes | 70.34 MiB | 70.34 MiB | effectively unchanged |
| Full-model CPU output | reference | bit-exact | no reconstruction change |
| CUDA logging syncs per 50 steps | 50 | 1 | 50x fewer |
| Explicit full-stage score tensor | 512 MiB fp32 | no explicit SDPA score tensor | estimated kernel-memory reduction |
| Inference checkpoint payload on GPU | 70-211 MiB typical | 0 MiB payload tensors | estimated residency reduction |

CPU 64x64 inference medians were 155.2 ms before and 126.0 ms after in
separate runs. Since the CPU attention path intentionally remains the same
and the measurements were not isolated, this difference is treated as timing
noise, not as a speed claim.

Unavailable:

- CUDA train-step latency.
- CUDA inference latency and throughput.
- Measured CUDA peak allocation.
- FLOPs.
- Real ARAD before/after metrics, because no dataset path or trained checkpoint
  was available in the workspace.

## 8. Optimization Roadmap

Immediate, low risk:

1. Run the CUDA equivalence test and benchmark SDPA backend selection on A100.
2. Train `recommended_retrain` from identical seeds and compare raw,
   unclipped ARAD validation metrics with the proven recipe.
3. Record `torch.cuda.max_memory_allocated()` for 128, 256, and full-resolution
   stages.
4. Export an inference-only checkpoint without optimizer and scheduler state.

Medium risk:

1. Add optional whole-SST activation checkpointing.
2. Compare `cat_rpe=True/False` while holding the recommended preset fixed.
3. Compare paper and branch-delta residual topology from identical seeds.
4. Test EMA only as a controlled training ablation.

High risk, high reward:

1. Train and evaluate `rectangular_candidate` against `recommended_retrain`
   from identical seeds.
2. Distill the proven model into a narrower deployment model.
3. Explore low-rank spectral bases only if ARAD quality and RGB back-projection
   metrics remain at or above the current result.

## 9. Final Verdict

**FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED**

The architecture is demonstrably strong for RGB-to-HSI reconstruction. Legacy
checkpoint behavior remains available, while fresh training now uses the most
defensible combination supported by the paper, source lineage, and numerical
conditioning probes. Real ARAD validation is still required before claiming
that the stage-head and residual corrections improve reconstruction metrics;
the identity-conditioned residual and native rectangular pairing remain opt-in
until controlled retraining confirms raw reconstruction quality.

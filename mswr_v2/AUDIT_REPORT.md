# MSWR-Net v2.1.2 - Bottleneck and Optimization Audit

**Date:** 2026-06-11
**Scope:** `mswr_v2/` architecture, data, training, validation, inference, export
**Audit host:** Windows 11, Python 3.11.14, PyTorch 2.11.0+cpu, 8 CPU threads
**Limitation:** CUDA kernel selection, peak VRAM, AMP throughput, and GPU latency require a GPU follow-up.

## 1. Inferred Task and Model Family

- **Task:** RGB-to-31-band hyperspectral reconstruction on ARAD-1K/NTIRE-style data.
- **Input/output:** `B x 3 x H x W` RGB to `B x 31 x H x W` reflectance.
- **Model:** encoder-decoder CNN/Transformer hybrid with local window attention,
  landmark attention, CNN DWT/IDWT branches, skip connections, and optional
  MST++-style spectral self-attention.
- **Primary metric:** MRAE; secondary metrics are RMSE, PSNR, SAM, and SSIM.
- **Deployment:** CUDA training/inference with AMP, channels-last, checkpointing,
  EMA, full-image or tiled inference, TTA, and MATLAB/NumPy/HDF5 export.

## 2. Critical Paths & Profiling Plan

Critical call graph:

`ARAD files -> dataloader.py -> IntegratedMSWRNet.forward -> wavelet + attention blocks -> loss -> backward/optimizer/scheduler/EMA -> validation metrics -> checkpoint -> inference/TTA/tiling -> export`

Highest-risk zones inspected and probed:

1. DWT/IDWT subband ordering, border reconstruction, and padding alignment.
2. Window/landmark/spectral attention shape, numerical stability, and memory.
3. AMP, accumulation, EMA, scheduler, resume, and validation metric semantics.
4. Checkpoint container normalization and inference preprocessing.
5. NTIRE result retention, TTA alignment/peak memory, and MATLAB format.

## 3. Bottleneck Summary

| severity | category | file:line | bottleneck | impact | proposed fix |
|---|---|---|---|---|---|
| HIGH | Memory | `mswr_test_ntire.py:1023` | Full prediction and GT tensors were retained for every test scene | About 2.85 GiB for 50 ARAD scenes, often on CUDA | **Patched:** retain scalar summaries only |
| HIGH | Memory | `mswr_test_ntire.py:139,255` | Statistical buffer could retain 32 full error cubes and concatenate them | Up to about 0.91 GiB plus concatenation | **Patched:** deterministic 500k-value cap, about 1.9 MiB |
| MED | Memory/Inference | `mswr_inference.py:358-376`, `mswr_test_ntire.py:878-945` | TTA stored all outputs and then allocated a stack | About 0.46 GiB output storage at 482x512 for eight-way TTA | **Patched:** float32 streaming average |
| MED | Inference | `mswr_test_ntire.py:876-945` | NTIRE `full` TTA used four transforms while production used eight | Inconsistent evaluation and incomplete ensemble | **Patched:** aligned eight-way dihedral TTA |
| MED | Inference | `mswr_test_ntire.py:1068` | `.mat` export used MATLAB v5 instead of repository v7.3 utility | ARAD tooling compatibility risk | **Patched:** use `save_matv73` |
| LOW | Implementation | `mswr_inference.py:29-41`, `mswr_test_ntire.py:27-42` | Plotting libraries were mandatory at import time | Headless metric/inference tools failed to import | **Patched:** optional imports with explicit visualization errors |
| MED | Architecture/Quality | `model/mswr_net_v212.py:808-839` | Default learned landmarks are a static dictionary, not content-derived spatial summaries | Weak global spatial mixing; likely quality ceiling | Benchmark `landmark_pooling: adaptive` |
| MED | Data/Memory | `dataloader.py:202-234,349-387` | Train and validation datasets preload every RGB/HSI scene | Roughly 30 GiB host RAM for a full ARAD training split | Add lazy loading plus bounded scene cache |
| MED | Training | `train_mswr_v212_logging.py:839-867` | Weight-decay exemption only recognizes names containing `norm` or `bias` | Layer scales, spectral temperatures, and landmarks are decayed | Add explicit no-decay rules and ablate |
| MED | Architecture/Quality | `model/mswr_net_v212.py:482-483` | The model-class fallback wavelet levels are `[1,2,3]` | Deep stage can collapse spatial evidence to 4x4 | Canonical `configs/train.yaml` uses `[1,1,1]` |
| LOW | Evaluation | `train_mswr_v212_logging.py:1951-1958` | Selection metric is clamped MRAE while MST++ comparison is raw MRAE | Benchmark ambiguity | Report both; use `mrae_unclamped` for MST++ tables |

## 4. Detailed Findings

### Evaluation memory retention

**Evidence:** `run_test()` previously appended the complete result dictionary,
including device-resident prediction and GT tensors, to `all_results`.

**Why it matters:** one prediction/GT pair at `31x482x512` float32 is about
58.4 MiB. Fifty scenes retain about 2.85 GiB before model activations.

**Implemented fix:** `_result_summary()` keeps only the name and scalar metrics.
The stronger alternative is a streaming JSONL writer for very large datasets.
There is no quality tradeoff.

### Statistical analysis buffer

**Evidence:** the previous 32-cube cap still approached 0.91 GiB at full
resolution, followed by `np.concatenate`.

**Implemented fix:** sample at most 10,000 deterministic elements per scene and
500,000 total. Peak retained statistics fall to about 1.9 MiB. The tradeoff is
that distribution plots are estimates rather than exhaustive pixel statistics.

### TTA correctness and peak memory

**Evidence:** production `full` TTA used eight transforms; NTIRE testing used
four. Both paths retained every prediction and then called `torch.stack`.

**Implemented fix:** align both paths to eight transforms with exact inverses
and accumulate in float32. This removes the list-plus-stack allocation and
avoids fp16 averaging noise. Runtime remains eight forwards by design.

### Remaining quality bottlenecks

The default learned-landmark path expands static learned tokens for every image;
only the adaptive path derives landmarks from image content. This is not a
correctness bug, but it limits the "global" branch's ability to summarize the
current scene. Run an architecture-controlled learned-vs-adaptive ablation.

The canonical config's `[1,1,1]` wavelet schedule is mathematically better
motivated than the model-class fallback `[1,2,3]` for a 128x128 crop because
the bottleneck attention sees 16x16 rather than 4x4 low-pass evidence.

## 5. Patches Implemented

- `mswr_test_ntire.py`
  - bounded statistical sampling
  - scalar-only result retention
  - complete and aligned eight-way TTA
  - streaming float32 TTA averaging
  - MATLAB v7.3 export
  - optional plotting imports and `--no_save_visualizations`
- `mswr_inference.py`
  - streaming float32 TTA averaging
  - optional plotting imports with a clear visualization dependency error
- `tests/test_ntire_runtime.py`
  - five regression tests for memory bounds, result retention, both TTA paths,
    spatial alignment, and v7.3 export routing

The prior DWT ordering, periodic IDWT border, window-padding alignment, SDPA
bias, scaler resume, EMA cold-start, OOM cleanup, and checkpoint-load fixes were
also regression-checked and remain intact.

## 6. Tests Added + How to Run

```powershell
cd mswr_v2
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' -m pytest -q --basetemp=.tmp_pytest
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' smoke_train.py
& '..\CSWIN v2\.venv-audit\Scripts\python.exe' smoke_infer.py
```

Result: **169 passed, 2 skipped** after the final patch set. CUDA-only tests are
skipped on this host.

## 7. Benchmark Results

| metric | result |
|---|---:|
| Base parameters | 2,597,500 |
| Base + spectral attention parameters | 3,035,684 |
| Tiny parameters | 184,072 |
| Tiny inference 64x64, CPU, warm | 31.68 ms |
| Tiny inference 128x128, CPU, warm | 71.24 ms |
| Smoke training, CPU, 5 steps | 396.4 ms/step |
| Smoke inference 128x128, CPU | 129.0 ms |
| Smoke inference 256x256, CPU | 441.7 ms |
| Retained 50-scene test tensors, before | about 2.85 GiB |
| Retained test tensors, after | scalar metadata only |
| Statistical buffer, before | up to about 0.91 GiB |
| Statistical buffer, after | at most about 1.9 MiB |
| Eight-way TTA output storage, before | about 0.46 GiB at 482x512 |
| Eight-way TTA accumulator, after | one fp32 accumulator plus current output |

The first 64x64 smoke inference includes initialization overhead (4.01 s);
warm-loop latency is reported separately above. Peak VRAM and CUDA throughput
were not measurable on this CPU-only host.

## 8. Optimization Roadmap

**Immediate low risk**

1. Use `configs/train.yaml` as the primary quality recipe and report raw plus
   EMA MRAE separately.
2. Add explicit optimizer no-decay handling for layer-scale `gamma`, spectral
   `rescale`, and learned landmarks, then run a controlled weight-decay sweep.
3. Convert dataset storage to lazy file reads with a small LRU scene cache.

**Medium risk**

1. Compare `landmark_pooling: learned` against `adaptive` at equal training
   budget and parameter count.
2. Sweep `drop_path` in `{0.1, 0.2, 0.3}` and EMA decay in
   `{0.999, 0.9995, 0.9999}` for the observed overfitting regime.
3. Stream tiled inference directly into the blend accumulator instead of
   retaining all processed tiles.

**High risk/high reward**

1. Replace the static landmark branch with content-derived low-rank spatial
   attention or remove it if spectral attention supplies the quality gain.
2. Distill the spectral-attention model into the base network for lower latency.
3. Benchmark `torch.compile`, SDPA backend selection, BF16, and channels-last on
   the target GPU before changing defaults.

## 9. Final Verdict

**FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED**

The current code is correctness-tested, numerically stable on CPU probes, and
safe enough for training and evaluation. The main deployment-blocking test-time
memory issues found in this audit are fixed. Remaining limitations are primarily
host-RAM data loading, GPU-unverified performance, and architecture/regularization
choices that need controlled experiments rather than speculative rewrites.

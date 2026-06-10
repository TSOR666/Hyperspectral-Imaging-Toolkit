# HSIFusion & SHARP Bottleneck Audit

## 1. Inferred Task and Model Family

- Task: RGB-to-31-band hyperspectral reconstruction on ARAD-1K/MST++-style data.
- HSIFusion: hierarchical encoder-decoder transformer with sliding-window spatial attention, spectral attention, optional MoE, and decoder cross-attention.
- SHARP: hierarchical encoder-decoder with dense multi-scale attention, streaming top-k attention, RBF features, and gated cross-attention.
- Inputs/outputs: `B x 3 x H x W` RGB to `B x 31 x H x W` spectra.
- Primary metrics: MRAE, RMSE, and PSNR.
- Intended hardware: CUDA GPU training/inference, with A100-oriented LSF launch templates.

## 2. Critical Paths & Profiling Plan

Critical call graph:

`split_txt + RGB/.mat -> optimized_dataloader -> model forward -> reconstruction/spectral loss -> AMP/backward -> AdamW/cosine schedule -> validation metrics -> raw+EMA checkpoint -> tiled inference`

The audit used static call-path inspection, synthetic shape and finite-value checks, one-step training/inference, exact-objective invariants, attention microbenchmarks, checkpoint round trips, and the complete local pytest suite. Real ARAD-1K data and CUDA hardware were unavailable, so dataset quality and GPU peak-memory claims remain unverified.

## 3. Bottleneck Summary

| Severity | Category | File:line | Bottleneck | Impact | Proposed fix |
|---|---|---|---|---|---|
| BLOCKER | Math/Algorithm | `sharp_v322_hardened.py:1073` | Perfect predictions had large positive loss from a non-orthonormal basis reconstruction | Optimizer pushed exact outputs away from targets | Compare second spectral derivatives of prediction and target |
| HIGH | Memory/Speed | `sharp_v322_hardened.py:510`, `:785` | Dense attention used all `H*W` keys at high resolution | Quadratic score compute in encoder and decoder | Pool K/V to a configurable token cap |
| HIGH | Memory/Speed | `hsifusion_v252_complete.py:1346` | Decoder cross-attention was quadratic at full resolution | Large latency and attention memory | Bound context K/V tokens while retaining full-resolution queries |
| HIGH | Architecture | `sharp_v322_hardened.py:287` | Long-sequence fallback used flattened 1D neighborhoods | Row-boundary wrap and wrong spatial inductive bias | Chunked 2D local neighborhoods |
| HIGH | Data | `optimized_dataloader.py:43`, `:164`, `:423` | Lazy mode cached 100 full float32 cubes per worker | About 11.4 GiB for four workers, HSI only | Float16 cache with default size 4 |
| HIGH | Training | `sharp_v322_hardened.py:1306`, `sharp_training_script_fixed.py:805` | Built-in trainer state was not checkpointed | Resume restarted optimizer, scheduler, scaler, and EMA | Full trainer state serialization |
| HIGH | Validation | `sharp_v322_hardened.py:1446` | Built-in validation ignored configured center crop and EMA | Best-model metric disagreed with protocol/inference | Crop-aware EMA validation and EMA checkpoint weights |
| HIGH | Quality | `sharp_training_script_fixed.py:107` | Default `mean` key projection collapsed each key to a repeated scalar | Sparse rankings were nearly query-independent | Default new training to learned `linear` projection |
| MEDIUM | Data | `optimized_dataloader.py:123`, `:193` | Floor patch counts omitted right/bottom borders | Biased patch coverage | Ceil counts with boundary-aligned final patches |
| MEDIUM | Inference | `sharp_inference.py:117`, `:179` | Blend weights rebuilt per patch; narrow images could mismatch weight shape | Avoidable overhead and shape failures | Cache weights by actual patch shape |
| MEDIUM | Training | `hsifusion_training.py:209` | Weight decay included norms, biases, and layer-scale vectors | Poor regularization semantics | Module-aware AdamW groups |
| MEDIUM | Speed | `train_job_HSI.sh:51`, `train_job_SHARP.sh:53` | `CUDA_LAUNCH_BLOCKING=1` enabled in production jobs | Forced GPU synchronization | Disable launch blocking and DSA debugging |

## 4. Detailed Findings

### SHARP Objective Was Mathematically Invalid

Before the patch, `compute_loss(x, x)` returned `24.1725` for a random perfect target. The generated basis had rank 29 and its Gram matrix differed from identity by mean absolute error `0.9823`. The regularizer therefore penalized exact reconstruction.

The replacement matches second-order spectral curvature to the target. Exact predictions now have zero loss. A stronger future alternative is a weighted MRAE + SAM + curvature objective validated against ARAD-1K metrics.

### Dense Global Attention Dominated Runtime

Both models retained full-resolution queries while also attending to every spatial key. The patch adaptively pools only K/V, preserving local branches and full-resolution output while reducing score pairs from `N^2` to `N*K`.

At a 128x128 stage, the default cap of 1024 reduces dense score pairs by 16x. The tradeoff is compressed global context, so final cap selection should be validated on real MRAE/SAM.

### SHARP Fallback Lost 2D Geometry

The fallback above `sparse_max_tokens` used neighboring flattened indices. Adjacent indices can belong to different image rows, while vertically adjacent pixels can be far apart in the sequence. The new implementation builds bounded 2D neighborhoods and chunks queries to control temporary memory.

### Lazy Loading Was Not Low-Memory

Lazy mode cached float32 RGB/HSI pairs and defaulted to 100 images per worker. For 31x482x512 HSI cubes and four workers, HSI cache capacity alone was about 11.4 GiB. The new float16, four-image cache is about 0.23 GiB for HSI, roughly 50x lower, plus a small RGB contribution.

### Resume, Validation, and Inference Disagreed

The built-in trainer previously omitted optimization and EMA state. Validation also bypassed configured center cropping, and inference loaded raw weights despite EMA tracking. Checkpoints now contain raw weights for resume, complete trainer state, and EMA weights for validation/inference.

## 5. Patches Implemented

- Corrected SHARP spectral loss and 2D local fallback.
- Added configurable bounded global context to SHARP and HSIFusion.
- Changed new SHARP training defaults to `key_rbf_mode=linear`.
- Made lazy loading bounded, float16-backed, border-covering, and alignment-checked.
- Added exact SHARP trainer resume state, crop-aware EMA validation, and EMA inference.
- Cached tiled-inference blend weights and handled non-square edge patches.
- Added HSIFusion AdamW parameter groups, accurate skipped-batch averaging, CLI plumbing, and disabled the untrained uncertainty head by default.
- Disabled production CUDA debugging synchronization.
- Corrected README commands, CLI names, dataset setup behavior, LSF labeling, and unsupported DDP claims.

## 6. Tests Added + How to Run

Added regressions for:

- zero loss on perfect SHARP predictions;
- bounded global K/V token counts;
- correct 2D local neighborhoods;
- crop-aware evaluation;
- float16 lazy cache and border coverage;
- trainer-state round trip;
- EMA checkpoint preference;
- narrow-image tiled inference;
- CLI/config forwarding.

Run:

```bash
python -m pytest -q
python smoke_train.py --model both --steps 1 --size 64 --device cpu
python smoke_pipeline.py --model both --size 64 --device cpu
```

## 7. Benchmark Results

CPU: PyTorch 2.12.0, four threads for attention microbenchmarks, batch 1, dim 64, four heads, 48x48 features, K/V cap 256.

| Module | Before | After | Speedup |
|---|---:|---:|---:|
| SHARP multi-scale global branch | 40.07 ms | 4.21 ms | 9.51x |
| SHARP decoder cross-attention | 31.29 ms | 5.20 ms | 6.02x |
| HSIFusion decoder cross-attention | 8.40 ms | 1.47 ms | 5.72x |

Additional results:

- SHARP perfect-target loss: `24.1725 -> 0.0`.
- SHARP tiny parameters: 7.30M.
- HSIFusion tiny parameters: 18.20M.
- SHARP 64x64 one-step CPU training after fixes: 0.90 s, finite loss.
- Full synthetic pipeline: train, validation, checkpoint load, tiled inference, and finite-output checks passed for both models.
- GPU latency, VRAM, throughput, and FLOPs were not measured because CUDA hardware was unavailable.

## 8. Optimization Roadmap

Immediate low risk:

- Benchmark `max_global_tokens` values 256/512/1024 on ARAD-1K.
- Add dataset range and spectral-band-order assertions.
- Record dataloader samples/s and GPU utilization in training logs.

Medium risk:

- Replace SHARP `tanh` output with a checkpoint-versioned nonnegative radiance parameterization.
- Evaluate MRAE + SAM + curvature loss weights on validation data.
- Batch tiled inference patches instead of processing them serially.

High risk/high reward:

- Replace streaming exact top-k with windowed/landmark attention at high resolution.
- Distill global context into lower-resolution stages.
- Add proper DDP with distributed samplers, rank-aware checkpoints, and reduced metrics.

## 9. Final Verdict

**FIT-FOR-PURPOSE BUT OPTIMIZATION NEEDED**

The confirmed correctness blockers and major CPU attention bottlenecks are fixed, and synthetic training/inference is stable. Deployment readiness is still limited by the absence of real-dataset quality comparisons, CUDA memory/throughput measurements, export validation, and distributed training support.

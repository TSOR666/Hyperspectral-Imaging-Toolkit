# SSTrans ARAD-1K performance investigation

## Observed run

The supplied `metrics 18.jsonl` contains 6,864 records. Its best validation
record is:

| global step | stage | MRAE | RMSE | PSNR | SAM |
|---:|---:|---:|---:|---:|---:|
| 334,000 | 256-pixel | 0.250837 | 0.036484 | 30.6894 | 0.091921 |

The run ended at step 334,850 of 400,000, before its nominal full-frame stage.
It recorded no non-finite optimizer skips, and its cosine learning-rate trace
matches the configured schedule. This rules out collapse and scheduler drift
as the primary explanation. During the 256-pixel stage, training L1 and
validation MRAE have correlation -0.284: L1 continues to improve while MRAE
does not.

## Provenance of the expected number

The expected value is the `SSTransformer_orig_no_rpe-ARAD-origin` result
(`MRAE 0.1468109139`) in the upstream `HSI-diffusion` result file. That result
uses the 50 validation/ARAD-origin scenes (`0901`-`0950`) and complete frames.
It is not the held-out test split or the separate center-crop MST++/NTIRE
number.

The local ARAD-1K contract is explicitly `900/50/50`: scenes `0001`-`0900`
train, `0901`-`0950` validation, and `0951`-`1000` test. Consequently, the
reported target is compared only on the middle 50 scenes; the final 50 test
scenes do not enter training, checkpoint selection, or the `0.1468`
comparison.

The upstream result generator adds `1e-5` to prediction and ground truth when
a scene contains zeros, then computes MRAE. The previous local metric instead
used `abs(target).clamp_min(1e-6)`, so the two values were not directly
comparable.

The upstream result notebook inspects
`SSTransformer_orig_no_rpe/lightning_logs/version_0/checkpoints/last.ckpt`.
Its native Lightning contents and path match the active two-GPU job using the
fixed-128, seven-epoch YAML (global batch 32). With 900 scenes and 2,254
stride-8 patches per scene, this is 443,758 optimizer updates. The repository
also contains older progressive YAMLs, but their launch lines are commented
and no tracked checkpoint, log, or hash links their continuation stages to
this result. Because trainers share output names and launchers resume existing
checkpoints, exact ancestry remains unprovable.

Source references:

- `result.txt` at commit `866bf038e97f6cf9a1ae21bfea10afea6816d941`
- `configs/sst/sstransformer_ori_no_rpe.yaml`
- `configs/sst/ablation/sstransformer_ori_no_rpe_256.yaml`
- `configs/sst/ablation/sstransformer_ori_no_rpe_512.yaml`
- `models/transformer/SST_ablation.py`
- `models/transformer/Base.py`
- `models/transformer/basemodel.py`
- `jobqueue/sstransformer-orig-ablation-2gpu.sh`
- `figure.ipynb`

## Blocking differences

The previous `recommended_retrain` configuration did not reproduce the model
that generated the target:

| Contract | reported source run | previous local run |
|---|---|---|
| encoder depths / bottleneck | `[2,2]` / 2 | `[1,2,3]` / 4 |
| spectral heads | effective 16 at every scale | 2/4/8/16 |
| residual graph | legacy double residual | paper graph |
| Conv/Linear initialization | trunc-normal, std 0.02 | Conv Kaiming |
| spatial cosine scale | 1 | 5 |
| attention math | eager normalize/matmul/softmax | guarded SDPA |
| objective | MRAE + 0.1 SAM + 0.1 DeltaE | L1 |
| crop sampling | exhaustive, stride 8 | 16 random crops/scene |
| schedule | fixed 128, 7 epochs (443,758 updates) | 300k + 50k + 50k |
| precision / clipping | FP32 / none | BF16 / grad-norm 1 |
| target metric | full frame, source `+1e-5` | full frame, clamp `1e-6` |

The architecture changes make the existing checkpoint incompatible with the
source-reproduction preset. It remains useful for diagnostic rescoring, but
reaching the source result requires a fresh run.

## Resolution

`configs/train_arad1k.json` now selects `source_reproduction` and restores the
source architecture, composite loss, exhaustive fixed-128 seven-epoch
schedule, FP32 optimizer path, and full-frame source validation. The public
test command exposes named
`source_arad_origin`, `ntire_center`, and `legacy_full` metric profiles so the
protocol cannot change silently.

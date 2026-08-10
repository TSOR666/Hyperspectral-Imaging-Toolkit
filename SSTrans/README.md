# HSIFormer

Self-contained PyTorch implementation and retraining pipeline for RGB-to-
hyperspectral reconstruction on ARAD-1K.

The repository contains only the model package, ARAD-1K split manifests,
training/evaluation commands, configurations, and tests. The public split is:

- 900 training scenes: `ARAD_1K_0001` to `ARAD_1K_0900`
- 50 validation scenes: `ARAD_1K_0901` to `ARAD_1K_0950`
- 50 test scenes: `ARAD_1K_0951` to `ARAD_1K_1000`

## Install

```powershell
uv sync --extra dev
```

Alternatively:

```powershell
python -m pip install -e ".[dev]"
```

## Dataset

The loader expects the NTIRE 2022 ARAD-1K layout:

```text
ARAD_1K/
|-- Train_RGB/
|   `-- ARAD_1K_0001.jpg
`-- Train_spectral/
    `-- ARAD_1K_0001.mat
```

Split-specific directories are resolved per scene, so redistributed copies also
work: `Test_RGB`/`Test_Spec`, `Valid_RGB`/`Valid_spectral`, `Validation_*`, and
`Val_Spec` are all searched before falling back to `Train_*`. RGB files may be
`.jpg`, `.png`, `.bmp`, or `.tif`.

Spectral `.mat` files should contain a `cube` dataset; the aliases
`reflectance`, `rad`, `hsi`, `hyper`, `data`, and `image` are accepted, as is a
single 31-band 3D dataset under any other name. Data is loaded lazily, so the
complete dataset is not held in memory.

The official ARAD-1K test release ships `Test_Spec/*.mat` files holding the raw
MSFA `mosaic` payload instead of a spectral cube. A mosaic file never shadows a
real cube: every candidate directory is probed, so cubes kept in, say,
`Test_spectral` alongside a mosaic `Test_Spec` are found and used. Point
`--spectral-dir` at any other location (a name under the root or an absolute
path). Scenes with no cube anywhere are reconstructed and exported but never
scored; see [Public Test](#public-test).

Check a local dataset:

```powershell
python scripts/inspect_arad.py "D:\datasets\ARAD_1K" --split test
```

## Training

`configs/train_arad1k.json` now targets the executable source run associated
with the reported ARAD-origin MRAE `0.1468109`. This matters because the
released training code and the manuscript recipe are not the same. The source
run uses:

- Adam with betas `(0.9, 0.999)`
- per-iteration cosine learning-rate decay to `1e-6`
- MRAE + `0.1 * SAM + 0.1 * DeltaE`, including the original 31-band camera
  response used by the color loss
- full-scene RGB min-max normalization before crop extraction, matching the
  reference loader and keeping train/validation input scales consistent
- exhaustive 128x128 patches at stride 8, rather than 16 random crops per scene
- seven fixed-128 epochs at global batch 32: 63,394 updates per epoch and
  443,758 total updates for the 900-scene training manifest
- validation and checkpointing every 2,000 iterations
- full-frame FP32 checkpoint selection with the source MRAE denominator:
  `abs(pred-target)/(target+1e-5)` when a scene contains zeros
- FP32 training without gradient clipping, matching the source optimizer path
- a finite-loss guard: optimizer steps with non-finite loss or gradients are
  skipped, and training aborts after 100 consecutive non-finite steps

This is the strongest tracked provenance: the result notebook inspects a
native Lightning `last.ckpt` in the path targeted by the active seven-epoch
job. Older commented job lines and standalone YAMLs describe a
400k/50k/400k progressive experiment, but no tracked checkpoint, log, or hash
links those continuation stages to the reported `0.1468` artifact. Exact
checkpoint ancestry is therefore not provable from the upstream repository.

Start training:

```powershell
python scripts/train.py `
  --data-root "D:\datasets\ARAD_1K" `
  --output-dir "runs\sstrans_source_arad1k" `
  --device cuda
```

After installation, the equivalent command is `hsiformer-train`.

Resume model, optimizer, scheduler, stage, and iteration state from a trainer
checkpoint:

```powershell
python scripts/train.py `
  --config configs/train_arad1k.json `
  --data-root "D:\datasets\ARAD_1K" `
  --resume "runs\sstrans_source_arad1k\checkpoints\latest.pt"
```

Each checkpoint includes the model architecture, optimizer, scheduler, AMP
scaler, stage position, global iteration, and resolved training configuration.
Mid-epoch resume is not bit-identical: the current trainer does not persist the
DataLoader cursor or Python/NumPy/Torch RNG streams, so the shuffle and
augmentation stream restarts.
Resume refuses NaN/Inf-poisoned checkpoints: non-finite parameters or
optimizer moments raise instead of silently continuing a collapsed run.
Training metrics are appended to `metrics.jsonl`.

When the crop, MRAE denominator, precision, tiling, normalization, or manifest
changes on resume, the trainer resets only the stored best-MRAE threshold.
Weights, optimizer, scheduler, stage, and iteration still resume normally.

The default is `source_reproduction`: the no-spectral-RPE architecture actually
used by the reported source run. It uses `[2, 2]` encoder depths, a depth-2 CAT
bottleneck, 16 effective spectral heads at every scale, the legacy double
residual graph, spatial attention scale 1, and trunc-normal initialization for
both convolutions and linear layers. It also uses the source's eager
normalize/matmul/softmax attention path (default normalization epsilon, no
scale clamp, and no SDPA substitution). These details are
checkpoint-incompatible with `recommended_retrain`. It keeps CAT relative bias
because the executable source does, even though the manuscript prose says
those blocks omit positional encoding.

Available presets are:

- `legacy`: historical local checkpoint layout with spectral RPE.
- `ablation_no_rpe`: the historical local no-RPE toggle; it does not include
  all source-run depths, scale, or initialization settings.
- `source_reproduction`: complete source-run no-RPE architecture and
  initialization associated with MRAE `0.1468`.
- `corrected_rpe`: places spectral RPE before softmax.
- `optimized_candidate`: no spectral/CAT RPE, corrected residual topology, and
  activation checkpointing.
- `recommended_retrain`: no spectral RPE, intended stage-wise spectral heads,
  CAT RPE, paper residual topology, Kaiming convolution initialization, spatial
  scale 5, and activation checkpointing. This is experimental and is not the
  architecture that produced `0.1468`.
- `rectangular_candidate`: the recommended retraining architecture plus native
  rectangular CSWin stripe pairing and local CAT patch padding. Square inputs
  remain bit-exact; rectangular behavior requires retraining validation.

`residual_mode="branch_delta"` is also available for controlled experiments.
It makes each neutral SST branch an exact identity, unlike the legacy and
literal paper graphs, but is not the default until an ARAD-1K ablation confirms
its reconstruction quality.

The rectangular candidate is intentionally separate from the default training
configuration. Select `"preset": "rectangular_candidate"` only for a fresh
training run; its square path is checkpoint-equivalent, but rectangular stripe
pairing changes the function and should be compared on raw ARAD metrics.

Change `preset`, model overrides, stages, loss weights, or validation tiling in
the JSON configuration for controlled ablations.

## Inference

Reconstruct an arbitrary folder of RGB images:

```powershell
python scripts/infer.py `
  --checkpoint "runs\hsiformer_arad1k\checkpoints\best.pt" `
  --rgb-dir "D:\datasets\ARAD_1K\Train_RGB" `
  --split test `
  --output-dir "outputs\test_cubes" `
  --device cuda `
  --amp
```

Or point at an ARAD-1K root and let the split select its RGB directory:

```powershell
python scripts/infer.py `
  --checkpoint "runs\hsiformer_arad1k\checkpoints\best.pt" `
  --data-root "D:\datasets\ARAD_1K" `
  --split test `
  --output-dir "outputs\test_cubes" `
  --device cuda `
  --amp
```

After installation, the equivalent command is `hsiformer-infer`.

Each output is an NTIRE-compatible HDF5 `.mat` file containing `cube`, `bands`,
and `norm_factor`. The layout is directly readable by
`NTIRE2022Util.loadCube`. Trainer checkpoints also carry their RGB
normalization mode, which inference restores automatically. Raw legacy weights
without that metadata require an explicit
`--rgb-normalization per_image|scale_255`.

For limited GPU memory, add `--tile-size 256 --overlap 32`. Omit
`--tile-size` for full-frame inference and benchmark-comparable metrics:
independent tiles change a global-context model's predictions. Add `--clip`
only when clipped `[0, 1]` exported cubes are desired; raw MRAE/RMSE remain
unclipped.

## Public Test

Reproduce the reported validation/ARAD-origin comparison, export cubes, and
compute per-scene plus mean MRAE, RMSE, PSNR, SAM, and SSIM:

```powershell
python scripts/test_ntire.py `
  --checkpoint "runs\sstrans_source_arad1k\checkpoints\latest.pt" `
  --data-root "D:\datasets\ARAD_1K" `
  --output-dir "outputs\source_validation" `
  --split validation `
  --metric-profile source_arad_origin `
  --visualize `
  --device cuda
```

`--visualize` hands the exported `cubes/*.mat` and `metrics.csv` directly to
the repository-level [`hsi_viz_suite`](../hsi_viz_suite). It writes 300-DPI
PNG/PDF qualitative panels, MRAE/SAM/RMSE maps, spectral/residual plots,
bandwise summaries, metric ECDFs, and a reconstruction scatter under
`outputs/source_validation/figures/`. Install its plotting dependencies once
if needed:

```powershell
pip install -r ..\hsi_viz_suite\requirements.txt
```

Use `--viz-output`, `--viz-max-samples`, `--viz-dpi`, or `--viz-style` to
customize that handoff. An installed SSTrans package can locate a sibling suite
checkout automatically; otherwise pass `--hsi-viz-suite <path-to-hsi_viz_suite>`.

For the distinct 50-image public test split, use `--split test` and a separate
output directory. Its score must not be compared directly with `0.1468`.

If the cubes for 0951-1000 sit outside the standard directories, name that
location once:

```powershell
python scripts/test_ntire.py `
  --checkpoint "runs\sstrans_source_arad1k\checkpoints\latest.pt" `
  --data-root "/work3/<user>/dataset" `
  --output-dir "outputs\public_test" `
  --split test `
  --target-dir "Test_spectral" `
  --require-targets
```

If that split carries no spectral ground truth — the official release puts an
MSFA `mosaic` payload in `Test_Spec` — the run prints which scenes are
unscorable, reconstructs every scene, writes the cubes plus `inference.json`,
and skips `summary.json`/`metrics.csv` instead of failing. Pass
`--require-targets` to make missing ground truth a hard error, and use
`--split validation` when metrics are the point. Partially annotated roots are
scored on the scenes that do have cubes; `summary.json` reports both `count`
and `skipped`.

This is the unavoidable boundary between local benchmarking and a blind
leaderboard: SSTrans can compute metrics for any test manifest only when you
provide the matching 31-band reference cubes (via `--target-dir`, also accepted
as `--spectral-dir`). The official mosaic-only test release can be submitted or
visualized with `--visualize`, but its leaderboard score is computed only by the
challenge server after submission.

After installation, the equivalent command is `hsiformer-test`.
The upstream result notebook loads the Lightning `last.ckpt`, so the matching
local artifact is `latest.pt`; `best.pt` is retained as a useful additional
checkpoint selected by validation MRAE.

Outputs:

```text
outputs/public_test/
|-- cubes/*.mat
|-- metrics.csv
|-- summary.json
`-- figures/                 # when --visualize is passed
```

The default `--metric-profile source_arad_origin` reproduces the full-frame
protocol used by the reported SSTrans `0.1468` line. Use
`--metric-profile ntire_center` for the separate 128-pixel center-crop
MST++/NTIRE comparison, or `--metric-profile legacy_full` to inspect older
full-frame `clamp_min(1e-6)` logs. `summary.json` records the named profile,
crop, denominator, RGB normalization, precision, tiling, and export clipping.
The native `sam` value is radians; each report also includes `sam_degrees` and
`sam_unit` so publication plots consistently display SAM in degrees. `ssim` is
reported alongside the four common reconstruction metrics using the same
box-filter convention as the MSWR NTIRE tester.

The `0.1468` artifact is a validation/ARAD-origin result, not the held-out test
split. The command above makes both the split and metric protocol explicit;
the CLI also defaults to this matching pair.

## Package API

```python
from hsiformer import build_model

model = build_model("source_reproduction")
```

The package also exposes `ARAD1KDataset`, `RGBImageDataset`, `TrainingConfig`,
`train`, `predict_hsi`, `evaluate_loader`, and NTIRE cube I/O helpers.

## Verification

```powershell
python -m pytest
python scripts/smoke_model.py --preset source_reproduction
```

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

Spectral `.mat` files must contain a `cube` dataset. Data is loaded lazily, so
the complete dataset is not held in memory.

Check a local dataset:

```powershell
python scripts/inspect_arad.py "D:\datasets\ARAD_1K" --split test
```

## Training

`configs/train_arad1k.json` follows the MST++/published HSIFormer training
style:

- Adam with betas `(0.9, 0.999)`
- per-iteration cosine learning-rate decay to `1e-6`
- L1 objective, matching the reported ARAD-1K training procedure
- 300,000 iterations with 128x128 crops
- 50,000 iterations with 256x256 crops
- 50,000 full-resolution iterations (the 512 stage uses 482x512 ARAD frames)
- validation and checkpointing every 2,000 iterations

Start training:

```powershell
python scripts/train.py `
  --data-root "D:\datasets\ARAD_1K" `
  --output-dir "runs\hsiformer_arad1k" `
  --device cuda
```

After installation, the equivalent command is `hsiformer-train`.

Resume exactly from a trainer checkpoint:

```powershell
python scripts/train.py `
  --config configs/train_arad1k.json `
  --data-root "D:\datasets\ARAD_1K" `
  --resume "runs\hsiformer_arad1k\checkpoints\latest.pt"
```

Each checkpoint includes the model architecture, optimizer, scheduler, AMP
scaler, stage position, global iteration, and resolved training configuration.
Training metrics are appended to `metrics.jsonl`.

The default fresh-training preset is `recommended_retrain`. It keeps the
reported no-spectral-RPE improvement (MRAE `0.1497` to `0.1468`, SAM `0.0824`
to `0.0774`), restores the intended 2/4/8/16 spectral head schedule, retains
CAT's valid spatial relative bias, and uses the paper residual topology.
Available presets are:

- `legacy`: published source behavior, for reproducing existing checkpoints.
- `ablation_no_rpe`: strongest reported ablation with legacy head/residual
  behavior.
- `corrected_rpe`: places spectral RPE before softmax.
- `optimized_candidate`: no spectral/CAT RPE, corrected residual topology, and
  activation checkpointing.
- `recommended_retrain`: no spectral RPE, intended stage-wise spectral heads,
  CAT RPE, paper residual topology, and activation checkpointing.
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

After installation, the equivalent command is `hsiformer-infer`.

Each output is an NTIRE-compatible HDF5 `.mat` file containing `cube`, `bands`,
and `norm_factor`. The layout is directly readable by
`NTIRE2022Util.loadCube`.

For limited GPU memory, add `--tile-size 256 --overlap 32`. Omit
`--tile-size` for full-frame inference. Add `--clip` only when clipped
`[0, 1]` submission values are desired.

## Public Test

Run inference on the now-public 50-image test split, export NTIRE cubes, and
compute per-scene plus mean MRAE, RMSE, PSNR, and SAM:

```powershell
python scripts/test_ntire.py `
  --checkpoint "runs\hsiformer_arad1k\checkpoints\best.pt" `
  --data-root "D:\datasets\ARAD_1K" `
  --output-dir "outputs\public_test" `
  --device cuda `
  --amp
```

After installation, the equivalent command is `hsiformer-test`.

Outputs:

```text
outputs/public_test/
|-- cubes/*.mat
|-- metrics.csv
`-- summary.json
```

## Package API

```python
from hsiformer import build_model

model = build_model("recommended_retrain")
```

The package also exposes `ARAD1KDataset`, `RGBImageDataset`, `TrainingConfig`,
`train`, `predict_hsi`, `evaluate_loader`, and NTIRE cube I/O helpers.

## Verification

```powershell
python -m pytest
python scripts/smoke_model.py --preset recommended_retrain
```

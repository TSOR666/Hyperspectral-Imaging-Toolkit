# Unified Checkpoint Benchmarking

`benchmark_hsi.py` evaluates RGB-to-HSI checkpoints from this repository and
the official MST++ model zoo on CAVE, ICVL, BGU, or custom hyperspectral
datasets. It produces aligned predictions, full-reference metrics, per-band
statistics, confidence intervals, publication figures, and paper tables.

## Install

Install PyTorch for the desired CUDA version first, then:

```bash
pip install -r requirements-benchmark.txt
```

Some adapters also need the requirements of their project directory. WaveDiff,
for example, uses `torchvision`, while MSWR uses `PyWavelets`.

## Supported checkpoints

| Type | Checkpoint source |
| --- | --- |
| `cswin` | `CSWIN v2` generator checkpoints |
| `mswr:tiny`, `mswr:small`, `mswr:base`, `mswr:large` | MSWR v2 checkpoints |
| `hsifusion:SIZE` | HSIFusion v2.5.3 checkpoints |
| `sharp:SIZE` | SHARP v3.2.2 checkpoints |
| `wavediff[:TYPE]` | WaveDiff base/wavelet/adaptive checkpoints |
| `mst:METHOD` | Official MST++ model-zoo checkpoints |

MST++ `METHOD` can be `mst_plus_plus`, `mst`, `mirnet`, `hinet`, `mprnet`,
`restormer`, `edsr`, `hdnet`, `hrnet`, `hscnn_plus`, or `awan`.

The official MST++ architecture checkout is supplied separately:

```bash
git clone https://github.com/caiyuanhao1998/MST-plus-plus.git
```

Pass its root with `--mst-root`. Downloaded model-zoo `.pth` files can remain
anywhere.

## Run all datasets

The `NAME=TYPE@CHECKPOINT` syntax gives every method a stable table/figure
label. Quote the whole argument on Windows.

```bash
python benchmark_hsi.py \
  --model "CSWIN=cswin@checkpoints/cswin_best.pth" \
  --model "MSWR=mswr:base@checkpoints/mswr_best.pth" \
  --model "MST++=mst:mst_plus_plus@model_zoo/mst_plus_plus.pth" \
  --mst-root ../MST-plus-plus \
  --dataset "cave=D:/datasets/CAVE" \
  --dataset "icvl=D:/datasets/ICVL" \
  --dataset "bgu=D:/datasets/BGU" \
  --output results/cross_dataset \
  --device cuda \
  --tile-size 256 \
  --tile-overlap 32 \
  --figures 5
```

Use `--model-config NAME=config.json` when an older CSWIN checkpoint does not
embed the architecture configuration. Loading is strict by default. Partial
weights are rejected unless `--allow-partial-load` is explicitly supplied,
and even then at least 90% parameter coverage is required.

PyTorch safe loading is used first. `--trust-checkpoint` permits pickle-based
configuration objects and should only be used with trusted checkpoints.

## Dataset input

The automatic loader recursively supports:

- `.mat`, MATLAB v7.3/HDF5, `.h5`, `.hdf5`, `.npy`, and `.npz` cubes.
- Common cube keys: `cube`, `reflectance`, `rad`, `hsi`, `hyper`, and `data`.
- CAVE-style directories containing one grayscale image per spectral band.
- Paired RGB images sharing a scene name or ending in `_RGB`.

For unusual layouts, provide a CSV manifest:

```csv
name,hsi,rgb
scene_001,cubes/scene_001.mat,rgb/scene_001.png
scene_002,cubes/scene_002.mat,rgb/scene_002.png
```

```bash
python benchmark_hsi.py \
  --model "Model=mswr:base@model.pth" \
  --dataset "custom=D:/my_dataset" \
  --manifest "custom=D:/my_dataset/test.csv"
```

Use `--hsi-key` and `--rgb-key` to select nonstandard variables.

## Spectral protocol

All targets are resampled to 31 bands from 400 to 700 nm by default. The
source wavelength vector is resolved in this order:

1. `--wavelengths-file`
2. A wavelength variable embedded in the cube
3. `--source-range MIN MAX`
4. 400-700 nm for an existing 31-band cube
5. 400-1000 nm for multi-band ICVL/BGU presets

For paper results, verify the dataset's actual wavelength calibration instead
of relying on a preset. Change the output grid with `--target-range` and
`--target-bands`.

Reflectance scaling is recorded in `sample_metadata.json`. `--hsi-scale auto`
uses common 8/12/14/16-bit divisors. Prefer an explicit numeric divisor when
the acquisition protocol is known.

## RGB protocol

`--rgb-source auto` uses a paired RGB image when one is found. Otherwise, it
synthesizes approximate CIE-like RGB from the hyperspectral target. Each
sample records the selected protocol.

For defensible cross-dataset comparisons, use paired RGB or the dataset's
camera response:

```bash
--rgb-source response --response-file camera_response.npy
```

The response matrix must have shape `bands x 3`. Approximate CIE inputs are
useful for qualitative generalization tests, but should not be mixed with
paired-camera benchmark numbers without clearly reporting that domain shift.

## Metrics and outputs

Metrics are computed on aligned reflectance cubes in `[0, 1]`:

- MRAE
- RMSE
- PSNR
- SAM in degrees
- band-averaged SSIM
- MAE

Set `--crop-border 128` to reproduce the NTIRE/ARAD border-crop convention;
leave it at zero for full-image CAVE/ICVL/BGU evaluation unless a paper
protocol specifies otherwise.

Each `OUTPUT/DATASET/METHOD` directory contains:

- `summary.json`: model, protocol, means, standard deviations, and 95% bootstrap CIs
- `metrics.csv`: per-scene metrics, runtime, throughput, and GPU memory
- `per_band_metrics.csv`: MRAE/RMSE/PSNR/MAE by wavelength
- `per_band_summary.csv`: cross-scene mean and standard deviation by wavelength
- `hsi/*.npy`: prediction/target pairs compatible with `hsi_viz_suite`
- `figures/*.png` and `figures/*.pdf`: qualitative paper figures

The output root also contains `paper_table.csv`, `paper_table.md`,
`paper_table.tex`, and cross-method comparison figures.

To generate the repository's extended visualization set:

```bash
python hsi_viz_suite/scripts/generate_all_visualizations.py \
  --results results/cross_dataset/cave/MSWR \
  --output results/cross_dataset/cave/MSWR/extended_figures
```

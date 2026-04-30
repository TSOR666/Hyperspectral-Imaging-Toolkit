# HSI Reconstruction Visualization Suite

Self-contained utilities for turning `.npy` hyperspectral reconstruction outputs into publication-quality figures. Model-agnostic — accepts outputs from any pipeline (CSWIN, MSWR, HSIFusion, SHARP, WaveDiff, or custom models) that follow the ARAD-1K MST++ folder convention.

---

## Table of Contents

1. [What This Suite Produces](#what-this-suite-produces)
2. [Environment Setup](#environment-setup)
3. [Input Format](#input-format)
4. [One-Shot Pipeline](#one-shot-pipeline)
5. [Comparing Multiple Methods](#comparing-multiple-methods)
6. [Individual Scripts](#individual-scripts)
7. [CLI Reference](#cli-reference)
8. [Colour Conversion Utility](#colour-conversion-utility)
9. [Outputs](#outputs)
10. [Examples](#examples)
11. [Troubleshooting](#troubleshooting)
12. [Project Structure](#project-structure)

---

## What This Suite Produces

Running `generate_all_visualizations.py` produces five figure types:

| Folder | Figure type | Description |
|---|---|---|
| `main_figures/` | Reconstruction grid | Qualitative RGB renders of HSI reconstructions side by side |
| `error_maps/` | Error heatmaps | Per-pixel absolute error, coloured by magnitude |
| `spectral_analysis/` | Spectral curves | Ground truth vs. predicted spectral signatures at sampled pixel locations |
| `statistics/` | Metric distributions | PSNR / SAM / MRAE histograms and box plots across images |
| `comparison_grids/` | Method comparison | RGB renders + error maps for multiple methods in a single figure |

All figures are saved as both `.png` and `.pdf` to support papers and slide decks.

---

## Environment Setup

```bash
cd hsi_viz_suite
python -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

Core requirements: `numpy`, `matplotlib`, `seaborn`, `torch` (lightweight, CPU-only build is fine).

Optional for higher-quality PDF output:
```bash
pip install "matplotlib[latex]"      # LaTeX text rendering in figures
```

---

## Input Format

Each method folder must follow this layout:

```
/path/to/method_outputs/
├── hsi/                        # reconstructed HSI cubes
│   ├── img001.npy              # shape: (31, H, W) or (H, W, 31)
│   ├── img002.npy
│   └── ...
├── rgb/                        # corresponding RGB inputs (optional, for rendering)
│   ├── img001.png
│   └── ...
└── metrics.json                # optional — per-image evaluation metrics
```

**`metrics.json` format** (optional but enables statistics plots):

```json
{
  "img001": {"psnr": 38.5, "ssim": 0.94, "mrae": 0.032, "rmse": 0.018, "sam": 0.048},
  "img002": {"psnr": 37.1, "ssim": 0.92, "mrae": 0.041, "rmse": 0.022, "sam": 0.055},
  ...
}
```

If `metrics.json` is absent, PSNR/SAM statistics are recomputed from `.npy` files when ground truth is available.

**HSI cube convention**: the loader handles both `(31, H, W)` (channel-first) and `(H, W, 31)` (channel-last). Non-contiguous tensors are handled safely.

---

## One-Shot Pipeline

Generate all figure types for a single method:

```bash
cd hsi_viz_suite
python scripts/generate_all_visualizations.py \
  --results /path/to/method_outputs \
  --output figs \
  --dpi 300 \
  --style paper
```

This runs all downstream scripts in sequence and writes output under `figs/`:

```
figs/
├── main_figures/      # PNG + PDF
├── error_maps/        # PNG + PDF
├── spectral_analysis/ # PNG + PDF
├── statistics/        # PNG + PDF
└── comparison_grids/  # PNG + PDF (only with --methods)
```

---

## Comparing Multiple Methods

Pass additional result folders with `--methods` to generate side-by-side comparison grids:

```bash
python scripts/generate_all_visualizations.py \
  --results outputs/our_method \
  --methods outputs/baseline_a outputs/baseline_b outputs/baseline_c \
  --method-names "Ours" "Baseline A" "Baseline B" "Baseline C" \
  --output figs \
  --dpi 300
```

`--method-names` controls the legend labels. If omitted, directory names are used as labels. Each method folder must follow the same `hsi/` + optional `rgb/` layout.

---

## Individual Scripts

You can also run each script directly for finer control.

### `visualize_results.py` — Qualitative reconstruction grid

```bash
python scripts/visualize_results.py \
  --results /path/to/method_outputs \
  --output figs/main_figures \
  --max-samples 10 \
  --dpi 300
```

Renders each HSI reconstruction as a false-color RGB (using Gaussian-approximated CIE 1931 CMFs) alongside the ground-truth RGB input.

### `generate_error_maps.py` — Per-pixel error heatmaps

```bash
python scripts/generate_error_maps.py \
  --results /path/to/method_outputs \
  --ground-truth /path/to/ground_truth \
  --output figs/error_maps \
  --colormap hot \
  --max-samples 5
```

Produces absolute error maps per band and a mean-across-bands summary heatmap. `--colormap` accepts any Matplotlib colormap name.

### `plot_spectral_curves.py` — Spectral signature comparison

```bash
python scripts/plot_spectral_curves.py \
  --results /path/to/method_outputs \
  --ground-truth /path/to/ground_truth \
  --output figs/spectral_analysis \
  --n-pixels 5 \
  --max-samples 3
```

For each sample, picks `--n-pixels` pixel locations (uniformly spaced) and overlays the predicted vs. ground-truth spectral curves across 400–700 nm.

### `plot_metrics_statistics.py` — Aggregate metric distributions

```bash
python scripts/plot_metrics_statistics.py \
  --results /path/to/method_outputs \
  --output figs/statistics
```

Reads `metrics.json` (or computes metrics from `.npy` files) and produces histograms and box plots for PSNR, SSIM, MRAE, RMSE, and SAM.

### `create_comparison_grid.py` — Multi-method side-by-side

```bash
python scripts/create_comparison_grid.py \
  --methods outputs/ours outputs/baseline_a \
  --method-names "Ours" "Baseline A" \
  --ground-truth /path/to/ground_truth \
  --output figs/comparison_grids \
  --max-samples 5
```

Renders a grid with rows = samples, columns = methods, plus a ground-truth column and an error-map column.

---

## CLI Reference

### `generate_all_visualizations.py` (orchestrator)

| Flag | Default | Description |
|---|---|---|
| `--results` | required | Path to primary method output folder |
| `--output` | `figs` | Root output directory |
| `--methods` | — | Additional method folders for comparison |
| `--method-names` | folder names | Display names for legend labels |
| `--ground-truth` | auto-detected | Ground truth HSI folder (if separate from results) |
| `--max-samples` | `10` | Maximum samples per figure type |
| `--dpi` | `300` | Figure resolution |
| `--style` | `paper` | Matplotlib style: `paper` / `poster` / `default` |
| `--formats` | `png,pdf` | Output formats, comma-separated |
| `--no-stats` | disabled | Skip statistics plots |
| `--no-spectral` | disabled | Skip spectral curve plots |
| `--no-error-maps` | disabled | Skip error map generation |

---

## Colour Conversion Utility

`hsi_model/utils.py` exports a standalone `hsi_to_rgb()` function for converting 31-band HSI cubes to display-ready RGB:

```python
from hsi_model.utils import hsi_to_rgb, get_cached_cmf

# Convert a numpy array (31, H, W) → RGB (H, W, 3) uint8
rgb = hsi_to_rgb(hsi_cube)           # hsi_cube: np.ndarray or torch.Tensor

# Access cached CIE 1931 CMF coefficients directly
cmf = get_cached_cmf()               # shape (31, 3) — precomputed Gaussian approx
```

**Implementation details:**
- Uses Gaussian-approximated CIE 1931 colour-matching functions at 10 nm intervals (400–700 nm)
- Cache is computed once at import and reused; thread-safe
- Handles non-contiguous tensors safely (calls `.contiguous()` before numpy conversion)
- Clips to [0, 1] before uint8 conversion

For the ARAD-1K centre-crop (used in all NTIRE/MST++ evaluations):

```python
from hsi_model.utils import crop_center_arad1k

cropped = crop_center_arad1k(hsi_cube)   # (31, 482, 512) → (31, 226, 256)
```

---

## Outputs

Each figure script writes files in both PNG and PDF format. Filenames are consistent across runs (based on sample index and method name), making it easy to version-control outputs or replace them after re-running experiments.

Example output tree after a full comparison run:

```
figs/
├── main_figures/
│   ├── sample_000_ours.png
│   ├── sample_000_ours.pdf
│   └── ...
├── error_maps/
│   ├── sample_000_error_ours.png
│   └── ...
├── spectral_analysis/
│   ├── sample_000_spectral.png
│   └── ...
├── statistics/
│   ├── psnr_distribution.png
│   ├── sam_distribution.png
│   └── ...
└── comparison_grids/
    ├── grid_sample_000.png
    └── ...
```

---

## Examples

### Executable walkthrough

```bash
python examples/example_usage.py
```

This script generates a synthetic HSI cube, saves it as a `.npy` file, and runs the full visualization pipeline, printing each command so you can adapt it.

### Typical paper figure workflow

```bash
# 1. Run inference with your model → .npy files in outputs/ours/hsi/
# 2. Run baselines → outputs/baseline_a/hsi/, outputs/baseline_b/hsi/

# 3. Generate all figures
python scripts/generate_all_visualizations.py \
  --results outputs/ours \
  --methods outputs/baseline_a outputs/baseline_b \
  --method-names "Our Method" "SHARP" "MSWR" \
  --output paper_figs \
  --dpi 600 \
  --style paper

# 4. Find your figures at paper_figs/comparison_grids/*.pdf
```

---

## Troubleshooting

### `hsi_to_rgb` produces all-black or all-white output

This usually means the HSI cube is not in [0, 1] range. Normalize before passing:

```python
cube = cube / cube.max()             # simple max-normalize
rgb = hsi_to_rgb(cube)
```

### `FileNotFoundError: hsi/ directory not found`

The scripts expect a `hsi/` subdirectory inside `--results`. Create it or symlink your `.npy` files:

```bash
mkdir -p outputs/ours/hsi
cp /path/to/reconstructions/*.npy outputs/ours/hsi/
```

### Figures look pixelated at print size

Increase `--dpi 600` for print-quality output. For LaTeX papers, use the PDF output directly.

### `metrics.json` not found warning

The statistics script falls back to computing metrics from `.npy` files when `metrics.json` is absent. Provide ground truth with `--ground-truth` to enable this fallback.

### Spectral curve x-axis shows indices instead of wavelengths

The 31-band x-axis defaults to wavelength labels 400–700 nm at 10 nm intervals. If your data uses different bands, edit the `WAVELENGTHS` constant in `hsi_model/utils.py`.

---

## Project Structure

```
hsi_viz_suite/
├── hsi_model/
│   └── utils.py                       # hsi_to_rgb, get_cached_cmf, crop_center_arad1k
├── scripts/
│   ├── generate_all_visualizations.py # Orchestrator — runs all figure scripts
│   ├── visualize_results.py           # Qualitative reconstruction grids
│   ├── generate_error_maps.py         # Per-pixel error heatmaps
│   ├── plot_spectral_curves.py        # Spectral signature comparisons
│   ├── plot_metrics_statistics.py     # PSNR/SAM/MRAE distribution charts
│   └── create_comparison_grid.py      # Multi-method side-by-side grids
├── examples/
│   └── example_usage.py               # End-to-end executable walkthrough
├── requirements.txt
└── README.md
```

---

## Related Projects

Consume outputs from any of the following models:

- [`../CSWIN v2`](../CSWIN%20v2/README.md)
- [`../HSIFUSION&SHARP`](../HSIFUSION%26SHARP/README.md)
- [`../mswr_v2`](../mswr_v2/README.md)
- [`../WaveDiff`](../WaveDiff/README.md)

---

## License

Distributed under the [MIT License](../LICENSE).

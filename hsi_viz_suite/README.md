# HSI Reconstruction Visualization Suite

Self-contained utilities for turning hyperspectral reconstruction results into publication-ready plots. The suite consumes `.npy`/`.npz` tensors, classic MATLAB files, and SSTrans/NTIRE HDF5 `.mat` cubes, plus optional per-sample or aggregate metric files. It produces PNG/PDF outputs for qualitative inspection, error analysis, spectral curve comparisons, and aggregate publication figures.

## Features

- **Turn-key pipeline**: `scripts/generate_all_visualizations.py` orchestrates every downstream figure with a single command.
- **Robust colour conversion**: `hsi_model/utils.py` ships a Gaussian-approximated CIE 1931 CMF cache and a safe `hsi_to_rgb` helper that copes with non-contiguous tensors.
- **Batch export**: Figures are saved as both PNG and PDF to make it easy to drop them into papers or slide decks.
- **Method comparison grids**: Supply additional result folders to highlight differences against baselines or competitors.
- **SSTrans compatible**: Read `outputs/.../cubes/<scene>.mat`, `bands`, `norm_factor`, `metrics.csv`, and prediction-only inference folders directly.
- **Model-agnostic inputs**: Consume `.npy` outputs from CSWIN, MSWR, HSIFusion, SHARP, or any compatible MST++ pipeline.
- **Publication figures**: Generate bandwise error summaries, prediction-vs-target density scatters, ECDFs, metric summary heatmaps, and prediction spectral overviews.

## Installation

```bash
cd hsi_viz_suite
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The requirements cover `numpy`, `matplotlib`, `seaborn`, `torch` (for colour matching and spatial metrics), `scipy`, `pandas`, and `h5py` (for SSTrans/NTIRE HDF5 `.mat` files).

## Usage

### One-shot pipeline

```bash
python scripts/generate_all_visualizations.py \
  --results /path/to/our_method \
  --output figs \
  --dpi 300 \
  --style paper
```

For an SSTrans evaluation or inference folder, point `--results` at the folder containing `cubes/` (the `cubes/` child itself is also accepted). Targets are optional for prediction-only figures; provide the ARAD spectral directory to enable error maps and paired plots:

```bash
python scripts/generate_all_visualizations.py \
  --results outputs/source_validation \
  --targets /path/to/ARAD_1K/Train_spectral \
  --output figs/sstrans \
  --dpi 300
```

Scored SSTrans runs record the reference directory in `summary.json`, so the
`--targets` argument can be omitted when that directory still exists. The
alias `--target-dir` is accepted as well. The pipeline prints the number of
prediction samples, metric rows, and paired targets it discovered; it stops
with the exact results/target paths when a requested target directory cannot
be paired.

SSTrans' `scripts/test_ntire.py --visualize` runs this exact command
automatically after cube export. Its reports retain native SAM radians while
also writing `sam_degrees`; the suite uses the degree-normalized value in
qualitative annotations and cross-method statistics.

This command will:

1. Render qualitative reconstructions under `figs/main_figures/`.
2. Produce error heat maps for up to five samples in `figs/error_maps/`.
3. Plot spectral curves for three samples at multiple pixel locations in `figs/spectral_analysis/`.
4. Add aggregate publication figures under `figs/publication/`: prediction spectral overview, bandwise MRAE/RMSE summary, and a density-aware reconstruction scatter when targets are available.
5. Aggregate metric distributions (e.g., PSNR/SAM) across methods in `figs/statistics/`, including ECDFs and a mean ± SD heatmap when metric rows are available.
6. Optionally generate comparison grids if `--methods` is provided.

### Comparing multiple methods

```bash
python scripts/generate_all_visualizations.py \
  --results outputs/ours \
  --methods outputs/baseline_a outputs/baseline_b \
  --method-names "Ours" "Baseline A" "Baseline B" \
  --output figs
```

Provide explicit `--method-names` to control legend labels; otherwise directory names are used. Each method folder may include per-sample prediction/target pairs (`hsi/<sample>.npy` or `hsi/<sample>_pred.npy`, plus `hsi/<sample>_target.npy`) or SSTrans/NTIRE cubes under `cubes/<sample>.mat`. Optional metric inputs are `metrics/<sample>_metrics.json` or SSTrans `metrics.csv`.

### Customising samples and styling

Key arguments exposed by the wrapper script:

| Flag | Description | Default |
| --- | --- | --- |
| `--max-samples` | Maximum number of samples processed per figure type. | `10` |
| `--targets`, `--target-dir` | Optional target directory; supports a direct scene-file directory such as `Train_spectral`. Scored SSTrans reports are auto-discovered when omitted. | `None` |
| `--dpi` | Rendering resolution for saved figures. | `300` |
| `--style` | Matplotlib style sheet (`paper`, `poster`, etc.). | `paper` |
| `--methods` | Additional result directories for comparisons. | `None` |
| `--method-names` | Display names corresponding to `--methods`. | derived from folder names |

For granular control you can run the individual scripts in `scripts/` directly (e.g., `visualize_results.py`, `generate_error_maps.py`, or `plot_publication_figures.py`), passing the same `--targets` argument when paired SSTrans analysis is desired.

## Example

An executable walkthrough is included in [`examples/example_usage.py`](examples/example_usage.py):

```bash
python examples/example_usage.py
```

It prints the full command to reproduce the end-to-end pipeline.

## Outputs

Every figure directory contains both `.png` and `.pdf` versions. File names are consistent across runs to simplify versioning. Use the generated folders as-is for reports or track them with Git LFS if you need to store artefacts alongside code.

## License

The visualization suite is provided under the [MIT License](../LICENSE). By contributing or redistributing, you agree to the terms outlined therein.

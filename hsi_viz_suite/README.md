# HSI Reconstruction Visualization Suite

Self-contained utilities for turning hyperspectral reconstruction results into publication-ready plots. The suite consumes `.npy` tensors and optional `metrics.json` files (ARAD-1K style) and produces PNG/PDF outputs for qualitative inspection, error analysis, and spectral curve comparisons.

## Features

- **Turn-key pipeline**: `scripts/generate_all_visualizations.py` orchestrates every downstream figure with a single command.
- **Robust colour conversion**: `hsi_model/utils.py` ships a Gaussian-approximated CIE 1931 CMF cache and a safe `hsi_to_rgb` helper that copes with non-contiguous tensors.
- **Batch export**: Figures are saved as both PNG and PDF to make it easy to drop them into papers or slide decks.
- **Method comparison grids**: Supply additional result folders to highlight differences against baselines or competitors.
- **Model-agnostic inputs**: Consume `.npy` outputs from CSWIN, MSWR, HSIFusion, SHARP, or any compatible MST++ pipeline.

## Installation

```bash
cd hsi_viz_suite
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The requirements cover `numpy`, `matplotlib`, `seaborn`, and `torch` (for the lightweight colour matching utilities).

## Usage

### One-shot pipeline

```bash
python scripts/generate_all_visualizations.py \
  --results /path/to/our_method \
  --output figs \
  --dpi 300 \
  --style paper
```

This command will:

1. Render qualitative reconstructions under `figs/main_figures/`.
2. Produce error heat maps for up to five samples in `figs/error_maps/`.
3. Plot spectral curves for three samples at multiple pixel locations in `figs/spectral_analysis/`.
4. Aggregate metric distributions (e.g., PSNR/SAM) across methods in `figs/statistics/`.
5. Optionally generate comparison grids if `--methods` is provided.

### Comparing multiple methods

```bash
python scripts/generate_all_visualizations.py \
  --results outputs/ours \
  --methods outputs/baseline_a outputs/baseline_b \
  --method-names "Ours" "Baseline A" "Baseline B" \
  --output figs
```

Provide explicit `--method-names` to control legend labels; otherwise directory names are used. Each method folder should mirror the ARAD-1K MST++ layout (`hsi/*.npy`, `rgb/*.png`, optional `metrics.json`).

### Customising samples and styling

Key arguments exposed by the wrapper script:

| Flag | Description | Default |
| --- | --- | --- |
| `--max-samples` | Maximum number of samples processed per figure type. | `10` |
| `--dpi` | Rendering resolution for saved figures. | `300` |
| `--style` | Matplotlib style sheet (`paper`, `poster`, etc.). | `paper` |
| `--methods` | Additional result directories for comparisons. | `None` |
| `--method-names` | Display names corresponding to `--methods`. | derived from folder names |

For granular control you can run the individual scripts in `scripts/` directly (e.g., `visualize_results.py`, `generate_error_maps.py`), passing the same arguments the orchestrator uses.

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

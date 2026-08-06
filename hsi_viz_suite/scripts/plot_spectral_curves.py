from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from hsi_io import HsiSample, load_hsi
from result_layout import prediction_path, target_path
from visualization_utils import compute_bandwise_errors, save_figure, setup_publication_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class SpectralAnalyzer:
    """Plot pixel signatures, image-level spectra, and bandwise errors."""

    def __init__(
        self,
        results_dir: str,
        output_dir: str,
        dpi: int = 300,
        targets_dir: str | None = None,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.targets_dir = Path(targets_dir) if targets_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_publication_style("paper", dpi)

    def _load_pair(self, sample: str) -> Tuple[Optional[HsiSample], Optional[HsiSample]]:
        """Load prediction and optional target cubes with band metadata."""
        pred_file = prediction_path(self.results_dir, sample)
        target_root = self.targets_dir or self.results_dir
        target_file = target_path(
            target_root,
            sample,
            allow_direct=self.targets_dir is not None,
        )
        if pred_file is None:
            return None, None
        pred = load_hsi(pred_file)
        target = load_hsi(target_file) if target_file is not None else None
        return pred, target

    def analyze_spectral_signatures(
        self,
        sample: str,
        pixel_locations: List[Tuple[int, int]],
    ) -> None:
        """Create a four-panel publication figure for one sample."""
        pred_sample, target_sample = self._load_pair(sample)
        if pred_sample is None:
            print(f"Skipping {sample}: prediction file not found")
            return
        pred = pred_sample.cube
        target = target_sample.cube if target_sample is not None else None
        if target is not None and target.shape != pred.shape:
            print(f"Skipping {sample}: shape mismatch {pred.shape} vs {target.shape}")
            return

        height, width = pred.shape[-2:]
        pixels = [
            (min(max(int(y), 0), height - 1), min(max(int(x), 0), width - 1))
            for y, x in pixel_locations
        ]
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.5), constrained_layout=True)
        self._plot_individual(axes[0, 0], pred_sample, target_sample, pixels[:5])
        self._plot_mean_spectra(axes[0, 1], pred_sample, target_sample)
        if target is not None:
            assert target_sample is not None
            self._plot_bandwise_errors(axes[1, 0], pred_sample, target_sample)
            self._plot_residual_distribution(axes[1, 1], pred, target)
        else:
            self._plot_prediction_spread(axes[1, 0], pred_sample)
            self._plot_band_quantiles(axes[1, 1], pred_sample)
        fig.suptitle(sample, y=1.02, fontweight="bold")
        save_figure(fig, self.output_dir / f"spectral_analysis_{sample}")

    def _plot_individual(
        self,
        ax: "Axes",
        pred: HsiSample,
        target: Optional[HsiSample],
        pixels: List[Tuple[int, int]],
    ) -> None:
        cmap = plt.get_cmap("tab10")
        for i, (y, x) in enumerate(pixels):
            color = cmap(i % 10)
            if target is not None:
                ax.plot(
                    target.wavelengths,
                    target.cube[:, y, x],
                    color=color,
                    linewidth=1.1,
                    label=f"GT ({x},{y})",
                )
            ax.plot(
                pred.wavelengths,
                pred.cube[:, y, x],
                "--" if target is not None else "-",
                color=color,
                linewidth=1.1,
                label=f"Pred ({x},{y})",
            )
        ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance", title="Pixel signatures")
        ax.grid(True, alpha=0.25)
        if pixels:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=False)

    def _plot_mean_spectra(
        self,
        ax: "Axes",
        pred: HsiSample,
        target: Optional[HsiSample],
    ) -> None:
        pred_values = pred.cube.reshape(pred.cube.shape[0], -1)
        pred_mean = pred_values.mean(axis=1)
        pred_low, pred_high = np.percentile(pred_values, [5, 95], axis=1)
        ax.plot(pred.wavelengths, pred_mean, color="#0072B2", label="Prediction")
        ax.fill_between(pred.wavelengths, pred_low, pred_high, color="#0072B2", alpha=0.16)
        if target is not None:
            target_values = target.cube.reshape(target.cube.shape[0], -1)
            target_mean = target_values.mean(axis=1)
            target_low, target_high = np.percentile(target_values, [5, 95], axis=1)
            ax.plot(target.wavelengths, target_mean, color="#D55E00", label="Ground truth")
            ax.fill_between(target.wavelengths, target_low, target_high, color="#D55E00", alpha=0.16)
        ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance", title="Image mean ± 90% range")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    def _plot_bandwise_errors(self, ax: "Axes", pred: HsiSample, target: HsiSample) -> None:
        errors = compute_bandwise_errors(pred.cube, target.cube)
        ax.plot(pred.wavelengths, errors["mrae"], color="#CC79A7", label="MRAE")
        ax.set(xlabel="Wavelength (nm)", ylabel="MRAE", title="Bandwise reconstruction error")
        ax2 = ax.twinx()
        ax2.plot(pred.wavelengths, errors["rmse"], color="#009E73", linestyle="--", label="RMSE")
        ax2.set_ylabel("RMSE")
        ax.grid(True, alpha=0.25)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)

    def _plot_residual_distribution(self, ax: "Axes", pred: np.ndarray, target: np.ndarray) -> None:
        residual = (pred - target).reshape(-1)
        residual = residual[np.isfinite(residual)]
        limit = max(float(np.percentile(np.abs(residual), 99)), 1e-6)
        ax.hist(residual, bins=50, range=(-limit, limit), color="#56B4E9", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set(xlabel="Prediction − ground truth", ylabel="Pixels × bands", title="Residual distribution")
        ax.grid(True, axis="y", alpha=0.25)

    def _plot_prediction_spread(self, ax: "Axes", pred: HsiSample) -> None:
        values = pred.cube.reshape(pred.cube.shape[0], -1)
        low, high = np.percentile(values, [5, 95], axis=1)
        ax.fill_between(pred.wavelengths, low, high, color="#009E73", alpha=0.2)
        ax.plot(pred.wavelengths, high - low, color="#009E73")
        ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance range", title="Spatial spectral spread")
        ax.grid(True, alpha=0.25)

    def _plot_band_quantiles(self, ax: "Axes", pred: HsiSample) -> None:
        values = pred.cube.reshape(pred.cube.shape[0], -1)
        low, median, high = np.percentile(values, [5, 50, 95], axis=1)
        ax.fill_between(pred.wavelengths, low, high, color="#E69F00", alpha=0.2)
        ax.plot(pred.wavelengths, median, color="#E69F00")
        ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance", title="Prediction quantiles")
        ax.grid(True, alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot HSI spectral signatures and errors.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--targets", help="Optional target directory, e.g. ARAD-1K/Train_spectral")
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pixels", nargs="+", type=int)
    args = ap.parse_args()
    analyzer = SpectralAnalyzer(args.results, args.output, args.dpi, args.targets)
    if args.pixels and len(args.pixels) % 2 == 0:
        iterator = iter(args.pixels)
        pixels = list(zip(iterator, iterator))
    else:
        pixels = [(64, 64), (128, 128), (192, 192)]
    for sample in args.samples:
        analyzer.analyze_spectral_signatures(sample, pixels)


if __name__ == "__main__":
    main()

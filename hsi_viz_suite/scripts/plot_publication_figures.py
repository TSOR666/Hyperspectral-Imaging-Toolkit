"""Additional publication figures for HSI reconstruction experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from hsi_io import HsiSample, load_hsi
from result_layout import find_prediction_samples, prediction_path, target_path
from visualization_utils import compute_bandwise_errors, save_figure, setup_publication_style


class PublicationFigureGenerator:
    """Generate aggregate spectra and prediction-vs-target density figures."""

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

    def _load_prediction(self, sample: str) -> Optional[HsiSample]:
        path = prediction_path(self.results_dir, sample)
        return load_hsi(path) if path is not None else None

    def _load_target(self, sample: str) -> Optional[HsiSample]:
        root = self.targets_dir or self.results_dir
        path = target_path(root, sample, allow_direct=self.targets_dir is not None)
        return load_hsi(path) if path is not None else None

    def prediction_spectral_overview(self, samples: list[str]) -> None:
        """Plot per-scene mean spectra so prediction-only SSTrans runs are useful."""
        loaded = [(sample, self._load_prediction(sample)) for sample in samples]
        loaded = [(sample, data) for sample, data in loaded if data is not None]
        if not loaded:
            return
        fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(loaded)))
        for color, (sample, data) in zip(colors, loaded, strict=True):
            assert data is not None
            mean = data.cube.reshape(data.cube.shape[0], -1).mean(axis=1)
            ax.plot(data.wavelengths, mean, color=color, linewidth=1.0, label=sample)
        ax.set(
            xlabel="Wavelength (nm)",
            ylabel="Mean reflectance",
            title="Prediction spectral overview",
        )
        ax.grid(True, alpha=0.25)
        if len(loaded) <= 12:
            ax.legend(frameon=False, ncol=2, fontsize=7)
        save_figure(fig, self.output_dir / "prediction_spectral_overview")

    def spectral_error_summary(self, samples: list[str]) -> None:
        """Plot median and interquartile bandwise MRAE/RMSE across scenes."""
        paired: list[tuple[HsiSample, HsiSample]] = []
        for sample in samples:
            pred = self._load_prediction(sample)
            target = self._load_target(sample)
            if pred is not None and target is not None and pred.cube.shape == target.cube.shape:
                paired.append((pred, target))
        if not paired:
            return

        reference_wavelengths = paired[0][0].wavelengths
        summaries: dict[str, list[np.ndarray]] = {"mrae": [], "rmse": []}
        for pred, target in paired:
            values = compute_bandwise_errors(pred.cube, target.cube)
            for name in summaries:
                series = values[name]
                if not np.array_equal(pred.wavelengths, reference_wavelengths):
                    series = np.interp(reference_wavelengths, pred.wavelengths, series)
                summaries[name].append(series)

        fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True, constrained_layout=True)
        for ax, name, color, label in zip(
            axes,
            ("mrae", "rmse"),
            ("#CC79A7", "#009E73"),
            ("MRAE", "RMSE"),
            strict=True,
        ):
            matrix = np.stack(summaries[name])
            median = np.median(matrix, axis=0)
            low, high = np.percentile(matrix, [25, 75], axis=0)
            ax.fill_between(reference_wavelengths, low, high, color=color, alpha=0.22, label="IQR")
            ax.plot(reference_wavelengths, median, color=color, linewidth=1.5, label="Median")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, loc="upper right")
        axes[-1].set_xlabel("Wavelength (nm)")
        fig.suptitle(f"Bandwise error across {len(paired)} scene(s)", fontweight="bold")
        save_figure(fig, self.output_dir / "bandwise_error_summary")

    def reconstruction_scatter(self, samples: list[str], max_points: int = 30_000) -> None:
        """Plot a density-aware prediction-vs-target scatter with identity line."""
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for sample in samples:
            pred = self._load_prediction(sample)
            target = self._load_target(sample)
            if pred is None or target is None or pred.cube.shape != target.cube.shape:
                continue
            prediction, reference = _paired_subsample(
                pred.cube,
                target.cube,
                max_points=max(max_points // max(len(samples), 1), 1),
            )
            predictions.append(prediction)
            targets.append(reference)
        if not predictions:
            return
        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        count = min(prediction.size, target.size)
        prediction, target = prediction[:count], target[:count]
        lower = float(min(np.min(prediction), np.min(target)))
        upper = float(max(np.max(prediction), np.max(target)))
        padding = max((upper - lower) * 0.03, 1e-6)
        lower -= padding
        upper += padding
        residual = prediction - target
        r2 = 1.0 - float(np.sum(residual**2) / max(np.sum((target - target.mean()) ** 2), 1e-12))
        correlation = float(np.corrcoef(prediction, target)[0, 1]) if count > 1 else float("nan")
        fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
        if count >= 500:
            image = ax.hexbin(target, prediction, gridsize=75, mincnt=1, bins="log", cmap="magma")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="log₁₀ density")
        else:
            ax.scatter(target, prediction, s=8, alpha=0.5, color="#0072B2", linewidth=0)
        ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.0)
        ax.set(
            xlim=(lower, upper),
            ylim=(lower, upper),
            xlabel="Ground-truth reflectance",
            ylabel="Predicted reflectance",
            title="Spectral reconstruction agreement",
        )
        ax.text(
            0.04,
            0.96,
            f"R² = {r2:.4f}\nr = {correlation:.4f}\nN = {count:,}",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "linewidth": 0},
        )
        ax.grid(True, alpha=0.2)
        save_figure(fig, self.output_dir / "reconstruction_scatter")

def _paired_subsample(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten aligned finite pixels and sample identical positions."""
    predicted = np.asarray(prediction).reshape(-1)
    reference = np.asarray(target).reshape(-1)
    mask = np.isfinite(predicted) & np.isfinite(reference)
    predicted = predicted[mask]
    reference = reference[mask]
    if predicted.size <= max_points:
        return predicted, reference
    indices = np.linspace(0, predicted.size - 1, max_points, dtype=np.int64)
    return predicted[indices], reference[indices]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate aggregate publication HSI figures.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--targets", help="Optional target directory, e.g. ARAD-1K/Train_spectral")
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", nargs="*")
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    samples = args.samples or find_prediction_samples(Path(args.results), max_samples=args.max_samples)
    generator = PublicationFigureGenerator(args.results, args.output, args.dpi, args.targets)
    generator.prediction_spectral_overview(samples)
    generator.spectral_error_summary(samples)
    generator.reconstruction_scatter(samples)


if __name__ == "__main__":
    main()

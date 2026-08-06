from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics_io import load_metric_rows
from visualization_utils import save_figure, setup_publication_style


class MetricsStatisticsPlotter:
    """Publication-quality distribution and summary plots for scalar metrics."""

    def __init__(
        self,
        results_dirs: Dict[str, str],
        output_dir: str,
        dpi: int = 300,
    ) -> None:
        self.results_dirs = {name: Path(directory) for name, directory in results_dirs.items()}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_publication_style("paper", dpi)
        self.df = self._load_all()

    def _load_all(self) -> pd.DataFrame:
        data: list[dict[str, object]] = []
        for method, result_dir in self.results_dirs.items():
            for row in load_metric_rows(result_dir):
                enriched = dict(row)
                enriched["method"] = method
                data.append(enriched)
        return pd.DataFrame(data)

    def violin(self, metrics: List[str] | None = None) -> None:
        """Save violin distributions with median and sample observations."""
        if metrics is None:
            metrics = ["mrae", "rmse", "psnr", "sam"]
        available = [metric for metric in metrics if metric in self.df.columns]
        if not available or self.df.empty:
            return
        fig, axes = plt.subplots(
            1,
            len(available),
            figsize=(2.9 * len(available), 4.0),
            squeeze=False,
        )
        palette = {
            method: color
            for method, color in zip(self.results_dirs, plt.get_cmap("tab10").colors)
        }
        methods = list(self.results_dirs)
        for index, metric in enumerate(available):
            ax = axes[0, index]
            groups = []
            labels = []
            colors = []
            for method in methods:
                values = pd.to_numeric(
                    self.df.loc[self.df["method"] == method, metric], errors="coerce"
                ).dropna().to_numpy()
                if values.size:
                    groups.append(values)
                    labels.append(method)
                    colors.append(palette[method])
            if not groups:
                ax.set_visible(False)
                continue
            parts = ax.violinplot(groups, showmeans=False, showmedians=True, showextrema=False)
            for body, color in zip(parts["bodies"], colors, strict=True):
                body.set_facecolor(color)
                body.set_edgecolor("white")
                body.set_alpha(0.72)
            for x, values, color in zip(range(1, len(groups) + 1), groups, colors, strict=True):
                jitter = np.linspace(-0.07, 0.07, values.size)
                ax.scatter(
                    np.full(values.size, x) + jitter,
                    values,
                    s=10,
                    color=color,
                    alpha=0.45,
                    linewidth=0,
                )
                ax.scatter(x, np.median(values), marker="_", s=180, color="black", linewidth=1.2, zorder=4)
            ax.set_xticks(range(1, len(labels) + 1), labels, rotation=40, ha="right")
            ax.set_title(metric.upper(), fontweight="bold")
            ax.grid(True, axis="y", alpha=0.25)
        fig.supxlabel("Method")
        fig.supylabel("Metric value")
        save_figure(fig, self.output_dir / "metrics_violin_plots")

    def ecdf(self, metrics: List[str] | None = None) -> None:
        """Save empirical CDFs, useful when aggregate means hide outliers."""
        if metrics is None:
            metrics = ["mrae", "rmse", "psnr", "sam"]
        available = [metric for metric in metrics if metric in self.df.columns]
        if not available or self.df.empty:
            return
        fig, axes = plt.subplots(
            1,
            len(available),
            figsize=(2.9 * len(available), 3.5),
            squeeze=False,
        )
        colors = plt.get_cmap("tab10").colors
        for index, metric in enumerate(available):
            ax = axes[0, index]
            for method_index, method in enumerate(self.results_dirs):
                values = pd.to_numeric(
                    self.df.loc[self.df["method"] == method, metric], errors="coerce"
                ).dropna().sort_values().to_numpy()
                if values.size:
                    y = np.arange(1, values.size + 1) / values.size
                    ax.step(
                        values,
                        y,
                        where="post",
                        label=method,
                        color=colors[method_index % len(colors)],
                    )
            ax.set_title(metric.upper(), fontweight="bold")
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.25)
        axes[0, 0].legend(frameon=False, loc="lower right")
        fig.supxlabel("Metric value")
        fig.supylabel("Empirical CDF")
        save_figure(fig, self.output_dir / "metrics_ecdf")

    def summary_table(self, metrics: List[str] | None = None) -> None:
        """Save a mean ± standard deviation heatmap and CSV summary."""
        if metrics is None:
            metrics = ["mrae", "rmse", "psnr", "sam"]
        valid_metrics = [metric for metric in metrics if metric in self.df.columns]
        if not valid_metrics or self.df.empty:
            return
        rows: list[dict[str, object]] = []
        matrix: list[list[float]] = []
        annotations: list[list[str]] = []
        for method in self.results_dirs:
            row: dict[str, object] = {"method": method}
            values: list[float] = []
            labels: list[str] = []
            for metric in valid_metrics:
                series = pd.to_numeric(
                    self.df.loc[self.df["method"] == method, metric], errors="coerce"
                ).dropna()
                mean = float(series.mean()) if not series.empty else float("nan")
                std = float(series.std(ddof=0)) if not series.empty else float("nan")
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                values.append(mean)
                labels.append("—" if not np.isfinite(mean) else f"{mean:.4g}\n± {std:.2g}")
            rows.append(row)
            matrix.append(values)
            annotations.append(labels)
        pd.DataFrame(rows).to_csv(self.output_dir / "metrics_summary.csv", index=False)
        fig, ax = plt.subplots(figsize=(1.8 + 1.15 * len(valid_metrics), 1.4 + 0.55 * len(rows)))
        image = np.asarray(matrix, dtype=float)
        masked = np.ma.masked_invalid(image)
        cmap = "mako" if "mako" in plt.colormaps() else "viridis"
        im = ax.imshow(masked, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(valid_metrics)), [metric.upper() for metric in valid_metrics])
        ax.set_yticks(range(len(rows)), [str(row["method"]) for row in rows])
        for r, annotation_row in enumerate(annotations):
            for c, label in enumerate(annotation_row):
                ax.text(c, r, label, ha="center", va="center", fontsize=7)
        ax.set_title("Mean ± population SD", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        save_figure(fig, self.output_dir / "metrics_summary_heatmap")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot HSI metric distributions.")
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--metrics", nargs="*", default=["mrae", "rmse", "psnr", "sam"])
    args = ap.parse_args()
    if len(args.names) != len(args.results):
        raise SystemExit("--names and --results must contain the same number of entries")
    plotter = MetricsStatisticsPlotter(
        dict(zip(args.names, args.results, strict=True)),
        args.output,
        args.dpi,
    )
    plotter.violin(args.metrics)
    plotter.ecdf(args.metrics)
    plotter.summary_table(args.metrics)


if __name__ == "__main__":
    main()

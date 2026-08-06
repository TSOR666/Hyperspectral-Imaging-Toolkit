from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make the sibling hsi_model package importable when this file is called by
# an absolute path from the repository root.
_SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(_SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUITE_ROOT))

from hsi_io import load_hsi
from hsi_model.utils import crop_center_arad1k, get_cached_cmf, hsi_to_rgb
from metrics_io import load_metric_for_sample
from result_layout import (
    HsiResultLayoutError,
    filter_prediction_samples,
    find_prediction_samples,
    no_prediction_samples_message,
    prediction_path,
    target_path,
)
from visualization_utils import (
    compute_mrae_map,
    compute_sam_map,
    robust_limits,
    save_figure,
    setup_publication_style,
)


class ResultsVisualizer:
    """Create compact qualitative figures from native or SSTrans outputs."""

    def __init__(
        self,
        results_dir: str,
        output_dir: str,
        style: str = "paper",
        dpi: int = 300,
        targets_dir: str | None = None,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.targets_dir = Path(targets_dir) if targets_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        setup_publication_style(style, dpi)
        self._cache: Dict[str, Any] = {}

    def _target_path(self, sample_name: str) -> Path | None:
        root = self.targets_dir or self.results_dir
        return target_path(root, sample_name, allow_direct=self.targets_dir is not None)

    def load_sample_data(self, sample_name: str) -> Dict[str, Any]:
        if sample_name in self._cache:
            return self._cache[sample_name]
        data: Dict[str, Any] = {}
        pred_path = prediction_path(self.results_dir, sample_name)
        if pred_path is not None:
            pred = load_hsi(pred_path)
            data["pred_hsi"] = pred.cube
            data["pred_wavelengths"] = pred.wavelengths
            data["pred_source"] = pred.source
        tgt_path = self._target_path(sample_name)
        if tgt_path is not None:
            target = load_hsi(tgt_path)
            data["target_hsi"] = target.cube
            data["target_wavelengths"] = target.wavelengths
            data["target_source"] = target.source
        metrics = load_metric_for_sample(self.results_dir, sample_name)
        if metrics is not None:
            data["metrics"] = metrics
        self._cache[sample_name] = data
        return data

    @staticmethod
    def _rgb(cube: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(cube).float().unsqueeze(0)
        cmf = get_cached_cmf(cube.shape[0], torch.device("cpu"))
        return hsi_to_rgb(tensor, cmf).squeeze(0).permute(1, 2, 0).numpy()

    def create_comparison_figure(
        self,
        sample_names: List[str],
        save_name: str = "comparison_figure",
        show_metrics: bool = True,
        crop_arad1k_flag: bool = False,
    ) -> None:
        sample_names = filter_prediction_samples(self.results_dir, sample_names)
        if not sample_names:
            raise HsiResultLayoutError(no_prediction_samples_message(self.results_dir))

        rows: list[dict[str, Any]] = []
        for name in sample_names[:4]:
            data = self.load_sample_data(name)
            pred = torch.from_numpy(data["pred_hsi"]).float().unsqueeze(0)
            target_data = data.get("target_hsi")
            target = (
                torch.from_numpy(target_data).float().unsqueeze(0)
                if target_data is not None
                else None
            )
            if crop_arad1k_flag:
                pred = crop_center_arad1k(pred)
                if target is not None:
                    target = crop_center_arad1k(target)
            pred_cube = pred.squeeze(0).numpy()
            row: dict[str, Any] = {
                "name": name,
                "pred_rgb": self._rgb(pred_cube),
                "pred_cube": pred_cube,
                "metrics": data.get("metrics"),
            }
            if target is not None and target.shape == pred.shape:
                target_cube = target.squeeze(0).numpy()
                row["target_rgb"] = self._rgb(target_cube)
                row["mrae"] = compute_mrae_map(pred, target)
                row["sam"] = compute_sam_map(pred, target)
                row["rgb_diff"] = np.abs(row["pred_rgb"] - row["target_rgb"]).mean(axis=-1)
            elif target is not None:
                row["target_rgb"] = self._rgb(target.squeeze(0).numpy())
                row["target_error"] = (
                    f"shape mismatch\n{tuple(pred.shape[1:])} vs {tuple(target.shape[1:])}"
                )
            rows.append(row)

        has_target = any("target_rgb" in row for row in rows)
        has_errors = any("mrae" in row for row in rows)
        ncols = 5 if has_errors else 2
        headers = (
            ["Ground truth", "Prediction", "MRAE", "SAM", "RGB |Δ|"]
            if has_errors
            else (["Ground truth", "Prediction"] if has_target else ["Prediction", "Spectral mean"])
        )
        fig, axes = plt.subplots(
            len(rows),
            ncols,
            figsize=(2.35 * ncols, 2.45 * len(rows)),
            squeeze=False,
            constrained_layout=True,
        )
        mrae_values = [row["mrae"] for row in rows if "mrae" in row]
        sam_values = [row["sam"] for row in rows if "sam" in row]
        mrae_limits = (
            robust_limits(np.concatenate([value.ravel() for value in mrae_values]), floor=0.0)
            if mrae_values
            else (0.0, 1.0)
        )
        sam_limits = (
            robust_limits(np.concatenate([value.ravel() for value in sam_values]), floor=0.0)
            if sam_values
            else (0.0, 1.0)
        )

        for row_index, row in enumerate(rows):
            for column, header in enumerate(headers):
                ax = axes[row_index, column]
                if row_index == 0:
                    ax.set_title(header, fontweight="bold")
                if has_errors:
                    if column == 0 and "target_rgb" in row:
                        ax.imshow(row["target_rgb"])
                    elif column == 1:
                        ax.imshow(row["pred_rgb"])
                        _add_metrics(ax, row.get("metrics"), show_metrics)
                    elif column == 2 and "mrae" in row:
                        im = ax.imshow(
                            row["mrae"], cmap="magma", vmin=mrae_limits[0], vmax=mrae_limits[1]
                        )
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                    elif column == 3 and "sam" in row:
                        im = ax.imshow(
                            row["sam"], cmap="viridis", vmin=sam_limits[0], vmax=sam_limits[1]
                        )
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                    elif column == 4 and "rgb_diff" in row:
                        vmax = max(float(np.nanpercentile(row["rgb_diff"], 99)), 1e-6)
                        im = ax.imshow(row["rgb_diff"], cmap="inferno", vmin=0.0, vmax=vmax)
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            row.get("target_error", "N/A"),
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                elif has_target:
                    if column == 0 and "target_rgb" in row:
                        ax.imshow(row["target_rgb"])
                    elif column == 1:
                        ax.imshow(row["pred_rgb"])
                        _add_metrics(ax, row.get("metrics"), show_metrics)
                else:
                    if column == 0:
                        ax.imshow(row["pred_rgb"])
                    else:
                        ax.imshow(row["pred_cube"].mean(axis=0), cmap="cividis")
                ax.axis("off")
            axes[row_index, 0].set_ylabel(row["name"], rotation=90, labelpad=12)

        save_figure(fig, self.output_dir / save_name)


def _add_metrics(ax: Any, metrics: Any, enabled: bool) -> None:
    if not enabled or not isinstance(metrics, dict):
        return
    labels = []
    for name, suffix, precision in (("mrae", "", 4), ("psnr", " dB", 2), ("sam", "°", 3)):
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            continue
        labels.append(f"{name.upper()}: {value:.{precision}f}{suffix}")
    if labels:
        ax.text(
            0.02,
            0.98,
            "\n".join(labels),
            transform=ax.transAxes,
            va="top",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.85,
                "linewidth": 0,
            },
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Render qualitative HSI reconstruction figures.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--targets", help="Optional target directory, e.g. ARAD-1K/Train_spectral")
    ap.add_argument("--samples", nargs="+")
    ap.add_argument("--style", default="paper", choices=["paper", "presentation"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--crop-arad1k", action="store_true")
    args = ap.parse_args()
    vis = ResultsVisualizer(args.results, args.output, args.style, args.dpi, args.targets)
    sample_names = args.samples or find_prediction_samples(Path(args.results), max_samples=10)
    try:
        vis.create_comparison_figure(sample_names, "comparison_figure", True, args.crop_arad1k)
    except HsiResultLayoutError as exc:
        raise SystemExit(str(exc)) from None
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

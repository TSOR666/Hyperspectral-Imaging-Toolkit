from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch

from hsi_io import load_hsi
from result_layout import prediction_path, target_path
from visualization_utils import (
    apply_gaussian_smoothing,
    compute_mrae_map,
    compute_rmse_map,
    compute_sam_map,
    robust_limits,
    save_figure,
    setup_publication_style,
)


class ErrorMapGenerator:
    """Generate robust, comparable spatial error figures for paired cubes."""

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
        self.dpi = dpi
        setup_publication_style("paper", dpi)

    def _load_pair(self, sample: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load prediction/target pair as ``(1,C,H,W)`` tensors."""
        pred_file = prediction_path(self.results_dir, sample)
        target_root = self.targets_dir or self.results_dir
        target_file = target_path(
            target_root,
            sample,
            allow_direct=self.targets_dir is not None,
        )
        if pred_file is None or target_file is None:
            return None, None
        pred = torch.from_numpy(load_hsi(pred_file).cube).float().unsqueeze(0)
        target = torch.from_numpy(load_hsi(target_file).cube).float().unsqueeze(0)
        return pred, target

    def create_error_figure(self, sample: str, save_name: Optional[str] = None) -> None:
        """Create MRAE, SAM, and RMSE maps with robust percentile scaling."""
        pred, target = self._load_pair(sample)
        if pred is None or target is None:
            print(f"Skipping {sample}: prediction or target file not found")
            return
        if pred.shape != target.shape:
            print(f"Skipping {sample}: shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
            return

        maps = {
            "MRAE": compute_mrae_map(pred, target),
            "SAM (degrees)": compute_sam_map(pred, target),
            "RMSE": compute_rmse_map(pred, target),
        }
        fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.1), constrained_layout=True)
        for ax, (title, values), cmap in zip(
            axes,
            maps.items(),
            ("magma", "viridis", "cividis"),
            strict=True,
        ):
            display_values = apply_gaussian_smoothing(values, 0.6)
            vmin, vmax = robust_limits(display_values, floor=0.0)
            im = ax.imshow(display_values, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.suptitle(sample, y=1.02, fontweight="bold")
        save_figure(fig, self.output_dir / (save_name or f"error_maps_{sample}"))

    def create_mrae_heatmap(self, sample: str, save_name: Optional[str] = None) -> None:
        """Backward-compatible single MRAE heatmap export."""
        pred, target = self._load_pair(sample)
        if pred is None or target is None or pred.shape != target.shape:
            print(f"Skipping {sample}: prediction/target pair unavailable or mismatched")
            return
        values = apply_gaussian_smoothing(compute_mrae_map(pred, target), 0.6)
        vmin, vmax = robust_limits(values, floor=0.0)
        fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
        im = ax.imshow(values, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"MRAE — {sample}", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="MRAE")
        save_figure(fig, self.output_dir / (save_name or f"mrae_heatmap_{sample}"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate spatial HSI error maps.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--targets", help="Optional target directory, e.g. ARAD-1K/Train_spectral")
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    generator = ErrorMapGenerator(args.results, args.output, args.dpi, args.targets)
    for sample in args.samples:
        generator.create_error_figure(sample)


if __name__ == "__main__":
    main()

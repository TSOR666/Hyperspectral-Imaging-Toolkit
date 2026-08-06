from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

_SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(_SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUITE_ROOT))

from hsi_io import load_hsi
from hsi_model.utils import get_cached_cmf, hsi_to_rgb
from result_layout import (
    HsiResultLayoutError,
    filter_prediction_samples,
    no_prediction_samples_message,
    prediction_path,
    target_path,
)
from visualization_utils import save_figure, setup_publication_style


class ComparisonGridGenerator:
    """Create aligned method grids from .npy or SSTrans .mat predictions."""

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

    def _load_hsi(self, dir_path: Path, sample: str) -> Optional[np.ndarray]:
        path = prediction_path(dir_path, sample)
        return load_hsi(path).cube if path is not None else None

    @staticmethod
    def _rgb(cube: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(cube).float().unsqueeze(0)
        cmf = get_cached_cmf(cube.shape[0], torch.device("cpu"))
        return hsi_to_rgb(tensor, cmf).squeeze(0).permute(1, 2, 0).numpy()

    def create_main_comparison_figure(
        self,
        sample_names: List[str],
        methods: Optional[Dict[str, str]] = None,
    ) -> None:
        methods = methods or {"Ours": str(self.results_dir)}
        sample_names = filter_prediction_samples(self.results_dir, sample_names)
        if not sample_names:
            raise HsiResultLayoutError(no_prediction_samples_message(self.results_dir))

        target_root = self.targets_dir or self.results_dir
        targets = {
            sample: target_path(
                target_root,
                sample,
                allow_direct=self.targets_dir is not None,
            )
            for sample in sample_names[:4]
        }
        has_targets = any(path is not None for path in targets.values())
        n_samples = min(4, len(sample_names))
        n_methods = len(methods)
        n_columns = n_methods + (1 if has_targets else 0)
        fig, axes = plt.subplots(
            n_samples,
            n_columns,
            figsize=(2.6 * n_columns, 2.7 * n_samples),
            squeeze=False,
            constrained_layout=True,
        )
        headers = (["Ground truth"] if has_targets else []) + list(methods.keys())
        for row_index, sample in enumerate(sample_names[:n_samples]):
            column_offset = 0
            if has_targets:
                ax = axes[row_index, 0]
                target_file = targets[sample]
                if target_file is not None:
                    ax.imshow(self._rgb(load_hsi(target_file).cube))
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                column_offset = 1
                ax.axis("off")
                if row_index == 0:
                    ax.set_title(headers[0], fontweight="bold")
            for method_index, (method, method_dir) in enumerate(methods.items()):
                ax = axes[row_index, method_index + column_offset]
                cube = self._load_hsi(Path(method_dir), sample)
                if cube is None:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.imshow(self._rgb(cube))
                ax.axis("off")
                if row_index == 0:
                    ax.set_title(method, fontweight="bold")
            axes[row_index, 0].set_ylabel(sample, rotation=90, labelpad=12)

        save_figure(fig, self.output_dir / "main_comparison")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create an HSI method comparison grid.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--targets", help="Optional target directory, e.g. ARAD-1K/Train_spectral")
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", nargs="+")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--method-names", nargs="*")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    from result_layout import find_prediction_samples

    samples = args.samples or find_prediction_samples(Path(args.results), max_samples=4)
    names = args.method_names or [Path(directory).name for directory in (args.methods or [])]
    if len(names) != len(args.methods or []):
        raise SystemExit("--method-names must match --methods when supplied")
    methods = {"Ours": args.results}
    methods.update(dict(zip(names, args.methods or [], strict=True)))
    generator = ComparisonGridGenerator(args.results, args.output, args.dpi, args.targets)
    generator.create_main_comparison_figure(samples, methods=methods)


if __name__ == "__main__":
    main()

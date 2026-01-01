
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from hsi_model.utils import crop_center_arad1k, get_cached_cmf, hsi_to_rgb
from visualization_utils import compute_mrae_map

class ResultsVisualizer:
    def __init__(
        self, results_dir: str, output_dir: str, style: str = "paper", dpi: int = 300
    ) -> None:
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self._setup_style(style)
        self.cmf = get_cached_cmf(31, torch.device('cpu'))
        self._cache: Dict[str, Any] = {}

    def _setup_style(self, style: str) -> None:
        plt.style.use('seaborn-v0_8-paper' if style == "paper" else 'seaborn-v0_8-talk')
        plt.rcParams.update({
            'font.size': 10 if style == "paper" else 14,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
        })

    def load_sample_data(self, sample_name: str) -> Dict[str, Any]:
        if sample_name in self._cache:
            return self._cache[sample_name]
        data: Dict[str, Any] = {}
        hsi_path = self.results_dir / "hsi" / f"{sample_name}.npy"
        if hsi_path.exists():
            data['pred_hsi'] = np.load(hsi_path)
        tgt_path = self.results_dir / "hsi" / f"{sample_name}_target.npy"
        if tgt_path.exists():
            data['target_hsi'] = np.load(tgt_path)
        metrics_path = self.results_dir / "metrics" / f"{sample_name}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                data['metrics'] = json.load(f)
        self._cache[sample_name] = data
        return data

    def create_comparison_figure(
        self,
        sample_names: List[str],
        save_name: str = "comparison_figure",
        show_metrics: bool = True,
        crop_arad1k_flag: bool = False,
    ) -> None:
        n = min(len(sample_names), 4)
        fig = plt.figure(figsize=(12, 3 * n))
        gs = gridspec.GridSpec(n, 4, hspace=0.3, wspace=0.1)
        im = None
        for i, name in enumerate(sample_names[:n]):
            d = self.load_sample_data(name)
            if not all(k in d for k in ['pred_hsi', 'target_hsi']):
                continue
            pred = torch.from_numpy(d['pred_hsi']).float()
            if d['pred_hsi'].ndim == 3:
                pred = pred.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
            targ = torch.from_numpy(d['target_hsi']).float()
            if d['target_hsi'].ndim == 3:
                targ = targ.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
            if crop_arad1k_flag:
                pred = crop_center_arad1k(pred)
                targ = crop_center_arad1k(targ)
            pred_rgb = hsi_to_rgb(pred, self.cmf).squeeze(0).permute(1, 2, 0).numpy()
            targ_rgb = hsi_to_rgb(targ, self.cmf).squeeze(0).permute(1, 2, 0).numpy()

            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(targ_rgb)
            ax1.set_title('Ground Truth' if i == 0 else '')
            ax1.axis('off')

            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(pred_rgb)
            ax2.set_title('Prediction' if i == 0 else '')
            ax2.axis('off')

            err = compute_mrae_map(pred, targ)  # (H,W)
            ax3 = fig.add_subplot(gs[i, 2])
            im = ax3.imshow(err, cmap='hot', vmin=0, vmax=0.1)
            ax3.set_title('MRAE' if i == 0 else '')
            ax3.axis('off')

            diff = np.abs(pred_rgb - targ_rgb).mean(-1)
            ax4 = fig.add_subplot(gs[i, 3])
            ax4.imshow(diff, cmap='viridis')
            ax4.set_title('RGB Diff' if i == 0 else '')
            ax4.axis('off')

            if show_metrics and 'metrics' in d:
                txt = (
                    f"MRAE: {d['metrics'].get('mrae', 0):.4f}\n"
                    f"PSNR: {d['metrics'].get('psnr', 0):.2f} dB"
                )
                ax2.text(
                    0.02, 0.98, txt, transform=ax2.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=8,
                )
        if im is not None:
            # Use tuple instead of list for add_axes (type checker requirement)
            cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
            plt.colorbar(im, cax=cbar_ax).set_label('MRAE')
        out_path = self.output_dir / f"{save_name}.pdf"
        plt.savefig(out_path)
        plt.savefig(out_path.with_suffix('.png'))
        plt.close()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", nargs='+')
    ap.add_argument("--style", default="paper", choices=["paper","presentation"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--crop-arad1k", action="store_true")
    args = ap.parse_args()
    vis = ResultsVisualizer(args.results, args.output, args.style, args.dpi)
    if args.samples:
        sample_names = args.samples
    else:
        from pathlib import Path
        sample_names = sorted([p.stem for p in (Path(args.results)/"hsi").glob("*.npy") if not p.stem.endswith("_target")])[:10]
    vis.create_comparison_figure(sample_names, "comparison_figure", True, args.crop_arad1k)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

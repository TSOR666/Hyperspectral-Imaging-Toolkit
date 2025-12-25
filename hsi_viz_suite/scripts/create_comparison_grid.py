
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from hsi_model.utils import get_cached_cmf, hsi_to_rgb

class ComparisonGridGenerator:
    def __init__(self, results_dir: str, output_dir: str, dpi: int = 300) -> None:
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.cmf = get_cached_cmf(31, torch.device('cpu'))

    def _load_hsi(self, dir_path: Path, sample: str) -> Optional[torch.Tensor]:
        """Load an HSI prediction as (B,C,H,W) tensor."""
        p = dir_path / "hsi" / f"{sample}.npy"
        if not p.exists(): return None
        x = torch.from_numpy(np.load(p)).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
        return x

    def create_main_comparison_figure(self, sample_names: List[str], methods: Optional[Dict[str, str]] = None) -> None:
        methods = methods or {"Ours": str(self.results_dir)}
        n_samples = min(4, len(sample_names)); n_methods = len(methods)
        fig = plt.figure(figsize=(2.6*(n_methods+1), 2.6*n_samples))
        gs = gridspec.GridSpec(n_samples, n_methods+1, wspace=0.02, hspace=0.02)
        headers = ['Ground Truth'] + list(methods.keys())
        for r, s in enumerate(sample_names[:n_samples]):
            tgt_path = self.results_dir / "hsi" / f"{s}_target.npy"
            if not tgt_path.exists(): continue
            tgt = torch.from_numpy(np.load(tgt_path)).float()
            if tgt.dim() == 3:
                tgt = tgt.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
            tgt_rgb = hsi_to_rgb(tgt, self.cmf).squeeze(0).permute(1, 2, 0).numpy()  # (1,3,H,W)->(H,W,3)
            ax = fig.add_subplot(gs[r,0]); ax.imshow(tgt_rgb); ax.set_title(headers[0] if r==0 else ''); ax.axis('off')
            for c,(mname,mdir) in enumerate(methods.items()):
                axm = fig.add_subplot(gs[r,c+1])
                pred = self._load_hsi(Path(mdir), s)
                if pred is None:
                    axm.text(0.5,0.5,'N/A',ha='center',va='center',transform=axm.transAxes); axm.axis('off'); continue
                rgb = hsi_to_rgb(pred, self.cmf).squeeze(0).permute(1, 2, 0).numpy()  # (1,3,H,W)->(H,W,3)
                axm.imshow(rgb); axm.set_title(mname if r==0 else ''); axm.axis('off')
        out = self.output_dir / "main_comparison.pdf"; plt.savefig(out); plt.savefig(out.with_suffix('.png')); plt.close()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", nargs='+', required=True)
    ap.add_argument("--methods", nargs='*')
    ap.add_argument("--method-names", nargs='*')
    args = ap.parse_args()
    methods = None
    if args.methods:
        names = args.method_names or [Path(d).name for d in args.methods]
        methods = {n:d for n,d in zip(names, args.methods)}
    gen = ComparisonGridGenerator(args.results, args.output)
    gen.create_main_comparison_figure(args.samples, methods=methods)

if __name__ == "__main__":
    main()


from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
from hsi_model.utils import get_cached_cmf, hsi_to_rgb
from visualization_utils import compute_mrae_map, compute_sam_map, compute_rmse_map, create_error_colormap, apply_gaussian_smoothing

class ErrorMapGenerator:
    def __init__(self, results_dir: str, output_dir: str, dpi: int = 300):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.cmf = get_cached_cmf(31, torch.device('cpu'))

    def _load_pair(self, sample: str):
        pred_path = self.results_dir / "hsi" / f"{sample}.npy"
        tgt_path  = self.results_dir / "hsi" / f"{sample}_target.npy"
        if not pred_path.exists() or not tgt_path.exists(): return None, None
        pred = torch.from_numpy(np.load(pred_path)).float()
        tgt  = torch.from_numpy(np.load(tgt_path)).float()
        if pred.dim() == 3: pred = pred.unsqueeze(0)
        if tgt.dim()  == 3: tgt  = tgt.unsqueeze(0)
        return pred, tgt

    def create_mrae_heatmap(self, sample: str, save_name: Optional[str] = None):
        pred, tgt = self._load_pair(sample)
        if pred is None: return
        mrae = compute_mrae_map(pred, tgt)
        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(apply_gaussian_smoothing(mrae, 0.6), cmap=create_error_colormap(), vmin=0, vmax=0.1)
        ax.axis('off'); plt.colorbar(im).set_label('MRAE')
        save_name = save_name or f"mrae_heatmap_{sample}"
        out = self.output_dir / f"{save_name}.pdf"; plt.savefig(out); plt.savefig(out.with_suffix('.png')); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", nargs='+', required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    eg = ErrorMapGenerator(args.results, args.output, args.dpi)
    for s in args.samples:
        eg.create_mrae_heatmap(s)

if __name__ == "__main__":
    main()

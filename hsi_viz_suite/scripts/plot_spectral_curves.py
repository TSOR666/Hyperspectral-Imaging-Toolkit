
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from visualization_utils import to_chw

class SpectralAnalyzer:
    def __init__(self, results_dir: str, output_dir: str, dpi: int = 300):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.wavelength_range = (400,700); self.n_bands = 31
        self.wavelengths = np.linspace(*self.wavelength_range, self.n_bands)
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({'savefig.dpi': self.dpi, 'savefig.bbox': 'tight'})

    def _load_pair(self, sample: str):
        pred_path = self.results_dir / "hsi" / f"{sample}.npy"
        tgt_path  = self.results_dir / "hsi" / f"{sample}_target.npy"
        if not pred_path.exists() or not tgt_path.exists(): return None, None
        pred = to_chw(np.load(pred_path)); tgt = to_chw(np.load(tgt_path))
        return pred, tgt

    def analyze_spectral_signatures(self, sample: str, pixel_locations: List[Tuple[int,int]]):
        pred, tgt = self._load_pair(sample)
        if pred is None: return
        fig = plt.figure(figsize=(10,8)); gs = gridspec.GridSpec(2,2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0,0]); self._plot_individual(ax1, pred, tgt, pixel_locations[:5])
        ax4 = fig.add_subplot(gs[1,1]); self._plot_bandwise_errors(ax4, pred, tgt)
        out = self.output_dir / f"spectral_analysis_{sample}.pdf"
        plt.savefig(out); plt.savefig(out.with_suffix('.png')); plt.close()

    def _plot_individual(self, ax, pred, tgt, pixels):
        colors = plt.cm.tab10(np.linspace(0,1,len(pixels)))
        for i,(y,x) in enumerate(pixels):
            ax.plot(self.wavelengths, tgt[:,y,x], '-', color=colors[i], label=f'GT ({x},{y})', alpha=0.85)
            ax.plot(self.wavelengths, pred[:,y,x], '--', color=colors[i], label=f'Pred ({x},{y})', alpha=0.85)
        ax.set(xlabel='Wavelength (nm)', ylabel='Reflectance', title='Spectral Signatures',
               xlim=self.wavelength_range, ylim=(0,1))
        ax.grid(True, alpha=0.3); ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=8)

    def _plot_bandwise_errors(self, ax, pred, tgt):
        mrae = np.mean(np.abs(pred - tgt)/(np.abs(tgt)+1e-8), axis=(1,2))
        rmse = np.sqrt(np.mean((pred - tgt)**2, axis=(1,2)))
        ax2 = ax.twinx()
        ax.plot(self.wavelengths, mrae, '-', label='MRAE')
        ax2.plot(self.wavelengths, rmse, '--', label='RMSE')
        ax.set(xlabel='Wavelength (nm)', ylabel='MRAE', xlim=self.wavelength_range)
        ax2.set(ylabel='RMSE')
        ax.grid(True, alpha=0.3)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines+lines2, labels+labels2, loc='upper right')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", nargs='+', required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pixels", nargs='+', type=int)
    args = ap.parse_args()
    sa = SpectralAnalyzer(args.results, args.output, args.dpi)
    if args.pixels and len(args.pixels)%2==0:
        it = iter(args.pixels); pixels = [(y,x) for y,x in zip(it, it)]
    else:
        pixels = [(64,64),(128,128),(192,192)]
    for s in args.samples:
        sa.analyze_spectral_signatures(s, pixels)

if __name__ == "__main__":
    main()

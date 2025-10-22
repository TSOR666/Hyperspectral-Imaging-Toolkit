
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MetricsStatisticsPlotter:
    def __init__(self, results_dirs, output_dir: str, dpi: int = 300):
        self.results_dirs = {name: Path(d) for name,d in results_dirs.items()}
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.df = self._load_all()

    def _load_all(self) -> pd.DataFrame:
        data = []
        for method, rdir in self.results_dirs.items():
            mdir = rdir / "metrics"
            if not mdir.exists(): continue
            for p in mdir.glob("*_metrics.json"):
                if 'overall' in p.name: continue
                with open(p,'r') as f: row = json.load(f)
                row['method'] = method; row['sample'] = p.stem.replace('_metrics','')
                data.append(row)
        return pd.DataFrame(data)

    def violin(self, metrics = ['mrae','rmse','psnr','sam']):
        n = len(metrics); fig, axes = plt.subplots(1, n, figsize=(3*n,4))
        axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
        for i, m in enumerate(metrics):
            d = self.df[['method', m]].dropna()
            if d.empty: continue
            groups = [g[m].values for _, g in d.groupby('method')]
            parts = axes[i].violinplot(groups, showmeans=True, showextrema=False)
            axes[i].set_xticks(range(1, len(groups)+1))
            axes[i].set_xticklabels(list(d['method'].unique()), rotation=45, ha='right')
            axes[i].set_title(m.upper()); axes[i].grid(True, axis='y', alpha=0.3)
        out = self.output_dir / "metrics_violin_plots.pdf"
        plt.tight_layout(); plt.savefig(out); plt.savefig(out.with_suffix('.png')); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs='+', required=True)
    ap.add_argument("--names", nargs='+', required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--metrics", nargs='*', default=['mrae','rmse','psnr','sam'])
    args = ap.parse_args()
    res = {n:d for n,d in zip(args.names, args.results)}
    ms = MetricsStatisticsPlotter(res, args.output, args.dpi)
    ms.violin(args.metrics)

if __name__ == "__main__":
    main()

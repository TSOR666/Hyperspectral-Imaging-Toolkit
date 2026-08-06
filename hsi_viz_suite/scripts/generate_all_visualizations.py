
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Allow the suite to be invoked from the repository root as well as from
# inside hsi_viz_suite/ (the latter is how the README examples are written).
_SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(_SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUITE_ROOT))

from result_layout import find_prediction_samples, find_sample_pairs, no_prediction_samples_message


def run_script(script: str, args: List[str]) -> None:
    cmd = [sys.executable, str(Path(__file__).parent / script)] + args
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--targets", help="Optional target directory for SSTrans predictions")
    ap.add_argument("--output", required=True)
    ap.add_argument("--methods", nargs='*')
    ap.add_argument("--method-names", nargs='*')
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--style", default="paper")
    ap.add_argument("--max-samples", type=int, default=10)
    args = ap.parse_args()

    rdir = Path(args.results)
    samples = find_prediction_samples(rdir, max_samples=args.max_samples)
    if not samples:
        raise SystemExit(no_prediction_samples_message(rdir))

    target_args = ["--targets", args.targets] if args.targets else []

    out_main = Path(args.output) / "main_figures"
    out_main.mkdir(parents=True, exist_ok=True)
    main_cmd = [
        "--results", args.results,
        "--output", str(out_main),
        "--dpi", str(args.dpi),
        "--style", args.style,
    ] + target_args
    if samples:
        main_cmd += ["--samples"] + samples
    run_script("visualize_results.py", main_cmd)

    out_err = Path(args.output) / "error_maps"
    out_err.mkdir(parents=True, exist_ok=True)
    paired_samples = find_sample_pairs(
        rdir,
        max_samples=min(5, len(samples)),
        target_dir=Path(args.targets) if args.targets else None,
    )
    if paired_samples:
        run_script(
            "generate_error_maps.py",
            ["--results", args.results, "--output", str(out_err), "--samples"]
            + paired_samples
            + target_args,
        )
    else:
        print("No target pairs found; skipping spatial error maps.")

    out_spec = Path(args.output) / "spectral_analysis"
    out_spec.mkdir(parents=True, exist_ok=True)
    px = ["--pixels", "64", "64", "128", "128", "192", "192"]
    run_script(
        "plot_spectral_curves.py",
        ["--results", args.results, "--output", str(out_spec), "--samples"]
        + samples[:min(3, len(samples))] + px + target_args,
    )

    out_pub = Path(args.output) / "publication"
    out_pub.mkdir(parents=True, exist_ok=True)
    run_script(
        "plot_publication_figures.py",
        ["--results", args.results, "--output", str(out_pub), "--samples"]
        + samples
        + target_args
        + ["--dpi", str(args.dpi)],
    )

    if args.methods:
        names = args.method_names or [Path(d).name for d in args.methods]
        out_cmp = Path(args.output) / "comparison_grids"
        out_cmp.mkdir(parents=True, exist_ok=True)
        run_script(
            "create_comparison_grid.py",
            ["--results", args.results, "--output", str(out_cmp), "--samples"]
            + samples[:min(4, len(samples))]
            + ["--methods"] + args.methods
            + ["--method-names"] + names
            + target_args
            + ["--dpi", str(args.dpi)],
        )

    all_dirs = [args.results] + (args.methods or [])
    if args.methods:
        all_names = ["Ours"] + (args.method_names or [Path(d).name for d in args.methods])
    else:
        all_names = ["Ours"]
    out_stats = Path(args.output) / "statistics"
    out_stats.mkdir(parents=True, exist_ok=True)
    run_script(
        "plot_metrics_statistics.py",
        ["--results"] + all_dirs + ["--names"] + all_names + ["--output", str(out_stats)],
    )

    print(f"Done. Figures saved under: {args.output}")

if __name__ == "__main__":
    main()

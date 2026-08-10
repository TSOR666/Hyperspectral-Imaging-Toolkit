
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

# Allow the suite to be invoked from the repository root as well as from
# inside hsi_viz_suite/ (the latter is how the README examples are written).
_SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(_SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUITE_ROOT))

from metrics_io import load_metric_rows
from result_layout import find_prediction_samples, find_sample_pairs, no_prediction_samples_message


def _reported_target_directory(results_dir: Path) -> Path | None:
    """Recover a usable target directory recorded by SSTrans."""
    for report_name in ("summary.json", "inference.json"):
        report_path = results_dir / report_name
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        raw_target = report.get("target_dir")
        if not raw_target:
            continue
        candidate = Path(str(raw_target)).expanduser()
        candidates = [candidate]
        if not candidate.is_absolute():
            candidates.extend((results_dir / candidate, results_dir.parent / candidate))
        for target_dir in candidates:
            if target_dir.is_dir():
                return target_dir.resolve()
        # Preserve a recorded-but-missing path so the caller can report the
        # stale reference location instead of silently downgrading to plots
        # without error maps.
        return candidate.resolve()
    return None


def _metric_rows_for_directories(directories: list[Path]) -> dict[Path, list[dict]]:
    return {directory: load_metric_rows(directory) for directory in directories}


def _normalize_results_directory(results_dir: Path) -> Path:
    """Accept either an SSTrans run directory or its ``cubes`` child."""
    if results_dir.name.lower() != "cubes":
        return results_dir
    parent = results_dir.parent
    if any((parent / name).is_file() for name in ("metrics.csv", "summary.json", "inference.json")):
        print(
            f"Results points at {results_dir}; using its SSTrans run directory "
            f"{parent} so metrics and reports are included."
        )
        return parent
    return results_dir


def run_script(script: str, args: List[str]) -> None:
    cmd = [sys.executable, str(Path(__file__).parent / script)] + args
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument(
        "--targets",
        "--target-dir",
        dest="targets",
        help="Optional target directory for SSTrans predictions",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--methods", nargs='*')
    ap.add_argument("--method-names", nargs='*')
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--style", default="paper")
    ap.add_argument("--max-samples", type=int, default=10)
    args = ap.parse_args()

    rdir = _normalize_results_directory(Path(args.results).expanduser().resolve())
    if not rdir.is_dir():
        raise SystemExit(f"Results directory does not exist: {rdir}")
    samples = find_prediction_samples(rdir, max_samples=args.max_samples)
    if not samples:
        raise SystemExit(no_prediction_samples_message(rdir))

    target_root = (
        Path(args.targets).expanduser().resolve()
        if args.targets
        else _reported_target_directory(rdir)
    )
    if target_root is not None and not target_root.is_dir():
        raise SystemExit(f"Target directory does not exist: {target_root}")
    target_args = ["--targets", str(target_root)] if target_root is not None else []
    if args.targets:
        print(f"Using explicitly supplied target directory: {target_root}")
    elif target_root is not None:
        print(f"Using target directory recorded by SSTrans: {target_root}")
    else:
        print("No target directory found; generating prediction-only figures.")

    method_dirs = [
        Path(directory).expanduser().resolve()
        for directory in (args.methods or [])
    ]
    metric_dirs = [rdir, *method_dirs]
    metric_rows = _metric_rows_for_directories(metric_dirs)
    metric_row_count = sum(len(rows) for rows in metric_rows.values())
    print(
        f"Discovered {len(samples)} prediction sample(s) and "
        f"{metric_row_count} metric row(s) under the supplied result folder(s)."
    )

    figure_root = Path(args.output).expanduser().resolve()
    out_main = figure_root / "main_figures"
    out_main.mkdir(parents=True, exist_ok=True)
    main_cmd = [
        "--results", str(rdir),
        "--output", str(out_main),
        "--dpi", str(args.dpi),
        "--style", args.style,
    ] + target_args
    if samples:
        main_cmd += ["--samples"] + samples
    run_script("visualize_results.py", main_cmd)

    out_err = figure_root / "error_maps"
    out_err.mkdir(parents=True, exist_ok=True)
    paired_samples = find_sample_pairs(
        rdir,
        max_samples=min(5, len(samples)),
        target_dir=target_root,
    )
    if paired_samples:
        run_script(
            "generate_error_maps.py",
            ["--results", str(rdir), "--output", str(out_err), "--samples"]
            + paired_samples
            + target_args
            + ["--dpi", str(args.dpi)],
        )
    elif target_root is not None:
        raise SystemExit(
            "A target directory was supplied, but no prediction/target pairs "
            f"were found.\nResults: {rdir}\nTargets: {target_root}\n"
            "Expected matching files such as cubes/<scene>.mat and "
            "<targets>/<scene>.mat (or .npy/.npz). Check that --results points "
            "to the folder containing cubes/ and that scene names match."
        )
    else:
        print("No target directory available; skipping spatial error maps.")

    out_spec = figure_root / "spectral_analysis"
    out_spec.mkdir(parents=True, exist_ok=True)
    px = ["--pixels", "64", "64", "128", "128", "192", "192"]
    run_script(
        "plot_spectral_curves.py",
        ["--results", str(rdir), "--output", str(out_spec), "--samples"]
        + samples[:min(3, len(samples))]
        + px
        + target_args
        + ["--dpi", str(args.dpi)],
    )

    out_pub = figure_root / "publication"
    out_pub.mkdir(parents=True, exist_ok=True)
    run_script(
        "plot_publication_figures.py",
        ["--results", str(rdir), "--output", str(out_pub), "--samples"]
        + samples
        + target_args
        + ["--dpi", str(args.dpi)],
    )

    if args.methods:
        names = args.method_names or [directory.name for directory in method_dirs]
        out_cmp = figure_root / "comparison_grids"
        out_cmp.mkdir(parents=True, exist_ok=True)
        run_script(
            "create_comparison_grid.py",
            ["--results", str(rdir), "--output", str(out_cmp), "--samples"]
            + samples[:min(4, len(samples))]
            + ["--methods"] + [str(directory) for directory in method_dirs]
            + ["--method-names"] + names
            + target_args
            + ["--dpi", str(args.dpi)],
        )

    all_dirs = [str(directory) for directory in metric_dirs]
    if args.methods:
        all_names = ["Ours"] + (args.method_names or [directory.name for directory in method_dirs])
    else:
        all_names = ["Ours"]
    if metric_row_count:
        out_stats = figure_root / "statistics"
        out_stats.mkdir(parents=True, exist_ok=True)
        run_script(
            "plot_metrics_statistics.py",
            ["--results"]
            + all_dirs
            + ["--names"]
            + all_names
            + ["--output", str(out_stats), "--dpi", str(args.dpi)],
        )
    else:
        report_files = [
            directory / "metrics.csv"
            for directory in metric_dirs
            if (directory / "metrics.csv").is_file()
        ]
        if report_files:
            print(
                "Metric report(s) were found but contain no usable rows; "
                f"skipping statistics: {', '.join(map(str, report_files))}"
            )
        else:
            print(
                "No usable metric rows found; skipping statistics. A blind SSTrans "
                "export cannot produce metrics until matching reference cubes are "
                "provided."
            )

    print(f"Done. Figures saved under: {figure_root}")

if __name__ == "__main__":
    main()

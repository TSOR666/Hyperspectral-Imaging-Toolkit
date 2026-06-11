from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .data import HSISample, spectral_to_rgb

LOGGER = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=_json_default)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fieldnames))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extras)
        writer.writeheader()
        writer.writerows(rows)


def save_result_arrays(
    method_dir: Path,
    sample: HSISample,
    prediction: np.ndarray,
    metrics: Mapping[str, float],
) -> None:
    hsi_dir = method_dir / "hsi"
    metrics_dir = method_dir / "metrics"
    hsi_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    np.save(hsi_dir / f"{sample.name}.npy", prediction.astype(np.float32))
    np.save(hsi_dir / f"{sample.name}_target.npy", sample.target.astype(np.float32))
    write_json(metrics_dir / f"{sample.name}_metrics.json", dict(metrics))


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError(
            "Paper figures require matplotlib. Install hsi_viz_suite requirements."
        ) from exc


def _robust_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(float(np.percentile(finite, percentile)), 1e-8)


def save_sample_figure(
    path_stem: Path,
    sample: HSISample,
    prediction: np.ndarray,
    metric_maps: Mapping[str, np.ndarray],
    metrics: Mapping[str, float],
    *,
    dpi: int,
) -> None:
    plt = _require_matplotlib()
    target_rgb = spectral_to_rgb(sample.target, sample.wavelengths)
    pred_rgb = spectral_to_rgb(prediction, sample.wavelengths)
    input_rgb = np.moveaxis(sample.rgb, 0, -1)
    rmse_map = metric_maps["map_rmse"]
    sam_map = metric_maps["map_sam"]
    error_y, error_x = np.unravel_index(np.argmax(rmse_map), rmse_map.shape)
    error_y += (sample.target.shape[1] - rmse_map.shape[0]) // 2
    error_x += (sample.target.shape[2] - rmse_map.shape[1]) // 2
    center_y, center_x = sample.target.shape[1] // 2, sample.target.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)
    axes[0, 0].imshow(np.clip(input_rgb, 0.0, 1.0))
    axes[0, 0].set_title("Model RGB input")
    axes[0, 1].imshow(np.clip(target_rgb, 0.0, 1.0))
    axes[0, 1].set_title("Ground truth")
    axes[0, 2].imshow(np.clip(pred_rgb, 0.0, 1.0))
    axes[0, 2].set_title("Reconstruction")

    image = axes[1, 0].imshow(
        rmse_map, cmap="magma", vmin=0.0, vmax=_robust_limit(rmse_map)
    )
    axes[1, 0].set_title("Per-pixel RMSE")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)
    image = axes[1, 1].imshow(
        sam_map, cmap="viridis", vmin=0.0, vmax=_robust_limit(sam_map)
    )
    axes[1, 1].set_title("SAM (degrees)")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].plot(
        sample.wavelengths,
        sample.target[:, center_y, center_x],
        color="#222222",
        linewidth=2,
        label="GT center",
    )
    axes[1, 2].plot(
        sample.wavelengths,
        prediction[:, center_y, center_x],
        color="#2878B5",
        linestyle="--",
        linewidth=2,
        label="Pred center",
    )
    axes[1, 2].plot(
        sample.wavelengths,
        sample.target[:, error_y, error_x],
        color="#C82423",
        linewidth=1.5,
        label="GT max-error",
    )
    axes[1, 2].plot(
        sample.wavelengths,
        prediction[:, error_y, error_x],
        color="#F8AC8C",
        linestyle="--",
        linewidth=1.5,
        label="Pred max-error",
    )
    axes[1, 2].set_xlabel("Wavelength (nm)")
    axes[1, 2].set_ylabel("Reflectance")
    axes[1, 2].grid(alpha=0.2)
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_title("Spectral signatures")

    for axis in axes.flat[:5]:
        axis.set_xticks([])
        axis.set_yticks([])
    metric_text = (
        f"MRAE {metrics['mrae']:.4f} | RMSE {metrics['rmse']:.4f} | "
        f"PSNR {metrics['psnr']:.2f} dB | SAM {metrics['sam']:.2f} deg"
    )
    fig.suptitle(f"{sample.name}\n{metric_text}", fontsize=12)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(
    path_stem: Path,
    sample: HSISample,
    predictions: Mapping[str, np.ndarray],
    metric_maps: Mapping[str, Mapping[str, np.ndarray]],
    *,
    dpi: int,
) -> None:
    plt = _require_matplotlib()
    methods = list(predictions)
    fig, axes = plt.subplots(
        2,
        len(methods) + 1,
        figsize=(3.2 * (len(methods) + 1), 6.0),
        constrained_layout=True,
        squeeze=False,
    )
    axes[0, 0].imshow(spectral_to_rgb(sample.target, sample.wavelengths))
    axes[0, 0].set_title("Ground truth")
    axes[1, 0].imshow(np.moveaxis(sample.rgb, 0, -1))
    axes[1, 0].set_title("Model RGB input")
    for column, method in enumerate(methods, start=1):
        axes[0, column].imshow(
            spectral_to_rgb(predictions[method], sample.wavelengths)
        )
        axes[0, column].set_title(method)
        error = metric_maps[method]["map_rmse"]
        image = axes[1, column].imshow(
            error, cmap="magma", vmin=0.0, vmax=_robust_limit(error)
        )
        axes[1, column].set_title("RMSE map")
        fig.colorbar(image, ax=axes[1, column], fraction=0.046)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(sample.name)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_method_report(
    method_dir: Path,
    *,
    dataset_name: str,
    method_name: str,
    model_info: Mapping[str, Any],
    protocol: Mapping[str, Any],
    sample_rows: Sequence[Mapping[str, Any]],
    per_band_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    write_csv(method_dir / "metrics.csv", sample_rows)
    write_csv(method_dir / "per_band_metrics.csv", per_band_rows)
    grouped_bands: Dict[tuple[int, float], List[Mapping[str, Any]]] = {}
    for row in per_band_rows:
        key = (int(row["band"]), float(row["wavelength_nm"]))
        grouped_bands.setdefault(key, []).append(row)
    per_band_summary: List[Dict[str, Any]] = []
    for (band, wavelength), rows in sorted(grouped_bands.items()):
        summary_row: Dict[str, Any] = {
            "dataset": dataset_name,
            "method": method_name,
            "band": band,
            "wavelength_nm": wavelength,
            "samples": len(rows),
        }
        for metric in ("mrae", "rmse", "psnr", "mae"):
            values = np.asarray([float(row[metric]) for row in rows])
            summary_row[metric] = float(values.mean())
            summary_row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )
        per_band_summary.append(summary_row)
    write_csv(method_dir / "per_band_summary.csv", per_band_summary)
    result = {
        "dataset": dataset_name,
        "method": method_name,
        "model": dict(model_info),
        "protocol": dict(protocol),
        "summary": dict(summary),
        "samples": list(sample_rows),
    }
    write_json(method_dir / "summary.json", result)
    return result


def _summary_row(result: Mapping[str, Any]) -> Dict[str, Any]:
    summary = result["summary"]
    row: Dict[str, Any] = {
        "dataset": result["dataset"],
        "method": result["method"],
        "parameters": result["model"].get("parameters"),
        "samples": next(iter(summary.values())).get("count", 0) if summary else 0,
    }
    for metric in ("mrae", "rmse", "psnr", "sam", "ssim", "mae", "runtime_s"):
        if metric in summary:
            row[metric] = summary[metric]["mean"]
            row[f"{metric}_std"] = summary[metric]["std"]
            row[f"{metric}_ci95_low"] = summary[metric]["ci95_low"]
            row[f"{metric}_ci95_high"] = summary[metric]["ci95_high"]
    return row


def write_paper_tables(output_dir: Path, results: Sequence[Mapping[str, Any]]) -> None:
    rows = [_summary_row(result) for result in results]
    write_csv(output_dir / "paper_table.csv", rows)

    headers = ["Dataset", "Method", "MRAE", "RMSE", "PSNR", "SAM", "SSIM"]
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    latex = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        "Dataset & Method & MRAE $\\downarrow$ & RMSE $\\downarrow$ & "
        "PSNR $\\uparrow$ & SAM $\\downarrow$ & SSIM $\\uparrow$ \\\\",
        r"\midrule",
    ]
    for row in rows:
        plain_values = [
            str(row["dataset"]),
            str(row["method"]),
            f"{row.get('mrae', float('nan')):.4f} +/- {row.get('mrae_std', float('nan')):.4f}",
            f"{row.get('rmse', float('nan')):.4f} +/- {row.get('rmse_std', float('nan')):.4f}",
            f"{row.get('psnr', float('nan')):.2f} +/- {row.get('psnr_std', float('nan')):.2f}",
            f"{row.get('sam', float('nan')):.2f} +/- {row.get('sam_std', float('nan')):.2f}",
            f"{row.get('ssim', float('nan')):.4f} +/- {row.get('ssim_std', float('nan')):.4f}",
        ]
        markdown.append("| " + " | ".join(plain_values) + " |")
        latex_values = [
            plain_values[0].replace("_", r"\_"),
            plain_values[1].replace("_", r"\_"),
        ]
        for metric, precision in (
            ("mrae", 4),
            ("rmse", 4),
            ("psnr", 2),
            ("sam", 2),
            ("ssim", 4),
        ):
            latex_values.append(
                f"{row.get(metric, float('nan')):.{precision}f} "
                f"$\\pm$ {row.get(f'{metric}_std', float('nan')):.{precision}f}"
            )
        latex.append(" & ".join(latex_values) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "paper_table.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    (output_dir / "paper_table.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )
    write_json(output_dir / "all_results.json", list(results))

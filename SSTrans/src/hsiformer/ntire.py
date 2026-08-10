from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

from .metrics import (
    MRAEDenominator,
    peak_signal_to_noise_ratio,
    spectral_metrics,
)

ARAD_BANDS_NM = np.arange(400, 701, 10, dtype=np.int32)
SAM_DEGREES_PER_RADIAN = 180.0 / np.pi


def save_ntire_cube(
    path: str | Path,
    cube: torch.Tensor | np.ndarray,
    *,
    bands: np.ndarray = ARAD_BANDS_NM,
    norm_factor: float = 1.0,
) -> None:
    """Save an HWC cube in the layout consumed by NTIRE2022Util.loadCube."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = _cube_to_hwc(cube)
    if array.shape[-1] != len(bands):
        raise ValueError(
            f"Cube has {array.shape[-1]} channels, but {len(bands)} bands were given."
        )

    with h5py.File(destination, "w") as handle:
        # NTIRE2022Util.loadCube reads this dataset and applies .T.
        handle.create_dataset("cube", data=array.T, compression="gzip")
        handle.create_dataset("bands", data=np.asarray(bands))
        handle.create_dataset("norm_factor", data=np.asarray(norm_factor))


def load_ntire_cube(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        cube = np.asarray(handle["cube"], dtype=np.float32).T
        bands = np.asarray(handle["bands"]).squeeze()
    return cube, bands


def predict_hsi(
    model: nn.Module,
    rgb: torch.Tensor,
    *,
    tile_size: int | None = None,
    overlap: int = 16,
) -> torch.Tensor:
    """Predict a batch, optionally averaging overlapping spatial tiles."""
    if rgb.ndim != 4:
        raise ValueError(f"Expected BCHW input, got {tuple(rgb.shape)}.")
    if tile_size is None:
        return model(rgb)
    if tile_size < 1:
        raise ValueError("tile_size must be positive.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size.")

    outputs = [
        _predict_single_tiled(model, sample.unsqueeze(0), tile_size, overlap)
        for sample in rgb
    ]
    return torch.cat(outputs, dim=0)


def evaluate_loader(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    tile_size: int | None = None,
    overlap: int = 16,
    amp: bool = False,
    output_dir: str | Path | None = None,
    clip: bool = False,
    crop_border: int = 0,
    mrae_denominator: MRAEDenominator = "clamp_abs",
    mrae_epsilon: float = 1e-6,
    psnr_clip: bool = False,
    include_ssim: bool = False,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    """Evaluate predictions under an explicitly selected metric protocol.

    The reported SSTrans ``0.1468`` result uses full ARAD-origin frames and
    the source evaluator's additive ``1e-5`` denominator. A 128-pixel center
    crop is a separate NTIRE/MST++ protocol. Exported cubes always retain the
    complete prediction regardless of the metric crop. MRAE/RMSE/PSNR/SAM are
    always reported; callers can opt into the MSWR-compatible SSIM diagnostic
    without adding its pooling cost to training-time validation.
    """
    if crop_border < 0:
        raise ValueError("crop_border cannot be negative.")
    if mrae_epsilon <= 0:
        raise ValueError("mrae_epsilon must be positive.")

    model.eval()
    rows: list[dict[str, float | str]] = []
    skipped: list[str] = []
    cube_dir = Path(output_dir) if output_dir is not None else None
    if cube_dir is not None:
        cube_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch in loader:
            rgb = _require_tensor(batch, "cond").to(device, non_blocking=True)
            # ARAD-1K test batches carry no cube; they are reconstructed and
            # exported, then excluded from the metric average.
            target = (
                _require_tensor(batch, "label").to(device, non_blocking=True)
                if "label" in batch
                else None
            )
            scene_ids = _scene_ids(batch["scene_id"], rgb.shape[0])
            amp_enabled = amp and device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype(device) if amp_enabled else None,
                enabled=amp_enabled,
            ):
                prediction = predict_hsi(
                    model,
                    rgb,
                    tile_size=tile_size,
                    overlap=overlap,
                )
            for index, scene_id in enumerate(scene_ids):
                predicted_scene = prediction[index : index + 1].float()
                if target is None:
                    skipped.append(scene_id)
                    if cube_dir is not None:
                        save_ntire_cube(
                            cube_dir / f"{scene_id}.mat",
                            (
                                predicted_scene.clamp(0.0, 1.0)
                                if clip
                                else predicted_scene
                            )[0],
                        )
                    continue
                target_scene = target[index : index + 1].float()
                metric_prediction, metric_target = _crop_metric_region(
                    predicted_scene,
                    target_scene,
                    crop_border,
                )
                if (
                    mrae_denominator == "source_additive"
                    and bool(torch.any(metric_target == 0))
                ):
                    # The result-generation script adds the same epsilon to
                    # both arrays before all metrics. This leaves absolute
                    # errors unchanged, supplies the MRAE denominator floor,
                    # and reproduces its tiny SAM shift at zero-valued bands.
                    metric_prediction = metric_prediction + mrae_epsilon
                    metric_target = metric_target + mrae_epsilon
                metrics = spectral_metrics(
                    metric_prediction,
                    metric_target,
                    eps=mrae_epsilon,
                    mrae_denominator=mrae_denominator,
                    include_ssim=include_ssim,
                )
                if psnr_clip:
                    metrics["psnr"] = peak_signal_to_noise_ratio(
                        metric_prediction.clamp(0.0, 1.0),
                        metric_target.clamp(0.0, 1.0),
                    )
                row: dict[str, float | str] = {"scene_id": scene_id}
                row.update(
                    {name: float(value.item()) for name, value in metrics.items()}
                )
                rows.append(row)
                if cube_dir is not None:
                    save_ntire_cube(
                        cube_dir / f"{scene_id}.mat",
                        (
                            predicted_scene.clamp(0.0, 1.0)
                            if clip
                            else predicted_scene
                        )[0],
                    )

    if not rows:
        if skipped:
            raise ValueError(
                f"None of the {len(skipped)} evaluated scenes carry a "
                "ground-truth cube; use infer_loader for prediction-only runs."
            )
        raise ValueError("Evaluation loader produced no samples.")
    metric_names = ("mrae", "rmse", "psnr", "sam")
    if include_ssim:
        metric_names = (*metric_names, "ssim")
    summary = {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in metric_names
    }
    summary["count"] = float(len(rows))
    summary["skipped"] = float(len(skipped))
    summary["crop_border"] = float(crop_border)
    return summary, rows


def infer_loader(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    output_dir: str | Path,
    tile_size: int | None = None,
    overlap: int = 16,
    amp: bool = False,
    clip: bool = False,
) -> list[str]:
    model.eval()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    scene_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            rgb = _require_tensor(batch, "cond").to(device, non_blocking=True)
            batch_scene_ids = _scene_ids(batch["scene_id"], rgb.shape[0])
            amp_enabled = amp and device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype(device) if amp_enabled else None,
                enabled=amp_enabled,
            ):
                prediction = predict_hsi(
                    model,
                    rgb,
                    tile_size=tile_size,
                    overlap=overlap,
                )
            if clip:
                prediction = prediction.clamp(0.0, 1.0)
            for index, scene_id in enumerate(batch_scene_ids):
                save_ntire_cube(
                    destination / f"{scene_id}.mat",
                    prediction[index].float(),
                )
                scene_ids.append(scene_id)
    return scene_ids


def write_metric_reports(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    rows: list[Mapping[str, float | str]],
) -> None:
    """Write machine-readable benchmark reports and paper-friendly SAM units.

    ``spectral_angle_mapper`` follows PyTorch's convention and returns radians.
    Keeping that native value in ``sam`` preserves SSTrans' source metric, while
    ``sam_degrees`` makes cross-method paper figures unambiguous (and matches
    the unit used by :mod:`hsi_viz_suite` spatial SAM maps).
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_summary = _with_sam_units(summary, include_metadata=True)
    (destination / "summary.json").write_text(
        json.dumps(report_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    report_rows = [_with_sam_units(row) for row in rows]
    with (destination / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scene_id",
                "mrae",
                "rmse",
                "psnr",
                "sam",
                "ssim",
                "sam_degrees",
                "sam_unit",
            ),
        )
        writer.writeheader()
        writer.writerows(report_rows)


def run_hsi_viz_suite(
    results_dir: str | Path,
    *,
    target_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    suite_path: str | Path | None = None,
    max_samples: int = 10,
    dpi: int = 300,
    style: str = "paper",
) -> Path:
    """Run the repository's publication-grade visualization suite.

    The handoff deliberately uses SSTrans' normal evaluation layout:
    ``results_dir/cubes/*.mat`` and ``results_dir/metrics.csv``.  Passing a
    target directory enables paired colour/error/spectral figures; omitting it
    remains useful for an official blind-test export and produces prediction-
    only figures without inventing metrics.
    """
    result_root = Path(results_dir).resolve()
    if not result_root.is_dir():
        raise FileNotFoundError(
            f"Cannot visualize missing SSTrans results directory: {result_root}"
        )
    target_root = Path(target_dir).resolve() if target_dir is not None else None
    if target_root is not None and not target_root.is_dir():
        raise FileNotFoundError(
            f"Cannot visualize with missing target directory: {target_dir}"
        )
    if max_samples < 1:
        raise ValueError("max_samples must be at least one.")
    if dpi < 1:
        raise ValueError("dpi must be at least one.")

    entrypoint = resolve_hsi_viz_entrypoint(suite_path)
    figure_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else result_root / "figures"
    )
    command = [
        sys.executable,
        str(entrypoint),
        "--results",
        str(result_root),
        "--output",
        str(figure_root),
        "--max-samples",
        str(max_samples),
        "--dpi",
        str(dpi),
        "--style",
        style,
    ]
    if target_root is not None:
        command.extend(("--targets", str(target_root)))

    # Run from the suite root so its local hsi_model package and scripts work
    # equally from an editable SSTrans checkout or an installed SSTrans wheel.
    subprocess.run(
        command,
        check=True,
        cwd=str(entrypoint.parents[1]),
    )
    return figure_root


def resolve_hsi_viz_entrypoint(
    suite_path: str | Path | None = None,
) -> Path:
    """Find ``hsi_viz_suite/scripts/generate_all_visualizations.py`` safely."""
    script_name = "generate_all_visualizations.py"
    candidates: list[Path] = []
    if suite_path is not None:
        supplied = Path(suite_path)
        candidates.extend(
            (
                supplied,
                supplied / script_name,
                supplied / "scripts" / script_name,
            )
        )
    else:
        # A source checkout has SSTrans and hsi_viz_suite as sibling folders.
        # Also inspect the caller's directory tree so a user can run the
        # console entry point from the repository root without extra flags.
        anchors = (Path(__file__).resolve(), Path.cwd().resolve())
        for anchor in anchors:
            roots = (
                anchor.parents
                if anchor.is_file()
                else (anchor, *anchor.parents)
            )
            candidates.extend(
                parent / "hsi_viz_suite" / "scripts" / script_name
                for parent in roots
            )

    for candidate in candidates:
        if candidate.is_file() and candidate.name == script_name:
            return candidate.resolve()
    location_hint = (
        f" at {suite_path}" if suite_path is not None else " beside this checkout"
    )
    raise FileNotFoundError(
        "Could not find hsi_viz_suite/scripts/generate_all_visualizations.py"
        f"{location_hint}. Pass --hsi-viz-suite <suite root, scripts dir, or "
        "generate_all_visualizations.py path>."
    )


def _with_sam_units(
    values: Mapping[str, Any],
    *,
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Add an explicit paper-friendly SAM degree value to metric records."""
    report = dict(values)
    if "sam" in report and report["sam"] not in (None, ""):
        unit = str(report.get("sam_unit", "radians")).strip().lower()
        if unit in {"rad", "radian", "radians"}:
            report["sam_unit"] = "radians"
            scale = SAM_DEGREES_PER_RADIAN
        elif unit in {"degree", "degrees", "deg"}:
            report["sam_unit"] = "degrees"
            scale = 1.0
        else:
            raise ValueError(
                "sam_unit must be 'radians' or 'degrees', "
                f"got {report['sam_unit']!r}."
            )
        report.setdefault("sam_degrees", float(report["sam"]) * scale)
    if include_metadata:
        metric_units = dict(report.get("metric_units", {}))
        metric_units.setdefault("mrae", "unitless")
        metric_units.setdefault("rmse", "reflectance")
        metric_units.setdefault("psnr", "dB")
        metric_units.setdefault("sam", report.get("sam_unit", "radians"))
        metric_units.setdefault("sam_degrees", "degrees")
        metric_units.setdefault("ssim", "unitless")
        report["metric_units"] = metric_units
    return report


def resolve_device(value: str = "auto") -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def autocast_dtype(device: torch.device, preferred: str = "bf16") -> torch.dtype:
    """Resolve the autocast compute dtype, preferring overflow-safe bf16.

    fp16 autocast (max ~65504) overflows in the model's unbounded depthwise
    convolutions and deep residual stack, which is the dominant source of NaN
    activations. bfloat16 carries the float32 exponent range and removes that
    failure class entirely, so it is preferred whenever the GPU supports it.
    fp16 is used only as a fallback on hardware without bf16.
    """
    if device.type != "cuda":
        return torch.float32
    if preferred == "fp16":
        return torch.float16
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except (AssertionError, RuntimeError):
        pass
    return torch.float16


def _predict_single_tiled(
    model: nn.Module,
    rgb: torch.Tensor,
    tile_size: int,
    overlap: int,
) -> torch.Tensor:
    height, width = rgb.shape[-2:]
    if tile_size >= height and tile_size >= width:
        return model(rgb)

    rows = _tile_starts(height, tile_size, overlap)
    columns = _tile_starts(width, tile_size, overlap)
    output: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    for row in rows:
        for column in columns:
            tile = rgb[
                :,
                :,
                row : min(row + tile_size, height),
                column : min(column + tile_size, width),
            ]
            predicted_tile = model(tile)
            if output is None:
                output = predicted_tile.new_zeros(
                    (1, predicted_tile.shape[1], height, width)
                )
                weights = predicted_tile.new_zeros((1, 1, height, width))
            tile_height, tile_width = predicted_tile.shape[-2:]
            output[
                :,
                :,
                row : row + tile_height,
                column : column + tile_width,
            ] += predicted_tile
            weights[
                :,
                :,
                row : row + tile_height,
                column : column + tile_width,
            ] += 1
    if output is None or weights is None:
        raise RuntimeError("Tiled inference produced no tiles.")
    return output / weights.clamp_min(1)


def _tile_starts(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def _cube_to_hwc(cube: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(cube, torch.Tensor):
        array = cube.detach().cpu().numpy()
    else:
        array = np.asarray(cube)
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim != 3:
        raise ValueError(f"Expected a three-dimensional cube, got {array.shape}.")
    if array.shape[0] == len(ARAD_BANDS_NM):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != len(ARAD_BANDS_NM):
        raise ValueError(f"Cannot identify spectral axis in cube {array.shape}.")
    return np.ascontiguousarray(array)


def _require_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch value '{key}' must be a tensor.")
    return value


def _scene_ids(value: Any, batch_size: int) -> list[str]:
    if isinstance(value, str):
        values = [value]
    else:
        values = [str(item) for item in value]
    if len(values) != batch_size:
        raise ValueError("Number of scene identifiers does not match batch size.")
    return values


def _crop_metric_region(
    prediction: torch.Tensor,
    target: torch.Tensor,
    crop_border: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if crop_border == 0:
        return prediction, target
    height, width = prediction.shape[-2:]
    if target.shape[-2:] != (height, width):
        raise ValueError(
            "Prediction and target spatial shapes must match before metric cropping."
        )
    if height <= 2 * crop_border or width <= 2 * crop_border:
        raise ValueError(
            f"crop_border={crop_border} leaves no scoring region for "
            f"spatial shape {(height, width)}."
        )
    region = (
        ...,
        slice(crop_border, -crop_border),
        slice(crop_border, -crop_border),
    )
    return prediction[region], target[region]

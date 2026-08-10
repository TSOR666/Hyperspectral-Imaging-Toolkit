"""Metric-file adapters for native and SSTrans result folders."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_metric_rows(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load per-sample metrics from JSON files and SSTrans ``metrics.csv``.

    The suite's visual SAM maps are expressed in degrees.  SSTrans preserves
    its native radians value under ``sam`` and writes ``sam_degrees`` alongside
    it, so normalize that one presentation field here before comparing methods.
    Older folders without unit metadata retain the conventional degree unit.
    """
    root = Path(results_dir)
    rows: list[dict[str, Any]] = []
    summary_sam_unit = _summary_sam_unit(root)
    metrics_dir = root / "metrics"
    if metrics_dir.is_dir():
        for path in sorted(metrics_dir.glob("*_metrics.json")):
            if "overall" in path.name.lower():
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                row = dict(row)
                row.setdefault("sample", path.stem.removesuffix("_metrics"))
                rows.append(_normalize_sam_for_visualization(row, summary_sam_unit))

    csv_path = root / "metrics.csv"
    if csv_path.is_file():
        try:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    normalized = dict(row)
                    normalized["sample"] = row.get("scene_id") or row.get("sample") or ""
                    rows.append(
                        _normalize_sam_for_visualization(
                            normalized,
                            summary_sam_unit,
                        )
                    )
        except (OSError, csv.Error):
            pass

    # Prefer the first row for a sample (the native JSON layout takes
    # precedence over an optional aggregate CSV export).
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample = str(row.get("sample", ""))
        if sample and sample not in deduplicated:
            deduplicated[sample] = row
    return list(deduplicated.values())


def _summary_sam_unit(root: Path) -> str | None:
    """Return the raw SAM unit advertised by an SSTrans summary, if present."""
    path = root / "summary.json"
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict):
        return None
    unit = summary.get("sam_unit")
    if unit is None and isinstance(summary.get("metric_units"), dict):
        unit = summary["metric_units"].get("sam")
    return str(unit) if unit is not None else None


def _normalize_sam_for_visualization(
    row: dict[str, Any],
    fallback_unit: str | None,
) -> dict[str, Any]:
    """Expose ``sam`` as degrees while retaining SSTrans' raw value."""
    normalized = dict(row)
    raw_value = normalized.get("sam")
    unit = str(normalized.get("sam_unit") or fallback_unit or "degrees").lower()
    degrees_value = normalized.get("sam_degrees")
    try:
        if degrees_value not in (None, ""):
            degrees = float(degrees_value)
        elif unit in {"rad", "radian", "radians"} and raw_value not in (None, ""):
            degrees = math.degrees(float(raw_value))
        else:
            degrees = float(raw_value)
    except (TypeError, ValueError):
        return normalized

    if unit in {"rad", "radian", "radians"}:
        normalized.setdefault("sam_radians", raw_value)
    normalized["sam"] = degrees
    normalized["sam_degrees"] = degrees
    normalized["sam_unit"] = "degrees"
    return normalized


def load_metric_for_sample(
    results_dir: str | Path,
    sample: str,
) -> dict[str, Any] | None:
    """Return one sample's metrics, including SSTrans CSV rows."""
    for row in load_metric_rows(results_dir):
        if str(row.get("sample", "")) == sample:
            return row
    return None


def numeric_metric(row: dict[str, Any], name: str) -> float | None:
    """Convert a metric field to a finite float when available."""
    value = row.get(name)
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def metric_names(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Return common scalar metric names present in at least one row."""
    preferred = ["mrae", "rmse", "psnr", "sam", "ssim", "mae"]
    return [name for name in preferred if any(numeric_metric(row, name) is not None for row in rows)]

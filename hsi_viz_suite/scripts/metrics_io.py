"""Metric-file adapters for native and SSTrans result folders."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_metric_rows(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load per-sample metrics from JSON files and SSTrans ``metrics.csv``."""
    root = Path(results_dir)
    rows: list[dict[str, Any]] = []
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
                rows.append(row)

    csv_path = root / "metrics.csv"
    if csv_path.is_file():
        try:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    normalized = dict(row)
                    normalized["sample"] = row.get("scene_id") or row.get("sample") or ""
                    rows.append(normalized)
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
    preferred = ["mrae", "rmse", "psnr", "sam", "mae"]
    return [name for name in preferred if any(numeric_metric(row, name) is not None for row in rows)]

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


PREDICTION_SUFFIX = "_pred"
TARGET_SUFFIX = "_target"


class HsiResultLayoutError(ValueError):
    """Raised when a results directory has no visualizable HSI pairs."""


def _sample_from_prediction_stem(stem: str) -> Optional[str]:
    if stem.endswith(TARGET_SUFFIX):
        return None
    if stem.endswith(PREDICTION_SUFFIX):
        return stem[: -len(PREDICTION_SUFFIX)]
    return stem


def _unique_sorted(values: Iterable[str]) -> List[str]:
    return sorted({v for v in values if v})


def prediction_path(results_dir: Path, sample: str) -> Optional[Path]:
    """Return the first supported prediction path for a sample."""
    candidates = [
        results_dir / "hsi" / f"{sample}.npy",
        results_dir / "hsi" / f"{sample}{PREDICTION_SUFFIX}.npy",
        results_dir / "predictions" / f"{sample}{PREDICTION_SUFFIX}.npy",
        results_dir / "predictions" / f"{sample}.npy",
    ]
    return next((p for p in candidates if p.exists()), None)


def target_path(results_dir: Path, sample: str) -> Optional[Path]:
    """Return the first supported target path for a sample."""
    candidates = [
        results_dir / "hsi" / f"{sample}{TARGET_SUFFIX}.npy",
        results_dir / "targets" / f"{sample}{TARGET_SUFFIX}.npy",
        results_dir / "targets" / f"{sample}.npy",
    ]
    return next((p for p in candidates if p.exists()), None)


def candidate_samples(results_dir: Path) -> List[str]:
    """Return sample names inferred from supported prediction locations."""
    samples: List[str] = []
    for directory in [results_dir / "hsi", results_dir / "predictions"]:
        if not directory.exists():
            continue
        for path in directory.glob("*.npy"):
            sample = _sample_from_prediction_stem(path.stem)
            if sample is not None:
                samples.append(sample)
    return _unique_sorted(samples)


def find_sample_pairs(results_dir: Path, max_samples: Optional[int] = None) -> List[str]:
    """Return samples that have both prediction and target arrays."""
    pairs = [
        sample
        for sample in candidate_samples(results_dir)
        if prediction_path(results_dir, sample) is not None
        and target_path(results_dir, sample) is not None
    ]
    return pairs if max_samples is None else pairs[:max_samples]


def filter_sample_pairs(results_dir: Path, samples: Iterable[str]) -> List[str]:
    """Keep requested samples that have both prediction and target arrays."""
    return [
        sample
        for sample in samples
        if prediction_path(results_dir, sample) is not None
        and target_path(results_dir, sample) is not None
    ]


def no_sample_pairs_message(results_dir: Path) -> str:
    """Build a helpful error for empty or incompatible result folders."""
    results_dir = Path(results_dir)
    hsi_dir = results_dir / "hsi"
    predictions_dir = results_dir / "predictions"
    targets_dir = results_dir / "targets"
    metrics_dir = results_dir / "metrics"

    parts = [
        f"No visualizable HSI prediction/target pairs found in: {results_dir}",
        (
            "Expected per-sample .npy pairs such as "
            "hsi/<sample>.npy (or hsi/<sample>_pred.npy) and "
            "hsi/<sample>_target.npy."
        ),
    ]

    if not hsi_dir.exists():
        parts.append(f"Missing expected directory: {hsi_dir}")
    else:
        pred_count = len(
            [
                p
                for p in hsi_dir.glob("*.npy")
                if _sample_from_prediction_stem(p.stem) is not None
            ]
        )
        target_count = len(list(hsi_dir.glob(f"*{TARGET_SUFFIX}.npy")))
        parts.append(
            f"Found {pred_count} prediction candidate(s) and "
            f"{target_count} target candidate(s) under {hsi_dir}."
        )

    if predictions_dir.exists():
        pred_count = len(list(predictions_dir.glob("*.npy")))
        parts.append(
            f"Found {pred_count} .npy file(s) under {predictions_dir}; "
            "matching target arrays are still required for comparison/error figures."
        )

    if targets_dir.exists():
        target_count = len(list(targets_dir.glob("*.npy")))
        parts.append(f"Found {target_count} .npy target file(s) under {targets_dir}.")

    if metrics_dir.exists():
        metric_count = len(list(metrics_dir.glob("*_metrics.json")))
        parts.append(f"Found {metric_count} per-sample metric file(s) under {metrics_dir}.")

    if (results_dir / "test_results.json").exists():
        parts.append(
            "This looks like an MSWR aggregate test output. Re-run/export with "
            "per-sample arrays, e.g. predictions as .npy plus matching targets, "
            "before using the HSI visualization suite."
        )

    return "\n".join(parts)

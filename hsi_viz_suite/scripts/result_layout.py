from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


PREDICTION_SUFFIX = "_pred"
TARGET_SUFFIX = "_target"
SUPPORTED_ARRAY_SUFFIXES = (".npy", ".npz", ".mat", ".h5", ".hdf5")

# SSTrans writes NTIRE-compatible HDF5 ``.mat`` cubes under ``cubes/``.
PREDICTION_DIRECTORIES = ("hsi", "predictions", "cubes")
TARGET_DIRECTORIES = (
    "hsi",
    "targets",
    "ground_truth",
    "references",
    "reference",
    "Train_spectral",
    "spectral",
)


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


def _candidate_filenames(sample: str, *, target: bool) -> List[str]:
    stems = (
        (f"{sample}{TARGET_SUFFIX}", sample)
        if target
        else (sample, f"{sample}{PREDICTION_SUFFIX}")
    )
    return [
        f"{stem}{suffix}"
        for stem in stems
        for suffix in SUPPORTED_ARRAY_SUFFIXES
    ]


def _candidate_directories(
    results_dir: Path,
    names: Iterable[str],
    *,
    include_root: bool,
) -> List[Path]:
    directories = [results_dir / name for name in names]
    if include_root:
        directories.append(results_dir)
    return directories


def _find_sample_file(
    results_dir: Path,
    sample: str,
    *,
    directories: Iterable[str],
    target: bool,
    include_root: bool,
) -> Optional[Path]:
    filenames = _candidate_filenames(sample, target=target)
    for directory in _candidate_directories(
        Path(results_dir), directories, include_root=include_root
    ):
        if not directory.is_dir():
            continue
        for filename in filenames:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return None


def prediction_path(results_dir: Path, sample: str) -> Optional[Path]:
    """Return the first supported prediction path for a sample.

    This recognizes both the original per-sample ``.npy`` layouts and
    SSTrans/NTIRE HDF5 ``.mat`` cubes in ``cubes/``.
    """
    return _find_sample_file(
        results_dir,
        sample,
        directories=PREDICTION_DIRECTORIES,
        target=False,
        include_root=True,
    )


def target_path(
    results_dir: Path,
    sample: str,
    *,
    allow_direct: bool = False,
) -> Optional[Path]:
    """Return the first supported target path for a sample.

    ``allow_direct`` is useful when ``results_dir`` is an explicitly supplied
    target directory such as ARAD-1K's ``Train_spectral``.  It defaults to
    false so a prediction-only SSTrans folder is never mistaken for a target.
    """
    return _find_sample_file(
        results_dir,
        sample,
        directories=TARGET_DIRECTORIES,
        target=True,
        include_root=allow_direct,
    )


def _iter_supported_files(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return ()
    return (
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_ARRAY_SUFFIXES
    )


def candidate_samples(results_dir: Path) -> List[str]:
    """Return sample names inferred from supported prediction locations."""
    samples: List[str] = []
    for directory in _candidate_directories(
        Path(results_dir), PREDICTION_DIRECTORIES, include_root=True
    ):
        for path in _iter_supported_files(directory):
            sample = _sample_from_prediction_stem(path.stem)
            if sample is not None:
                samples.append(sample)
    return _unique_sorted(samples)


def find_prediction_samples(
    results_dir: Path,
    max_samples: Optional[int] = None,
) -> List[str]:
    """Return samples with a prediction, including prediction-only SSTrans runs."""
    samples = candidate_samples(Path(results_dir))
    return samples if max_samples is None else samples[:max_samples]


def find_sample_pairs(
    results_dir: Path,
    max_samples: Optional[int] = None,
    target_dir: Optional[Path] = None,
) -> List[str]:
    """Return samples that have both prediction and target arrays."""
    target_root = Path(target_dir) if target_dir is not None else Path(results_dir)
    pairs = [
        sample
        for sample in candidate_samples(results_dir)
        if prediction_path(results_dir, sample) is not None
        and target_path(
            target_root,
            sample,
            allow_direct=target_dir is not None,
        )
        is not None
    ]
    return pairs if max_samples is None else pairs[:max_samples]


def filter_prediction_samples(results_dir: Path, samples: Iterable[str]) -> List[str]:
    """Keep requested samples that have a prediction."""
    return [sample for sample in samples if prediction_path(results_dir, sample) is not None]


def filter_sample_pairs(
    results_dir: Path,
    samples: Iterable[str],
    target_dir: Optional[Path] = None,
) -> List[str]:
    """Keep requested samples that have both prediction and target arrays."""
    target_root = Path(target_dir) if target_dir is not None else Path(results_dir)
    return [
        sample
        for sample in samples
        if prediction_path(results_dir, sample) is not None
        and target_path(
            target_root,
            sample,
            allow_direct=target_dir is not None,
        )
        is not None
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
            "Expected per-sample pairs such as hsi/<sample>.npy (or "
            "hsi/<sample>_pred.npy) and hsi/<sample>_target.npy, or "
            "SSTrans cubes/<sample>.mat plus a target directory."
        ),
    ]

    if not hsi_dir.exists():
        parts.append(f"Missing expected directory: {hsi_dir}")
    else:
        pred_count = len(
            [
                p
                for p in _iter_supported_files(hsi_dir)
                if _sample_from_prediction_stem(p.stem) is not None
            ]
        )
        target_count = len(
            [
                p
                for p in _iter_supported_files(hsi_dir)
                if p.stem.endswith(TARGET_SUFFIX)
            ]
        )
        parts.append(
            f"Found {pred_count} prediction candidate(s) and "
            f"{target_count} target candidate(s) under {hsi_dir}."
        )

    if predictions_dir.exists():
        pred_count = len(list(_iter_supported_files(predictions_dir)))
        parts.append(
            f"Found {pred_count} supported array file(s) under {predictions_dir}; "
            "matching target arrays are still required for comparison/error figures."
        )

    if targets_dir.exists():
        target_count = len(list(_iter_supported_files(targets_dir)))
        parts.append(f"Found {target_count} target array file(s) under {targets_dir}.")

    cubes_dir = results_dir / "cubes"
    if cubes_dir.exists():
        cube_count = len(
            [p for p in _iter_supported_files(cubes_dir) if p.suffix.lower() == ".mat"]
        )
        parts.append(
            f"Found {cube_count} SSTrans/NTIRE .mat cube(s) under {cubes_dir}; "
            "pass --targets <Train_spectral> for target-dependent figures."
        )

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


def no_prediction_samples_message(results_dir: Path) -> str:
    """Build a helpful error for folders without prediction files."""
    results_dir = Path(results_dir)
    return "\n".join(
        [
            f"No supported HSI predictions found in: {results_dir}",
            (
                "Expected .npy/.npz files under hsi/ or predictions/, or "
                "SSTrans/NTIRE HDF5 .mat cubes under cubes/."
            ),
        ]
    )

"""Input adapters for common hyperspectral reconstruction result formats.

SSTrans exports NTIRE-compatible HDF5 ``.mat`` files.  The stored ``cube``
dataset is transposed before writing, so this module reverses that convention
and exposes every loaded sample as a channel-first ``(C,H,W)`` cube.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from visualization_utils import to_chw


@dataclass(frozen=True)
class HsiSample:
    """A loaded HSI cube and its optional wavelength metadata."""

    cube: np.ndarray
    wavelengths: np.ndarray
    source: Path
    metadata: dict[str, Any]


def load_hsi(path: str | Path, *, cube_key: str = "cube") -> HsiSample:
    """Load ``.npy``, ``.npz``, SSTrans ``.mat``, or HDF5 HSI output.

    Arrays are normalized to ``(C,H,W)``.  SSTrans/NTIRE files are recognized
    by their ``bands`` dataset and decoded with the same transpose convention
    as ``hsiformer.ntire.load_ntire_cube``.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        array = np.load(source, allow_pickle=False)
        bands = None
        metadata: dict[str, Any] = {}
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            key = cube_key if cube_key in archive.files else archive.files[0]
            array = archive[key]
            bands = archive["bands"] if "bands" in archive.files else None
        metadata = {}
    elif suffix in {".mat", ".h5", ".hdf5"}:
        array, bands, metadata = _load_mat_or_hdf5(source, cube_key)
    else:
        raise ValueError(f"Unsupported HSI file extension: {source.suffix}")

    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional HSI cube, got {array.shape} from {source}."
        )

    band_values: Optional[np.ndarray]
    if bands is None:
        band_values = None
    else:
        band_values = np.asarray(bands, dtype=np.float32).reshape(-1)

    cube = to_chw(
        array,
        expected_bands=len(band_values) if band_values is not None else None,
    )
    if cube.ndim != 3:
        raise ValueError(
            f"Could not normalize HSI cube {array.shape} from {source} to CHW."
        )
    if band_values is None or len(band_values) != cube.shape[0]:
        band_values = np.linspace(400.0, 700.0, cube.shape[0], dtype=np.float32)

    return HsiSample(
        cube=np.ascontiguousarray(cube, dtype=np.float32),
        wavelengths=band_values,
        source=source,
        metadata=metadata,
    )


def _load_mat_or_hdf5(
    path: Path,
    cube_key: str,
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, Any]]:
    """Load an HDF5 NTIRE file, with a classic-MAT fallback."""
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            key = cube_key if cube_key in handle else _first_3d_dataset(handle)
            if key is None:
                raise KeyError(f"{path} contains no three-dimensional HSI dataset.")
            array = np.asarray(handle[key], dtype=np.float32)
            bands = np.asarray(handle["bands"]) if "bands" in handle else None
            metadata: dict[str, Any] = {
                "format": "ntire_hdf5" if "bands" in handle else "hdf5",
            }
            if "norm_factor" in handle:
                metadata["norm_factor"] = float(
                    np.asarray(handle["norm_factor"]).squeeze()
                )
            # SSTrans writes an HWC cube as array.T.  ``.T`` is intentionally
            # used here to preserve rectangular-frame orientation as well.
            if "bands" in handle and key == cube_key:
                array = array.T
            return array, bands, metadata
    except (OSError, KeyError, ValueError):
        from scipy.io import loadmat

        payload = loadmat(path)
        if cube_key not in payload:
            candidates = [
                value
                for name, value in payload.items()
                if not name.startswith("__")
                and isinstance(value, np.ndarray)
                and value.ndim == 3
            ]
            if not candidates:
                raise KeyError(f"{path} contains no three-dimensional HSI dataset.")
            array = candidates[0]
        else:
            array = payload[cube_key]
        bands = payload.get("bands")
        return array, bands, {"format": "matlab"}


def _first_3d_dataset(handle: Any) -> Optional[str]:
    found: Optional[str] = None

    def visit(name: str, node: Any) -> None:
        nonlocal found
        if found is None and getattr(node, "ndim", 0) == 3:
            found = name

    handle.visititems(visit)
    return found


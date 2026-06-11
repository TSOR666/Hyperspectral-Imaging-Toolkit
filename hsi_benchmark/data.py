from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat

LOGGER = logging.getLogger(__name__)

_CUBE_SUFFIXES = {".mat", ".h5", ".hdf5", ".npy", ".npz"}
_IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_CUBE_KEYS = ("cube", "reflectance", "rad", "hsi", "hyper", "data", "image")
_RGB_KEYS = ("rgb", "RGB", "bgr", "color", "image")
_WAVELENGTH_KEYS = ("wavelengths", "wavelength", "bands", "lambda")


@dataclass(frozen=True)
class SampleRecord:
    name: str
    hsi_path: Path
    rgb_path: Optional[Path] = None


@dataclass
class HSISample:
    name: str
    rgb: np.ndarray
    target: np.ndarray
    wavelengths: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class DatasetOptions:
    preset: str
    root: Path
    manifest: Optional[Path] = None
    rgb_root: Optional[Path] = None
    hsi_key: Optional[str] = None
    rgb_key: Optional[str] = None
    rgb_source: str = "auto"
    response_file: Optional[Path] = None
    wavelengths_file: Optional[Path] = None
    source_range: Optional[Tuple[float, float]] = None
    target_range: Tuple[float, float] = (400.0, 700.0)
    target_bands: int = 31
    hsi_scale: str = "auto"
    allow_spatial_resize: bool = False


def _natural_key(path: Path) -> List[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "sample"


def _read_manifest(path: Path, root: Path) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "hsi"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain columns {sorted(required)}")
        for row in reader:
            hsi_path = Path(row["hsi"])
            if not hsi_path.is_absolute():
                hsi_path = root / hsi_path
            rgb_value = (row.get("rgb") or "").strip()
            rgb_path = Path(rgb_value) if rgb_value else None
            if rgb_path is not None and not rgb_path.is_absolute():
                rgb_path = root / rgb_path
            records.append(
                SampleRecord(_safe_name(row["name"]), hsi_path, rgb_path)
            )
    return records


def _is_probable_cube_file(path: Path) -> bool:
    lower = path.stem.lower()
    return path.suffix.lower() in _CUBE_SUFFIXES and not any(
        token in lower for token in ("rgb", "response", "wavelength", "camera")
    )


def _band_stack_directories(root: Path) -> Iterable[Path]:
    for directory in sorted((path for path in root.rglob("*") if path.is_dir())):
        images = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in _IMAGE_SUFFIXES
            and "rgb" not in path.stem.lower()
        ]
        if len(images) >= 16:
            try:
                first = np.asarray(Image.open(images[0]))
            except OSError:
                continue
            if first.ndim == 2:
                yield directory


def _rgb_index(root: Optional[Path]) -> Dict[str, Path]:
    if root is None or not root.exists():
        return {}
    index: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            stem = path.stem.lower()
            index.setdefault(stem, path)
            index.setdefault(stem.replace("_rgb", ""), path)
    return index


def _find_paired_rgb(hsi_path: Path, index: Dict[str, Path]) -> Optional[Path]:
    stem = hsi_path.stem
    parent = hsi_path if hsi_path.is_dir() else hsi_path.parent
    scene = parent.parent.name if hsi_path.is_dir() and stem.lower().endswith("_ms") else stem
    bases = {
        stem,
        stem.removesuffix("_ms"),
        stem.replace("_MS", "").replace("_ms", ""),
        scene,
    }
    candidates: List[Path] = []
    search_dirs = [parent, parent.parent]
    for directory in search_dirs:
        for base in bases:
            for suffix in _IMAGE_SUFFIXES:
                candidates.extend(
                    [
                        directory / f"{base}_RGB{suffix}",
                        directory / f"{base}_rgb{suffix}",
                        directory / f"{base}{suffix}",
                    ]
                )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    for base in bases:
        match = index.get(base.lower())
        if match is not None:
            return match
    return None


def discover_samples(options: DatasetOptions) -> List[SampleRecord]:
    if options.manifest is not None:
        records = _read_manifest(options.manifest, options.root)
    else:
        cube_files = [
            path for path in options.root.rglob("*") if path.is_file() and _is_probable_cube_file(path)
        ]
        cube_dirs = list(_band_stack_directories(options.root))
        paths = sorted(cube_files, key=lambda path: str(path).lower())
        covered_parents = {path.parent for path in paths}
        paths.extend(path for path in cube_dirs if path not in covered_parents)
        rgb_lookup = _rgb_index(options.rgb_root or options.root)
        records = []
        seen: set[str] = set()
        for path in paths:
            stem = path.stem.removesuffix("_ms")
            name = _safe_name(stem)
            if name in seen:
                name = _safe_name(f"{path.parent.name}_{stem}")
            seen.add(name)
            records.append(SampleRecord(name, path, _find_paired_rgb(path, rgb_lookup)))
    missing = [str(record.hsi_path) for record in records if not record.hsi_path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing HSI paths; first: {missing[0]}")
    if not records:
        raise FileNotFoundError(
            f"No HSI cubes found under {options.root}. Use --manifest for custom layouts."
        )
    return records


def _arrays_from_file(path: Path) -> Dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return {"data": np.load(path)}
    if suffix == ".npz":
        with np.load(path) as archive:
            return {key: archive[key] for key in archive.files}
    if suffix in {".h5", ".hdf5"}:
        arrays: Dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as handle:
            def visit(name: str, value: object) -> None:
                if isinstance(value, h5py.Dataset):
                    arrays[name.split("/")[-1]] = np.asarray(value)

            handle.visititems(visit)
        return arrays
    try:
        return {
            key: np.asarray(value)
            for key, value in loadmat(path).items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        }
    except (NotImplementedError, ValueError, OSError):
        arrays = {}
        with h5py.File(path, "r") as handle:
            def visit(name: str, value: object) -> None:
                if isinstance(value, h5py.Dataset):
                    arrays[name.split("/")[-1]] = np.asarray(value)

            handle.visititems(visit)
        return arrays


def _choose_array(
    arrays: Dict[str, np.ndarray],
    *,
    requested_key: Optional[str],
    preferred_keys: Sequence[str],
    dimensions: int,
) -> Tuple[str, np.ndarray]:
    if requested_key:
        if requested_key not in arrays:
            raise KeyError(f"Key {requested_key!r} not found; available: {sorted(arrays)}")
        return requested_key, arrays[requested_key]
    for key in preferred_keys:
        if key in arrays and arrays[key].ndim == dimensions:
            return key, arrays[key]
    candidates = [
        (key, value)
        for key, value in arrays.items()
        if value.ndim == dimensions and min(value.shape) > 1
    ]
    if not candidates:
        raise ValueError(f"No {dimensions}D image array found; keys: {sorted(arrays)}")
    return max(candidates, key=lambda item: item[1].size)


def _load_band_stack(directory: Path) -> np.ndarray:
    paths = sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in _IMAGE_SUFFIXES
            and "rgb" not in path.stem.lower()
        ],
        key=_natural_key,
    )
    bands = [np.asarray(Image.open(path), dtype=np.float32) for path in paths]
    if not bands or any(band.ndim != 2 for band in bands):
        raise ValueError(f"Expected grayscale spectral band images in {directory}")
    if len({band.shape for band in bands}) != 1:
        raise ValueError(f"Spectral bands have inconsistent sizes in {directory}")
    return np.stack(bands, axis=0)


def _to_chw(cube: np.ndarray, preferred_bands: Optional[int] = None) -> np.ndarray:
    cube = np.asarray(cube)
    cube = np.squeeze(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D HSI cube, got {cube.shape}")
    matching_axes = (
        [axis for axis, size in enumerate(cube.shape) if size == preferred_bands]
        if preferred_bands is not None
        else []
    )
    spectral_axis = matching_axes[0] if len(matching_axes) == 1 else int(np.argmin(cube.shape))
    return np.moveaxis(cube, spectral_axis, 0)


def _scale_cube(cube: np.ndarray, mode: str) -> Tuple[np.ndarray, float]:
    raw = np.asarray(cube)
    if mode == "none":
        factor = 1.0
    elif mode != "auto":
        factor = float(mode)
        if factor <= 0:
            raise ValueError("hsi_scale must be auto, none, or a positive number")
    else:
        maximum = float(np.nanmax(raw))
        if maximum <= 1.5:
            factor = 1.0
        elif maximum <= 255:
            factor = 255.0
        elif maximum <= 4095:
            factor = 4095.0
        elif maximum <= 10000:
            factor = 10000.0
        else:
            factor = 65535.0
    scaled = np.asarray(raw, dtype=np.float32) / factor
    return np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0), factor


def _load_wavelength_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path)
    elif path.suffix.lower() in {".txt", ".csv"}:
        values = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    else:
        arrays = _arrays_from_file(path)
        values = None
        for key in _WAVELENGTH_KEYS:
            if key in arrays and arrays[key].ndim in {1, 2}:
                values = arrays[key]
                break
        if values is None:
            candidates = [
                value
                for value in arrays.values()
                if value.ndim in {1, 2} and 1 in value.shape
            ]
            if not candidates:
                raise ValueError(f"No wavelength vector found in {path}")
            values = max(candidates, key=lambda item: item.size)
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _source_wavelengths(
    arrays: Dict[str, np.ndarray],
    bands: int,
    options: DatasetOptions,
) -> Tuple[np.ndarray, str]:
    if options.wavelengths_file is not None:
        wavelengths = _load_wavelength_file(options.wavelengths_file)
        source = str(options.wavelengths_file)
    else:
        wavelengths = np.array([], dtype=np.float32)
        source = ""
        for key in _WAVELENGTH_KEYS:
            if key in arrays and arrays[key].size == bands:
                wavelengths = np.asarray(arrays[key], dtype=np.float32).reshape(-1)
                source = f"cube:{key}"
                break
        if wavelengths.size == 0:
            if options.source_range is not None:
                start, end = options.source_range
                source = "cli-source-range"
            elif bands == 31:
                start, end = 400.0, 700.0
                source = "31-band-default"
            elif options.preset.lower() in {"icvl", "bgu"}:
                start, end = 400.0, 1000.0
                source = f"{options.preset.lower()}-preset"
            else:
                raise ValueError(
                    f"Cannot infer wavelengths for {bands} bands. Pass --source-range "
                    "or --wavelengths-file."
                )
            wavelengths = np.linspace(start, end, bands, dtype=np.float32)
    if wavelengths.size != bands:
        raise ValueError(f"Found {wavelengths.size} wavelengths for a {bands}-band cube")
    return wavelengths, source


def resample_spectra(
    cube: np.ndarray,
    source_wavelengths: np.ndarray,
    target_wavelengths: np.ndarray,
) -> np.ndarray:
    if target_wavelengths.min() < source_wavelengths.min() or target_wavelengths.max() > source_wavelengths.max():
        raise ValueError(
            "Target wavelength range lies outside the source wavelength range: "
            f"target={target_wavelengths[[0, -1]]}, source={source_wavelengths[[0, -1]]}"
        )
    indices = np.searchsorted(source_wavelengths, target_wavelengths, side="left")
    indices = np.clip(indices, 1, len(source_wavelengths) - 1)
    left = indices - 1
    right = indices
    denominator = source_wavelengths[right] - source_wavelengths[left]
    weight = (target_wavelengths - source_wavelengths[left]) / np.maximum(denominator, 1e-12)
    return (
        cube[left] * (1.0 - weight[:, None, None])
        + cube[right] * weight[:, None, None]
    ).astype(np.float32)


def _load_rgb(path: Path, key: Optional[str]) -> np.ndarray:
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    else:
        arrays = _arrays_from_file(path)
        _, rgb = _choose_array(
            arrays, requested_key=key, preferred_keys=_RGB_KEYS, dimensions=3
        )
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
            rgb = np.moveaxis(rgb, 0, -1)
        if rgb.shape[-1] != 3:
            raise ValueError(f"Paired RGB must have three channels, got {rgb.shape}")
        maximum = float(np.nanmax(rgb))
        if maximum > 1.5:
            rgb /= 255.0 if maximum <= 255 else maximum
    return np.clip(np.nan_to_num(rgb), 0.0, 1.0)


def _load_response(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        response = np.load(path)
    else:
        arrays = _arrays_from_file(path)
        candidates = [
            value for value in arrays.values() if value.ndim == 2 and 3 in value.shape
        ]
        if not candidates:
            raise ValueError(f"No spectral response matrix found in {path}")
        response = max(candidates, key=lambda value: value.size)
    response = np.asarray(response, dtype=np.float32)
    if response.shape[0] == 3:
        response = response.T
    if response.shape[1] != 3:
        raise ValueError(f"Expected response shape bands x 3, got {response.shape}")
    return response


def spectral_to_rgb(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    response: Optional[np.ndarray] = None,
) -> np.ndarray:
    if response is None:
        response = np.stack(
            [
                np.exp(-0.5 * ((wavelengths - 605.0) / 45.0) ** 2),
                np.exp(-0.5 * ((wavelengths - 545.0) / 38.0) ** 2),
                np.exp(-0.5 * ((wavelengths - 450.0) / 30.0) ** 2),
            ],
            axis=1,
        ).astype(np.float32)
    if response.shape[0] != cube.shape[0]:
        raise ValueError(
            f"Response has {response.shape[0]} bands but cube has {cube.shape[0]}"
        )
    response = response / np.maximum(response.sum(axis=0, keepdims=True), 1e-12)
    rgb = np.einsum("chw,cr->hwr", cube, response, optimize=True)
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _resize_target(target: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = np.moveaxis(target, 0, -1)
    bands = []
    for band in range(tensor.shape[-1]):
        image = Image.fromarray(tensor[..., band], mode="F")
        bands.append(np.asarray(image.resize((width, height), Image.Resampling.BILINEAR)))
    return np.stack(bands, axis=0).astype(np.float32)


def load_sample(record: SampleRecord, options: DatasetOptions) -> HSISample:
    arrays: Dict[str, np.ndarray] = {}
    if record.hsi_path.is_dir():
        raw_cube = _load_band_stack(record.hsi_path)
        cube_key = "band-stack"
        cube_chw = raw_cube
    else:
        arrays = _arrays_from_file(record.hsi_path)
        cube_key, raw_cube = _choose_array(
            arrays,
            requested_key=options.hsi_key,
            preferred_keys=_CUBE_KEYS,
            dimensions=3,
        )
        cube_chw = _to_chw(raw_cube, preferred_bands=options.target_bands)
    source_cube, scale_factor = _scale_cube(cube_chw, options.hsi_scale)
    source_wavelengths, wavelength_source = _source_wavelengths(
        arrays, source_cube.shape[0], options
    )
    order = np.argsort(source_wavelengths)
    source_cube = source_cube[order]
    source_wavelengths = source_wavelengths[order]
    target_wavelengths = np.linspace(
        options.target_range[0],
        options.target_range[1],
        options.target_bands,
        dtype=np.float32,
    )
    if source_cube.shape[0] == options.target_bands and np.allclose(
        source_wavelengths, target_wavelengths, atol=1e-3
    ):
        target = source_cube.astype(np.float32, copy=False)
    else:
        target = resample_spectra(source_cube, source_wavelengths, target_wavelengths)

    response = _load_response(options.response_file) if options.response_file else None
    if response is not None and response.shape[0] == len(order):
        response = response[order]
    rgb_mode = options.rgb_source
    if rgb_mode not in {"auto", "paired", "cie", "response"}:
        raise ValueError("rgb_source must be auto, paired, cie, or response")
    use_paired = record.rgb_path is not None and rgb_mode in {"auto", "paired"}
    if use_paired:
        rgb = _load_rgb(record.rgb_path, options.rgb_key)
        rgb_protocol = f"paired:{record.rgb_path}"
    elif rgb_mode == "paired":
        raise FileNotFoundError(f"No paired RGB found for {record.name}")
    else:
        if rgb_mode == "response" and response is None:
            raise ValueError("rgb_source=response requires --response-file")
        synthesis_cube = target
        synthesis_wavelengths = target_wavelengths
        if response is not None and response.shape[0] == source_cube.shape[0]:
            synthesis_cube = source_cube
            synthesis_wavelengths = source_wavelengths
        rgb = spectral_to_rgb(synthesis_cube, synthesis_wavelengths, response)
        rgb_protocol = "camera-response" if response is not None else "approximate-cie"

    if rgb.shape[:2] == target.shape[1:]:
        pass
    elif rgb.shape[:2] == target.shape[1:][::-1]:
        target = target.transpose(0, 2, 1)
    elif options.allow_spatial_resize:
        target = _resize_target(target, rgb.shape[0], rgb.shape[1])
    else:
        raise ValueError(
            f"Spatial mismatch for {record.name}: rgb={rgb.shape[:2]} "
            f"hsi={target.shape[1:]}. Pass --allow-spatial-resize only if the "
            "dataset protocol permits interpolation."
        )

    return HSISample(
        name=record.name,
        rgb=np.moveaxis(rgb.astype(np.float32), -1, 0),
        target=np.clip(target.astype(np.float32), 0.0, 1.0),
        wavelengths=target_wavelengths,
        metadata={
            "hsi_path": str(record.hsi_path),
            "rgb_path": str(record.rgb_path) if record.rgb_path else None,
            "cube_key": cube_key,
            "source_bands": int(source_cube.shape[0]),
            "source_wavelength_min_nm": float(source_wavelengths.min()),
            "source_wavelength_max_nm": float(source_wavelengths.max()),
            "wavelength_source": wavelength_source,
            "scale_factor": scale_factor,
            "rgb_protocol": rgb_protocol,
        },
    )

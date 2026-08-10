from __future__ import annotations

from importlib.resources import files
from itertools import permutations
from pathlib import Path
from collections.abc import Sequence
from typing import Literal

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Split = Literal["train", "validation", "test"]
RGBNormalization = Literal["scale_255", "per_image"]

RGB_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPECTRAL_SUFFIXES = (".mat",)

# Redistributed ARAD-1K copies disagree on directory names, and some keep every
# scene inside Train_*. Lookup is per scene: the first directory that actually
# holds the file wins, so a Train-only root still serves validation and test.
RGB_DIRS: dict[str, tuple[str, ...]] = {
    "train": ("Train_RGB", "Train_rgb"),
    "validation": ("Valid_RGB", "Validation_RGB", "Val_RGB", "Train_RGB"),
    "test": ("Test_RGB", "Testing_RGB", "Valid_RGB", "Train_RGB"),
}
SPECTRAL_DIRS: dict[str, tuple[str, ...]] = {
    "train": ("Train_spectral", "Train_Spec", "Train_Spectral"),
    "validation": (
        "Valid_spectral",
        "Valid_Spec",
        "Validation_spectral",
        "Validation_Spec",
        "Val_Spec",
        "Train_spectral",
        "Train_Spec",
    ),
    "test": (
        "Test_spectral",
        "Test_Spec",
        "Testing_Spec",
        "Train_spectral",
        "Train_Spec",
    ),
}

# Variable names seen in ARAD-1K/NTIRE redistributions, lower-cased.
CUBE_DATASET_KEYS = (
    "cube",
    "reflectance",
    "rad",
    "hsi",
    "hyper",
    "data",
    "image",
)

# ARAD-1K's public ``Test_Spec`` uses this name for a raw MSFA measurement,
# not for a reconstructed 31-band cube. Exclude it before any shape-based
# fallback so a repackaged, cube-shaped mosaic cannot become a false target.
NON_CUBE_DATASET_KEYS = frozenset({"mosaic"})


def resolve_arad_directory(
    root: str | Path,
    split: Split,
    kind: Literal["rgb", "spectral"] = "rgb",
) -> Path:
    """Locate the ARAD-1K directory holding ``split`` data of ``kind``."""
    if split not in RGB_DIRS:
        raise ValueError(f"Unknown ARAD split: {split}")
    base = Path(root)
    candidates = RGB_DIRS[split] if kind == "rgb" else SPECTRAL_DIRS[split]
    suffixes = RGB_SUFFIXES if kind == "rgb" else SPECTRAL_SUFFIXES
    for name in candidates:
        directory = base / name
        if not directory.is_dir():
            continue
        if any(
            path.is_file() and path.suffix.lower() in suffixes
            for path in directory.iterdir()
        ):
            return directory
    raise FileNotFoundError(
        f"No {kind} directory with {split} data under {base}. Searched "
        f"{list(candidates)}."
    )


def load_arad_manifest(
    split: Split,
    manifest_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Load an ARAD-1K scene manifest without depending on the working directory."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown ARAD split: {split}")
    if manifest_path is None:
        filename = {
            "train": "arad1k_train.txt",
            "validation": "arad1k_validation.txt",
            "test": "arad1k_test.txt",
        }[split]
        manifest = files("hsiformer").joinpath("resources", filename)
        text = manifest.read_text(encoding="utf-8")
    else:
        text = Path(manifest_path).read_text(encoding="utf-8")

    scene_ids = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not scene_ids:
        raise ValueError("The ARAD manifest is empty.")
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("The ARAD manifest contains duplicate scene identifiers.")
    return scene_ids


class ARAD1KDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Lazy paired loader for the NTIRE 2022 ARAD-1K directory layout.

    Expected layout::

        root/
        |-- Train_RGB/ARAD_1K_0001.jpg
        `-- Train_spectral/ARAD_1K_0001.mat

    Split-specific directories (``Test_RGB``/``Test_Spec``,
    ``Valid_RGB``/``Valid_spectral``, ...) are resolved per scene, so a root
    that keeps everything under ``Train_*`` still works.

    Spectral files are read only when requested, avoiding the tens of gigabytes
    of RAM used by the original eager dataset.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: Split = "train",
        manifest_path: str | Path | None = None,
        crop_size: int | tuple[int, int] | None = None,
        stride: int | tuple[int, int] | None = None,
        random_crop: bool | None = None,
        crops_per_scene: int = 1,
        augment: bool | None = None,
        rgb_normalization: RGBNormalization = "scale_255",
        include_ycrcb: bool = False,
        spectral_channels: int = 31,
        cube_key: str = "cube",
        image_size: tuple[int, int] | None = None,
        require_targets: bool = True,
        probe_targets: bool | None = None,
        rgb_dirs: Sequence[str] | None = None,
        spectral_dirs: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if split not in RGB_DIRS:
            raise ValueError(f"Unknown ARAD split: {split}")

        self.split = split
        # Explicit directories are searched first; names may also be absolute
        # paths, for roots that keep cubes outside the ARAD tree.
        self.rgb_dirs = (*(rgb_dirs or ()), *RGB_DIRS[split])
        self.spectral_dirs = (*(spectral_dirs or ()), *SPECTRAL_DIRS[split])
        self.scene_ids = load_arad_manifest(split, manifest_path)
        self.crop_size = _pair(crop_size) if crop_size is not None else None
        self.stride = _pair(stride or crop_size or 1)
        self.random_crop = (
            split == "train" and self.crop_size is not None
            if random_crop is None
            else random_crop
        )
        self.crops_per_scene = crops_per_scene
        self.augment = split == "train" if augment is None else augment
        self.rgb_normalization = rgb_normalization
        self.include_ycrcb = include_ycrcb
        self.spectral_channels = spectral_channels
        self.cube_key = cube_key

        if crops_per_scene < 1:
            raise ValueError("crops_per_scene must be at least one.")
        if rgb_normalization not in {"scale_255", "per_image"}:
            raise ValueError(
                f"Unknown RGB normalization mode: {rgb_normalization}"
            )
        if self.random_crop and self.crop_size is None:
            raise ValueError("random_crop requires crop_size.")

        self._resolve_scene_files()
        self._resolve_targets(require_targets, probe_targets)

        self.image_size = image_size or self._read_image_size(self.scene_ids[0])
        self.full_frame_crop = bool(
            self.crop_size is not None
            and self.crop_size[0] >= self.image_size[0]
            and self.crop_size[1] >= self.image_size[1]
        )
        self._crop_positions = self._build_crop_positions()

    def _resolve_scene_files(self) -> None:
        """Bind every manifest scene to a concrete RGB and spectral file."""
        self._rgb_paths: dict[str, Path] = {}
        # Every candidate is kept: a mosaic-only Test_Spec entry must not hide
        # a real cube stored under a later directory name.
        self._spectral_candidates: dict[str, tuple[Path, ...]] = {}
        self._spectral_paths: dict[str, Path | None] = {}
        missing_rgb: list[str] = []

        for scene_id in self.scene_ids:
            rgb_path = _find_scene_file(
                self.root,
                scene_id,
                self.rgb_dirs,
                RGB_SUFFIXES,
            )
            if rgb_path is None:
                missing_rgb.append(scene_id)
                continue
            self._rgb_paths[scene_id] = rgb_path
            candidates = _find_scene_files(
                self.root,
                scene_id,
                self.spectral_dirs,
                SPECTRAL_SUFFIXES,
            )
            self._spectral_candidates[scene_id] = candidates
            self._spectral_paths[scene_id] = candidates[0] if candidates else None

        if missing_rgb:
            raise FileNotFoundError(
                f"No RGB file for {len(missing_rgb)} {self.split} scenes under "
                f"{self.root} (searched {list(self.rgb_dirs)} with suffixes "
                f"{list(RGB_SUFFIXES)}); first missing: {missing_rgb[:5]}."
            )

        self.rgb_root = next(iter(self._rgb_paths.values())).parent

    def _resolve_targets(
        self,
        require_targets: bool,
        probe_targets: bool | None,
    ) -> None:
        """Record which scenes carry a usable ground-truth cube.

        The official ARAD-1K test release ships ``Test_Spec`` files that hold
        only the raw MSFA ``mosaic`` payload, not a 31-band cube. Probing every
        candidate directory both recovers a cube stored elsewhere in the root
        and turns a genuinely absent target into an explicit, actionable state
        instead of an opaque cube-key ``KeyError`` mid-evaluation.
        """
        # Train_spectral is always cube data; skip 900 file opens at startup.
        probe = self.split != "train" if probe_targets is None else probe_targets
        self.unusable_targets: dict[str, str] = {}

        for scene_id in self.scene_ids:
            candidates = self._spectral_candidates[scene_id]
            if not candidates:
                self.unusable_targets[scene_id] = (
                    f"no spectral file found in {list(self.spectral_dirs)}"
                )
                continue
            if not probe:
                continue

            reasons: list[str] = []
            for candidate in candidates:
                reason = _probe_cube_file(candidate, self.spectral_channels)
                if reason is None:
                    self._spectral_paths[scene_id] = candidate
                    break
                reasons.append(reason)
            else:
                self.unusable_targets[scene_id] = reasons[0]

        spectral_parents = [
            self._spectral_paths[scene_id].parent
            for scene_id in self.scene_ids
            if self.has_target(scene_id)
            and self._spectral_paths[scene_id] is not None
        ]
        self.spectral_root = spectral_parents[0] if spectral_parents else None
        self.scene_ids_with_targets = tuple(
            scene_id
            for scene_id in self.scene_ids
            if scene_id not in self.unusable_targets
        )
        if require_targets and self.unusable_targets:
            examples = "; ".join(
                f"{scene_id}: {reason}"
                for scene_id, reason in list(self.unusable_targets.items())[:3]
            )
            raise FileNotFoundError(
                f"No usable ground-truth cube for "
                f"{len(self.unusable_targets)}/{len(self.scene_ids)} "
                f"{self.split} scenes under {self.root} (searched "
                f"{list(self.spectral_dirs)}). {examples}. Point "
                "spectral_dirs at the directory holding the cubes, pass "
                "require_targets=False to reconstruct without metrics, or "
                "evaluate a split that ships cubes (e.g. validation)."
            )

    def has_target(self, scene_id: str) -> bool:
        return scene_id not in self.unusable_targets

    def _rgb_path(self, scene_id: str) -> Path:
        return self._rgb_paths[scene_id]

    def _spectral_path(self, scene_id: str) -> Path | None:
        return self._spectral_paths[scene_id]

    def _read_image_size(self, scene_id: str) -> tuple[int, int]:
        path = self._rgb_path(scene_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
        return height, width

    def _build_crop_positions(self) -> tuple[tuple[int, int], ...]:
        if self.crop_size is None or self.random_crop:
            return ((0, 0),)
        if self.full_frame_crop:
            return ((0, 0),)
        image_height, image_width = self.image_size
        crop_height, crop_width = self.crop_size
        if crop_height > image_height or crop_width > image_width:
            raise ValueError(
                f"Crop {self.crop_size} exceeds image size {self.image_size}."
            )
        rows = _grid_starts(image_height, crop_height, self.stride[0])
        columns = _grid_starts(image_width, crop_width, self.stride[1])
        return tuple((row, column) for row in rows for column in columns)

    def __len__(self) -> int:
        if self.random_crop:
            return len(self.scene_ids) * self.crops_per_scene
        return len(self.scene_ids) * len(self._crop_positions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        if self.random_crop:
            scene_index = index // self.crops_per_scene
            crop_position = None
        else:
            patches_per_scene = len(self._crop_positions)
            scene_index, patch_index = divmod(index, patches_per_scene)
            crop_position = self._crop_positions[patch_index]

        scene_id = self.scene_ids[scene_index]
        rgb_uint8, ycrcb_uint8 = self._load_rgb(scene_id)
        height, width = rgb_uint8.shape[-2:]
        # Match the reference ARAD pipeline: estimate per-image normalization
        # statistics from the complete scene, then select a training crop.
        # Normalizing after cropping gives every random crop a different RGB
        # scale while validation uses full-scene statistics, creating a
        # train/validation input-distribution mismatch.
        cond = _normalize_rgb(rgb_uint8, self.rgb_normalization)
        ycrcb = (
            _normalize_rgb(ycrcb_uint8, self.rgb_normalization)
            if ycrcb_uint8 is not None
            else None
        )
        if (
            self.crop_size is not None
            and not self.random_crop
            and (height, width) != self.image_size
        ):
            raise ValueError(
                f"Scene {scene_id} has size {(height, width)}, but the grid "
                f"was built for {self.image_size}."
            )

        augment = self.augment
        active_crop_size: tuple[int, int] | None = None
        if self.crop_size is not None and not self.full_frame_crop:
            crop_fits = (
                self.crop_size[0] <= height and self.crop_size[1] <= width
            )
            if not (self.random_crop and not crop_fits):
                if crop_position is None:
                    crop_position = _random_crop_position(
                        (height, width),
                        self.crop_size,
                    )
                active_crop_size = self.crop_size
                cond = _crop(cond, crop_position, self.crop_size)
                if ycrcb is not None:
                    ycrcb = _crop(
                        ycrcb,
                        crop_position,
                        self.crop_size,
                    )

        label = (
            self._load_cube(
                scene_id,
                height,
                width,
                crop_position=(
                    crop_position if active_crop_size is not None else None
                ),
                crop_size=active_crop_size,
            )
            if self.has_target(scene_id)
            else None
        )

        sample: dict[str, torch.Tensor | str] = {
            "cond": cond,
            "scene_id": scene_id,
        }
        # Ground-truth-less scenes (ARAD-1K test) omit the key entirely so
        # consumers fail loudly rather than scoring against a placeholder.
        if label is not None:
            sample["label"] = label
        if self.include_ycrcb:
            if ycrcb is None:
                raise RuntimeError("YCbCr input was requested but not loaded.")
            sample["ycrcb"] = torch.cat([cond, ycrcb], dim=0)

        if augment:
            tensor_keys = [
                key for key in ("cond", "label", "ycrcb") if key in sample
            ]
            tensors = [sample[key] for key in tensor_keys]
            augmented = _paired_augmentation(
                [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
            )
            for key, tensor in zip(tensor_keys, augmented, strict=True):
                sample[key] = tensor
        return sample

    def _load_rgb(
        self,
        scene_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        path = self._rgb_path(scene_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.include_ycrcb:
            return _load_rgb_tensors(path)
        return _load_rgb_tensor(path), None

    def _load_cube(
        self,
        scene_id: str,
        height: int,
        width: int,
        *,
        crop_position: tuple[int, int] | None = None,
        crop_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        path = self._spectral_path(scene_id)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"No spectral file for scene '{scene_id}' under {self.root}."
            )
        with h5py.File(path, "r") as handle:
            dataset = handle[
                _resolve_cube_key(
                    handle,
                    path,
                    channels=self.spectral_channels,
                    preferred=self.cube_key,
                )
            ]
            cube = None
            if crop_position is not None and crop_size is not None:
                cube = _read_cube_crop(
                    dataset,
                    self.spectral_channels,
                    height,
                    width,
                    crop_position,
                    crop_size,
                )
            if cube is None:
                cube = np.asarray(dataset, dtype=np.float32).squeeze()
                cube = _to_chw(
                    cube,
                    self.spectral_channels,
                    height,
                    width,
                )
                if crop_position is not None and crop_size is not None:
                    row, column = crop_position
                    crop_height, crop_width = crop_size
                    cube = cube[
                        :,
                        row : row + crop_height,
                        column : column + crop_width,
                    ]
        return torch.from_numpy(np.ascontiguousarray(cube))


class RGBImageDataset(Dataset[dict[str, torch.Tensor | str]]):
    """RGB-only loader for NTIRE-style inference and blind test folders."""

    _DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    def __init__(
        self,
        root: str | Path,
        *,
        scene_ids: Sequence[str] | None = None,
        manifest_path: str | Path | None = None,
        rgb_normalization: RGBNormalization = "scale_255",
        extensions: Sequence[str] = _DEFAULT_EXTENSIONS,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if scene_ids is not None and manifest_path is not None:
            raise ValueError("Pass scene_ids or manifest_path, not both.")
        if rgb_normalization not in {"scale_255", "per_image"}:
            raise ValueError(
                f"Unknown RGB normalization mode: {rgb_normalization}"
            )

        self.rgb_normalization = rgb_normalization
        self.extensions = tuple(_normalize_extension(value) for value in extensions)
        if manifest_path is not None:
            scene_ids = tuple(
                line.strip()
                for line in Path(manifest_path)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )

        if scene_ids is None:
            paths = sorted(
                path
                for path in self.root.iterdir()
                if path.is_file() and path.suffix.lower() in self.extensions
            )
        else:
            paths = [self._resolve_scene_path(scene_id) for scene_id in scene_ids]
        if not paths:
            raise ValueError(f"No supported RGB images found under {self.root}.")
        self.paths = tuple(paths)

    def _resolve_scene_path(self, scene_id: str) -> Path:
        candidate = self.root / scene_id
        if candidate.is_file():
            return candidate
        for extension in self.extensions:
            candidate = self.root / f"{scene_id}{extension}"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Could not find RGB image for scene '{scene_id}' under {self.root}."
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        rgb = _load_rgb_tensor(path)
        return {
            "cond": _normalize_rgb(rgb, self.rgb_normalization),
            "scene_id": path.stem,
            "source_path": str(path),
        }


def _find_scene_files(
    root: Path,
    scene_id: str,
    subdirs: Sequence[str],
    suffixes: Sequence[str],
) -> tuple[Path, ...]:
    """Return every file matching ``scene_id``, in candidate-directory order."""
    matches: list[Path] = []
    seen: set[Path] = set()
    for subdir in subdirs:
        # An absolute subdir replaces root, so callers may pass full paths.
        directory = root / subdir
        if not directory.is_dir():
            continue
        for suffix in suffixes:
            candidate = directory / f"{scene_id}{suffix}"
            resolved = candidate.resolve()
            if candidate.is_file() and resolved not in seen:
                seen.add(resolved)
                matches.append(candidate)
    return tuple(matches)


def _find_scene_file(
    root: Path,
    scene_id: str,
    subdirs: Sequence[str],
    suffixes: Sequence[str],
) -> Path | None:
    matches = _find_scene_files(root, scene_id, subdirs, suffixes)
    return matches[0] if matches else None


def _resolve_cube_key(
    handle: h5py.File,
    path: Path,
    *,
    channels: int = 31,
    preferred: str | None = "cube",
) -> str:
    """Return the name of the HDF5 dataset holding the HSI cube.

    Prefers the canonical ``cube`` key, then known aliases (case-insensitive),
    then any non-mosaic 3D dataset with a matching spectral axis, then a lone
    non-mosaic 3D dataset. Raises a descriptive ``KeyError`` listing the
    available datasets when none qualifies, so a wrong or MSFA-only file is
    diagnosable.
    """
    available = [
        key
        for key in handle.keys()
        if not key.startswith("#") and isinstance(handle[key], h5py.Dataset)
    ]
    by_lower = {key.lower(): key for key in available}
    candidates = CUBE_DATASET_KEYS
    if preferred:
        candidates = (preferred.lower(), *CUBE_DATASET_KEYS)
    for candidate in candidates:
        if candidate in NON_CUBE_DATASET_KEYS:
            continue
        if candidate in by_lower:
            return by_lower[candidate]

    banded = [
        key
        for key in available
        if key.lower() not in NON_CUBE_DATASET_KEYS
        and handle[key].ndim == 3
        and channels in tuple(int(value) for value in handle[key].shape)
    ]
    if banded:
        return banded[0]
    cubes = [
        key
        for key in available
        if key.lower() not in NON_CUBE_DATASET_KEYS and handle[key].ndim == 3
    ]
    if len(cubes) == 1:
        return cubes[0]

    hint = ""
    if "mosaic" in by_lower:
        hint = (
            " This file holds the ARAD-1K MSFA 'mosaic' payload; the official "
            "test split ships without spectral ground truth, so it can only be "
            "reconstructed, not scored."
        )
    raise KeyError(
        f"No HSI cube dataset found in {path}. Expected a variable named "
        f"'cube' (ARAD-1K/NTIRE convention) or one of {CUBE_DATASET_KEYS}; "
        f"available datasets: {available or '<none>'}.{hint}"
    )


def _probe_cube_file(path: Path, channels: int) -> str | None:
    """Return why ``path`` is unusable as a target, or ``None`` if it is fine."""
    try:
        with h5py.File(path, "r") as handle:
            key = _resolve_cube_key(handle, path, channels=channels)
            shape = tuple(int(value) for value in handle[key].shape)
    except (OSError, KeyError, ValueError) as error:
        # KeyError's str() escapes its message; args[0] keeps paths readable.
        return str(error.args[0]) if error.args else str(error)
    if len(shape) != 3 or channels not in shape:
        return (
            f"dataset '{key}' in {path} has shape {shape}; expected a 3D cube "
            f"with a {channels}-band axis"
        )
    return None


def _to_chw(
    cube: np.ndarray,
    channels: int,
    height: int,
    width: int,
) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected a three-dimensional cube, got {cube.shape}.")
    order = _cube_axis_order(cube.shape, channels, height, width)
    if order is None:
        raise ValueError(
            f"Cannot align spectral cube {cube.shape} to CHW target "
            f"{(channels, height, width)}."
        )
    return np.transpose(cube, order)


def _cube_axis_order(
    shape: Sequence[int],
    channels: int,
    height: int,
    width: int,
) -> tuple[int, int, int] | None:
    target = (channels, height, width)
    matching = [
        order
        for order in permutations(range(3))
        if tuple(shape[index] for index in order) == target
    ]
    return matching[0] if matching else None


def _read_cube_crop(
    dataset: h5py.Dataset,
    channels: int,
    height: int,
    width: int,
    position: tuple[int, int],
    crop_size: tuple[int, int],
) -> np.ndarray | None:
    if dataset.ndim != 3:
        return None
    order = _cube_axis_order(dataset.shape, channels, height, width)
    if order is None:
        return None

    row, column = position
    crop_height, crop_width = crop_size
    slices = [slice(None), slice(None), slice(None)]
    slices[order[1]] = slice(row, row + crop_height)
    slices[order[2]] = slice(column, column + crop_width)
    cube = np.asarray(dataset[tuple(slices)], dtype=np.float32)
    return np.transpose(cube, order)


def _normalize_rgb(
    image: torch.Tensor,
    mode: RGBNormalization,
) -> torch.Tensor:
    image = image.to(torch.float32)
    if mode == "scale_255":
        return image / 255.0
    minimum = image.amin()
    scale = (image.amax() - minimum).clamp_min(torch.finfo(image.dtype).eps)
    return (image - minimum) / scale


def _load_rgb_tensors(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        rgb = np.asarray(rgb_image, dtype=np.uint8).copy()
        # PIL returns Y, Cb, Cr; the historical code used OpenCV's Y, Cr, Cb.
        ycbcr = np.asarray(rgb_image.convert("YCbCr"), dtype=np.uint8).copy()
    ycrcb = ycbcr[..., [0, 2, 1]]
    return (
        torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
        torch.from_numpy(ycrcb).permute(2, 0, 1).contiguous(),
    )


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def _normalize_extension(value: str) -> str:
    value = value.lower()
    return value if value.startswith(".") else f".{value}"


def _grid_starts(length: int, crop: int, stride: int) -> tuple[int, ...]:
    if stride < 1:
        raise ValueError("Stride must be positive.")
    starts = list(range(0, length - crop + 1, stride))
    final = length - crop
    if not starts or starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def _random_crop_position(
    image_size: tuple[int, int],
    crop_size: tuple[int, int],
) -> tuple[int, int]:
    max_row = image_size[0] - crop_size[0]
    max_column = image_size[1] - crop_size[1]
    if max_row < 0 or max_column < 0:
        raise ValueError(f"Crop {crop_size} exceeds image size {image_size}.")
    row = int(torch.randint(max_row + 1, ()).item())
    column = int(torch.randint(max_column + 1, ()).item())
    return row, column


def _crop(
    tensor: torch.Tensor,
    position: tuple[int, int],
    crop_size: tuple[int, int],
) -> torch.Tensor:
    row, column = position
    height, width = crop_size
    return tensor[:, row : row + height, column : column + width]


def _paired_augmentation(
    tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    if not tensors:
        return tensors
    height, width = tensors[0].shape[-2:]
    rotations = (0, 2) if height != width else (0, 1, 2, 3)
    rotation = rotations[int(torch.randint(len(rotations), ()).item())]
    vertical_flip = bool(torch.randint(2, ()).item())
    horizontal_flip = bool(torch.randint(2, ()).item())

    outputs = []
    for tensor in tensors:
        tensor = torch.rot90(tensor, rotation, dims=(-2, -1))
        if vertical_flip:
            tensor = torch.flip(tensor, dims=(-2,))
        if horizontal_flip:
            tensor = torch.flip(tensor, dims=(-1,))
        outputs.append(tensor.contiguous())
    return outputs


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))

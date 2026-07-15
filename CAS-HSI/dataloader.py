"""ARAD-1K dataloaders for CAS-HSI, MST++/NTIRE-faithful.

Layout expected under ``data_root`` (the standard ARAD-1K / MST++ layout used by
every other project in this repo):

    data_root/
        Train_RGB/    ARAD_1K_0001.jpg ...
        Train_Spec/   ARAD_1K_0001.mat ...      (h5 with a 'cube' dataset)
        Valid_RGB/    (optional; falls back to Train_RGB)
        Valid_Spec/   (optional; falls back to Train_Spec)
        split_txt/
            train_list.txt
            valid_list.txt
            test_list.txt     <- see prepare_splits.py

Conventions, all matching MST++ and the sibling projects:

* RGB read with cv2 (BGR), converted to RGB, then **per-image min-max normalized**
  to [0, 1].  This is what MST++'s own loader does; it is *not* a plain /255.
* HSI cubes are stored ``(bands, width, height)`` and transposed to
  ``(bands, height, width)``.
* Training samples 128x128 patches on a stride-8 grid, with rot90 + h/v flips.
  Geometric augmentation only: photometric jitter would break the RGB->spectrum
  physics the model is supposed to learn.
* Validation/test return whole scenes at their native resolution.
"""

from __future__ import annotations

import bisect
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["TrainDataset", "EvalDataset", "DatasetConfig", "load_rgb_image", "load_hsi_cube"]

_RGB_DIRS = {
    "train": ("Train_RGB",),
    "valid": ("Valid_RGB", "Validation_RGB", "Val_RGB", "Train_RGB"),
    "test": ("Test_RGB", "Valid_RGB", "Validation_RGB", "Train_RGB"),
}
_HSI_DIRS = {
    "train": ("Train_Spec",),
    "valid": ("Valid_Spec", "Validation_Spec", "Val_Spec", "Train_Spec"),
    "test": ("Test_Spec", "Valid_Spec", "Validation_Spec", "Train_Spec"),
}
_SPLIT_FILES = {
    "train": ("train_list.txt",),
    "valid": ("valid_list.txt", "val_list.txt"),
    "test": ("test_list.txt",),
}
_RGB_SUFFIXES = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")

_CACHE_DTYPES = {"float32": np.float32, "float16": np.float16}


def _resolve_cache_dtype(name: str) -> np.dtype:
    key = str(name).strip().lower()
    if key not in _CACHE_DTYPES:
        raise ValueError(f"cache_dtype must be one of {sorted(_CACHE_DTYPES)}, got {name!r}")
    return np.dtype(_CACHE_DTYPES[key])


def load_hsi_cube(path: Path) -> np.ndarray:
    """Read an ARAD .mat cube as ``(bands, height, width)`` float32."""
    with h5py.File(path, "r") as mat:
        if "cube" not in mat:
            raise KeyError(f"Missing 'cube' dataset in {path} (keys: {list(mat.keys())})")
        cube = np.array(mat["cube"], dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Unexpected cube shape {cube.shape} in {path}")
    # ARAD stores (bands, width, height).
    return np.transpose(cube, (0, 2, 1))


def load_rgb_image(path: Path, *, bgr2rgb: bool = True, norm: str = "minmax") -> np.ndarray:
    """Read an RGB image as ``(3, height, width)`` float32 in [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)

    if norm == "minmax":
        # MST++'s own normalization. A flat image has zero range: preserve its mean
        # intensity rather than dividing by zero or collapsing it to black, which
        # would destroy calibration targets and uniform regions.
        denom = float(image.max() - image.min())
        if denom < 1e-6:
            image = np.full_like(image, float(np.clip(image.mean() / 255.0, 0.0, 1.0)))
        else:
            image = (image - image.min()) / denom
    elif norm == "div255":
        image = image / 255.0
    else:
        raise ValueError(f"rgb_norm must be 'minmax' or 'div255', got {norm!r}")

    return np.transpose(image, (2, 0, 1))


@dataclass
class DatasetConfig:
    data_root: Path
    logger: logging.Logger
    bgr2rgb: bool = True
    rgb_norm: str = "minmax"

    def _resolve(
        self, stem: str, subdirs: Sequence[str], suffixes: Sequence[str], kind: str
    ) -> Optional[Path]:
        for subdir in subdirs:
            for suffix in suffixes:
                candidate = self.data_root / subdir / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate
        checked = ", ".join(f"{d}/*{s}" for d in subdirs for s in suffixes)
        self.logger.warning("%s missing for %s (checked %s)", kind, stem, checked)
        return None

    def rgb_path(self, stem: str, split: str) -> Optional[Path]:
        return self._resolve(stem, _RGB_DIRS[split], _RGB_SUFFIXES, "RGB image")

    def hsi_path(self, stem: str, split: str) -> Optional[Path]:
        return self._resolve(stem, _HSI_DIRS[split], (".mat",), "HSI cube")


def _read_split(data_root: Path, split: str) -> Tuple[Path, List[str]]:
    split_dir = data_root / "split_txt"
    for name in _SPLIT_FILES[split]:
        path = split_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                stems = [Path(line.strip()).stem for line in handle if line.strip()]
            if not stems:
                raise RuntimeError(f"Split file is empty: {path}")
            return path, stems

    checked = ", ".join(str(split_dir / n) for n in _SPLIT_FILES[split])
    hint = ""
    if split == "test":
        hint = (
            "\n\nARAD-1K's public release has NO ground-truth test split (the NTIRE 2022 "
            "challenge withheld it). Run prepare_splits.py to materialize a test split "
            "and to see exactly which protocol you are choosing."
        )
    raise FileNotFoundError(f"Split file not found for '{split}'; checked: {checked}{hint}")


def _patch_starts(extent: int, crop_size: int, stride: int) -> List[int]:
    """Tail-inclusive patch starts for one spatial axis."""
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if extent <= crop_size:
        return [0]
    last = extent - crop_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _pad_to_crop(array: np.ndarray, crop_size: int) -> np.ndarray:
    """Pad CHW data up to ``crop_size`` without tripping reflect mode on 1-pixel axes."""
    height, width = array.shape[1:]
    pad_h = max(0, crop_size - height)
    pad_w = max(0, crop_size - width)
    if pad_h == 0 and pad_w == 0:
        return array
    mode = "reflect" if height > 1 and width > 1 else "edge"
    return np.pad(array, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)


class TrainDataset(Dataset):
    """Patch-based ARAD-1K training set (MST++: 128x128 crops on a stride-8 grid)."""

    def __init__(
        self,
        data_root: str | Path,
        crop_size: int = 128,
        *,
        stride: int = 8,
        bgr2rgb: bool = True,
        rgb_norm: str = "minmax",
        augment: bool = True,
        cache_dtype: str = "float32",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()
        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        self.config = DatasetConfig(
            data_root=root,
            logger=logger or logging.getLogger(__name__),
            bgr2rgb=bgr2rgb,
            rgb_norm=rgb_norm,
        )
        self.crop_size = int(crop_size)
        self.stride = int(stride)
        if self.crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {self.crop_size}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        self.augment = bool(augment)
        self._cache_dtype = _resolve_cache_dtype(cache_dtype)

        split_file, stems = _read_split(root, "train")
        self.config.logger.info(
            "Loading %d training scenes from %s (%s)", len(stems), root, split_file.name
        )

        self.rgb_images: List[np.ndarray] = []
        self.hsi_cubes: List[np.ndarray] = []
        self._patch_grids: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        self._patch_offsets: List[int] = []
        total_patches = 0

        for stem in stems:
            rgb_path = self.config.rgb_path(stem, "train")
            hsi_path = self.config.hsi_path(stem, "train")
            if rgb_path is None or hsi_path is None:
                continue
            try:
                rgb = load_rgb_image(rgb_path, bgr2rgb=bgr2rgb, norm=rgb_norm)
                hsi = load_hsi_cube(hsi_path)
            except Exception as exc:  # pragma: no cover - defensive
                self.config.logger.warning("Skipping %s: %s", stem, exc)
                continue

            if rgb.shape[1:] != hsi.shape[1:]:
                self.config.logger.warning(
                    "Skipping %s: RGB %s and HSI %s disagree spatially",
                    stem, rgb.shape[1:], hsi.shape[1:],
                )
                continue

            height, width = hsi.shape[1:]
            if height < self.crop_size or width < self.crop_size:
                rgb = _pad_to_crop(rgb, self.crop_size)
                hsi = _pad_to_crop(hsi, self.crop_size)
                height, width = hsi.shape[1:]

            self.rgb_images.append(rgb.astype(self._cache_dtype, copy=False))
            self.hsi_cubes.append(hsi.astype(self._cache_dtype, copy=False))

            y_starts = tuple(_patch_starts(height, self.crop_size, self.stride))
            x_starts = tuple(_patch_starts(width, self.crop_size, self.stride))
            self._patch_grids.append((y_starts, x_starts))
            total_patches += len(y_starts) * len(x_starts)
            self._patch_offsets.append(total_patches)

        if not self.rgb_images:
            raise RuntimeError(
                f"No valid training samples loaded from {root}. Check Train_RGB/, "
                "Train_Spec/ and split_txt/train_list.txt."
            )

        self.total_patches = total_patches
        self.config.logger.info(
            "Prepared %d patches across %d scenes (crop=%d, stride=%d)",
            self.total_patches, len(self.rgb_images), self.crop_size, self.stride,
        )

    def __len__(self) -> int:
        return self.total_patches

    @staticmethod
    def _augment(rgb: np.ndarray, hsi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rotations = random.randint(0, 3)
        if rotations:
            rgb = np.rot90(rgb, rotations, axes=(1, 2))
            hsi = np.rot90(hsi, rotations, axes=(1, 2))
        if random.random() < 0.5:
            rgb, hsi = rgb[:, ::-1, :], hsi[:, ::-1, :]
        if random.random() < 0.5:
            rgb, hsi = rgb[:, :, ::-1], hsi[:, :, ::-1]
        return rgb, hsi

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_index = bisect.bisect_right(self._patch_offsets, index)
        previous = self._patch_offsets[image_index - 1] if image_index > 0 else 0
        local = index - previous

        rgb = self.rgb_images[image_index]
        hsi = self.hsi_cubes[image_index]
        y_starts, x_starts = self._patch_grids[image_index]

        row, col = divmod(local, len(x_starts))
        y = y_starts[row]
        x = x_starts[col]

        rgb_patch = rgb[:, y : y + self.crop_size, x : x + self.crop_size]
        hsi_patch = hsi[:, y : y + self.crop_size, x : x + self.crop_size]

        if self.augment:
            rgb_patch, hsi_patch = self._augment(rgb_patch, hsi_patch)

        return (
            torch.from_numpy(np.ascontiguousarray(rgb_patch, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(hsi_patch, dtype=np.float32)),
        )


class EvalDataset(Dataset):
    """Full-scene ARAD-1K validation / test set (native resolution, batch size 1)."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        split: str = "valid",
        bgr2rgb: bool = True,
        rgb_norm: str = "minmax",
        cache_dtype: str = "float32",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()
        if split not in {"valid", "test"}:
            raise ValueError(f"split must be 'valid' or 'test', got {split!r}")

        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        self.split = split
        self.config = DatasetConfig(
            data_root=root,
            logger=logger or logging.getLogger(__name__),
            bgr2rgb=bgr2rgb,
            rgb_norm=rgb_norm,
        )
        cache = _resolve_cache_dtype(cache_dtype)

        split_file, stems = _read_split(root, split)
        self.config.logger.info(
            "Loading %d %s scenes from %s (%s)", len(stems), split, root, split_file.name
        )

        self.rgb_images: List[torch.Tensor] = []
        self.hsi_cubes: List[torch.Tensor] = []
        self.stems: List[str] = []
        missing: List[str] = []

        for stem in stems:
            rgb_path = self.config.rgb_path(stem, split)
            hsi_path = self.config.hsi_path(stem, split)
            if rgb_path is None or hsi_path is None:
                missing.append(stem)
                continue
            try:
                rgb = load_rgb_image(rgb_path, bgr2rgb=bgr2rgb, norm=rgb_norm)
                hsi = load_hsi_cube(hsi_path)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"Failed to load {split} scene {stem}: {exc}") from exc

            if rgb.shape[1:] != hsi.shape[1:]:
                raise ValueError(
                    f"{split} scene {stem}: RGB {rgb.shape[1:]} and HSI {hsi.shape[1:]} "
                    "disagree spatially"
                )

            self.rgb_images.append(torch.from_numpy(np.ascontiguousarray(rgb, dtype=cache)))
            self.hsi_cubes.append(torch.from_numpy(np.ascontiguousarray(hsi, dtype=cache)))
            self.stems.append(stem)

        if missing:
            raise RuntimeError(
                f"{split} split has {len(missing)} scene(s) without both RGB and HSI data, "
                f"e.g. {missing[:3]}. Run prepare_splits.py before evaluating."
            )

        if not self.rgb_images:
            raise RuntimeError(
                f"No {split} samples loaded from {root} via {split_file}. "
                f"Checked RGB dirs {list(_RGB_DIRS[split])} and HSI dirs {list(_HSI_DIRS[split])}."
            )

    def __len__(self) -> int:
        return len(self.rgb_images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rgb_images[index].float(), self.hsi_cubes[index].float()

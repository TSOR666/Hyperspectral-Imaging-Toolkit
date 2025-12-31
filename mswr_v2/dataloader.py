#!/usr/bin/env python3
"""MSWR-Net training dataloaders with ARAD-1K compatibility.

This module provides the legacy ``TrainDataset`` and ``ValidDataset``
interfaces that the original MSWR release expected.  The previous
monorepo snapshot required an external ``dataloader.py`` to be supplied
manually which made the training scripts fail immediately.  The
implementations below mirror the behaviour of the public ARAD-1K loaders
with a couple of quality-of-life improvements:

* explicit error messages when the dataset structure is incomplete
* graceful fallback between ``.jpg`` and ``.png`` RGB files
* patch extraction that honours the true spatial resolution of each
  scene (instead of assuming every image is 482×512)
* optional logging hooks so the training driver can surface dataset
  statistics

Both datasets return ``torch.float32`` tensors that the training script
moves to the GPU.  No additional normalisation is applied beyond the
per-image min/max scaling used by MST++ style pipelines.
"""

from __future__ import annotations

import bisect
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

__all__ = ["TrainDataset", "ValidDataset", "DatasetConfig"]


@dataclass
class DatasetConfig:
    """Common configuration shared between the datasets."""

    data_root: Path
    logger: logging.Logger
    bgr2rgb: bool

    def rgb_path(self, stem: str) -> Optional[Path]:
        """Return the RGB image path for ``stem`` if it exists."""
        rgb_dir = self.data_root / "Train_RGB"
        jpg_path = rgb_dir / f"{stem}.jpg"
        if jpg_path.exists():
            return jpg_path
        png_path = rgb_dir / f"{stem}.png"
        if png_path.exists():
            return png_path
        self.logger.warning("RGB image missing for %s (checked .jpg/.png)", stem)
        return None

    def hsi_path(self, stem: str) -> Optional[Path]:
        """Return the hyperspectral cube path for ``stem`` if it exists."""
        hsi_dir = self.data_root / "Train_Spec"
        mat_path = hsi_dir / f"{stem}.mat"
        if mat_path.exists():
            return mat_path
        self.logger.warning("HSI cube missing for %s", stem)
        return None


def _load_hsi_cube(path: Path) -> np.ndarray:
    """Load a hyperspectral cube stored under the ``cube`` key."""
    with h5py.File(path, "r") as mat:
        if "cube" not in mat:
            raise KeyError(f"Missing 'cube' dataset in {path}")
        cube = np.array(mat["cube"], dtype=np.float32)
    # Public ARAD files are stored as (bands, height, width)
    if cube.ndim != 3:
        raise ValueError(f"Unexpected cube shape {cube.shape} in {path}")
    return np.transpose(cube, (0, 2, 1))  # -> (bands, height, width)


def _load_rgb_image(path: Path, *, bgr2rgb: bool) -> np.ndarray:
    """Load and normalise an RGB image."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    denom = image.max() - image.min()
    if denom < 1e-6:
        # EDGE CASE FIX: Preserve mean intensity for flat/constant-value images
        # instead of zeroing, which loses all information and breaks reconstruction
        # for calibration targets or uniform regions
        mean_val = image.mean()
        # Normalize mean to [0, 1] range (assuming 0-255 input range)
        normalized_mean = np.clip(mean_val / 255.0, 0.0, 1.0)
        image = np.full_like(image, normalized_mean)
    else:
        image = (image - image.min()) / denom
    image = np.transpose(image, (2, 0, 1))  # -> (channels, height, width)
    return image


def _read_split_file(path: Path) -> List[str]:
    """Read a MST++ style split file and return scene stems."""
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open("r") as handle:
        stems = [line.strip() for line in handle if line.strip()]
    return stems


class TrainDataset(Dataset):
    """Patch-based ARAD-1K training dataset used by MSWR-Net."""

    def __init__(
        self,
        data_root: str,
        crop_size: int = 128,
        *,
        bgr2rgb: bool = True,
        arg: bool = True,
        stride: int = 8,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()
        data_root_path = Path(data_root)
        if not data_root_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {data_root_path}")
        self.config = DatasetConfig(
            data_root=data_root_path,
            logger=logger or logging.getLogger(__name__),
            bgr2rgb=bgr2rgb,
        )
        self.crop_size = int(crop_size)
        self.stride = int(stride)
        self.augment = bool(arg)

        split_file = data_root_path / "split_txt" / "train_list.txt"
        stems = _read_split_file(split_file)
        if not stems:
            raise RuntimeError(f"No entries found in {split_file}")

        self.config.logger.info("Loading %d training scenes from %s", len(stems), data_root_path)

        self.rgb_images: List[np.ndarray] = []
        self.hsi_cubes: List[np.ndarray] = []
        self._patch_offsets: List[int] = []
        total_patches = 0

        for stem in stems:
            rgb_path = self.config.rgb_path(stem)
            hsi_path = self.config.hsi_path(stem)
            if rgb_path is None or hsi_path is None:
                continue
            try:
                rgb = _load_rgb_image(rgb_path, bgr2rgb=self.config.bgr2rgb)
                hsi = _load_hsi_cube(hsi_path)
            except Exception as exc:  # pragma: no cover - defensive
                self.config.logger.warning("Skipping %s due to load failure: %s", stem, exc)
                continue

            h, w = hsi.shape[1:]

            # Handle undersized images by padding to at least crop_size
            if h < self.crop_size or w < self.crop_size:
                pad_h = max(0, self.crop_size - h)
                pad_w = max(0, self.crop_size - w)
                # Pad with reflect mode to maintain continuity
                rgb = np.pad(rgb, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                hsi = np.pad(hsi, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                h, w = hsi.shape[1:]
                self.config.logger.debug(
                    "Padded image to (%d, %d) for crop_size=%d", h, w, self.crop_size
                )

            self.rgb_images.append(rgb)
            self.hsi_cubes.append(hsi)

            patches_h = max(1, (h - self.crop_size) // self.stride + 1)
            patches_w = max(1, (w - self.crop_size) // self.stride + 1)
            total_patches += patches_h * patches_w
            self._patch_offsets.append(total_patches)

        if not self.rgb_images:
            raise RuntimeError("No valid training samples were loaded; check dataset integrity.")

        self.total_patches = total_patches
        self.config.logger.info(
            "Prepared %d patches across %d scenes (crop=%d, stride=%d)",
            self.total_patches,
            len(self.rgb_images),
            self.crop_size,
            self.stride,
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.total_patches

    # -------------------------- augmentation helpers -------------------------
    @staticmethod
    def _apply_augmentations(rgb: np.ndarray, hsi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rot_times = random.randint(0, 3)
        v_flip = random.random() < 0.5
        h_flip = random.random() < 0.5

        if rot_times:
            rgb = np.rot90(rgb, rot_times, axes=(1, 2)).copy()
            hsi = np.rot90(hsi, rot_times, axes=(1, 2)).copy()
        if v_flip:
            rgb = rgb[:, ::-1, :].copy()
            hsi = hsi[:, ::-1, :].copy()
        if h_flip:
            rgb = rgb[:, :, ::-1].copy()
            hsi = hsi[:, :, ::-1].copy()
        return rgb, hsi

    # -------------------------------------------------------------------------
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx = bisect.bisect_right(self._patch_offsets, index)
        prev_total = self._patch_offsets[img_idx - 1] if img_idx > 0 else 0
        patch_local_idx = index - prev_total

        rgb = self.rgb_images[img_idx]
        hsi = self.hsi_cubes[img_idx]
        h, w = hsi.shape[1:]

        patches_w = max(1, (w - self.crop_size) // self.stride + 1)
        row = patch_local_idx // patches_w
        col = patch_local_idx % patches_w

        # Calculate patch position with proper clamping to ensure valid coordinates
        # max(0, ...) ensures non-negative; min(..., h/w - crop_size) ensures we don't exceed bounds
        y = max(0, min(row * self.stride, h - self.crop_size))
        x = max(0, min(col * self.stride, w - self.crop_size))

        rgb_patch = rgb[:, y : y + self.crop_size, x : x + self.crop_size]
        hsi_patch = hsi[:, y : y + self.crop_size, x : x + self.crop_size]

        # Validate patch shape (defensive check)
        if rgb_patch.shape[1:] != (self.crop_size, self.crop_size):
            raise RuntimeError(
                f"Unexpected patch shape {rgb_patch.shape} for crop_size={self.crop_size} "
                f"(image size: {h}x{w}, position: y={y}, x={x})"
            )

        if self.augment:
            rgb_patch, hsi_patch = self._apply_augmentations(rgb_patch, hsi_patch)

        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_patch, dtype=np.float32))
        hsi_tensor = torch.from_numpy(np.ascontiguousarray(hsi_patch, dtype=np.float32))
        return rgb_tensor, hsi_tensor


class ValidDataset(Dataset):
    """Full-image ARAD-1K validation dataset."""

    def __init__(
        self,
        data_root: str,
        *,
        bgr2rgb: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()
        data_root_path = Path(data_root)
        if not data_root_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {data_root_path}")
        self.config = DatasetConfig(
            data_root=data_root_path,
            logger=logger or logging.getLogger(__name__),
            bgr2rgb=bgr2rgb,
        )

        split_file = data_root_path / "split_txt" / "val_list.txt"
        stems = _read_split_file(split_file)
        if not stems:
            raise RuntimeError(f"No entries found in {split_file}")

        self.rgb_images: List[torch.Tensor] = []
        self.hsi_cubes: List[torch.Tensor] = []

        self.config.logger.info("Loading %d validation scenes from %s", len(stems), data_root_path)

        for stem in stems:
            rgb_path = self.config.rgb_path(stem)
            hsi_path = self.config.hsi_path(stem)
            if rgb_path is None or hsi_path is None:
                continue
            try:
                rgb = _load_rgb_image(rgb_path, bgr2rgb=self.config.bgr2rgb)
                hsi = _load_hsi_cube(hsi_path)
            except Exception as exc:  # pragma: no cover - defensive
                self.config.logger.warning("Skipping %s due to load failure: %s", stem, exc)
                continue

            self.rgb_images.append(torch.from_numpy(np.ascontiguousarray(rgb)))
            self.hsi_cubes.append(torch.from_numpy(np.ascontiguousarray(hsi)))

        if not self.rgb_images:
            raise RuntimeError("No validation samples were loaded; check dataset integrity.")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.rgb_images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rgb_images[index], self.hsi_cubes[index]

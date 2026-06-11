# src/hsi_model/utils/data/mst_dataset.py
"""
MST++ Dataset Classes

Implements exact MST++ data loading protocol:
- Loads ALL data into memory at initialization
- Uses h5py exclusively for .mat files (v7.3 format)
- Per-image min-max normalization for RGB
- No normalization for HSI (loads as float32 directly)
- On-the-fly patch extraction during __getitem__

Memory intensive but follows original MST++ implementation exactly.
"""

import os
import logging
import random
from pathlib import Path
import numpy as np
import h5py
import cv2
import torch
from torch.utils.data import Dataset
from typing import Any, Optional, Sequence, Tuple

from ...constants import ARAD1K_NUM_BANDS, DEFAULT_PATCH_SIZE, DEFAULT_STRIDE

logger = logging.getLogger(__name__)

_VALID_SPEC_DIRS = (
    "Valid_Spec",
    "Validation_Spec",
    "Val_Spec",
    "Test_Spec",
    "Train_Spec",
)
_VALID_RGB_DIRS = (
    "Valid_RGB",
    "Validation_RGB",
    "Val_RGB",
    "Test_RGB",
    "Train_RGB",
)
_RGB_SUFFIXES = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")


def _read_split_stems(split_path: Path) -> list[str]:
    with split_path.open("r", encoding="utf-8") as fin:
        return [Path(line.strip()).stem for line in fin if line.strip()]


def _resolve_dataset_file(
    data_root: Path,
    stem: str,
    subdirs: Sequence[str],
    suffixes: Sequence[str],
) -> Optional[Path]:
    for subdir in subdirs:
        for suffix in suffixes:
            candidate = data_root / subdir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _cube_to_chw(cube: np.ndarray, source: Path) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D HSI cube in {source}, got shape {cube.shape}")
    cube = np.float32(cube)
    if cube.shape[0] == ARAD1K_NUM_BANDS:
        return cube
    if cube.shape[-1] == ARAD1K_NUM_BANDS:
        return np.transpose(cube, (2, 0, 1))
    raise ValueError(
        f"Expected a {ARAD1K_NUM_BANDS}-band HSI cube in {source}, got shape {cube.shape}"
    )


def _load_mst_cube(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as mat:
        cube = np.array(mat["cube"])
    cube_chw = _cube_to_chw(cube, path)
    # MST++ ARAD files are commonly stored as C,W,H; this converts them to C,H,W.
    return np.transpose(cube_chw, [0, 2, 1])


def _align_hyper_to_rgb(hyper: np.ndarray, bgr_hw: Tuple[int, int], source: Path) -> np.ndarray:
    hyper_hw = (hyper.shape[1], hyper.shape[2])
    if hyper_hw == bgr_hw:
        return hyper
    if hyper_hw == (bgr_hw[1], bgr_hw[0]):
        logger.warning(
            "HSI spatial axes were swapped after MST transpose for %s; correcting to match RGB %s",
            source,
            bgr_hw,
        )
        return np.transpose(hyper, [0, 2, 1])
    raise ValueError(f"Spatial mismatch for {source}: hyper={hyper_hw} rgb={bgr_hw}")


class MST_TrainDataset(Dataset):
    """
    MST++ Training Dataset - Exact Implementation.
    
    Loads all training data into memory and extracts patches on-the-fly.
    
    Args:
        data_root: Root directory containing Train_Spec and Train_RGB folders
        crop_size: Patch size for training (default: 128)
        arg: Enable data augmentation (rotations, flips)
        bgr2rgb: Convert BGR to RGB
        stride: Stride for patch extraction (default: 8)
    
    Example:
        >>> dataset = MST_TrainDataset(
        ...     data_root="/path/to/ARAD_1K",
        ...     crop_size=128,
        ...     arg=True
        ... )
        >>> rgb_patch, hsi_patch = dataset[0]
    """

    def __init__(
        self,
        data_root: str,
        crop_size: int = DEFAULT_PATCH_SIZE,
        arg: bool = True,
        bgr2rgb: bool = True,
        stride: int = DEFAULT_STRIDE,
        memory_mode: str = "standard",
        **_: object,
    ):
        self.crop_size = crop_size
        self.hypers = []  # All HSI data in memory
        self.bgrs = []  # All RGB data in memory
        self.arg = arg
        self.memory_mode = memory_mode

        self.stride = stride
        self.patch_per_line = 0
        self.patch_per_column = 0
        self.patch_per_img = 0

        # MST++ file paths
        hyper_data_path = f"{data_root}/Train_Spec/"
        bgr_data_path = f"{data_root}/Train_RGB/"

        # Load file lists from MST++ split files
        with open(f"{data_root}/split_txt/train_list.txt", "r") as fin:
            hyper_list = [line.replace("\n", ".mat") for line in fin]
            bgr_list = [line.replace("mat", "jpg") for line in hyper_list]

        hyper_list.sort()
        bgr_list.sort()

        logger.info(f"MST++ Train Dataset - Loading {len(hyper_list)} scenes")

        # Load ALL data into memory (MST++ approach)
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if "mat" not in hyper_path:
                continue

            # Load HSI using h5py (MST++ way)
            try:
                hyper = _load_mst_cube(Path(hyper_path))
            except Exception as exc:
                logger.warning(f"Failed to load {hyper_path}: {exc}")
                continue

            # Load RGB using OpenCV (MST++ way)
            bgr_path = bgr_data_path + bgr_list[i]
            assert (
                hyper_list[i].split(".")[0] == bgr_list[i].split(".")[0]
            ), "Hyper and RGB come from different scenes."

            bgr = cv2.imread(bgr_path)
            if bgr is None:
                logger.warning("Failed to read RGB image: %s", bgr_path)
                continue
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if hyper.ndim != 3 or bgr.ndim != 3:
                logger.warning("Unexpected shapes for %s: hyper=%s rgb=%s", hyper_path, hyper.shape, bgr.shape)
                continue
            bgr_hw = (bgr.shape[0], bgr.shape[1])
            try:
                hyper = _align_hyper_to_rgb(hyper, bgr_hw, Path(hyper_path))
            except ValueError as exc:
                logger.warning(str(exc))
                continue

            # MST++ RGB normalization: per-image min-max
            bgr = np.float32(bgr)
            bgr_min = float(np.min(bgr))
            bgr_max = float(np.max(bgr))
            bgr_range = bgr_max - bgr_min
            if not np.isfinite(bgr_range) or bgr_range < 1e-12:
                logger.warning(
                    "Degenerate RGB dynamic range (min=%.6f, max=%.6f) for %s; using zeros",
                    bgr_min,
                    bgr_max,
                    bgr_path,
                )
                bgr = np.zeros_like(bgr, dtype=np.float32)
            else:
                bgr = (bgr - bgr_min) / bgr_range
            bgr = np.nan_to_num(bgr, nan=0.0, posinf=1.0, neginf=0.0)
            bgr = np.transpose(bgr, [2, 0, 1])  # [H, W, 3] -> [3, H, W]

            # Store in memory
            self.hypers.append(hyper)
            self.bgrs.append(bgr)

            if (i + 1) % 10 == 0 or (i + 1) == len(hyper_list):
                logger.info(f"Loaded {i+1}/{len(hyper_list)} training scenes")

        self.img_num = len(self.hypers)
        if self.img_num > 0:
            h, w = self.bgrs[0].shape[1:]
            if h < crop_size or w < crop_size:
                raise ValueError(
                    f"Loaded MST images are smaller than crop_size={crop_size}: {(h, w)}"
                )
            for sample_idx, (bgr, hyper) in enumerate(zip(self.bgrs, self.hypers)):
                if bgr.shape[1:] != (h, w) or hyper.shape[1:] != (h, w):
                    raise ValueError(
                        "MST_TrainDataset requires consistent spatial sizes; "
                        f"sample {sample_idx} has rgb={bgr.shape[1:]} hsi={hyper.shape[1:]}, "
                        f"expected {(h, w)}"
                    )
            self.patch_per_line = (w - crop_size) // stride + 1
            self.patch_per_column = (h - crop_size) // stride + 1
            self.patch_per_img = self.patch_per_line * self.patch_per_column
        logger.info("Finished loading MST++ training dataset")

    def set_patch_geometry(self, crop_size: int, stride: Optional[int] = None) -> None:
        """Re-index the in-RAM scenes for a new patch size (progressive stages).

        The stored scenes are full resolution; only the crop indexing depends on
        ``crop_size``/``stride``, so progressive-training stage transitions can
        reuse this dataset instead of re-loading ~30 GB of scenes.
        """
        crop_size = int(crop_size)
        stride = int(stride) if stride is not None else self.stride
        if self.img_num == 0:
            raise RuntimeError("Cannot set patch geometry on an empty dataset.")
        h, w = self.bgrs[0].shape[1:]
        if h < crop_size or w < crop_size:
            raise ValueError(
                f"crop_size={crop_size} exceeds loaded image size {(h, w)}"
            )
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.crop_size = crop_size
        self.stride = stride
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_column = (h - crop_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_column

    @staticmethod
    def arguement(
        data: np.ndarray, rot_times: int, v_flip: int, h_flip: int
    ) -> np.ndarray:
        """
        MST++ data augmentation: rotations and flips.
        """
        # Rotate
        if rot_times:
            data = np.rot90(data, rot_times, axes=(1, 2))

        # Vertical flip
        if v_flip:
            data = data[:, ::-1, :]

        # Horizontal flip
        if h_flip:
            data = data[:, :, ::-1]

        return data

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a training patch pair.

        Args:
            idx: Patch index

        Returns:
            Tuple of (rgb_patch, hsi_patch) as contiguous numpy arrays
        """
        stride = self.stride
        crop_size = self.crop_size

        # MST++ patch indexing
        img_idx = idx // self.patch_per_img
        patch_idx = idx % self.patch_per_img
        h_idx = patch_idx // self.patch_per_line
        w_idx = patch_idx % self.patch_per_line

        # Get full images from memory
        bgr = self.bgrs[img_idx]  # [3, H, W]
        hyper = self.hypers[img_idx]  # [31, H, W]

        # Extract patches (MST++ way)
        bgr_patch = bgr[
            :,
            h_idx * stride : h_idx * stride + crop_size,
            w_idx * stride : w_idx * stride + crop_size,
        ]
        hyper_patch = hyper[
            :,
            h_idx * stride : h_idx * stride + crop_size,
            w_idx * stride : w_idx * stride + crop_size,
        ]

        # MST++ augmentation
        if self.arg:
            rot_times = random.randint(0, 3)
            v_flip = random.randint(0, 1)
            h_flip = random.randint(0, 1)
            bgr_patch = self.arguement(bgr_patch, rot_times, v_flip, h_flip)
            hyper_patch = self.arguement(hyper_patch, rot_times, v_flip, h_flip)

        # MST++ returns contiguous arrays
        return np.ascontiguousarray(bgr_patch), np.ascontiguousarray(hyper_patch)

    def __len__(self) -> int:
        return self.patch_per_img * self.img_num


class MST_ValidDataset(Dataset):
    """
    MST++ Validation Dataset - Exact Implementation.
    
    Loads full validation images (no patches).
    
    Args:
        data_root: Root directory containing Train_Spec and Train_RGB folders
        bgr2rgb: Convert BGR to RGB
    
    Example:
        >>> dataset = MST_ValidDataset(data_root="/path/to/ARAD_1K")
        >>> rgb_full, hsi_full = dataset[0]
    """

    def __init__(
        self,
        data_root: str,
        bgr2rgb: bool = True,
        memory_mode: str = "standard",
        **_: Any,
    ):
        data_root_path = Path(data_root)
        self.hypers = []
        self.bgrs = []
        self.memory_mode = memory_mode

        # Load validation split
        split_path = data_root_path / "split_txt" / "valid_list.txt"
        if not split_path.exists():
            split_path = data_root_path / "split_txt" / "val_list.txt"
        scene_stems = sorted(_read_split_stems(split_path))

        logger.info(f"MST++ Valid Dataset - Loading {len(scene_stems)} scenes")

        # Load all validation data into memory
        for i, scene_stem in enumerate(scene_stems):
            hyper_path = _resolve_dataset_file(
                data_root_path,
                scene_stem,
                _VALID_SPEC_DIRS,
                (".mat",),
            )
            if hyper_path is None:
                logger.warning(
                    "Failed to load validation %s: .mat file not found in %s",
                    scene_stem,
                    ", ".join(_VALID_SPEC_DIRS),
                )
                continue

            # Load HSI using h5py (MST++ way)
            try:
                hyper = _load_mst_cube(hyper_path)
            except Exception as exc:
                logger.warning(f"Failed to load validation {hyper_path}: {exc}")
                continue

            # Load RGB using OpenCV (MST++ way)
            bgr_path = _resolve_dataset_file(
                data_root_path,
                scene_stem,
                _VALID_RGB_DIRS,
                _RGB_SUFFIXES,
            )
            if bgr_path is None:
                logger.warning(
                    "Failed to load validation %s: RGB file not found in %s",
                    scene_stem,
                    ", ".join(_VALID_RGB_DIRS),
                )
                continue

            bgr = cv2.imread(str(bgr_path))
            if bgr is None:
                logger.warning("Failed to read validation RGB image: %s", bgr_path)
                continue
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                hyper = _align_hyper_to_rgb(hyper, (bgr.shape[0], bgr.shape[1]), hyper_path)
            except ValueError as exc:
                logger.warning(str(exc))
                continue

            # MST++ RGB normalization
            bgr = np.float32(bgr)
            bgr_min = float(np.min(bgr))
            bgr_max = float(np.max(bgr))
            bgr_range = bgr_max - bgr_min
            if not np.isfinite(bgr_range) or bgr_range < 1e-12:
                logger.warning(
                    "Degenerate RGB dynamic range (min=%.6f, max=%.6f) for %s; using zeros",
                    bgr_min,
                    bgr_max,
                    bgr_path,
                )
                bgr = np.zeros_like(bgr, dtype=np.float32)
            else:
                bgr = (bgr - bgr_min) / bgr_range
            bgr = np.nan_to_num(bgr, nan=0.0, posinf=1.0, neginf=0.0)
            bgr = np.transpose(bgr, [2, 0, 1])

            self.hypers.append(hyper)
            self.bgrs.append(bgr)

            logger.info(f"Loaded validation scene {i+1}/{len(scene_stems)}")

        logger.info("MST++ Valid Dataset initialized with %d images", len(self.hypers))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a full validation image.
        
        Args:
            idx: Image index
            
        Returns:
            Tuple of (rgb_full, hsi_full) as contiguous numpy arrays
        """
        hyper = self.hypers[idx]  # [31, H, W]
        bgr = self.bgrs[idx]  # [3, H, W]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self) -> int:
        return len(self.hypers)

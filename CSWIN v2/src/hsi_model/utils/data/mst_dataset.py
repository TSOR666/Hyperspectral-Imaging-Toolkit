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
import numpy as np
import h5py
import cv2
import torch
from torch.utils.data import Dataset
from typing import Tuple

from ...constants import (
    ARAD1K_FULL_HEIGHT,
    ARAD1K_FULL_WIDTH,
    DEFAULT_PATCH_SIZE,
    DEFAULT_STRIDE,
)

logger = logging.getLogger(__name__)


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

        # ARAD-1K image dimensions (MST++ hardcoded)
        h, w = ARAD1K_FULL_HEIGHT, ARAD1K_FULL_WIDTH
        self.stride = stride
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_column = (h - crop_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_column

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
                with h5py.File(hyper_path, "r") as mat:
                    # MST++ loads 'cube' variable as float32
                    hyper = np.float32(np.array(mat["cube"]))
                # MST++ exact transpose: [31, H, W] -> [31, W, H] -> [H, W, 31]
                hyper = np.transpose(hyper, [0, 2, 1])
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
            hyper_hw = (hyper.shape[1], hyper.shape[2])
            bgr_hw = (bgr.shape[0], bgr.shape[1])
            if hyper_hw != bgr_hw and hyper_hw != (bgr_hw[1], bgr_hw[0]):
                logger.warning(
                    "Spatial mismatch for %s: hyper=%s rgb=%s",
                    hyper_path,
                    hyper_hw,
                    bgr_hw,
                )
                continue

            # MST++ RGB normalization: per-image min-max
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])  # [H, W, 3] -> [3, H, W]

            # Store in memory
            self.hypers.append(hyper)
            self.bgrs.append(bgr)

            if (i + 1) % 10 == 0 or (i + 1) == len(hyper_list):
                logger.info(f"Loaded {i+1}/{len(hyper_list)} training scenes")

        self.img_num = len(self.hypers)
        logger.info("Finished loading MST++ training dataset")

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
        self.hypers = []
        self.bgrs = []
        self.memory_mode = memory_mode

        # MST++ file paths
        hyper_data_path = f"{data_root}/Train_Spec/"
        bgr_data_path = f"{data_root}/Train_RGB/"

        # Load validation split
        with open(f"{data_root}/split_txt/valid_list.txt", "r") as fin:
            hyper_list = [line.replace("\n", ".mat") for line in fin]
            bgr_list = [line.replace("mat", "jpg") for line in hyper_list]

        hyper_list.sort()
        bgr_list.sort()

        logger.info(f"MST++ Valid Dataset - Loading {len(hyper_list)} scenes")

        # Load all validation data into memory
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if "mat" not in hyper_path:
                continue

            # Load HSI using h5py (MST++ way)
            try:
                with h5py.File(hyper_path, "r") as mat:
                    hyper = np.float32(np.array(mat["cube"]))
                hyper = np.transpose(hyper, [0, 2, 1])
            except Exception as exc:
                logger.warning(f"Failed to load validation {hyper_path}: {exc}")
                continue

            # Load RGB using OpenCV (MST++ way)
            bgr_path = bgr_data_path + bgr_list[i]
            assert (
                hyper_list[i].split(".")[0] == bgr_list[i].split(".")[0]
            ), "Hyper and RGB come from different scenes."

            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # MST++ RGB normalization
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])

            self.hypers.append(hyper)
            self.bgrs.append(bgr)

            logger.info(f"Loaded validation scene {i+1}/{len(hyper_list)}")

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

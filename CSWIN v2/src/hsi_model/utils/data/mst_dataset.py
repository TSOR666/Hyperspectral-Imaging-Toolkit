# src/hsi_model/utils/data/mst_dataset.py
"""
MST++ Dataset Classes

Implements exact MST++ data loading protocol:
- Supports resident float32/float16 data or file-backed lazy reads
- Uses h5py exclusively for .mat files (v7.3 format)
- Per-image min-max normalization for RGB
- No normalization for HSI (loads as float32 directly)
- On-the-fly patch extraction during __getitem__

The standard mode follows the original resident-memory MST++ behavior.
"""

import os
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _MSTSceneRef:
    hyper_path: Path
    rgb_path: Path
    height: int
    width: int


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


def _cube_base_hw(shape: Tuple[int, ...], source: Path) -> Tuple[int, int]:
    """Return the H,W produced by ``_load_mst_cube`` before RGB alignment."""
    if len(shape) != 3:
        raise ValueError(f"Expected 3D HSI cube in {source}, got shape {shape}")
    if shape[0] == ARAD1K_NUM_BANDS:
        return int(shape[2]), int(shape[1])
    if shape[-1] == ARAD1K_NUM_BANDS:
        return int(shape[1]), int(shape[0])
    raise ValueError(
        f"Expected a {ARAD1K_NUM_BANDS}-band HSI cube in {source}, got shape {shape}"
    )


def _validate_cube_hw(
    shape: Tuple[int, ...],
    rgb_hw: Tuple[int, int],
    source: Path,
) -> None:
    base_hw = _cube_base_hw(shape, source)
    if base_hw != rgb_hw and base_hw != (rgb_hw[1], rgb_hw[0]):
        raise ValueError(f"Spatial mismatch for {source}: hyper={base_hw} rgb={rgb_hw}")


def _load_mst_cube_patch(
    cube: Any,
    source: Path,
    rgb_hw: Tuple[int, int],
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Read one aligned C,H,W patch without materializing the full HSI cube."""
    shape = tuple(int(value) for value in cube.shape)
    base_hw = _cube_base_hw(shape, source)
    aligned_without_swap = base_hw == rgb_hw
    if not aligned_without_swap and base_hw != (rgb_hw[1], rgb_hw[0]):
        raise ValueError(f"Spatial mismatch for {source}: hyper={base_hw} rgb={rgb_hw}")

    if shape[0] == ARAD1K_NUM_BANDS:
        if aligned_without_swap:
            patch = np.asarray(
                cube[:, left : left + width, top : top + height],
                dtype=np.float32,
            )
            patch = np.transpose(patch, (0, 2, 1))
        else:
            patch = np.asarray(
                cube[:, top : top + height, left : left + width],
                dtype=np.float32,
            )
    else:
        if aligned_without_swap:
            patch = np.asarray(
                cube[left : left + width, top : top + height, :],
                dtype=np.float32,
            )
            patch = np.transpose(patch, (2, 1, 0))
        else:
            patch = np.asarray(
                cube[top : top + height, left : left + width, :],
                dtype=np.float32,
            )
            patch = np.transpose(patch, (2, 0, 1))

    expected = (ARAD1K_NUM_BANDS, height, width)
    if patch.shape != expected:
        raise ValueError(
            f"HSI patch read from {source} has shape {patch.shape}, expected {expected}"
        )
    return patch


def _load_normalized_rgb(path: Path, bgr2rgb: bool) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    if bgr2rgb:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    bgr = np.float32(bgr)
    bgr_min = float(np.min(bgr))
    bgr_max = float(np.max(bgr))
    bgr_range = bgr_max - bgr_min
    if not np.isfinite(bgr_range) or bgr_range < 1e-12:
        logger.warning(
            "Degenerate RGB dynamic range (min=%.6f, max=%.6f) for %s; using zeros",
            bgr_min,
            bgr_max,
            path,
        )
        bgr = np.zeros_like(bgr, dtype=np.float32)
    else:
        bgr = (bgr - bgr_min) / bgr_range
    bgr = np.nan_to_num(bgr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.transpose(bgr, (2, 0, 1))


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
    
    Supports resident standard/float16 storage and file-backed lazy patch reads.
    
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
        lazy_cache_size: int = 3,
        **_: object,
    ):
        self.crop_size = crop_size
        self.hypers = []  # All HSI data in memory
        self.bgrs = []  # All RGB data in memory
        self.arg = arg
        self.memory_mode = str(memory_mode).strip().lower()
        if self.memory_mode not in {"standard", "float16", "lazy"}:
            raise ValueError(
                "memory_mode must be 'standard', 'float16', or 'lazy', "
                f"got {memory_mode!r}"
            )
        self.lazy_cache_size = max(1, int(lazy_cache_size))
        self._bgr2rgb = bool(bgr2rgb)
        self._scene_refs: list[_MSTSceneRef] = []
        self._scene_hw: list[Tuple[int, int]] = []
        self._rgb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._h5_files: OrderedDict[int, h5py.File] = OrderedDict()

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

        if self.memory_mode == "lazy":
            self._initialize_lazy(
                hyper_data_path,
                bgr_data_path,
                hyper_list,
                bgr_list,
                crop_size,
                stride,
            )
            logger.info(
                "Finished indexing lazy MST++ training dataset (%d scenes, cache=%d)",
                self.img_num,
                self.lazy_cache_size,
            )
            return

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

            try:
                bgr = _load_normalized_rgb(Path(bgr_path), self._bgr2rgb)
            except Exception as exc:
                logger.warning("%s", exc)
                continue

            if hyper.ndim != 3 or bgr.ndim != 3:
                logger.warning("Unexpected shapes for %s: hyper=%s rgb=%s", hyper_path, hyper.shape, bgr.shape)
                continue
            bgr_hw = (bgr.shape[1], bgr.shape[2])
            try:
                hyper = _align_hyper_to_rgb(hyper, bgr_hw, Path(hyper_path))
            except ValueError as exc:
                logger.warning(str(exc))
                continue

            if self.memory_mode == "float16":
                hyper = hyper.astype(np.float16)
                bgr = bgr.astype(np.float16)

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
            self._scene_hw = [(h, w)] * self.img_num
            self._set_patch_geometry(crop_size, stride)
        logger.info("Finished loading MST++ training dataset")

    def _initialize_lazy(
        self,
        hyper_data_path: str,
        bgr_data_path: str,
        hyper_list: Sequence[str],
        bgr_list: Sequence[str],
        crop_size: int,
        stride: int,
    ) -> None:
        expected_hw: Optional[Tuple[int, int]] = None
        for hyper_name, bgr_name in zip(hyper_list, bgr_list):
            hyper_path = Path(hyper_data_path) / hyper_name
            bgr_path = Path(bgr_data_path) / bgr_name
            if hyper_path.stem != bgr_path.stem:
                raise ValueError(
                    f"Hyper and RGB come from different scenes: {hyper_path}, {bgr_path}"
                )
            try:
                rgb = _load_normalized_rgb(bgr_path, self._bgr2rgb)
                rgb_hw = (int(rgb.shape[1]), int(rgb.shape[2]))
                with h5py.File(hyper_path, "r") as mat:
                    if "cube" not in mat:
                        raise KeyError(f"Missing 'cube' dataset in {hyper_path}")
                    _validate_cube_hw(tuple(mat["cube"].shape), rgb_hw, hyper_path)
            except Exception as exc:
                logger.warning("Failed to index lazy scene %s: %s", hyper_path, exc)
                continue

            if expected_hw is None:
                expected_hw = rgb_hw
            elif rgb_hw != expected_hw:
                raise ValueError(
                    "MST_TrainDataset requires consistent spatial sizes; "
                    f"{bgr_path} has {rgb_hw}, expected {expected_hw}"
                )
            self._scene_refs.append(
                _MSTSceneRef(hyper_path, bgr_path, rgb_hw[0], rgb_hw[1])
            )
            self._scene_hw.append(rgb_hw)

        self.img_num = len(self._scene_refs)
        if self.img_num > 0:
            self._set_patch_geometry(crop_size, stride)

    def _set_patch_geometry(self, crop_size: int, stride: int) -> None:
        h, w = self._scene_hw[0]
        if h < crop_size or w < crop_size:
            raise ValueError(
                f"Loaded MST images are smaller than crop_size={crop_size}: {(h, w)}"
            )
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.crop_size = int(crop_size)
        self.stride = int(stride)
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_column = (h - crop_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_column

    def _get_lazy_rgb(self, scene_idx: int) -> np.ndarray:
        cached = self._rgb_cache.get(scene_idx)
        if cached is not None:
            self._rgb_cache.move_to_end(scene_idx)
            return cached
        scene = self._scene_refs[scene_idx]
        rgb = _load_normalized_rgb(scene.rgb_path, self._bgr2rgb)
        self._rgb_cache[scene_idx] = rgb
        while len(self._rgb_cache) > self.lazy_cache_size:
            self._rgb_cache.popitem(last=False)
        return rgb

    def _get_lazy_cube(self, scene_idx: int) -> Any:
        cached = self._h5_files.get(scene_idx)
        if cached is not None:
            self._h5_files.move_to_end(scene_idx)
            return cached["cube"]
        scene = self._scene_refs[scene_idx]
        handle = h5py.File(scene.hyper_path, "r")
        self._h5_files[scene_idx] = handle
        while len(self._h5_files) > self.lazy_cache_size:
            _, evicted = self._h5_files.popitem(last=False)
            evicted.close()
        return handle["cube"]

    def close(self) -> None:
        for handle in getattr(self, "_h5_files", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        if hasattr(self, "_h5_files"):
            self._h5_files.clear()
        if hasattr(self, "_rgb_cache"):
            self._rgb_cache.clear()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rgb_cache"] = OrderedDict()
        state["_h5_files"] = OrderedDict()
        return state

    def __del__(self) -> None:
        self.close()

    def set_patch_geometry(self, crop_size: int, stride: Optional[int] = None) -> None:
        """Re-index resident or lazy scenes for a new progressive geometry."""
        crop_size = int(crop_size)
        stride = int(stride) if stride is not None else self.stride
        if self.img_num == 0:
            raise RuntimeError("Cannot set patch geometry on an empty dataset.")
        if not hasattr(self, "_scene_hw") or not self._scene_hw:
            self._scene_hw = [tuple(self.bgrs[0].shape[1:])] * self.img_num
        h, w = self._scene_hw[0]
        if h < crop_size or w < crop_size:
            raise ValueError(
                f"crop_size={crop_size} exceeds loaded image size {(h, w)}"
            )
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self._set_patch_geometry(crop_size, stride)

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

        top = h_idx * stride
        left = w_idx * stride
        if self.memory_mode == "lazy":
            bgr = self._get_lazy_rgb(img_idx)
            scene = self._scene_refs[img_idx]
            bgr_patch = bgr[:, top : top + crop_size, left : left + crop_size]
            hyper_patch = _load_mst_cube_patch(
                self._get_lazy_cube(img_idx),
                scene.hyper_path,
                (scene.height, scene.width),
                top,
                left,
                crop_size,
                crop_size,
            )
        else:
            bgr = self.bgrs[img_idx]
            hyper = self.hypers[img_idx]
            bgr_patch = bgr[:, top : top + crop_size, left : left + crop_size]
            hyper_patch = hyper[:, top : top + crop_size, left : left + crop_size]

        # MST++ augmentation
        if self.arg:
            rot_times = random.randint(0, 3)
            v_flip = random.randint(0, 1)
            h_flip = random.randint(0, 1)
            bgr_patch = self.arguement(bgr_patch, rot_times, v_flip, h_flip)
            hyper_patch = self.arguement(hyper_patch, rot_times, v_flip, h_flip)

        if self.memory_mode == "float16":
            bgr_patch = bgr_patch.astype(np.float32)
            hyper_patch = hyper_patch.astype(np.float32)

        # MST++ returns contiguous arrays
        return np.ascontiguousarray(bgr_patch), np.ascontiguousarray(hyper_patch)

    def __len__(self) -> int:
        return self.patch_per_img * self.img_num


class MST_ValidDataset(Dataset):
    """
    MST++ Validation Dataset - Exact Implementation.
    
    Loads full validation images, either resident or on demand in lazy mode.
    
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
        lazy_cache_size: int = 1,
        **_: Any,
    ):
        data_root_path = Path(data_root)
        self.hypers = []
        self.bgrs = []
        self.memory_mode = str(memory_mode).strip().lower()
        if self.memory_mode not in {"standard", "float16", "lazy"}:
            raise ValueError(
                "memory_mode must be 'standard', 'float16', or 'lazy', "
                f"got {memory_mode!r}"
            )
        self.lazy_cache_size = max(1, int(lazy_cache_size))
        self._bgr2rgb = bool(bgr2rgb)
        self._scene_refs: list[_MSTSceneRef] = []
        self._rgb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._h5_files: OrderedDict[int, h5py.File] = OrderedDict()

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

            try:
                bgr = _load_normalized_rgb(bgr_path, self._bgr2rgb)
                rgb_hw = (int(bgr.shape[1]), int(bgr.shape[2]))
                if self.memory_mode == "lazy":
                    with h5py.File(hyper_path, "r") as mat:
                        if "cube" not in mat:
                            raise KeyError(f"Missing 'cube' dataset in {hyper_path}")
                        _validate_cube_hw(tuple(mat["cube"].shape), rgb_hw, hyper_path)
                    self._scene_refs.append(
                        _MSTSceneRef(
                            hyper_path,
                            bgr_path,
                            rgb_hw[0],
                            rgb_hw[1],
                        )
                    )
                    logger.info(
                        "Indexed validation scene %d/%d",
                        i + 1,
                        len(scene_stems),
                    )
                    continue

                hyper = _load_mst_cube(hyper_path)
                hyper = _align_hyper_to_rgb(hyper, rgb_hw, hyper_path)
            except Exception as exc:
                logger.warning("Failed to load validation %s: %s", hyper_path, exc)
                continue

            if self.memory_mode == "float16":
                hyper = hyper.astype(np.float16)
                bgr = bgr.astype(np.float16)

            self.hypers.append(hyper)
            self.bgrs.append(bgr)

            logger.info(f"Loaded validation scene {i+1}/{len(scene_stems)}")

        logger.info("MST++ Valid Dataset initialized with %d images", len(self))

    def _get_lazy_rgb(self, scene_idx: int) -> np.ndarray:
        cached = self._rgb_cache.get(scene_idx)
        if cached is not None:
            self._rgb_cache.move_to_end(scene_idx)
            return cached
        scene = self._scene_refs[scene_idx]
        rgb = _load_normalized_rgb(scene.rgb_path, self._bgr2rgb)
        self._rgb_cache[scene_idx] = rgb
        while len(self._rgb_cache) > self.lazy_cache_size:
            self._rgb_cache.popitem(last=False)
        return rgb

    def _get_lazy_cube(self, scene_idx: int) -> Any:
        cached = self._h5_files.get(scene_idx)
        if cached is not None:
            self._h5_files.move_to_end(scene_idx)
            return cached["cube"]
        scene = self._scene_refs[scene_idx]
        handle = h5py.File(scene.hyper_path, "r")
        self._h5_files[scene_idx] = handle
        while len(self._h5_files) > self.lazy_cache_size:
            _, evicted = self._h5_files.popitem(last=False)
            evicted.close()
        return handle["cube"]

    def close(self) -> None:
        for handle in getattr(self, "_h5_files", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        if hasattr(self, "_h5_files"):
            self._h5_files.clear()
        if hasattr(self, "_rgb_cache"):
            self._rgb_cache.clear()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rgb_cache"] = OrderedDict()
        state["_h5_files"] = OrderedDict()
        return state

    def __del__(self) -> None:
        self.close()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a full validation image.
        
        Args:
            idx: Image index
            
        Returns:
            Tuple of (rgb_full, hsi_full) as contiguous numpy arrays
        """
        if self.memory_mode == "lazy":
            scene = self._scene_refs[idx]
            bgr = self._get_lazy_rgb(idx)
            raw = np.asarray(self._get_lazy_cube(idx), dtype=np.float32)
            hyper = np.transpose(_cube_to_chw(raw, scene.hyper_path), (0, 2, 1))
            hyper = _align_hyper_to_rgb(
                hyper,
                (scene.height, scene.width),
                scene.hyper_path,
            )
        else:
            hyper = self.hypers[idx]
            bgr = self.bgrs[idx]
        if self.memory_mode == "float16":
            bgr = bgr.astype(np.float32)
            hyper = hyper.astype(np.float32)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self) -> int:
        if self.memory_mode == "lazy":
            return len(self._scene_refs)
        return len(self.hypers)

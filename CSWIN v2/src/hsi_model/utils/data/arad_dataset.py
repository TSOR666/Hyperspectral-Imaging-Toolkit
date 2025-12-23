# src/hsi_model/utils/data/arad_dataset.py
"""
ARAD-1K validation dataset with optional caching.
"""

import logging
import os
import sys
import threading
import warnings
import weakref
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Set, Callable, Union

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ...constants import ARAD1K_NUM_BANDS

logger = logging.getLogger(__name__)

ImageTransform = Callable[[Image.Image], Union[Image.Image, np.ndarray]]

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logger.warning(
        "h5py not available. MATLAB v7.3 files will not be readable."
    )


class DatasetCache:
    """
    Thread-safe LRU cache for dataset items.
    """

    def __init__(self, max_size_gb: float = 4.0):
        self.cache: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()
        self.max_size_bytes = max_size_gb * 1024**3
        self.current_size_bytes = 0
        self.access_count: Dict[str, int] = {}

    def get(self, key: str) -> Optional[np.ndarray]:
        with self.lock:
            if key in self.cache:
                self.access_count[key] += 1
                return self.cache[key].copy()
        return None

    def put(self, key: str, value: np.ndarray) -> bool:
        item_size = value.nbytes * 1.2  # 20% overhead for Python object

        with self.lock:
            while (
                self.current_size_bytes + item_size > self.max_size_bytes
                and len(self.cache) > 0
            ):
                lru_key = min(self.access_count, key=self.access_count.get)
                evicted = self.cache.pop(lru_key)
                self.current_size_bytes -= evicted.nbytes * 1.2
                del self.access_count[lru_key]
                logger.debug("Evicted %s from cache", lru_key)

            if self.current_size_bytes + item_size <= self.max_size_bytes:
                self.cache[key] = value.copy()
                self.access_count[key] = 1
                self.current_size_bytes += item_size
                return True

            return False

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.access_count.clear()
            self.current_size_bytes = 0


class ARAD1KDataset(Dataset):
    """
    ARAD-1K validation/test dataset with optional caching.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[ImageTransform] = None,
        resize_hsi: bool = False,
        cache_data: bool = False,
        raise_on_size_mismatch: bool = False,
        max_samples: Optional[int] = None,
        start_idx: int = 0,
        validate_data: bool = True,
        cache_size_gb: float = 4.0,
        allow_legacy_mat: bool = True,
        allowed_extensions: Optional[Set[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.resize_hsi = resize_hsi
        self.raise_on_size_mismatch = raise_on_size_mismatch
        self.max_samples = max_samples
        self.start_idx = start_idx
        self.validate_data = validate_data
        self.allow_legacy_mat = allow_legacy_mat
        self.allowed_extensions = allowed_extensions or {".mat"}
        self.cache = DatasetCache(cache_size_gb) if cache_data else None

        self.rgb_dir = self.data_dir / "ValidationRGB"
        self.hsi_dir = self.data_dir / "ValidationHSI"

        if not self.rgb_dir.exists() or not self.hsi_dir.exists():
            raise FileNotFoundError(
                f"Expected ARAD-1K directories at {self.data_dir}"
            )

        self.file_pairs = self._discover_file_pairs()

        if self.max_samples is not None:
            end_idx = min(self.start_idx + self.max_samples, len(self.file_pairs))
            self.file_pairs = self.file_pairs[self.start_idx:end_idx]

        logger.info("ARAD1KDataset initialised with %s samples", len(self.file_pairs))

    # --------------------------------------------------------------------- #
    # File discovery
    # --------------------------------------------------------------------- #

    def _discover_file_pairs(self) -> List[Tuple[str, str]]:
        rgb_files = sorted(
            f for f in os.listdir(self.rgb_dir) if f.lower().endswith(".png")
        )
        hsi_files = sorted(
            f for f in os.listdir(self.hsi_dir) if f.lower().endswith(tuple(self.allowed_extensions))
        )

        pairs: List[Tuple[str, str]] = []
        missing: List[str] = []

        for rgb_name in rgb_files:
            stem = Path(rgb_name).stem
            hsi_match = f"{stem}.mat"
            if hsi_match in hsi_files:
                pairs.append((rgb_name, hsi_match))
            else:
                missing.append(stem)

        if missing:
            warnings.warn(
                f"Missing HSI files for {len(missing)} RGB images: {missing[:5]}"
            )

        if not pairs:
            raise FileNotFoundError(
                f"No RGB/HSI pairs found in {self.rgb_dir} and {self.hsi_dir}"
            )

        return pairs

    # --------------------------------------------------------------------- #
    # Data loading helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _is_path_safe(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    def _load_hsi(self, file_path: Path) -> np.ndarray:
        if self.cache:
            cached = self.cache.get(str(file_path))
            if cached is not None:
                return cached

        if file_path.suffix == ".mat":
            data = self._load_mat_file(file_path)
        else:
            raise ValueError(f"Unsupported HSI format: {file_path.suffix}")

        if self.validate_data:
            self._validate_hsi_data(data, file_path)

        if self.cache:
            self.cache.put(str(file_path), data)

        return data

    def _load_mat_file(self, file_path: Path) -> np.ndarray:
        if file_path.suffix != ".mat":
            raise ValueError(f"Unsupported file suffix: {file_path.suffix}")

        if file_path.stat().st_size == 0:
            raise ValueError(f"Empty file: {file_path}")

        if HAS_H5PY:
            try:
                with h5py.File(file_path, "r") as mat:
                    if "cube" in mat:
                        data = np.array(mat["cube"])
                    else:
                        keys = [k for k in mat.keys() if not k.startswith("#")]
                        if not keys:
                            raise ValueError(f"No datasets found in {file_path}")
                        data = np.array(mat[keys[0]])
            except OSError as exc:
                if not self.allow_legacy_mat:
                    raise
                logger.debug(
                    "Failed to read %s with h5py (%s). Falling back to scipy.io.loadmat.",
                    file_path,
                    exc,
                )
                data = self._load_legacy_mat(file_path)
        else:
            data = self._load_legacy_mat(file_path)

        data = self._ensure_last_dim_is_band(data)
        return data.astype(np.float32, copy=False)

    @staticmethod
    def _load_legacy_mat(file_path: Path) -> np.ndarray:
        mat = sio.loadmat(file_path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No data keys found in {file_path}")
        return mat[keys[0]]

    @staticmethod
    def _ensure_last_dim_is_band(data: np.ndarray) -> np.ndarray:
        if data.ndim != 3:
            raise ValueError(f"Expected 3D data, got shape {data.shape}")

        if data.shape[0] == ARAD1K_NUM_BANDS:
            data = np.transpose(data, (1, 2, 0))
        elif data.shape[2] != ARAD1K_NUM_BANDS:
            raise ValueError(
                f"Expected one dimension to be {ARAD1K_NUM_BANDS}, got {data.shape}"
            )
        return data

    def _validate_hsi_data(self, hsi_data: np.ndarray, file_path: Path) -> None:
        if np.any(np.isnan(hsi_data)):
            raise ValueError(f"NaN values in {file_path}")
        if np.any(np.isinf(hsi_data)):
            raise ValueError(f"Inf values in {file_path}")
        if hsi_data.ndim != 3:
            raise ValueError(f"Expected 3D data, got {hsi_data.ndim}D in {file_path}")
        if hsi_data.shape[2] != ARAD1K_NUM_BANDS:
            raise ValueError(
                f"Expected {ARAD1K_NUM_BANDS} channels, got {hsi_data.shape[2]} in {file_path}"
            )

    # --------------------------------------------------------------------- #
    # Dataset API
    # --------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_file, hsi_file = self.file_pairs[idx]

        try:
            rgb_path = self.rgb_dir / rgb_file
            if not self._is_path_safe(rgb_path, self.rgb_dir):
                raise ValueError(
                    f"RGB path outside data directory: {rgb_path}"
                )

            rgb_image = Image.open(rgb_path).convert("RGB")
            if self.transform:
                rgb_image = self.transform(rgb_image)

            rgb_array = np.array(rgb_image).astype(np.float32) / 255.0
            rgb_tensor = torch.from_numpy(rgb_array.transpose(2, 0, 1))

            hsi_path = self.hsi_dir / hsi_file
            hsi_data = self._load_hsi(hsi_path).astype(np.float32)

            if hsi_data.max() > 1.5 or hsi_data.min() < -0.1:
                logger.warning(
                    "HSI has unexpected range [%.3f, %.3f] in %s",
                    hsi_data.min(),
                    hsi_data.max(),
                    hsi_file,
                )
                hsi_data = np.clip(hsi_data, 0, 1)

            hsi_tensor = torch.from_numpy(hsi_data.transpose(2, 0, 1))

            if rgb_tensor.shape[1:] != hsi_tensor.shape[1:]:
                if self.raise_on_size_mismatch:
                    raise ValueError(
                        f"Spatial mismatch: RGB {rgb_tensor.shape[1:]} vs "
                        f"HSI {hsi_tensor.shape[1:]}"
                    )

                if self.resize_hsi:
                    hsi_tensor = torch.nn.functional.interpolate(
                        hsi_tensor.unsqueeze(0),
                        size=rgb_tensor.shape[1:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                else:
                    rgb_tensor = torch.nn.functional.interpolate(
                        rgb_tensor.unsqueeze(0),
                        size=hsi_tensor.shape[1:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

            return rgb_tensor, hsi_tensor

        except Exception as exc:
            logger.error("Error loading sample %s (%s): %s", idx, rgb_file, exc)
            raise


def create_arad1k_dataloader(
    data_dir: str,
    batch_size: int = 1,
    num_workers: Optional[int] = None,
    shuffle: bool = False,
    **dataset_kwargs: object,
) -> DataLoader:
    """
    Create DataLoader for ARAD-1K validation dataset.
    """
    if num_workers is None:
        if os.name == "nt":
            num_workers = 0
        else:
            num_workers = min(4, os.cpu_count() or 1)

    dataset = ARAD1KDataset(data_dir=data_dir, **dataset_kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return dataloader

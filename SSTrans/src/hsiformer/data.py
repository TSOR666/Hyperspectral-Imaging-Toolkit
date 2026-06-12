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
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.rgb_root = self.root / "Train_RGB"
        self.spectral_root = self.root / "Train_spectral"
        if not self.rgb_root.is_dir() or not self.spectral_root.is_dir():
            raise FileNotFoundError(
                "Expected ARAD directories 'Train_RGB' and 'Train_spectral' "
                f"under {self.root}."
            )

        self.split = split
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

        self.image_size = image_size or self._read_image_size(self.scene_ids[0])
        self._crop_positions = self._build_crop_positions()

    def _rgb_path(self, scene_id: str) -> Path:
        return self.rgb_root / f"{scene_id}.jpg"

    def _spectral_path(self, scene_id: str) -> Path:
        return self.spectral_root / f"{scene_id}.mat"

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
        if (
            self.crop_size is not None
            and not self.random_crop
            and (height, width) != self.image_size
        ):
            raise ValueError(
                f"Scene {scene_id} has size {(height, width)}, but the grid "
                f"was built for {self.image_size}."
            )
        label = self._load_cube(scene_id, height, width)

        augment = self.augment
        if self.crop_size is not None:
            crop_fits = (
                self.crop_size[0] <= height and self.crop_size[1] <= width
            )
            if self.random_crop and not crop_fits:
                # The published 512 stage uses complete 482x512 ARAD frames.
                augment = False
            else:
                if crop_position is None:
                    crop_position = _random_crop_position(
                        (height, width),
                        self.crop_size,
                    )
                rgb_uint8 = _crop(rgb_uint8, crop_position, self.crop_size)
                ycrcb_uint8 = _crop(ycrcb_uint8, crop_position, self.crop_size)
                label = _crop(label, crop_position, self.crop_size)

        cond = _normalize_rgb(rgb_uint8, self.rgb_normalization)
        sample: dict[str, torch.Tensor | str] = {
            "cond": cond,
            "label": label,
            "scene_id": scene_id,
        }
        if self.include_ycrcb:
            ycrcb = _normalize_rgb(ycrcb_uint8, self.rgb_normalization)
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

    def _load_rgb(self, scene_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        path = self._rgb_path(scene_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        return _load_rgb_tensors(path)

    def _load_cube(
        self,
        scene_id: str,
        height: int,
        width: int,
    ) -> torch.Tensor:
        path = self._spectral_path(scene_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            if self.cube_key not in handle:
                raise KeyError(f"{path} does not contain '{self.cube_key}'.")
            cube = np.asarray(handle[self.cube_key], dtype=np.float32).squeeze()
        cube = _to_chw(cube, self.spectral_channels, height, width)
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
        rgb, _ = _load_rgb_tensors(path)
        return {
            "cond": _normalize_rgb(rgb, self.rgb_normalization),
            "scene_id": path.stem,
            "source_path": str(path),
        }


def _to_chw(
    cube: np.ndarray,
    channels: int,
    height: int,
    width: int,
) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected a three-dimensional cube, got {cube.shape}.")
    target = (channels, height, width)
    matching = [
        order
        for order in permutations(range(3))
        if tuple(cube.shape[index] for index in order) == target
    ]
    if not matching:
        raise ValueError(
            f"Cannot align spectral cube {cube.shape} to CHW target {target}."
        )
    return np.transpose(cube, matching[0])


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

"""Hugging Face ARAD HSDB adapter.

The public ``mhmdjouni/arad_hsdb`` dataset is exposed through Hugging Face as an
``imagefolder`` dataset.  This adapter keeps the training scripts on the
existing MST-style contract: every sample is returned as ``(rgb, hsi)`` with
channel-first tensors/arrays.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from ...constants import ARAD1K_NUM_BANDS, DEFAULT_PATCH_SIZE, DEFAULT_STRIDE

logger = logging.getLogger(__name__)

HF_ARAD_DATASET_NAME = "mhmdjouni/arad_hsdb"

_RGB_LABEL_KEYWORDS = ("realworld", "rgb", "input", "srgb")
_RGB_FALLBACK_KEYWORDS = ("clean",)
_HSI_LABEL_KEYWORDS = ("hsi", "hyper", "hyperspectral", "spectral", "spec", "target", "gt", "clean")
_BAND_RE = re.compile(
    r"(?i)(?:[_\-. ](?:band|b|lambda|wl|wave)[_\-. ]?(\d{1,3})|"
    r"(?:band|lambda|wl|wave)(\d{1,3}))$"
)
_TOKEN_RE = re.compile(
    r"(?i)(^|[_\-. ])(?:rgb|srgb|clean|realworld|hsi|hyper|hyperspectral|"
    r"spectral|spec|target|gt|train|training|valid|validation|test)(?=[_\-. ]|$)"
)


@dataclass(frozen=True)
class _RecordRef:
    index: int
    label_name: str
    name: str
    key: str
    band_index: int


@dataclass(frozen=True)
class _PairRef:
    rgb: _RecordRef
    hsi: Tuple[_RecordRef, ...]


def load_hf_arad_source(
    dataset_name: str = HF_ARAD_DATASET_NAME,
    split: str = "train",
) -> Any:
    """Load the Hugging Face dataset lazily enough to keep this dependency optional."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset loading requires the optional 'datasets' package. "
            "Install it with `pip install datasets` or use dataset_source=mst."
        ) from exc

    return load_dataset(dataset_name, split=split)


def _as_list(value: Optional[Any]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if value.strip().lower() in {"", "none", "null", "all", "*"}:
            return []
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    return [str(part).strip().lower() for part in value if str(part).strip()]


def _label_names(source: Any) -> Optional[List[str]]:
    features = getattr(source, "features", None)
    if not features:
        return None

    label_feature = features.get("label") if isinstance(features, Mapping) else None
    names = getattr(label_feature, "names", None)
    if names:
        return [str(name) for name in names]
    return None


def _label_to_name(record: Mapping[str, Any], label_names: Optional[Sequence[str]]) -> str:
    label = record.get("label", "")
    if isinstance(label, str):
        return label
    if isinstance(label, (int, np.integer)) and label_names is not None:
        label_idx = int(label)
        if 0 <= label_idx < len(label_names):
            return str(label_names[label_idx])
    return str(label)


def _record_name(record: Mapping[str, Any], index: int) -> str:
    for key in ("filename", "file_name", "path", "image_path"):
        if record.get(key):
            return str(record[key])

    image = record.get("image")
    if isinstance(image, Mapping):
        for key in ("path", "filename", "file_name"):
            if image.get(key):
                return str(image[key])

    image_name = getattr(image, "filename", "")
    if image_name:
        return str(image_name)

    return f"sample_{index:06d}"


def _sample_key(name: str) -> str:
    stem = Path(str(name)).stem.lower()
    stem = _BAND_RE.sub("", stem)
    stem = _TOKEN_RE.sub("_", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or Path(str(name)).stem.lower()


def _band_index(name: str, fallback: int) -> int:
    match = _BAND_RE.search(Path(str(name)).stem)
    if match:
        return int(next(group for group in match.groups() if group is not None))
    return fallback


def _label_matches(label_name: str, terms: Sequence[str]) -> bool:
    if not terms:
        return True
    lowered = label_name.lower()
    return any(term in lowered for term in terms)


def _select_labels(
    available: Sequence[str],
    explicit: Optional[Any],
    keywords: Sequence[str],
    fallback_keywords: Sequence[str] = (),
) -> List[str]:
    explicit_terms = _as_list(explicit)
    lowered_to_name = {name.lower(): name for name in available}

    if explicit_terms:
        selected: List[str] = []
        for term in explicit_terms:
            if term in lowered_to_name:
                selected.append(lowered_to_name[term])
            else:
                selected.extend(name for name in available if term in name.lower())
        return sorted(set(selected), key=selected.index)

    selected = [
        name
        for name in available
        if any(keyword in name.lower() for keyword in keywords)
    ]
    if selected:
        return selected

    return [
        name
        for name in available
        if any(keyword in name.lower() for keyword in fallback_keywords)
    ]


def _build_pair_refs(
    source: Any,
    include_label_keywords: Optional[Any] = None,
    rgb_label: Optional[Any] = None,
    hsi_label: Optional[Any] = None,
) -> List[_PairRef]:
    label_filter = _as_list(include_label_keywords)
    names = _label_names(source)

    refs: List[_RecordRef] = []
    labels_in_scope: List[str] = []
    for index in range(len(source)):
        record = source[index]
        label_name = _label_to_name(record, names)
        if not _label_matches(label_name, label_filter):
            continue

        record_name = _record_name(record, index)
        ref = _RecordRef(
            index=index,
            label_name=label_name,
            name=record_name,
            key=_sample_key(record_name),
            band_index=_band_index(record_name, index),
        )
        refs.append(ref)
        if label_name not in labels_in_scope:
            labels_in_scope.append(label_name)

    rgb_labels = _select_labels(
        labels_in_scope,
        rgb_label,
        _RGB_LABEL_KEYWORDS,
        _RGB_FALLBACK_KEYWORDS,
    )
    hsi_labels = _select_labels(labels_in_scope, hsi_label, _HSI_LABEL_KEYWORDS)

    rgb_by_key: Dict[str, _RecordRef] = {}
    for ref in refs:
        if ref.label_name in rgb_labels and ref.key not in rgb_by_key:
            rgb_by_key[ref.key] = ref

    hsi_by_key: Dict[str, List[_RecordRef]] = {}
    for ref in refs:
        if ref.label_name in hsi_labels:
            hsi_by_key.setdefault(ref.key, []).append(ref)

    pairs: List[_PairRef] = []
    for key, hsi_refs in sorted(hsi_by_key.items()):
        rgb_ref = rgb_by_key.get(key)
        if rgb_ref is None:
            # If a caller explicitly asks for the same class as input and target,
            # use the target record as RGB input as a last resort.
            if any(ref.label_name in rgb_labels for ref in hsi_refs):
                rgb_ref = hsi_refs[0]
            else:
                continue

        ordered_hsi = tuple(sorted(hsi_refs, key=lambda ref: (ref.band_index, ref.name)))
        pairs.append(_PairRef(rgb=rgb_ref, hsi=ordered_hsi))

    if not pairs:
        raise ValueError(
            "No RGB/HSI pairs could be inferred from Hugging Face ARAD rows. "
            f"Labels in scope={labels_in_scope}, selected rgb_labels={rgb_labels}, "
            f"hsi_labels={hsi_labels}, label_filter={label_filter or 'all'}. "
            "Set hf_rgb_label/hf_hsi_label explicitly if your local cache uses "
            "different class names."
        )

    logger.info(
        "Inferred %d Hugging Face ARAD pairs from labels rgb=%s target=%s filter=%s",
        len(pairs),
        rgb_labels,
        hsi_labels,
        label_filter or "all",
    )
    return pairs


def _normalise_array(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        return array.astype(np.float32) / float(info.max)

    array = array.astype(np.float32, copy=False)
    if array.size == 0:
        return array
    if np.nanmax(array) > 1.5:
        array = array / 255.0
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)


def _image_array(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    return np.asarray(image)


def _image_to_rgb_chw(image: Any) -> np.ndarray:
    if not isinstance(image, np.ndarray) and hasattr(image, "convert"):
        array = np.asarray(image.convert("RGB"))
    else:
        array = _image_array(image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.transpose(array, (1, 2, 0))
        elif array.ndim == 3 and array.shape[-1] > 3:
            array = array[:, :, :3]

    array = _normalise_array(array)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image convertible to HxWx3, got {array.shape}")

    return np.ascontiguousarray(array.transpose(2, 0, 1))


def _single_band_array(image: Any) -> np.ndarray:
    array = _image_array(image)
    if array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[:, :, 0]
        elif array.shape[0] == 1:
            array = array[0]
        else:
            array = array.mean(axis=-1)
    return _normalise_array(array).astype(np.float32, copy=False)


def _multiframe_hsi(image: Any, output_channels: int) -> Optional[np.ndarray]:
    n_frames = int(getattr(image, "n_frames", 1))
    if n_frames < output_channels or not hasattr(image, "seek"):
        return None

    bands: List[np.ndarray] = []
    for frame_idx in range(output_channels):
        image.seek(frame_idx)
        bands.append(_single_band_array(image))
    return np.stack(bands, axis=0)


def _image_to_hsi_chw(
    image: Any,
    output_channels: int,
    allow_pseudo_hsi: bool,
    sample_name: str,
) -> np.ndarray:
    multi = _multiframe_hsi(image, output_channels)
    if multi is not None:
        return np.ascontiguousarray(multi.astype(np.float32, copy=False))

    array = _normalise_array(_image_array(image))
    if array.ndim == 3 and array.shape[-1] == output_channels:
        return np.ascontiguousarray(array.transpose(2, 0, 1))
    if array.ndim == 3 and array.shape[0] == output_channels:
        return np.ascontiguousarray(array)

    if allow_pseudo_hsi:
        band = _single_band_array(image)
        return np.ascontiguousarray(np.repeat(band[None, :, :], output_channels, axis=0))

    raise ValueError(
        f"Hugging Face ARAD target '{sample_name}' decoded as shape {array.shape}, "
        f"not a {output_channels}-band hyperspectral target. Set hf_hsi_label to "
        "a class containing spectral cubes/bands, use the local MST layout, or "
        "set hf_allow_pseudo_hsi=true only for smoke tests."
    )


class HuggingFaceARADHSDBDataset(Dataset):
    """MST-compatible Dataset wrapper around ``mhmdjouni/arad_hsdb`` rows."""

    def __init__(
        self,
        source: Optional[Any] = None,
        dataset_name: str = HF_ARAD_DATASET_NAME,
        split: str = "train",
        training: bool = True,
        crop_size: int = DEFAULT_PATCH_SIZE,
        stride: int = DEFAULT_STRIDE,
        arg: bool = True,
        include_label_keywords: Optional[Any] = None,
        rgb_label: Optional[Any] = None,
        hsi_label: Optional[Any] = None,
        output_channels: int = ARAD1K_NUM_BANDS,
        patches_per_image: int = 1,
        max_samples: Optional[int] = None,
        allow_pseudo_hsi: bool = False,
    ):
        self.source = source if source is not None else load_hf_arad_source(dataset_name, split)
        self.training = training
        self.crop_size = int(crop_size)
        self.stride = int(stride)
        self.arg = bool(arg)
        self.output_channels = int(output_channels)
        self.patches_per_image = max(1, int(patches_per_image if training else 1))
        self.allow_pseudo_hsi = bool(allow_pseudo_hsi)

        pairs = _build_pair_refs(
            self.source,
            include_label_keywords=include_label_keywords,
            rgb_label=rgb_label,
            hsi_label=hsi_label,
        )
        if max_samples is not None:
            pairs = pairs[: int(max_samples)]
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    @staticmethod
    def argument(data: np.ndarray, rot_times: int, v_flip: int, h_flip: int) -> np.ndarray:
        if rot_times:
            data = np.rot90(data, rot_times, axes=(1, 2))
        if v_flip:
            data = data[:, ::-1, :]
        if h_flip:
            data = data[:, :, ::-1]
        return data

    def _record(self, ref: _RecordRef) -> Mapping[str, Any]:
        return self.source[ref.index]

    def _load_pair(self, pair: _PairRef) -> Tuple[np.ndarray, np.ndarray]:
        rgb = _image_to_rgb_chw(self._record(pair.rgb)["image"])

        if len(pair.hsi) >= self.output_channels:
            bands = [
                _single_band_array(self._record(ref)["image"])
                for ref in pair.hsi[: self.output_channels]
            ]
            hsi = np.stack(bands, axis=0)
        else:
            hsi_ref = pair.hsi[0]
            hsi = _image_to_hsi_chw(
                self._record(hsi_ref)["image"],
                self.output_channels,
                self.allow_pseudo_hsi,
                hsi_ref.name,
            )

        if rgb.shape[1:] != hsi.shape[1:]:
            raise ValueError(
                f"Spatial mismatch for Hugging Face ARAD pair {pair.rgb.name}: "
                f"RGB={rgb.shape[1:]} HSI={hsi.shape[1:]}"
            )

        return rgb.astype(np.float32, copy=False), hsi.astype(np.float32, copy=False)

    def _crop_pair(self, rgb: np.ndarray, hsi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.training:
            return rgb, hsi

        _, height, width = rgb.shape
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(
                f"HF ARAD sample {height}x{width} is smaller than crop_size={self.crop_size}"
            )

        top = random.randint(0, height - self.crop_size)
        left = random.randint(0, width - self.crop_size)
        rgb = rgb[:, top : top + self.crop_size, left : left + self.crop_size]
        hsi = hsi[:, top : top + self.crop_size, left : left + self.crop_size]

        if self.arg:
            rot_times = random.randint(0, 3)
            v_flip = random.randint(0, 1)
            h_flip = random.randint(0, 1)
            rgb = self.argument(rgb, rot_times, v_flip, h_flip)
            hsi = self.argument(hsi, rot_times, v_flip, h_flip)

        return rgb, hsi

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        pair = self.pairs[idx // self.patches_per_image]
        rgb, hsi = self._load_pair(pair)
        rgb, hsi = self._crop_pair(rgb, hsi)
        return np.ascontiguousarray(rgb), np.ascontiguousarray(hsi)


def create_hf_arad_datasets(
    config: Mapping[str, Any],
    seed: int = 42,
) -> Tuple[HuggingFaceARADHSDBDataset, HuggingFaceARADHSDBDataset]:
    del seed  # Pair discovery is deterministic; workers handle crop randomness.

    dataset_name = str(config.get("hf_dataset_name", HF_ARAD_DATASET_NAME))
    split = str(config.get("hf_split", "train"))
    source = load_hf_arad_source(dataset_name, split)

    common_kwargs = dict(
        source=source,
        dataset_name=dataset_name,
        split=split,
        crop_size=int(config.get("patch_size", DEFAULT_PATCH_SIZE)),
        stride=int(config.get("stride", DEFAULT_STRIDE)),
        rgb_label=config.get("hf_rgb_label"),
        hsi_label=config.get("hf_hsi_label"),
        output_channels=int(config.get("output_channels", ARAD1K_NUM_BANDS)),
        allow_pseudo_hsi=bool(config.get("hf_allow_pseudo_hsi", False)),
    )

    train_dataset = HuggingFaceARADHSDBDataset(
        **common_kwargs,
        training=True,
        arg=True,
        include_label_keywords=config.get("hf_train_label_filter", "train"),
        patches_per_image=int(config.get("hf_patches_per_image", 1)),
        max_samples=config.get("hf_max_train_samples"),
    )
    val_dataset = HuggingFaceARADHSDBDataset(
        **common_kwargs,
        training=False,
        arg=False,
        include_label_keywords=config.get("hf_val_label_filter", "validation,val"),
        patches_per_image=1,
        max_samples=config.get("hf_max_val_samples"),
    )

    return train_dataset, val_dataset

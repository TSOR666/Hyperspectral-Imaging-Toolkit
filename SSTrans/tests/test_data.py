from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from hsiformer import (
    ARAD1KDataset,
    load_arad_manifest,
    spectral_metrics,
)


def _write_scene(root, scene_id: str = "ARAD_1K_0001"):
    rgb_root = root / "Train_RGB"
    spectral_root = root / "Train_spectral"
    rgb_root.mkdir(parents=True)
    spectral_root.mkdir(parents=True)

    height, width = 6, 8
    rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(
        height,
        width,
        3,
    )
    Image.fromarray(rgb, mode="RGB").save(rgb_root / f"{scene_id}.jpg")

    cube = np.arange(31 * height * width, dtype=np.float32).reshape(
        31,
        height,
        width,
    )
    with h5py.File(spectral_root / f"{scene_id}.mat", "w") as handle:
        # Match the transposed MATLAB/HDF5 layout used by the original loader.
        handle.create_dataset("cube", data=cube.transpose(0, 2, 1))
    return cube


def test_packaged_arad_manifests_cover_public_1000_scene_split() -> None:
    train = load_arad_manifest("train")
    validation = load_arad_manifest("validation")
    test = load_arad_manifest("test")
    assert len(train) == 900
    assert len(validation) == 50
    assert len(test) == 50
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert len(set(train) | set(validation) | set(test)) == 1000
    assert test[0] == "ARAD_1K_0951"
    assert test[-1] == "ARAD_1K_1000"


def test_lazy_dataset_aligns_cube_and_covers_image_boundaries(tmp_path) -> None:
    expected_cube = _write_scene(tmp_path)
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0001\n", encoding="utf-8")

    dataset = ARAD1KDataset(
        tmp_path,
        manifest_path=manifest,
        crop_size=(4, 4),
        stride=(4, 4),
        random_crop=False,
        augment=False,
        include_ycrcb=True,
    )

    assert len(dataset) == 4
    first = dataset[0]
    last = dataset[-1]
    assert first["cond"].shape == (3, 4, 4)
    assert first["label"].shape == (31, 4, 4)
    assert first["ycrcb"].shape == (6, 4, 4)
    torch.testing.assert_close(
        first["label"],
        torch.from_numpy(expected_cube[:, :4, :4]),
    )
    torch.testing.assert_close(
        last["label"],
        torch.from_numpy(expected_cube[:, 2:6, 4:8]),
    )
    assert 0.0 <= float(first["cond"].min())
    assert float(first["cond"].max()) <= 1.0


def test_random_crop_length_is_controlled_per_scene(tmp_path) -> None:
    _write_scene(tmp_path)
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0001\n", encoding="utf-8")
    dataset = ARAD1KDataset(
        tmp_path,
        manifest_path=manifest,
        crop_size=4,
        random_crop=True,
        crops_per_scene=3,
        augment=True,
    )
    assert len(dataset) == 3
    assert dataset[0]["cond"].shape == (3, 4, 4)


def test_oversized_training_crop_uses_full_arad_frame(tmp_path) -> None:
    _write_scene(tmp_path)
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0001\n", encoding="utf-8")
    dataset = ARAD1KDataset(
        tmp_path,
        manifest_path=manifest,
        crop_size=9,
        random_crop=True,
        augment=True,
    )
    sample = dataset[0]
    assert sample["cond"].shape == (3, 6, 8)
    assert sample["label"].shape == (31, 6, 8)


def test_spectral_metrics_match_simple_reference() -> None:
    target = torch.ones(2, 3, 4, 4)
    prediction = torch.zeros_like(target)
    metrics = spectral_metrics(prediction, target)
    torch.testing.assert_close(metrics["mrae"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["rmse"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["psnr"], torch.tensor(0.0))
    torch.testing.assert_close(
        metrics["sam"],
        torch.tensor(torch.pi / 2),
    )


def test_dataset_rejects_missing_layout(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ARAD1KDataset(tmp_path)

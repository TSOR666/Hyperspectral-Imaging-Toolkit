from __future__ import annotations

from itertools import permutations

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

import hsiformer.data as data_module
from hsiformer import (
    ARAD1KDataset,
    load_arad_manifest,
    spectral_metrics,
)
from hsiformer.data import _read_cube_crop


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


def test_per_image_normalization_uses_full_scene_before_crop(tmp_path) -> None:
    _write_scene(tmp_path)
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0001\n", encoding="utf-8")
    dataset = ARAD1KDataset(
        tmp_path,
        manifest_path=manifest,
        crop_size=(4, 4),
        stride=(4, 4),
        random_crop=False,
        augment=False,
        rgb_normalization="per_image",
    )

    full_rgb = data_module._load_rgb_tensor(
        tmp_path / "Train_RGB" / "ARAD_1K_0001.jpg"
    )
    expected = data_module._normalize_rgb(full_rgb, "per_image")[:, :4, :4]
    actual = dataset[0]["cond"]

    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)
    # This crop excludes the full scene's brightest pixels, so crop-local
    # normalization (the old behavior) would incorrectly force its max to 1.
    assert float(actual.max()) < 1.0


@pytest.mark.parametrize("storage_order", tuple(permutations(range(3))))
def test_hdf5_crop_reader_preserves_all_axis_orders(
    tmp_path,
    storage_order,
) -> None:
    channels, height, width = 31, 6, 8
    cube = np.arange(channels * height * width, dtype=np.float32).reshape(
        channels,
        height,
        width,
    )
    stored = np.transpose(cube, storage_order)
    path = tmp_path / "cube.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cube", data=stored)

    inverse_order = tuple(storage_order.index(axis) for axis in range(3))
    with h5py.File(path, "r") as handle:
        cropped = _read_cube_crop(
            handle["cube"],
            channels,
            height,
            width,
            (1, 2),
            (3, 4),
        )

    assert cropped is not None
    np.testing.assert_array_equal(
        cropped,
        np.transpose(stored, inverse_order)[:, 1:4, 2:6],
    )


def test_oversized_training_crop_uses_full_arad_frame(
    tmp_path,
    monkeypatch,
) -> None:
    _write_scene(tmp_path)
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0001\n", encoding="utf-8")
    augmentation_called = False

    def record_augmentation(tensors):
        nonlocal augmentation_called
        augmentation_called = True
        return tensors

    monkeypatch.setattr(
        data_module,
        "_paired_augmentation",
        record_augmentation,
    )
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
    assert augmentation_called


def test_oversized_grid_crop_is_one_full_frame_per_scene(tmp_path) -> None:
    expected_cube = _write_scene(tmp_path)
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0001\n", encoding="utf-8")

    dataset = ARAD1KDataset(
        tmp_path,
        manifest_path=manifest,
        crop_size=9,
        stride=4,
        random_crop=False,
        augment=False,
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["cond"].shape == (3, 6, 8)
    torch.testing.assert_close(
        sample["label"],
        torch.from_numpy(expected_cube),
    )


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


def _write_arad_test_scene(
    root,
    scene_id: str,
    *,
    spectral_key: str | None = "cube",
    rgb_dir: str = "Test_RGB",
    spectral_dir: str = "Test_Spec",
    suffix: str = ".jpg",
):
    rgb_root = root / rgb_dir
    spectral_root = root / spectral_dir
    rgb_root.mkdir(parents=True, exist_ok=True)
    spectral_root.mkdir(parents=True, exist_ok=True)

    height, width = 6, 8
    rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    Image.fromarray(rgb, mode="RGB").save(rgb_root / f"{scene_id}{suffix}")

    cube = np.arange(31 * height * width, dtype=np.float32).reshape(31, height, width)
    with h5py.File(spectral_root / f"{scene_id}.mat", "w") as handle:
        if spectral_key is None:
            # The official ARAD-1K Test_Spec payload: a raw MSFA mosaic.
            handle.create_dataset("mosaic", data=np.zeros((height, width), np.uint16))
        else:
            handle.create_dataset(spectral_key, data=cube.transpose(0, 2, 1))
    return cube


def test_dataset_reads_arad_test_split_directories(tmp_path) -> None:
    expected_cube = _write_arad_test_scene(tmp_path, "ARAD_1K_0951")
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0951\n", encoding="utf-8")

    dataset = ARAD1KDataset(
        tmp_path,
        split="test",
        manifest_path=manifest,
        augment=False,
    )

    assert dataset.rgb_root == tmp_path / "Test_RGB"
    assert dataset.spectral_root == tmp_path / "Test_Spec"
    torch.testing.assert_close(
        dataset[0]["label"],
        torch.from_numpy(expected_cube),
    )


def test_dataset_accepts_cube_key_aliases_and_png_rgb(tmp_path) -> None:
    expected_cube = _write_arad_test_scene(
        tmp_path,
        "ARAD_1K_0951",
        spectral_key="rad",
        suffix=".png",
    )
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0951\n", encoding="utf-8")

    dataset = ARAD1KDataset(
        tmp_path,
        split="test",
        manifest_path=manifest,
        augment=False,
    )

    torch.testing.assert_close(
        dataset[0]["label"],
        torch.from_numpy(expected_cube),
    )


def test_mosaic_only_test_spec_is_reported_not_silently_scored(tmp_path) -> None:
    _write_arad_test_scene(tmp_path, "ARAD_1K_0951", spectral_key=None)
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0951\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No usable ground-truth cube"):
        ARAD1KDataset(tmp_path, split="test", manifest_path=manifest)

    dataset = ARAD1KDataset(
        tmp_path,
        split="test",
        manifest_path=manifest,
        augment=False,
        require_targets=False,
    )
    assert dataset.scene_ids_with_targets == ()
    assert "mosaic" in dataset.unusable_targets["ARAD_1K_0951"]
    sample = dataset[0]
    assert "label" not in sample
    assert sample["cond"].shape == (3, 6, 8)


def test_mosaic_file_does_not_shadow_a_cube_in_another_directory(
    tmp_path,
) -> None:
    _write_arad_test_scene(tmp_path, "ARAD_1K_0951", spectral_key=None)
    expected_cube = _write_arad_test_scene(
        tmp_path,
        "ARAD_1K_0951",
        spectral_dir="Test_spectral",
    )
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0951\n", encoding="utf-8")

    dataset = ARAD1KDataset(tmp_path, split="test", manifest_path=manifest,
                            augment=False)

    assert dataset.spectral_root == tmp_path / "Test_spectral"
    torch.testing.assert_close(
        dataset[0]["label"],
        torch.from_numpy(expected_cube),
    )


def test_explicit_spectral_dir_overrides_the_standard_search(tmp_path) -> None:
    _write_arad_test_scene(tmp_path, "ARAD_1K_0951", spectral_key=None)
    expected_cube = _write_arad_test_scene(
        tmp_path,
        "ARAD_1K_0951",
        spectral_dir="cubes_elsewhere",
    )
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0951\n", encoding="utf-8")

    dataset = ARAD1KDataset(
        tmp_path,
        split="test",
        manifest_path=manifest,
        augment=False,
        spectral_dirs=[str(tmp_path / "cubes_elsewhere")],
    )

    assert dataset.spectral_root == tmp_path / "cubes_elsewhere"
    torch.testing.assert_close(
        dataset[0]["label"],
        torch.from_numpy(expected_cube),
    )


def test_missing_spectral_file_is_a_missing_target_not_a_crash(tmp_path) -> None:
    _write_arad_test_scene(tmp_path, "ARAD_1K_0951", spectral_key=None)
    (tmp_path / "Test_Spec" / "ARAD_1K_0951.mat").unlink()
    manifest = tmp_path / "split.txt"
    manifest.write_text("ARAD_1K_0951\n", encoding="utf-8")

    dataset = ARAD1KDataset(
        tmp_path,
        split="test",
        manifest_path=manifest,
        augment=False,
        require_targets=False,
    )
    assert dataset.unusable_targets["ARAD_1K_0951"].startswith("no spectral file")
    assert "label" not in dataset[0]


def test_split_directory_resolution_falls_back_to_train(tmp_path) -> None:
    _write_scene(tmp_path, "ARAD_1K_0901")
    assert (
        data_module.resolve_arad_directory(tmp_path, "validation")
        == tmp_path / "Train_RGB"
    )
    assert (
        data_module.resolve_arad_directory(tmp_path, "validation", kind="spectral")
        == tmp_path / "Train_spectral"
    )
    with pytest.raises(FileNotFoundError):
        data_module.resolve_arad_directory(tmp_path / "empty", "test")

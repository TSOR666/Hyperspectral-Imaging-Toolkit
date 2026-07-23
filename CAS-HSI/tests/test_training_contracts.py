"""Regression tests for data and training-entry contracts."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import h5py
import yaml

import train_cas_hsi
import test_cas_ntire
from cas_hsi import CASHSIConfig, Depths, build_cas_hsi
from dataloader import TrainDataset, _pad_to_crop, _patch_starts, load_hsi_cube


def test_load_config_cli_resolves_postponed_annotations():
    config = train_cas_hsi.load_config(
        ["--data_root", "synthetic", "--epochs", "1", "--amp", "false", "--lr", "0.001"]
    )

    assert config.data_root == "synthetic"
    assert config.epochs == 1
    assert config.amp is False
    assert config.lr == pytest.approx(0.001)


@pytest.mark.parametrize(
    "config_name",
    ["cas_hsi_tiny.yaml", "cas_hsi_base.yaml", "cas_hsi_edge.yaml"],
)
def test_shipped_configs_preserve_mstpp_optimizer_contract(config_name):
    """Reference configs must not silently warm up or clip the MST++ Adam updates."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / config_name
    config = train_cas_hsi.load_config(["--config", str(config_path)])

    assert config.optimizer == "adam"
    assert config.lr == pytest.approx(4e-4)
    assert config.min_lr == pytest.approx(1e-6)
    assert config.weight_decay == 0.0
    assert config.warmup_epochs == 0
    assert config.gradient_clip == 0.0


def test_checkpoint_rng_state_restores_python_numpy_and_torch_streams():
    original = train_cas_hsi.capture_rng_state()
    try:
        train_cas_hsi.set_seed(1234, deterministic=False)
        saved = train_cas_hsi.capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))

        train_cas_hsi.restore_rng_state(saved)
        actual = (random.random(), np.random.rand(), torch.rand(3))

        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        assert torch.equal(actual[2], expected[2])
    finally:
        train_cas_hsi.restore_rng_state(original)


def test_evaluation_inherits_rgb_normalization_from_checkpoint():
    checkpoint = {"train_config": {"rgb_norm": "div255"}}

    assert test_cas_ntire.resolve_rgb_norm(None, checkpoint) == "div255"
    assert test_cas_ntire.resolve_rgb_norm("minmax", checkpoint) == "minmax"


def test_evaluation_uses_documented_legacy_rgb_normalization_fallback():
    assert test_cas_ntire.resolve_rgb_norm(None, {}) == "minmax"


def test_hsi_loader_restores_arad_spatial_axis_order_and_band_axis(tmp_path):
    # ARAD HDF5 data is (bands, width, height); coordinate values make an accidental
    # H/W transpose observable rather than merely checking the resulting shape.
    stored = np.arange(31 * 7 * 5, dtype=np.float32).reshape(31, 7, 5)
    path = tmp_path / "cube.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cube", data=stored)

    cube = load_hsi_cube(path, expected_bands=31)

    assert cube.shape == (31, 5, 7)
    assert np.array_equal(cube, np.transpose(stored, (0, 2, 1)))


def test_hsi_loader_rejects_nonfinite_or_wrong_band_count(tmp_path):
    path = tmp_path / "invalid.mat"
    stored = np.zeros((30, 7, 5), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cube", data=stored)
    with pytest.raises(ValueError, match="30 bands"):
        load_hsi_cube(path, expected_bands=31)

    stored[0, 0, 0] = np.nan
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cube", data=stored)
    with pytest.raises(ValueError, match="NaN or Inf"):
        load_hsi_cube(path, expected_bands=30)


def test_patch_grid_includes_tail_when_stride_does_not_divide_extent():
    assert _patch_starts(482, crop_size=128, stride=8)[-1] == 354
    assert _patch_starts(512, crop_size=128, stride=8)[-1] == 384


def test_train_dataset_getitem_reaches_bottom_right_tail_patch():
    dataset = TrainDataset.__new__(TrainDataset)
    dataset.crop_size = 4
    dataset.stride = 3
    dataset.augment = False
    dataset.rgb_images = [np.zeros((3, 7, 9), dtype=np.float32)]
    hsi = np.arange(31 * 7 * 9, dtype=np.float32).reshape(31, 7, 9)
    dataset.hsi_cubes = [hsi]
    dataset._patch_grids = [(tuple(_patch_starts(7, 4, 3)), tuple(_patch_starts(9, 4, 3)))]
    dataset._patch_offsets = [len(dataset._patch_grids[0][0]) * len(dataset._patch_grids[0][1])]
    dataset.total_patches = dataset._patch_offsets[-1]

    _, hsi_patch = dataset[len(dataset) - 1]

    expected = hsi[:, 3:7, 5:9]
    assert torch.equal(hsi_patch, torch.from_numpy(expected.copy()))


def test_pad_to_crop_handles_one_pixel_axes():
    image = np.ones((3, 1, 1), dtype=np.float32)
    padded = _pad_to_crop(image, crop_size=4)

    assert padded.shape == (3, 4, 4)
    assert np.all(padded == 1.0)


def test_untrained_low_rank_head_starts_close_to_linear_prior():
    torch.manual_seed(0)
    model = build_cas_hsi(
        CASHSIConfig(
            name="low_rank_init_test",
            base_width=32,
            spectral_head="low_rank",
            depths=Depths(
                encoder_full=1,
                encoder_half=1,
                bottleneck=3,
                decoder_half=1,
                decoder_full=1,
                refinement=1,
            ),
        )
    ).eval()
    rgb = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        output = model(rgb)
        prior = model.rgb_prior(rgb)

    residual = (output - prior).abs().mean()
    prior_scale = prior.abs().mean()
    assert residual < 0.05 * prior_scale


def test_config_rejects_nonpositive_head_dim_cleanly():
    with pytest.raises(ValueError, match="head_dim must be positive"):
        CASHSIConfig.from_dict({"head_dim": 0})

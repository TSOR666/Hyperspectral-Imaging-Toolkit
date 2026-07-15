"""Regression tests for data and training-entry contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import train_cas_hsi
from cas_hsi import CASHSIConfig, Depths, build_cas_hsi
from dataloader import TrainDataset, _pad_to_crop, _patch_starts


def test_load_config_cli_resolves_postponed_annotations():
    config = train_cas_hsi.load_config(
        ["--data_root", "synthetic", "--epochs", "1", "--amp", "false", "--lr", "0.001"]
    )

    assert config.data_root == "synthetic"
    assert config.epochs == 1
    assert config.amp is False
    assert config.lr == pytest.approx(0.001)


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

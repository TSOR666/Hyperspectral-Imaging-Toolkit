"""Tests for scripts/visualization_utils.py - Error metrics and utilities."""

from __future__ import annotations

import pytest
import torch
import numpy as np

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from visualization_utils import (
    compute_mrae_map,
    compute_sam_map,
    compute_rmse_map,
    to_chw,
    _ensure_bchw,
)


class TestToChw:
    """Tests for to_chw function."""

    def test_hwc_to_chw_31_bands(self) -> None:
        """Test HWC to CHW conversion for 31-band HSI."""
        x = np.random.rand(256, 256, 31)
        result = to_chw(x)
        assert result.shape == (31, 256, 256)

    def test_hwc_to_chw_32_bands(self) -> None:
        """Test HWC to CHW conversion for 32-band HSI."""
        x = np.random.rand(256, 256, 32)
        result = to_chw(x)
        assert result.shape == (32, 256, 256)

    def test_hwc_to_chw_64_bands(self) -> None:
        """Test HWC to CHW conversion for 64-band HSI."""
        x = np.random.rand(256, 256, 64)
        result = to_chw(x)
        assert result.shape == (64, 256, 256)

    def test_chw_unchanged(self) -> None:
        """Test CHW input is unchanged."""
        x = np.random.rand(31, 256, 256)
        result = to_chw(x)
        assert result.shape == (31, 256, 256)

    def test_explicit_band_count(self) -> None:
        """Test with explicit expected_bands parameter."""
        x = np.random.rand(256, 256, 31)
        result = to_chw(x, expected_bands=31)
        assert result.shape == (31, 256, 256)

    def test_2d_input_unchanged(self) -> None:
        """Test 2D input is returned unchanged."""
        x = np.random.rand(256, 256)
        result = to_chw(x)
        assert result.shape == (256, 256)

    def test_4d_input_unchanged(self) -> None:
        """Test 4D input is returned unchanged."""
        x = np.random.rand(4, 31, 256, 256)
        result = to_chw(x)
        assert result.shape == (4, 31, 256, 256)

    def test_small_image_with_bands(self) -> None:
        """Test small image with band count similar to spatial dims."""
        # 31x31 image with 31 bands - ambiguous case
        x = np.random.rand(31, 31, 31)
        result = to_chw(x, expected_bands=31)
        # With explicit band count, should convert
        assert result.shape == (31, 31, 31)


class TestEnsureBchw:
    """Tests for _ensure_bchw function."""

    def test_3d_to_4d(self) -> None:
        """Test 3D input is converted to 4D."""
        pred = torch.rand(31, 64, 64)
        target = torch.rand(31, 64, 64)
        pred_out, target_out = _ensure_bchw(pred, target)
        assert pred_out.shape == (1, 31, 64, 64)
        assert target_out.shape == (1, 31, 64, 64)

    def test_4d_unchanged(self) -> None:
        """Test 4D input is unchanged."""
        pred = torch.rand(2, 31, 64, 64)
        target = torch.rand(2, 31, 64, 64)
        pred_out, target_out = _ensure_bchw(pred, target)
        assert pred_out.shape == (2, 31, 64, 64)
        assert target_out.shape == (2, 31, 64, 64)

    def test_shape_mismatch_error(self) -> None:
        """Test error raised for shape mismatch."""
        pred = torch.rand(2, 31, 64, 64)
        target = torch.rand(2, 31, 32, 32)
        with pytest.raises(ValueError, match="shape mismatch"):
            _ensure_bchw(pred, target)

    def test_dtype_alignment(self) -> None:
        """Test dtypes are aligned."""
        pred = torch.rand(31, 64, 64, dtype=torch.float32)
        target = torch.rand(31, 64, 64, dtype=torch.float64)
        pred_out, target_out = _ensure_bchw(pred, target)
        assert target_out.dtype == pred_out.dtype


class TestComputeMraeMap:
    """Tests for compute_mrae_map function."""

    def test_output_shape_3d(self) -> None:
        """Test output shape for 3D input."""
        pred = torch.rand(31, 64, 64)
        target = torch.rand(31, 64, 64)
        mrae = compute_mrae_map(pred, target)
        assert mrae.shape == (64, 64)

    def test_output_shape_4d(self) -> None:
        """Test output shape for 4D input with batch > 1."""
        pred = torch.rand(4, 31, 64, 64)
        target = torch.rand(4, 31, 64, 64)
        mrae = compute_mrae_map(pred, target)
        assert mrae.shape == (4, 64, 64)

    def test_identical_inputs(self) -> None:
        """Test MRAE is zero for identical inputs."""
        x = torch.rand(31, 64, 64) + 0.1  # Avoid near-zero values
        mrae = compute_mrae_map(x, x)
        assert np.allclose(mrae, 0.0, atol=1e-6)

    def test_mrae_non_negative(self) -> None:
        """Test MRAE is always non-negative."""
        pred = torch.rand(31, 64, 64)
        target = torch.rand(31, 64, 64)
        mrae = compute_mrae_map(pred, target)
        assert (mrae >= 0).all(), "MRAE should be non-negative"

    def test_near_zero_target_stability(self) -> None:
        """Test MRAE doesn't explode for near-zero targets."""
        pred = torch.ones(31, 64, 64) * 0.5
        target = torch.ones(31, 64, 64) * 1e-10  # Near-zero
        mrae = compute_mrae_map(pred, target)
        # Should not explode - capped error handling
        assert not np.isinf(mrae).any(), "MRAE should not be infinite"
        assert not np.isnan(mrae).any(), "MRAE should not be NaN"
        # With new robust implementation, should be reasonably bounded
        assert mrae.max() <= 100.0, f"MRAE max {mrae.max()} is too large"

    def test_known_error(self) -> None:
        """Test MRAE with known error magnitude."""
        target = torch.ones(31, 64, 64) * 1.0
        pred = torch.ones(31, 64, 64) * 1.1  # 10% error
        mrae = compute_mrae_map(pred, target)
        # Expected MRAE = |1.1 - 1.0| / 1.0 = 0.1
        assert np.allclose(mrae, 0.1, atol=1e-5)

    def test_output_is_numpy(self) -> None:
        """Test output is numpy array."""
        pred = torch.rand(31, 64, 64)
        target = torch.rand(31, 64, 64)
        mrae = compute_mrae_map(pred, target)
        assert isinstance(mrae, np.ndarray)


class TestComputeSamMap:
    """Tests for compute_sam_map function."""

    def test_output_shape_3d(self) -> None:
        """Test output shape for 3D input."""
        pred = torch.rand(31, 64, 64)
        target = torch.rand(31, 64, 64)
        sam = compute_sam_map(pred, target)
        assert sam.shape == (64, 64)

    def test_identical_inputs(self) -> None:
        """Test SAM is near-zero for identical inputs."""
        x = torch.rand(31, 64, 64) + 0.1
        sam = compute_sam_map(x, x)
        # Floating point errors in normalization can produce small non-zero values
        assert np.allclose(sam, 0.0, atol=0.1), f"SAM max: {sam.max()}"

    def test_orthogonal_vectors(self) -> None:
        """Test SAM is 90 degrees for orthogonal vectors."""
        pred = torch.zeros(2, 64, 64)
        pred[0, :, :] = 1.0  # [1, 0] vectors
        target = torch.zeros(2, 64, 64)
        target[1, :, :] = 1.0  # [0, 1] vectors
        sam = compute_sam_map(pred, target)
        assert np.allclose(sam, 90.0, atol=0.1)

    def test_opposite_vectors(self) -> None:
        """Test SAM is 180 degrees for opposite vectors."""
        pred = torch.ones(31, 64, 64)
        target = -torch.ones(31, 64, 64)
        sam = compute_sam_map(pred, target)
        assert np.allclose(sam, 180.0, atol=0.1)

    def test_sam_range(self) -> None:
        """Test SAM is in valid range [0, 180] degrees."""
        pred = torch.rand(31, 64, 64)
        target = torch.rand(31, 64, 64)
        sam = compute_sam_map(pred, target)
        assert (sam >= 0).all(), "SAM should be >= 0"
        assert (sam <= 180).all(), "SAM should be <= 180"

    def test_sam_non_nan(self) -> None:
        """Test SAM doesn't produce NaN."""
        pred = torch.rand(31, 64, 64)
        target = torch.rand(31, 64, 64)
        sam = compute_sam_map(pred, target)
        assert not np.isnan(sam).any(), "SAM should not contain NaN"


class TestComputeRmseMap:
    """Tests for compute_rmse_map function."""

    def test_output_shape_3d(self) -> None:
        """Test output shape for 3D input."""
        pred = torch.rand(31, 64, 64)
        target = torch.rand(31, 64, 64)
        rmse = compute_rmse_map(pred, target)
        assert rmse.shape == (64, 64)

    def test_identical_inputs(self) -> None:
        """Test RMSE is zero for identical inputs."""
        x = torch.rand(31, 64, 64)
        rmse = compute_rmse_map(x, x)
        assert np.allclose(rmse, 0.0, atol=1e-6)

    def test_rmse_non_negative(self) -> None:
        """Test RMSE is always non-negative."""
        pred = torch.rand(31, 64, 64)
        target = torch.rand(31, 64, 64)
        rmse = compute_rmse_map(pred, target)
        assert (rmse >= 0).all(), "RMSE should be non-negative"

    def test_known_error(self) -> None:
        """Test RMSE with known error magnitude."""
        target = torch.zeros(31, 64, 64)
        pred = torch.ones(31, 64, 64) * 0.1
        rmse = compute_rmse_map(pred, target)
        # Expected RMSE = sqrt(mean(0.1^2)) = 0.1
        assert np.allclose(rmse, 0.1, atol=1e-5)

    def test_output_is_numpy(self) -> None:
        """Test output is numpy array."""
        pred = torch.rand(31, 64, 64)
        target = torch.rand(31, 64, 64)
        rmse = compute_rmse_map(pred, target)
        assert isinstance(rmse, np.ndarray)


class TestNumericalStability:
    """Tests for numerical stability across all metrics."""

    def test_very_small_values(self) -> None:
        """Test all metrics handle very small values."""
        pred = torch.rand(31, 32, 32) * 1e-10
        target = torch.rand(31, 32, 32) * 1e-10

        mrae = compute_mrae_map(pred, target)
        sam = compute_sam_map(pred, target)
        rmse = compute_rmse_map(pred, target)

        for name, metric in [("MRAE", mrae), ("SAM", sam), ("RMSE", rmse)]:
            assert not np.isnan(metric).any(), f"{name} contains NaN"
            assert not np.isinf(metric).any(), f"{name} contains Inf"

    def test_large_values(self) -> None:
        """Test all metrics handle large values."""
        pred = torch.rand(31, 32, 32) * 1e6
        target = torch.rand(31, 32, 32) * 1e6

        mrae = compute_mrae_map(pred, target)
        sam = compute_sam_map(pred, target)
        rmse = compute_rmse_map(pred, target)

        for name, metric in [("MRAE", mrae), ("SAM", sam), ("RMSE", rmse)]:
            assert not np.isnan(metric).any(), f"{name} contains NaN"
            assert not np.isinf(metric).any(), f"{name} contains Inf"

    def test_mixed_positive_negative(self) -> None:
        """Test metrics handle mixed positive/negative values."""
        pred = torch.rand(31, 32, 32) - 0.5  # Range [-0.5, 0.5]
        target = torch.rand(31, 32, 32) - 0.5

        mrae = compute_mrae_map(pred, target)
        sam = compute_sam_map(pred, target)
        rmse = compute_rmse_map(pred, target)

        for name, metric in [("MRAE", mrae), ("SAM", sam), ("RMSE", rmse)]:
            assert not np.isnan(metric).any(), f"{name} contains NaN"
            assert not np.isinf(metric).any(), f"{name} contains Inf"

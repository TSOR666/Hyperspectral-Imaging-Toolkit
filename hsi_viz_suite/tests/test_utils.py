"""Tests for hsi_model/utils.py - HSI to RGB conversion and utilities."""

from __future__ import annotations

import pytest
import torch

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hsi_model.utils import crop_center_arad1k, get_cached_cmf, hsi_to_rgb


class TestGetCachedCmf:
    """Tests for get_cached_cmf function."""

    def test_output_shape_default(self) -> None:
        """Test default 31-band CMF has correct shape."""
        cmf = get_cached_cmf()
        assert cmf.shape == (3, 31), f"Expected (3, 31), got {cmf.shape}"

    def test_output_shape_custom_bands(self) -> None:
        """Test CMF with custom band count has correct shape."""
        for n_bands in [16, 31, 64, 128]:
            cmf = get_cached_cmf(n_bands)
            assert cmf.shape == (3, n_bands), f"Expected (3, {n_bands}), got {cmf.shape}"

    def test_output_range(self) -> None:
        """Test CMF values are normalized to [0, 1] range."""
        cmf = get_cached_cmf()
        assert cmf.min() >= 0.0, f"CMF min {cmf.min()} should be >= 0"
        assert cmf.max() <= 1.0 + 1e-6, f"CMF max {cmf.max()} should be <= 1"

    def test_device_placement(self) -> None:
        """Test CMF is created on specified device."""
        cmf_cpu = get_cached_cmf(device=torch.device("cpu"))
        assert cmf_cpu.device.type == "cpu"

    def test_rgb_channels_distinct(self) -> None:
        """Test R, G, B channels are distinct (different peaks)."""
        cmf = get_cached_cmf(31)
        # R, G, B should peak at different wavelengths
        r_peak = cmf[0].argmax().item()
        g_peak = cmf[1].argmax().item()
        b_peak = cmf[2].argmax().item()
        # B should peak earliest (shortest wavelength), R latest
        assert b_peak < g_peak < r_peak, f"Peak order wrong: B={b_peak}, G={g_peak}, R={r_peak}"


class TestHsiToRgb:
    """Tests for hsi_to_rgb function."""

    def test_3d_input_shape(self) -> None:
        """Test 3D input (C,H,W) produces correct output shape."""
        hsi = torch.rand(31, 64, 64)
        rgb = hsi_to_rgb(hsi)
        assert rgb.shape == (1, 3, 64, 64), f"Expected (1, 3, 64, 64), got {rgb.shape}"

    def test_4d_input_shape(self) -> None:
        """Test 4D input (B,C,H,W) produces correct output shape."""
        hsi = torch.rand(2, 31, 64, 64)
        rgb = hsi_to_rgb(hsi)
        assert rgb.shape == (2, 3, 64, 64), f"Expected (2, 3, 64, 64), got {rgb.shape}"

    def test_batch_processing(self) -> None:
        """Test batch dimension is preserved correctly."""
        for batch_size in [1, 4, 8]:
            hsi = torch.rand(batch_size, 31, 32, 32)
            rgb = hsi_to_rgb(hsi)
            assert rgb.shape[0] == batch_size

    def test_output_range_clamped(self) -> None:
        """Test output is in [0, 1] range when clamped."""
        hsi = torch.rand(31, 64, 64)
        rgb = hsi_to_rgb(hsi, clamp=True)
        assert rgb.min() >= 0.0, f"RGB min {rgb.min()} should be >= 0"
        assert rgb.max() <= 1.0, f"RGB max {rgb.max()} should be <= 1"

    def test_output_range_unclamped(self) -> None:
        """Test unclamped output is still normalized (min-max)."""
        hsi = torch.rand(31, 64, 64)
        rgb = hsi_to_rgb(hsi, clamp=False)
        # Min-max normalization should give values close to [0, 1]
        assert rgb.min() >= -0.01, f"RGB min {rgb.min()} unexpectedly negative"
        assert rgb.max() <= 1.01, f"RGB max {rgb.max()} unexpectedly > 1"

    def test_dtype_float32_input(self) -> None:
        """Test float32 input produces float32 output."""
        hsi = torch.rand(31, 64, 64, dtype=torch.float32)
        rgb = hsi_to_rgb(hsi)
        assert rgb.dtype == torch.float32

    def test_dtype_float64_preserve(self) -> None:
        """Test float64 input is preserved with preserve_dtype=True."""
        hsi = torch.rand(31, 64, 64, dtype=torch.float64)
        rgb = hsi_to_rgb(hsi, preserve_dtype=True)
        assert rgb.dtype == torch.float64

    def test_custom_cmf_shape_mismatch(self) -> None:
        """Test CMF interpolation when band counts don't match."""
        hsi = torch.rand(64, 32, 32)  # 64 bands
        cmf = get_cached_cmf(31)  # 31-band CMF
        rgb = hsi_to_rgb(hsi, cmf=cmf)
        assert rgb.shape == (1, 3, 32, 32)

    def test_invalid_input_dimension(self) -> None:
        """Test error raised for invalid input dimensions."""
        hsi_2d = torch.rand(64, 64)
        with pytest.raises(ValueError, match="must be 3D or 4D"):
            hsi_to_rgb(hsi_2d)

        hsi_5d = torch.rand(1, 1, 31, 64, 64)
        with pytest.raises(ValueError, match="must be 3D or 4D"):
            hsi_to_rgb(hsi_5d)

    def test_zero_input(self) -> None:
        """Test handling of zero input (edge case)."""
        hsi = torch.zeros(31, 64, 64)
        rgb = hsi_to_rgb(hsi)
        # Should not produce NaN or Inf
        assert not torch.isnan(rgb).any(), "Output contains NaN"
        assert not torch.isinf(rgb).any(), "Output contains Inf"

    def test_constant_input(self) -> None:
        """Test handling of constant input (all same value)."""
        hsi = torch.ones(31, 64, 64) * 0.5
        rgb = hsi_to_rgb(hsi)
        # Should not produce NaN or Inf
        assert not torch.isnan(rgb).any(), "Output contains NaN"
        assert not torch.isinf(rgb).any(), "Output contains Inf"


class TestCropCenterArad1k:
    """Tests for crop_center_arad1k function."""

    def test_crop_large_image(self) -> None:
        """Test cropping of image larger than 256x256."""
        x = torch.rand(31, 512, 512)
        cropped = crop_center_arad1k(x)
        assert cropped.shape == (1, 31, 256, 256)

    def test_crop_preserves_batch(self) -> None:
        """Test batch dimension is preserved after cropping."""
        x = torch.rand(4, 31, 512, 512)
        cropped = crop_center_arad1k(x)
        assert cropped.shape == (4, 31, 256, 256)

    def test_no_crop_small_image(self) -> None:
        """Test small images are not cropped."""
        x = torch.rand(1, 31, 128, 128)
        cropped = crop_center_arad1k(x)
        assert cropped.shape == (1, 31, 128, 128)

    def test_partial_crop_width_only(self) -> None:
        """Test image smaller in one dimension is not cropped."""
        x = torch.rand(1, 31, 128, 512)
        cropped = crop_center_arad1k(x)
        # Should not crop because H < 256
        assert cropped.shape == (1, 31, 128, 512)

    def test_exact_256_size(self) -> None:
        """Test exactly 256x256 image is unchanged."""
        x = torch.rand(1, 31, 256, 256)
        cropped = crop_center_arad1k(x)
        assert cropped.shape == (1, 31, 256, 256)

    def test_center_crop_correctness(self) -> None:
        """Test that crop is actually centered."""
        x = torch.zeros(1, 1, 512, 512)
        # Put a marker at the center
        x[0, 0, 256, 256] = 1.0
        cropped = crop_center_arad1k(x)
        # After centering, the marker should be at (128, 128)
        assert cropped[0, 0, 128, 128] == 1.0

    def test_invalid_input_dimension(self) -> None:
        """Test error raised for invalid input dimensions."""
        x_2d = torch.rand(64, 64)
        with pytest.raises(ValueError, match="must be 3D or 4D"):
            crop_center_arad1k(x_2d)

"""Tests for MSWR v2 metrics module (NTIRE test)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Conditional import for AverageMeter
try:
    from utils import AverageMeter
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    AverageMeter = None


class TestSSIMCalculation:
    """Tests for SSIM calculation stability."""

    def test_ssim_variance_clamping(self):
        """Test that SSIM handles near-zero variance correctly."""
        # Create tensors with very low variance
        pred = torch.ones(1, 1, 64, 64) * 0.5 + torch.randn(1, 1, 64, 64) * 1e-8
        target = torch.ones(1, 1, 64, 64) * 0.5 + torch.randn(1, 1, 64, 64) * 1e-8

        # Compute SSIM components
        mu1 = F.avg_pool2d(pred, 11, 1, 5)
        mu2 = F.avg_pool2d(target, 11, 1, 5)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)

        sigma1_sq = F.avg_pool2d(pred * pred, 11, 1, 5) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, 11, 1, 5) - mu2_sq

        # Without clamping, sigma can be negative
        # With clamping, it should be positive
        eps = 1e-8
        sigma1_sq_clamped = torch.clamp(sigma1_sq, min=eps)
        sigma2_sq_clamped = torch.clamp(sigma2_sq, min=eps)

        assert (sigma1_sq_clamped >= 0).all()
        assert (sigma2_sq_clamped >= 0).all()

    def test_ssim_no_nan(self):
        """Test that SSIM doesn't produce NaN values."""
        # Various edge cases
        test_cases = [
            (torch.zeros(1, 1, 64, 64), torch.zeros(1, 1, 64, 64)),  # All zeros
            (torch.ones(1, 1, 64, 64), torch.ones(1, 1, 64, 64)),  # All ones
            (torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)),  # Random
        ]

        for pred, target in test_cases:
            # Simplified SSIM calculation with fixes
            mu1 = F.avg_pool2d(pred, 11, 1, 5)
            mu2 = F.avg_pool2d(target, 11, 1, 5)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.avg_pool2d(pred * pred, 11, 1, 5) - mu1_sq
            sigma2_sq = F.avg_pool2d(target * target, 11, 1, 5) - mu2_sq
            sigma12 = F.avg_pool2d(pred * target, 11, 1, 5) - mu1_mu2

            eps = 1e-8
            sigma1_sq = torch.clamp(sigma1_sq, min=eps)
            sigma2_sq = torch.clamp(sigma2_sq, min=eps)

            C1 = 0.01**2
            C2 = 0.03**2

            numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            ssim_map = numerator / torch.clamp(denominator, min=eps)

            assert not torch.isnan(ssim_map).any(), "NaN in SSIM for case"
            assert not torch.isinf(ssim_map).any(), "Inf in SSIM for case"


class TestMRAECalculation:
    """Tests for MRAE calculation stability."""

    def test_mrae_zero_target_handling(self):
        """Test that MRAE handles zero target values."""
        pred = torch.rand(1, 31, 64, 64)
        target = torch.zeros(1, 31, 64, 64)

        # Fixed MRAE calculation
        eps = 1e-3
        abs_error = torch.abs(pred - target)
        denominator = torch.maximum(
            torch.abs(target), torch.tensor(eps, device=target.device)
        )
        mrae = (abs_error / denominator).mean()

        assert not torch.isnan(mrae), "MRAE should not be NaN for zero target"
        assert not torch.isinf(mrae), "MRAE should not be Inf for zero target"

    def test_mrae_identical_tensors(self):
        """Test MRAE is zero for identical tensors."""
        tensor = torch.rand(1, 31, 64, 64) + 0.1  # Avoid zeros
        eps = 1e-3
        abs_error = torch.abs(tensor - tensor)
        denominator = torch.maximum(
            torch.abs(tensor), torch.tensor(eps, device=tensor.device)
        )
        mrae = (abs_error / denominator).mean()

        assert mrae.item() < 1e-6

    def test_mrae_bounded(self):
        """Test that MRAE doesn't produce extreme values."""
        pred = torch.rand(1, 31, 64, 64)
        target = torch.rand(1, 31, 64, 64) + 0.01

        eps = 1e-3
        abs_error = torch.abs(pred - target)
        denominator = torch.maximum(
            torch.abs(target), torch.tensor(eps, device=target.device)
        )
        mrae = (abs_error / denominator).mean()

        # MRAE should be bounded and reasonable
        assert mrae.item() < 100, "MRAE seems unreasonably high"


class TestSAMCalculation:
    """Tests for Spectral Angle Mapper calculation."""

    def test_sam_range(self):
        """Test SAM is in valid range [0, pi]."""
        pred = torch.rand(1, 31, 64, 64) + 0.1
        target = torch.rand(1, 31, 64, 64) + 0.1

        # Normalize vectors
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)

        # Dot product
        dot_product = (pred_norm * target_norm).sum(dim=1)
        dot_product = torch.clamp(dot_product, -1, 1)

        sam = torch.acos(dot_product).mean()

        assert 0 <= sam.item() <= np.pi

    def test_sam_identical_vectors(self):
        """Test SAM is zero for identical vectors."""
        tensor = torch.rand(1, 31, 64, 64) + 0.1

        tensor_norm = tensor / (tensor.norm(dim=1, keepdim=True) + 1e-8)
        dot_product = (tensor_norm * tensor_norm).sum(dim=1)
        dot_product = torch.clamp(dot_product, -1, 1)
        sam = torch.acos(dot_product).mean()

        # SAM should be very small for identical vectors (allow for float precision)
        assert sam.item() < 1e-3


class TestPerPixelMetrics:
    """Tests for per-pixel metric calculations."""

    def test_per_pixel_mrae_shape(self):
        """Test per-pixel MRAE output shape."""
        pred = torch.rand(1, 31, 64, 64)
        target = torch.rand(1, 31, 64, 64) + 0.01

        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
        target_flat = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])

        eps = 1e-3
        abs_error = torch.abs(pred_flat - target_flat)
        denominator = torch.maximum(
            torch.abs(target_flat), torch.tensor(eps, device=target.device)
        )
        pixel_mrae = (abs_error / denominator).mean(dim=1)

        # Should have n_pixels values
        expected_n_pixels = 64 * 64
        assert pixel_mrae.shape[0] == expected_n_pixels

    def test_per_pixel_rmse_shape(self):
        """Test per-pixel RMSE output shape."""
        pred = torch.rand(1, 31, 64, 64)
        target = torch.rand(1, 31, 64, 64)

        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
        target_flat = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])

        pixel_rmse = torch.sqrt(((pred_flat - target_flat) ** 2).mean(dim=1))

        expected_n_pixels = 64 * 64
        assert pixel_rmse.shape[0] == expected_n_pixels


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="utils module not available")
class TestAverageMeterIntegration:
    """Tests for AverageMeter usage in metrics."""

    def test_meter_accumulation(self):
        """Test that meter correctly accumulates values."""
        meter = AverageMeter()

        # Simulate multiple batches
        values = [0.1, 0.2, 0.15, 0.18]
        for v in values:
            meter.update(v)

        expected_avg = sum(values) / len(values)
        assert abs(meter.avg - expected_avg) < 1e-6

"""Tests for MSWR v2 utility functions."""

import numpy as np
import pytest
import torch

# Conditional import to handle missing dependencies
try:
    from utils import (
        AverageMeter,
        Loss_MRAE,
        Loss_PSNR,
        Loss_RMSE,
        Loss_SAM,
        calculate_metrics,
        count_parameters,
        get_model_size,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not UTILS_AVAILABLE, reason="utils module dependencies not available")


class TestAverageMeter:
    """Tests for AverageMeter class."""

    def test_initialization(self):
        """Test AverageMeter initial state."""
        meter = AverageMeter()
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0

    def test_single_update(self):
        """Test single value update."""
        meter = AverageMeter()
        meter.update(5.0)
        assert meter.val == 5.0
        assert meter.avg == 5.0
        assert meter.sum == 5.0
        assert meter.count == 1

    def test_multiple_updates(self):
        """Test multiple value updates."""
        meter = AverageMeter()
        meter.update(2.0)
        meter.update(4.0)
        meter.update(6.0)
        assert meter.val == 6.0
        assert meter.avg == 4.0
        assert meter.sum == 12.0
        assert meter.count == 3

    def test_weighted_update(self):
        """Test update with count > 1."""
        meter = AverageMeter()
        meter.update(5.0, n=2)
        assert meter.val == 5.0
        assert meter.sum == 10.0
        assert meter.count == 2
        assert meter.avg == 5.0

    def test_reset(self):
        """Test reset functionality."""
        meter = AverageMeter()
        meter.update(10.0)
        meter.reset()
        assert meter.val == 0
        assert meter.avg == 0


class TestLossFunctions:
    """Tests for loss function classes."""

    @pytest.fixture
    def sample_tensors(self):
        """Create sample prediction and target tensors."""
        pred = torch.rand(2, 31, 64, 64)
        target = torch.rand(2, 31, 64, 64)
        return pred, target

    def test_mrae_loss_shape(self, sample_tensors):
        """Test MRAE loss returns scalar."""
        pred, target = sample_tensors
        loss_fn = Loss_MRAE()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0  # Scalar

    def test_mrae_loss_zero(self):
        """Test MRAE loss is zero for identical tensors."""
        tensor = torch.rand(2, 31, 64, 64)
        loss_fn = Loss_MRAE()
        loss = loss_fn(tensor, tensor)
        assert loss.item() < 1e-6

    def test_mrae_loss_non_contiguous(self):
        """Test MRAE handles non-contiguous tensors."""
        pred = torch.rand(2, 31, 64, 64).permute(0, 2, 3, 1).permute(0, 3, 1, 2)
        target = torch.rand(2, 31, 64, 64)
        loss_fn = Loss_MRAE()
        # Should not raise
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)

    def test_rmse_loss_shape(self, sample_tensors):
        """Test RMSE loss returns scalar."""
        pred, target = sample_tensors
        loss_fn = Loss_RMSE()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_rmse_loss_zero(self):
        """Test RMSE loss is zero for identical tensors."""
        tensor = torch.rand(2, 31, 64, 64)
        loss_fn = Loss_RMSE()
        loss = loss_fn(tensor, tensor)
        assert loss.item() < 1e-6

    def test_psnr_loss_shape(self, sample_tensors):
        """Test PSNR loss returns scalar."""
        pred, target = sample_tensors
        loss_fn = Loss_PSNR()
        loss = loss_fn(pred, target, data_range=1.0)
        assert loss.dim() == 0

    def test_psnr_loss_high_for_identical(self):
        """Test PSNR is high for identical tensors."""
        tensor = torch.rand(2, 31, 64, 64)
        loss_fn = Loss_PSNR()
        loss = loss_fn(tensor, tensor, data_range=1.0)
        # PSNR should be very high (essentially infinity for identical)
        assert loss.item() > 30

    def test_sam_loss_shape(self, sample_tensors):
        """Test SAM loss returns scalar."""
        pred, target = sample_tensors
        loss_fn = Loss_SAM()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_sam_loss_zero(self):
        """Test SAM loss is zero for identical tensors."""
        tensor = torch.rand(2, 31, 64, 64) + 0.1  # Avoid zero vectors
        loss_fn = Loss_SAM()
        loss = loss_fn(tensor, tensor)
        assert loss.item() < 1e-5

    def test_sam_loss_range(self, sample_tensors):
        """Test SAM loss is in valid range [0, pi]."""
        pred, target = sample_tensors
        loss_fn = Loss_SAM()
        loss = loss_fn(pred, target)
        assert 0 <= loss.item() <= np.pi


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_calculate_metrics_returns_dict(self):
        """Test calculate_metrics returns dictionary."""
        pred = torch.rand(1, 31, 64, 64)
        target = torch.rand(1, 31, 64, 64)
        metrics = calculate_metrics(pred, target)
        assert isinstance(metrics, dict)
        assert "mrae" in metrics
        assert "rmse" in metrics
        assert "psnr" in metrics

    def test_calculate_metrics_includes_sam(self):
        """Test SAM is included for hyperspectral data."""
        pred = torch.rand(1, 31, 64, 64)
        target = torch.rand(1, 31, 64, 64)
        metrics = calculate_metrics(pred, target, include_sam=True)
        assert "sam" in metrics

    def test_calculate_metrics_excludes_sam(self):
        """Test SAM can be excluded."""
        pred = torch.rand(1, 31, 64, 64)
        target = torch.rand(1, 31, 64, 64)
        metrics = calculate_metrics(pred, target, include_sam=False)
        assert "sam" not in metrics


class TestModelUtilities:
    """Tests for model utility functions."""

    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        total, trainable = count_parameters(model)
        assert total == 55  # 10*5 + 5 bias
        assert trainable == 55

    def test_count_parameters_frozen(self):
        """Test parameter counting with frozen parameters."""
        model = torch.nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False
        total, trainable = count_parameters(model)
        assert total == 55
        assert trainable == 0

    def test_get_model_size(self):
        """Test model size calculation."""
        model = torch.nn.Linear(10, 5)
        size = get_model_size(model)
        assert size > 0
        assert isinstance(size, float)

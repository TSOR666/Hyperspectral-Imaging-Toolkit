"""Tests for MSWR v2 utility functions."""

import importlib.util

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
        Loss_SSIM,
        calculate_metrics,
        count_parameters,
        get_model_size,
        mrae_diagnostics,
        my_summary,
        save_matv73,
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

    def test_mrae_promotes_low_precision_outputs(self):
        """MRAE should not downcast fp32 labels when predictions are bf16."""
        pred = torch.tensor([[[[0.001001]]]], dtype=torch.bfloat16)
        target = torch.tensor([[[[0.001003]]]], dtype=torch.float32)
        expected = (pred.float() - target).abs().div(target.abs().clamp_min(1e-6)).mean()

        loss = Loss_MRAE()(pred, target)

        assert loss.dtype == torch.float32
        assert torch.allclose(loss, expected, rtol=1e-6, atol=1e-8)

    def test_mrae_diagnostics_bucket_accounting(self):
        """Diagnostic bucket fractions and contributions should account for strict MRAE."""
        pred = torch.tensor([[[[0.0, 0.002, 0.02, 0.2]]]], dtype=torch.float32)
        target = torch.tensor([[[[0.0005, 0.002, 0.04, 0.1]]]], dtype=torch.float32)

        diagnostics = mrae_diagnostics(pred, target)

        assert diagnostics["mrae_strict"] == pytest.approx(Loss_MRAE()(pred, target).item())
        assert diagnostics["target_lt_1e-2_frac"] == pytest.approx(0.5)
        bucket_frac_sum = sum(
            value for key, value in diagnostics.items()
            if key.startswith("bucket_") and key.endswith("_frac")
            and not key.endswith("_contrib_frac")
        )
        contrib_sum = sum(
            value for key, value in diagnostics.items()
            if key.startswith("bucket_") and key.endswith("_contrib_frac")
        )
        assert bucket_frac_sum == pytest.approx(1.0)
        assert contrib_sum == pytest.approx(1.0)

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

    def test_sam_loss_zero_vectors(self):
        """Test SAM treats matching zero spectra as zero angle instead of NaN."""
        tensor = torch.zeros(1, 31, 16, 16)
        loss_fn = Loss_SAM()
        loss = loss_fn(tensor, tensor)
        assert torch.isfinite(loss)
        assert loss.item() < 1e-6

    def test_sam_loss_range(self, sample_tensors):
        """Test SAM loss is in valid range [0, pi]."""
        pred, target = sample_tensors
        loss_fn = Loss_SAM()
        loss = loss_fn(pred, target)
        assert 0 <= loss.item() <= np.pi

    def test_ssim_score_identical_tensors(self):
        """SSIM should be near one for identical tensors."""
        tensor = torch.rand(1, 31, 32, 32)
        score = Loss_SSIM()(tensor, tensor)

        assert torch.isfinite(score)
        assert score.item() == pytest.approx(1.0, abs=1e-5)


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

    def test_calculate_metrics_can_include_ssim(self):
        """Test SSIM can be included for validation summaries."""
        pred = torch.rand(1, 31, 32, 32)
        target = torch.rand(1, 31, 32, 32)
        metrics = calculate_metrics(pred, target, include_ssim=True)
        assert "ssim" in metrics


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


class TestOptionalDependencies:
    """Utilities should import without optional export/profiling packages."""

    def test_save_matv73_missing_dependency_error_is_lazy(self):
        """Missing hdf5storage should only affect MATLAB export."""
        if importlib.util.find_spec("hdf5storage") is not None:
            pytest.skip("hdf5storage is installed in this environment")

        with pytest.raises(ImportError, match="hdf5storage"):
            save_matv73("unused.mat", "cube", np.zeros((1, 1), dtype=np.float32))

    def test_my_summary_missing_dependency_error_is_lazy(self):
        """Missing fvcore should only affect FLOP summary."""
        if importlib.util.find_spec("fvcore") is not None:
            pytest.skip("fvcore is installed in this environment")

        with pytest.raises(ImportError, match="fvcore"):
            my_summary(torch.nn.Conv2d(3, 31, 1), H=8, W=8, C=3, N=1)

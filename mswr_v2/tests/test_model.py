"""Tests for MSWR v2 model architecture."""

import pytest
import torch

# Conditional import to handle missing dependencies
try:
    from model.mswr_net_v212 import (
        MSWRDualConfig,
        create_mswr_tiny,
        create_mswr_small,
        create_mswr_base,
    )
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    MSWRDualConfig = None
    create_mswr_tiny = None
    create_mswr_small = None
    create_mswr_base = None

pytestmark = pytest.mark.skipif(not MODEL_AVAILABLE, reason="model module dependencies not available")

# Skip if CUDA not available for GPU-specific tests
cuda_available = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TestModelConfiguration:
    """Tests for model configuration."""

    def test_config_validation(self):
        """Test that invalid config raises ValueError."""
        # base_channels not divisible by num_heads should fail
        with pytest.raises(ValueError):
            MSWRDualConfig(base_channels=65, num_heads=8)

    def test_valid_config(self):
        """Test that valid config is accepted."""
        config = MSWRDualConfig(
            base_channels=64,
            num_heads=8,
            num_stages=3,
            use_wavelet=True,
            wavelet_type="db2",
        )
        assert config.base_channels == 64
        assert config.num_heads == 8


class TestModelCreation:
    """Tests for model factory functions."""

    def test_create_tiny(self):
        """Test tiny model creation."""
        model = create_mswr_tiny()
        assert model is not None

    def test_create_small(self):
        """Test small model creation."""
        model = create_mswr_small()
        assert model is not None

    def test_create_base(self):
        """Test base model creation."""
        model = create_mswr_base()
        assert model is not None

    def test_model_with_wavelets(self):
        """Test model with wavelets enabled."""
        model = create_mswr_tiny(use_wavelet=True, wavelet_type="db2")
        assert model is not None


class TestModelForward:
    """Tests for model forward pass."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        model = create_mswr_tiny()
        model.eval()

        # Input: (batch, 3, H, W) RGB
        x = torch.randn(1, 3, 128, 128)

        with torch.no_grad():
            output = model(x)

        # Output: (batch, 31, H, W) HSI
        assert output.shape == (1, 31, 128, 128)

    def test_forward_batch(self):
        """Test forward with batch > 1."""
        model = create_mswr_tiny()
        model.eval()

        x = torch.randn(4, 3, 128, 128)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 31, 128, 128)

    def test_forward_no_nan(self):
        """Test that forward doesn't produce NaN."""
        model = create_mswr_tiny()
        model.eval()

        x = torch.randn(1, 3, 128, 128)

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_different_sizes(self):
        """Test forward with different input sizes."""
        model = create_mswr_tiny()
        model.eval()

        # Test various sizes (should be divisible by model stride)
        sizes = [(64, 64), (128, 128), (256, 256)]

        for h, w in sizes:
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 31, h, w), f"Failed for size {h}x{w}"


class TestModelGradients:
    """Tests for model gradient computation."""

    def test_backward_pass(self):
        """Test that backward pass works."""
        model = create_mswr_tiny(use_checkpoint=False)  # Disable checkpointing for simpler gradient test
        model.train()

        # Use smaller tensors to reduce memory usage on CPU
        x = torch.randn(1, 3, 64, 64)
        target = torch.randn(1, 31, 64, 64)

        output = model(x)
        loss = torch.nn.functional.l1_loss(output, target)
        loss.backward()

        # Check that gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break

        assert has_gradients

    def test_gradient_no_nan(self):
        """Test that gradients don't contain NaN."""
        model = create_mswr_tiny(use_checkpoint=False)  # Disable checkpointing for simpler gradient test
        model.train()

        # Use smaller tensors to reduce memory usage on CPU
        x = torch.randn(1, 3, 64, 64)
        target = torch.randn(1, 31, 64, 64)

        output = model(x)
        loss = torch.nn.functional.l1_loss(output, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN in gradient: {name}"


class TestModelInfo:
    """Tests for model information methods."""

    def test_get_model_info(self):
        """Test model info retrieval."""
        model = create_mswr_tiny()
        info = model.get_model_info()

        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "total_memory_mb" in info
        assert info["total_parameters"] > 0

    def test_parameter_count(self):
        """Test parameter counting."""
        model = create_mswr_tiny()
        info = model.get_model_info()

        # Manually count
        manual_count = sum(p.numel() for p in model.parameters())
        assert info["total_parameters"] == manual_count


@cuda_available
class TestModelCUDA:
    """Tests for model on CUDA (GPU)."""

    def test_forward_cuda(self):
        """Test forward pass on CUDA."""
        model = create_mswr_tiny().cuda()
        model.eval()

        x = torch.randn(1, 3, 128, 128).cuda()

        with torch.no_grad():
            output = model(x)

        assert output.device.type == "cuda"
        assert output.shape == (1, 31, 128, 128)

    def test_backward_cuda(self):
        """Test backward pass on CUDA."""
        model = create_mswr_tiny().cuda()
        model.train()

        x = torch.randn(1, 3, 128, 128).cuda()
        target = torch.randn(1, 31, 128, 128).cuda()

        output = model(x)
        loss = torch.nn.functional.l1_loss(output, target)
        loss.backward()

        assert True  # If we get here, backward worked

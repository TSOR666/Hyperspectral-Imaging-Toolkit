import pytest
import torch

from hsi_model.models.discriminator_v2 import (
    DiscriminatorWithSinkhorn,
    SNTransformerDiscriminator,
    compute_gradient_penalty,
)
from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator


def test_generator_forward_shape_and_iteration():
    config = {
        "in_channels": 3,
        "out_channels": 31,
        "base_channels": 16,
        "split_sizes": [2, 2, 2],
        "num_heads": 4,
        "norm_groups": 4,
        "output_activation": "none",
    }
    gen = NoiseRobustCSWinGenerator(config)
    gen.train()
    x = torch.randn(2, 3, 8, 8)
    out = gen(x)
    assert out.shape == (2, 31, 8, 8)
    assert gen._iteration_count == 1


def test_discriminator_with_sinkhorn_outputs():
    config = {
        "discriminator_base_dim": 8,
        "discriminator_num_heads": 2,
        "discriminator_num_blocks": [1, 1, 1],
    }
    disc = DiscriminatorWithSinkhorn(config)
    rgb = torch.randn(2, 3, 16, 16)
    hsi = torch.randn(2, 31, 16, 16)
    output, features = disc(rgb, hsi, return_features=True)
    assert output.shape[0] == 2
    assert output.shape[1] == 1
    assert features.shape[0] == 2
    assert features.shape[1] == output.shape[2] * output.shape[3]


def test_gradient_penalty_scalar():
    config = {
        "discriminator_base_dim": 8,
        "discriminator_num_heads": 2,
        "discriminator_num_blocks": [1, 1, 1],
    }
    disc = SNTransformerDiscriminator(config)
    real_rgb = torch.randn(2, 3, 16, 16)
    real_hsi = torch.randn(2, 31, 16, 16)
    fake_hsi = torch.randn(2, 31, 16, 16)
    penalty = compute_gradient_penalty(disc, real_rgb, real_hsi, fake_hsi)
    assert penalty.ndim == 0
    assert torch.isfinite(penalty)


# GATE 4.1: Gradient Tests
def test_generator_gradients_flow():
    """Test that gradients flow correctly through the generator (Finding 4.1)."""
    config = {
        "in_channels": 3,
        "out_channels": 31,
        "base_channels": 16,
        "split_sizes": [2, 2],
        "num_heads": 2,
        "norm_groups": 4,
        "output_activation": "none",
    }
    gen = NoiseRobustCSWinGenerator(config).double()  # Use fp64 for gradcheck
    x = torch.randn(1, 3, 8, 8, dtype=torch.float64, requires_grad=True)

    # Gradcheck is expensive, so test on small input
    # We test if generator is differentiable
    def gen_fn(inp):
        return gen(inp).sum()

    # Use numerical gradcheck
    assert torch.autograd.gradcheck(gen_fn, x, eps=1e-6, atol=1e-4), "Generator gradient check failed"


def test_discriminator_gradients_flow():
    """Test that gradients flow correctly through the discriminator (Finding 4.1)."""
    config = {
        "discriminator_base_dim": 8,
        "discriminator_num_heads": 2,
        "discriminator_num_blocks": [1, 1],
    }
    disc = SNTransformerDiscriminator(config).double()
    rgb = torch.randn(1, 3, 16, 16, dtype=torch.float64, requires_grad=True)
    hsi = torch.randn(1, 31, 16, 16, dtype=torch.float64, requires_grad=True)

    def disc_fn(r, h):
        return disc(r, h).sum()

    # Test discriminator gradients
    assert torch.autograd.gradcheck(
        disc_fn, (rgb, hsi), eps=1e-6, atol=1e-4
    ), "Discriminator gradient check failed"


# GATE 4.4: Determinism Test
def test_generator_determinism():
    """Test that same seed produces same outputs (Finding 4.4)."""
    config = {
        "in_channels": 3,
        "out_channels": 31,
        "base_channels": 16,
        "split_sizes": [2, 2],
        "num_heads": 2,
        "norm_groups": 4,
        "output_activation": "none",
    }

    # Run 1
    torch.manual_seed(42)
    gen1 = NoiseRobustCSWinGenerator(config)
    gen1.eval()
    torch.manual_seed(42)
    x1 = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        out1 = gen1(x1)

    # Run 2 (same seed)
    torch.manual_seed(42)
    gen2 = NoiseRobustCSWinGenerator(config)
    gen2.eval()
    torch.manual_seed(42)
    x2 = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        out2 = gen2(x2)

    assert torch.equal(x1, x2), "Input mismatch"
    assert torch.equal(out1, out2), f"Output mismatch: max diff = {(out1 - out2).abs().max()}"

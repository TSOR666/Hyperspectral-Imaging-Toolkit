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

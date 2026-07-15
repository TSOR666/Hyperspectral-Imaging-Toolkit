"""Gradient validity, identity behaviour, and that the thing can actually learn (spec 4.8)."""

from __future__ import annotations

import pytest
import torch

from cas_hsi import build_cas_hsi
from cas_hsi.blocks import CASBlock, CASLiteBlock
from cas_hsi.config import Depths, CASHSIConfig


def _blocks():
    return [
        CASBlock(channels=64, spatial_mixer="dilated_local_attention", head_dim=32, dilations=(1, 2)),
        CASBlock(
            channels=128, spatial_mixer="hybrid_local_stripe_attention",
            head_dim=32, dilations=(1, 2, 3),
        ),
        CASBlock(channels=64, spatial_mixer="dilated_depthwise_conv", head_dim=32, dilations=(1, 2)),
        CASLiteBlock(channels=64, head_dim=32),
    ]


@pytest.mark.parametrize("block", _blocks(), ids=lambda b: f"{type(b).__name__}-{b.channels}")
def test_input_gradients_are_finite(block):
    x = torch.randn(2, block.channels, 32, 32, requires_grad=True)
    block(x).square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("block", _blocks(), ids=lambda b: f"{type(b).__name__}-{b.channels}")
def test_every_parameter_gets_a_finite_gradient(block):
    x = torch.randn(2, block.channels, 24, 32)
    block(x).square().mean().backward()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"{name} received no gradient (it is dead weight)"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"


@pytest.mark.parametrize(
    "mixer,channels",
    [
        ("dilated_local_attention", 64),
        ("hybrid_local_stripe_attention", 128),
        ("dilated_depthwise_conv", 64),
    ],
)
def test_cas_block_is_identity_at_zero_layer_scale(mixer, channels):
    """Spec 4.8: with every LayerScale at zero, CAS(X) == X exactly.

    This is what guarantees the identity path is never scaled -- if any branch leaked
    into the residual stream unscaled, this would fail.
    """
    block = CASBlock(
        channels=channels, spatial_mixer=mixer, head_dim=32,
        dilations=(1, 2, 3), layer_scale_init=0.0,
    )
    x = torch.randn(2, channels, 32, 48)
    assert torch.allclose(x, block(x), atol=1e-6)


def test_cas_lite_block_is_identity_at_zero_layer_scale():
    block = CASLiteBlock(channels=64, head_dim=32, layer_scale_init=0.0)
    x = torch.randn(2, 64, 32, 48)
    assert torch.allclose(x, block(x), atol=1e-6)


def test_untrained_model_is_close_to_its_linear_prior():
    """Spec 7.3: the near-zero residual head makes the fresh model ~= the linear prior.

    This is the property a blanket weight re-init would silently destroy.
    """
    torch.manual_seed(0)
    model = build_cas_hsi("tiny").eval()
    rgb = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        output = model(rgb)
        prior = model.rgb_prior(rgb)

    residual = (output - prior).abs().mean()
    prior_scale = prior.abs().mean()
    assert residual < 0.05 * prior_scale, (
        f"the deep residual ({residual:.4f}) is not small next to the prior "
        f"({prior_scale:.4f}); the near-zero head init was lost"
    )


def test_model_gradients_are_finite_end_to_end():
    model = build_cas_hsi(
        CASHSIConfig(
            base_width=32,
            depths=Depths(encoder_full=1, encoder_half=1, bottleneck=3, decoder_half=1,
                          decoder_full=1, refinement=1),
        )
    )
    rgb = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 31, 32, 32)

    loss = (model(rgb) - target).abs().mean()
    loss.backward()

    assert torch.isfinite(rgb.grad).all()
    dead = [name for name, p in model.named_parameters() if p.grad is None]
    assert not dead, f"parameters with no gradient path: {dead}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_model_can_actually_learn():
    """Overfit a single fixed batch. A model that runs but cannot learn passes every
    shape and gradient test above -- this is the one that would catch it."""
    torch.manual_seed(0)
    model = build_cas_hsi(
        CASHSIConfig(
            base_width=32,
            depths=Depths(encoder_full=1, encoder_half=1, bottleneck=2, decoder_half=1,
                          decoder_full=1, refinement=1),
        )
    ).train()

    rgb = torch.rand(2, 3, 32, 32)
    # A learnable target: a smooth linear function of the input, not noise.
    weight = torch.randn(31, 3, 1, 1) * 0.3
    target = torch.nn.functional.conv2d(rgb, weight).sigmoid()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    def mrae() -> float:
        prediction = model(rgb)
        loss = ((prediction - target).abs() / target.abs().clamp_min(1e-6)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    first = mrae()
    for _ in range(30):
        last = mrae()

    assert last < 0.5 * first, f"MRAE barely moved: {first:.4f} -> {last:.4f}"


def test_drop_path_is_identity_in_eval():
    """Spec 9.8 requires stochastic depth to be off before export; eval() must suffice."""
    block = CASBlock(
        channels=64, spatial_mixer="dilated_local_attention", head_dim=32, drop_path=0.5
    ).eval()
    x = torch.randn(2, 64, 16, 16)
    with torch.no_grad():
        a, b = block(x), block(x)
    assert torch.equal(a, b), "eval-mode forward is still stochastic"

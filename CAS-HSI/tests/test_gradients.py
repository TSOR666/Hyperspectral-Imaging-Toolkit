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


def test_untrained_model_starts_near_but_not_collapsed_onto_its_linear_prior():
    """Spec 7.3: the near-zero residual head makes the fresh model START near the prior.

    Two-sided on purpose. The upper bound is the original contract (a blanket weight
    re-init would destroy it). The LOWER bound is the one that was missing, and its
    absence cost a 46,000-step training run: with layer_scale_init=1e-3 the residual
    branch is gated to ~1% of the prior AND the trunk is affine to 4e-5, so the network
    is a linear Conv1x1 RGB->31 map whose ARAD-1K MRAE floor is ~0.6-0.7. That state
    passes every shape, gradient and "loss went down" test in this suite.
    """
    torch.manual_seed(0)
    model = build_cas_hsi("tiny").eval()
    rgb = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        output = model(rgb)
        prior = model.rgb_prior(rgb)

    residual = (output - prior).abs().mean()
    prior_scale = prior.abs().mean()
    ratio = float(residual / prior_scale)

    assert ratio < 0.35, (
        f"the deep residual ({residual:.4f}) is not small next to the prior "
        f"({prior_scale:.4f}); the near-zero head init was lost"
    )
    assert ratio > 0.005, (
        f"the deep residual is only {ratio:.2%} of the prior: the trunk is gated off at "
        "init and the model is effectively a linear RGB->31 map. Check layer_scale_init "
        "(1e-3 over-damps this depth; 1.0 is the validated value) and head_init_std."
    )


def test_untrained_model_is_not_an_affine_function_of_its_input():
    """The trunk must supply real nonlinearity at init, not just in principle.

    Measures deviation from additivity: f(a+b) vs f(a)+f(b)-f(0). An exactly linear
    network scores 0. Measured on the tiny/research model: 4e-5 at layer_scale_init=1e-3
    versus 3.4e-2 at 1.0 -- an 850x difference that the parameter count, the loss curve
    and every other test in this file are all blind to.
    """
    torch.manual_seed(0)
    model = build_cas_hsi("tiny").eval()
    a = torch.rand(1, 3, 32, 32)
    b = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        zero = model(torch.zeros_like(a))
        joint = model(a + b) - zero
        split = (model(a) - zero) + (model(b) - zero)

    deviation = float((joint - split).norm() / joint.norm().clamp_min(1e-12))
    assert deviation > 1e-3, (
        f"the untrained network is affine in its input to {deviation:.2e}; it cannot "
        "resolve metamers or use spatial context, and will plateau at the MRAE of a "
        "linear colour->spectrum map. Raise layer_scale_init."
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
    shape and gradient test above -- this is the one that would catch it.

    The target is deliberately NOT a function of RGB alone. It is a spatially-mixed
    transform, so the Conv1x1 rgb_prior cannot fit it and progress here is evidence the
    TRUNK is learning. The previous version of this test used a per-pixel linear target,
    which the prior alone can fit: it passed with a completely dead trunk, which is
    exactly the failure that reached a GPU.
    """
    torch.manual_seed(0)
    model = build_cas_hsi(
        CASHSIConfig(
            base_width=32,
            depths=Depths(encoder_full=1, encoder_half=1, bottleneck=2, decoder_half=1,
                          decoder_full=1, refinement=1),
        )
    ).train()

    rgb = torch.rand(2, 3, 32, 32)
    weight = torch.randn(31, 3, 5, 5) * 0.1
    target = torch.nn.functional.conv2d(rgb, weight, padding=2).sigmoid()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    def mrae() -> float:
        prediction = model(rgb)
        loss = ((prediction - target).abs() / target.abs().clamp_min(1e-6)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    first = mrae()
    for _ in range(60):
        last = mrae()

    assert last < 0.5 * first, f"MRAE barely moved: {first:.4f} -> {last:.4f}"

    # And the trunk -- not just the linear prior -- must be carrying some of it.
    with torch.no_grad():
        from cas_hsi.layers.padding import pad_to_multiple

        padded, _ = pad_to_multiple(rgb, multiple=model.config.size_multiple)
        features, _ = model._forward_padded_features(padded)
        share = float(
            model.spectral_head(features).norm() / model.rgb_prior(padded).norm().clamp_min(1e-12)
        )
    assert share > 0.05, (
        f"the deep residual is {share:.2%} of the linear prior after training: the loss "
        "fell but the trunk is dead, so this only fit what a Conv1x1 can fit"
    )


def test_drop_path_is_identity_in_eval():
    """Spec 9.8 requires stochastic depth to be off before export; eval() must suffice."""
    block = CASBlock(
        channels=64, spatial_mixer="dilated_local_attention", head_dim=32, drop_path=0.5
    ).eval()
    x = torch.randn(2, 64, 16, 16)
    with torch.no_grad():
        a, b = block(x), block(x)
    assert torch.equal(a, b), "eval-mode forward is still stochastic"

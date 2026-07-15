"""Shape contracts (specification 4.8, 6.7, 7.7) and head allocation (4.5.2)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cas_hsi.blocks import (
    CASBlock,
    CASLiteBlock,
    CrossChannelAttention,
    GatedConvFFN,
    MultiDilationDepthwiseMixer,
    allocate_head_groups,
)
from cas_hsi.layers import PixelShuffleUpsample, PixelUnshuffleDownsample


# --------------------------------------------------------------------- blocks --


@pytest.mark.parametrize("shape", [(1, 64, 32, 32), (2, 64, 31, 47), (1, 128, 17, 29)])
def test_cas_block_preserves_shape(shape):
    """Spec 4.8. Odd, non-square, non-power-of-two sizes must all survive."""
    batch, channels, height, width = shape
    block = CASBlock(
        channels=channels,
        spatial_mixer="dilated_local_attention",
        head_dim=32,
        dilations=(1, 2),
    )
    x = torch.randn(*shape)
    assert block(x).shape == x.shape


@pytest.mark.parametrize("shape", [(1, 128, 16, 16), (2, 128, 21, 13)])
def test_hybrid_stripe_block_preserves_shape(shape):
    """Spec 4.5.4: stripe attention must not require divisibility by the stripe width."""
    block = CASBlock(
        channels=shape[1],
        spatial_mixer="hybrid_local_stripe_attention",
        head_dim=32,
        dilations=(1, 2, 3),
        stripe_width=8,
    )
    x = torch.randn(*shape)
    assert block(x).shape == x.shape


@pytest.mark.parametrize("shape", [(1, 32, 32, 32), (2, 64, 31, 47)])
def test_cas_lite_block_preserves_shape(shape):
    block = CASLiteBlock(channels=shape[1], head_dim=32)
    x = torch.randn(*shape)
    assert block(x).shape == x.shape


def test_cas_lite_uses_no_unfold(small_model):
    """Spec 5.5: CAS-Lite must not unfold. Assert no attention mixer at full resolution."""
    from cas_hsi.blocks import DilatedLocalAttention, HybridLocalStripeAttention

    for name in ("encoder_full", "decoder_full", "refinement"):
        stack = getattr(small_model, name)
        offenders = [
            module
            for module in stack.modules()
            if isinstance(module, (DilatedLocalAttention, HybridLocalStripeAttention))
        ]
        assert not offenders, f"{name} contains spatial attention: {offenders}"


def test_gated_ffn_shape_and_hidden_width():
    ffn = GatedConvFFN(channels=64, expansion=2.0)
    assert ffn.hidden == 128
    x = torch.randn(2, 64, 15, 23)
    assert ffn(x).shape == x.shape


def test_gated_ffn_rejects_degenerate_expansion():
    with pytest.raises(ValueError):
        GatedConvFFN(channels=64, expansion=0.0)


def test_cross_channel_attention_shape_and_head_split():
    attention = CrossChannelAttention(channels=128, head_dim=32)
    assert attention.num_heads == 4
    x = torch.randn(2, 128, 9, 11)
    assert attention(x).shape == x.shape


def test_cross_channel_attention_rejects_indivisible_head_dim():
    """This is the exact contradiction in the spec's own CAS-HSI-Base config."""
    with pytest.raises(ValueError):
        CrossChannelAttention(channels=48, head_dim=32)


def test_cross_channel_attention_is_not_a_pooled_gate():
    """Spec 4.6.1: it must aggregate over ALL spatial locations, not pool to a scalar.

    Changing one pixel must move the output at *other* pixels -- an SE-style pooled
    sigmoid gate would too, but a purely per-pixel op would not. This pins the
    "attention over the channel axis with spatial aggregation" property: the attention
    matrix depends on the whole feature map.
    """
    torch.manual_seed(0)
    attention = CrossChannelAttention(channels=64, head_dim=32).eval()
    x = torch.randn(1, 64, 8, 8)

    with torch.no_grad():
        base = attention(x)
        perturbed = x.clone()
        perturbed[0, :, 0, 0] += 5.0
        moved = attention(perturbed)

    # A far-away pixel, outside the 3x3 depthwise footprint of the perturbation.
    far = (moved[0, :, 6, 6] - base[0, :, 6, 6]).abs().max()
    assert far > 1e-6, "output at a distant pixel did not react: attention is not global"


# ---------------------------------------------------------------- down / up ----


@pytest.mark.parametrize("height,width", [(32, 32), (48, 64), (128, 96)])
def test_down_up_roundtrip_shape(height, width):
    """Spec 6.7."""
    down = PixelUnshuffleDownsample(32, 64)
    up = PixelShuffleUpsample(64, 32)
    x = torch.randn(1, 32, height, width)
    assert up(down(x)).shape == x.shape


def test_downsample_rejects_odd_input():
    down = PixelUnshuffleDownsample(32, 64)
    with pytest.raises(ValueError, match="divisible by 2"):
        down(torch.randn(1, 32, 31, 32))


# ------------------------------------------------------------ head allocation --


def test_head_allocation_sums_to_num_heads():
    """Spec 4.5.2: 'The number of heads assigned to each group must sum to the total'."""
    for num_heads in range(1, 17):
        for dilations in [(1,), (1, 2), (1, 2, 3)]:
            groups = allocate_head_groups(num_heads, dilations, use_stripe=False)
            assert sum(g.heads for g in groups) == num_heads
            if num_heads >= 3:
                hybrid = allocate_head_groups(num_heads, dilations, use_stripe=True)
                assert sum(g.heads for g in hybrid) == num_heads


def test_head_allocation_matches_spec_worked_example():
    """Spec 4.5.2, C=128 d_h=32 -> 4 heads over dilations (1,2,3): heads 0/1/2/3 -> 1/2/3/3."""
    groups = allocate_head_groups(4, (1, 2, 3), use_stripe=False)
    assert [(g.kind, g.heads, g.dilation) for g in groups] == [
        ("local", 1, 1),
        ("local", 1, 2),
        ("local", 2, 3),
    ]


def test_hybrid_head_allocation_matches_spec_worked_example():
    """Spec 4.5.2 hybrid: 4 heads -> dilation 1, dilation 2, horizontal, vertical."""
    groups = allocate_head_groups(4, (1, 2, 3), use_stripe=True)
    assert [(g.kind, g.heads, g.dilation, g.orientation) for g in groups] == [
        ("local", 1, 1, "horizontal"),   # orientation is unused for local groups
        ("local", 1, 2, "horizontal"),
        ("stripe", 1, 1, "horizontal"),
        ("stripe", 1, 1, "vertical"),
    ]


def test_hybrid_requires_enough_heads():
    with pytest.raises(ValueError, match="at least 3 heads"):
        allocate_head_groups(2, (1, 2), use_stripe=True)


def test_multi_dilation_mixer_handles_indivisible_channels():
    """The spec's own Tiny config hits this: 128 channels, 3 dilations, 128 % 3 != 0."""
    mixer = MultiDilationDepthwiseMixer(channels=128, dilations=(1, 2, 3))
    assert sum(mixer.group_channels) == 128
    x = torch.randn(1, 128, 9, 11)
    assert mixer(x).shape == x.shape

    with pytest.raises(ValueError, match="divisible"):
        MultiDilationDepthwiseMixer(channels=128, dilations=(1, 2, 3), strict_split=True)


# ------------------------------------------------------------------- model -----


def test_model_output_shape(small_model):
    """Spec 7.7."""
    out = small_model(torch.randn(2, 3, 127, 193))
    assert out.shape == (2, 31, 127, 193)


def test_no_batchnorm_anywhere(tiny_model):
    """Definition of Done: 'the model has no BatchNorm layers'."""
    offenders = [
        name
        for name, module in tiny_model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    assert not offenders, f"BatchNorm found at {offenders}"


def test_latent_stays_wider_than_output_bands(tiny_model):
    """Definition of Done: latent features stay wider than 31 channels until the head."""
    _, features, _, _ = tiny_model.forward_features(torch.randn(1, 3, 32, 32))
    for name, tensor in features.items():
        assert tensor.shape[1] > tiny_model.config.output_bands, (
            f"{name} has {tensor.shape[1]} channels, not wider than "
            f"{tiny_model.config.output_bands}"
        )


def test_no_output_activation(small_model):
    """Spec 7.4: negative values must remain representable after the 31-band projection."""
    torch.manual_seed(1)
    # Drive the prior negative: a sigmoid/ReLU/tanh head would clip or squash this.
    small_model.rgb_prior.projection.bias.data.fill_(-5.0)
    with torch.no_grad():
        out = small_model(torch.zeros(1, 3, 16, 16))
    assert (out < 0).any(), "output cannot represent negative values: an activation was applied"
    small_model.rgb_prior.projection.bias.data.zero_()

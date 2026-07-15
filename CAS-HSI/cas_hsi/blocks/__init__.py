"""CAS and CAS-Lite blocks and their interchangeable spatial mixers."""

from .cas_block import CASBlock, build_bottleneck
from .cas_lite_block import CASLiteBlock
from .channel_attention import CrossChannelAttention
from .gated_ffn import GatedConvFFN
from .spatial_attention import (
    ATTENTION_MIXERS,
    CONV_MIXERS,
    ConvSpatialMixer,
    DilatedLocalAttention,
    HeadGroup,
    HybridLocalStripeAttention,
    MultiDilationDepthwiseMixer,
    ReparamConvSpatialMixer,
    allocate_head_groups,
    build_spatial_mixer,
    local_attention_reference,
    neighborhood_attention,
    stripe_attention,
    unfold_heads,
)

__all__ = [
    "ATTENTION_MIXERS",
    "CASBlock",
    "CASLiteBlock",
    "CONV_MIXERS",
    "ConvSpatialMixer",
    "CrossChannelAttention",
    "DilatedLocalAttention",
    "GatedConvFFN",
    "HeadGroup",
    "HybridLocalStripeAttention",
    "MultiDilationDepthwiseMixer",
    "ReparamConvSpatialMixer",
    "allocate_head_groups",
    "build_bottleneck",
    "build_spatial_mixer",
    "local_attention_reference",
    "neighborhood_attention",
    "stripe_attention",
    "unfold_heads",
]

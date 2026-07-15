"""CAS-Lite block for full-resolution processing (specification section 5).

Identical shell to :class:`~cas_hsi.blocks.cas_block.CASBlock`, but the spatial
mixer is a large-kernel depthwise convolution instead of explicit attention.

At full resolution H x W, neighborhood attention is memory-bound (its activation
scales with H*W*K) and needs an operator most runtimes do not have.  A 7x7
depthwise kernel buys a comparable per-layer receptive field for a fraction of
the memory and exports everywhere -- while cross-channel attention, whose cost is
independent of H*W, is *kept*, so the block still models global spectral
structure at full resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.layer_scale import DropPath, LayerScale
from ..layers.normalization import build_norm
from .channel_attention import CrossChannelAttention
from .gated_ffn import GatedConvFFN
from .spatial_attention import ConvSpatialMixer, ReparamConvSpatialMixer

__all__ = ["CASLiteBlock"]


class CASLiteBlock(nn.Module):
    """Conv spatial mixer + cross-channel attention + gated FFN."""

    def __init__(
        self,
        channels: int,
        head_dim: int = 32,
        ffn_expansion: float = 2.0,
        layer_scale_init: float = 1e-3,
        spatial_kernel: int = 7,
        drop_path: float = 0.0,
        *,
        norm: str = "layernorm",
        norm_eps: float = 1e-6,
        reparam: bool = False,
        reparam_kernels: tuple[int, ...] = (3, 5, 7),
        fp32_attention: bool = True,
        ffn_use_activation: bool = False,
        ffn_gate: str = "simple",
        softplus_temperature: bool = False,
    ) -> None:
        super().__init__()

        self.channels = int(channels)

        self.norm_spatial = build_norm(norm, channels, eps=norm_eps)
        self.norm_channel = build_norm(norm, channels, eps=norm_eps)
        self.norm_ffn = build_norm(norm, channels, eps=norm_eps)

        if reparam:
            self.spatial_mixer: nn.Module = ReparamConvSpatialMixer(
                channels, kernel_sizes=reparam_kernels
            )
        else:
            self.spatial_mixer = ConvSpatialMixer(channels, kernel_size=spatial_kernel)

        self.channel_attention = CrossChannelAttention(
            channels,
            head_dim=head_dim,
            softplus_temperature=softplus_temperature,
            fp32_attention=fp32_attention,
        )

        self.ffn = GatedConvFFN(
            channels,
            expansion=ffn_expansion,
            use_activation=ffn_use_activation,
            gate=ffn_gate,
        )

        self.gamma_spatial = LayerScale(channels, layer_scale_init)
        self.gamma_channel = LayerScale(channels, layer_scale_init)
        self.gamma_ffn = LayerScale(channels, layer_scale_init)

        # Identity at drop_path=0, which is the default for CAS-Lite: the spec's
        # reference block has no stochastic depth, and this preserves it exactly.
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(
            self.gamma_spatial(self.spatial_mixer(self.norm_spatial(x)))
        )
        x = x + self.drop_path(
            self.gamma_channel(self.channel_attention(self.norm_channel(x)))
        )
        x = x + self.drop_path(
            self.gamma_ffn(self.ffn(self.norm_ffn(x)))
        )
        return x

    def extra_repr(self) -> str:
        return f"channels={self.channels}"

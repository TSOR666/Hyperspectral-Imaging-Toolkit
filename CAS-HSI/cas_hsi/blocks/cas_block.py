"""The Convolutional Attention Stack block (specification section 4).

Three pre-normalized residual branches, each gated by its own LayerScale:

    X1 = X  + gamma_s * S(Norm(X))     spatial mixer
    X2 = X1 + gamma_c * C(Norm(X1))    cross-channel attention
    Y  = X2 + gamma_f * F(Norm(X2))    gated conv FFN

Each branch answers a different question -- *where* to look, *which spectra* to
mix, and *how* to transform -- and separating them is what lets the same shell
host either an attention or a convolutional spatial mixer without touching the
rest of the network (spec 9.2).

The identity path is never scaled, so with ``layer_scale_init=0`` the block is an
exact identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn as nn

from ..layers.layer_scale import DropPath, LayerScale
from ..layers.normalization import build_norm
from .channel_attention import CrossChannelAttention
from .gated_ffn import GatedConvFFN
from .spatial_attention import build_spatial_mixer

if TYPE_CHECKING:  # pragma: no cover
    from ..config import CASHSIConfig

__all__ = ["CASBlock", "build_bottleneck"]


class CASBlock(nn.Module):
    """Spatial mixer + cross-channel attention + gated FFN, pre-norm, LayerScale-gated."""

    def __init__(
        self,
        channels: int,
        spatial_mixer: str,
        head_dim: int = 32,
        dilations: Sequence[int] = (1, 2),
        ffn_expansion: float = 2.0,
        layer_scale_init: float = 1e-3,
        drop_path: float = 0.0,
        *,
        norm: str = "layernorm",
        norm_eps: float = 1e-6,
        kernel_size: int = 3,
        spatial_kernel: int = 7,
        large_kernel: int = 11,
        stripe_width: int = 8,
        relative_position_bias: bool = True,
        mask_padding: bool = True,
        fp32_attention: bool = True,
        ffn_use_activation: bool = False,
        ffn_gate: str = "simple",
        softplus_temperature: bool = False,
    ) -> None:
        super().__init__()

        self.channels = int(channels)
        self.spatial_mixer_name = str(spatial_mixer)

        self.norm_spatial = build_norm(norm, channels, eps=norm_eps)
        self.norm_channel = build_norm(norm, channels, eps=norm_eps)
        self.norm_ffn = build_norm(norm, channels, eps=norm_eps)

        self.spatial_mixer = build_spatial_mixer(
            name=spatial_mixer,
            channels=channels,
            head_dim=head_dim,
            dilations=dilations,
            kernel_size=kernel_size,
            spatial_kernel=spatial_kernel,
            large_kernel=large_kernel,
            stripe_width=stripe_width,
            relative_position_bias=relative_position_bias,
            mask_padding=mask_padding,
            fp32_attention=fp32_attention,
        )

        self.channel_attention = CrossChannelAttention(
            channels=channels,
            head_dim=head_dim,
            softplus_temperature=softplus_temperature,
            fp32_attention=fp32_attention,
        )

        self.ffn = GatedConvFFN(
            channels=channels,
            expansion=ffn_expansion,
            use_activation=ffn_use_activation,
            gate=ffn_gate,
        )

        self.gamma_spatial = LayerScale(channels, init_value=layer_scale_init)
        self.gamma_channel = LayerScale(channels, init_value=layer_scale_init)
        self.gamma_ffn = LayerScale(channels, init_value=layer_scale_init)

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
        return f"channels={self.channels}, spatial_mixer={self.spatial_mixer_name!r}"


def build_bottleneck(config: "CASHSIConfig", channels: int) -> nn.Sequential:
    """Bottleneck stack (spec 3.5).

    Mostly dilated local attention, with a hybrid local+stripe block every
    ``stripe_frequency`` blocks.  For 5 blocks at the default frequency of 3 that
    is: local, local, hybrid, local, local -- the stripe block injects a long-range
    axial receptive field periodically without paying for it in every block.
    """
    blocks = []
    mixers = config.bottleneck_mixers()

    for index, mixer in enumerate(mixers):
        blocks.append(
            CASBlock(
                channels=channels,
                spatial_mixer=mixer,
                head_dim=config.head_dim,
                dilations=config.dilations_quarter,
                ffn_expansion=config.ffn_expansion,
                layer_scale_init=config.layer_scale_init,
                drop_path=config.drop_path,
                norm=config.norm,
                norm_eps=config.norm_eps,
                kernel_size=config.attention_kernel_size,
                spatial_kernel=config.spatial_kernel,
                large_kernel=config.large_kernel,
                stripe_width=config.stripe_width,
                relative_position_bias=config.relative_position_bias,
                mask_padding=config.mask_padding,
                fp32_attention=config.fp32_attention,
                ffn_use_activation=config.ffn_use_activation,
                ffn_gate=config.ffn_gate,
                softplus_temperature=config.softplus_temperature,
            )
        )
        del index

    return nn.Sequential(*blocks)

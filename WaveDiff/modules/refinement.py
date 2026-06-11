import torch.nn as nn
import torch.nn.functional as F

from modules.attention import SpectralAttention, SpectralSpatialAttention
from modules.encoders import ResidualBlock
from modules.normalization import make_norm, resolve_norm_type


class SpectralRefinementBlock(nn.Module):
    """Residual refinement block with spectral attention and depthwise filtering."""

    def __init__(
        self,
        channels,
        use_batchnorm=True,
        norm_type=None,
        norm_groups=8,
    ):
        super().__init__()
        self.residual = ResidualBlock(
            channels, use_batchnorm, norm_type, norm_groups
        )
        self.spectral_attn = SpectralAttention(channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        res = self.residual(x)
        res = self.depthwise(res)
        res = self.pointwise(res)
        res = self.spectral_attn(res)
        return F.silu(res + x)


class SpectralRefinementHead(nn.Module):
    """
    Lightweight refinement head applied after diffusion decoding to enhance spectra.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        num_blocks=3,
        use_batchnorm=True,
        norm_type=None,
        norm_groups=8,
    ):
        super().__init__()
        hidden_channels = max(hidden_channels, in_channels)

        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList(
            [
                SpectralRefinementBlock(
                    hidden_channels,
                    use_batchnorm,
                    norm_type,
                    norm_groups,
                )
                for _ in range(num_blocks)
            ]
        )

        self.spatial_attn = SpectralSpatialAttention(hidden_channels)
        self.output_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.spatial_attn(h)
        h = self.output_proj(h)
        return residual + h


class PixelRefinementBlock(nn.Module):
    """Simple spatial refinement block with residual connections."""

    def __init__(
        self,
        channels,
        expansion=2,
        use_batchnorm=True,
        norm_type=None,
        norm_groups=8,
    ):
        super().__init__()
        hidden = channels * expansion
        resolved_norm = resolve_norm_type(use_batchnorm, norm_type)
        use_normalization = resolved_norm != "none"

        layers = [
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=3,
                padding=1,
                bias=not use_normalization,
            ),
        ]

        if use_normalization:
            layers.append(
                make_norm(
                    hidden,
                    use_batchnorm,
                    resolved_norm,
                    norm_groups,
                )
            )

        layers.extend(
            [
                nn.GELU(),
                nn.Conv2d(
                    hidden,
                    channels,
                    kernel_size=3,
                    padding=1,
                    bias=not use_normalization,
                ),
            ]
        )

        if use_normalization:
            layers.append(
                make_norm(
                    channels,
                    use_batchnorm,
                    resolved_norm,
                    norm_groups,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.silu(self.net(x) + x)


class PixelRefinementHead(nn.Module):
    """Optional lightweight refinement head operating purely in pixel space."""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        num_blocks=2,
        use_batchnorm=True,
        expansion=2,
        norm_type=None,
        norm_groups=8,
    ):
        super().__init__()
        hidden_channels = hidden_channels or in_channels

        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                PixelRefinementBlock(
                    hidden_channels,
                    expansion=expansion,
                    use_batchnorm=use_batchnorm,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.output_proj(h)
        return residual + h

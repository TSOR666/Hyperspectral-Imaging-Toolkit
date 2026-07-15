"""PixelShuffle upsampling (specification section 6.4).

``1x1 expansion -> PixelShuffle(2)``.  Transposed convolution is avoided because
its overlapping kernel footprint produces checkerboard artifacts, which in a
spectral-reconstruction setting show up as periodic banding in the reconstructed
cube rather than as an obvious visual grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PixelShuffleUpsample"]


class PixelShuffleUpsample(nn.Module):
    """``B x C_in x H x W  ->  B x C_out x 2H x 2W``."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.projection = nn.Conv2d(
            self.in_channels,
            4 * self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.shuffle(x)
        return x

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}"

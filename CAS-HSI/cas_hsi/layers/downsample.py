"""PixelUnshuffle downsampling (specification section 6.2).

``PixelUnshuffle(2) -> 1x1 projection`` keeps the sub-pixel information (nothing
is thrown away before the network gets to choose what matters), halves the
spatial dims exactly, and widens the latent -- all with export-friendly ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PixelUnshuffleDownsample"]


def _is_compiling() -> bool:
    """Whether the current call is being captured by ``torch.export``/Dynamo."""
    compiler = getattr(torch, "compiler", None)
    return bool(compiler is not None and compiler.is_compiling())


class PixelUnshuffleDownsample(nn.Module):
    """``B x C_in x H x W  ->  B x C_out x H/2 x W/2``."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.unshuffle = nn.PixelUnshuffle(2)
        self.projection = nn.Conv2d(
            4 * self.in_channels,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # These checks are valuable in eager mode, but Python branching on a SymInt
        # specializes the dimensions during torch.export. The enclosing CASHSI
        # forward pads to a multiple of four before this module, and the exported
        # graph retains that symbolic padding relationship.
        if not _is_compiling():
            if x.shape[-2] % 2 != 0:
                raise ValueError(
                    f"Input height must be divisible by 2, got {int(x.shape[-2])}. "
                    "Pad the network input with pad_to_multiple(x, multiple=4) first."
                )
            if x.shape[-1] % 2 != 0:
                raise ValueError(
                    f"Input width must be divisible by 2, got {int(x.shape[-1])}. "
                    "Pad the network input with pad_to_multiple(x, multiple=4) first."
                )

        x = self.unshuffle(x)
        x = self.projection(x)
        return x

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}"

"""LayerScale and stochastic depth (specification sections 4.2 and 4.4).

LayerScale gates each residual *branch* with a learnable per-channel vector.  The
identity path is never scaled, so a zero-initialized LayerScale makes the block
an exact identity -- that property is asserted in the test-suite and is what lets
deep stacks train stably from step 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["LayerScale", "DropPath"]


class LayerScale(nn.Module):
    """Per-channel learnable scale applied to a residual branch."""

    def __init__(self, channels: int, init_value: float = 1e-3) -> None:
        super().__init__()
        self.channels = int(channels)
        self.init_value = float(init_value)
        self.scale = nn.Parameter(torch.full((self.channels,), float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.view(1, -1, 1, 1).to(x.dtype)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, init_value={self.init_value}"


class DropPath(nn.Module):
    """Per-sample stochastic depth on a residual branch.

    Identity when ``drop_prob == 0`` or in eval mode, so exporting and the
    zero-LayerScale identity test are unaffected.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        if not 0.0 <= float(drop_prob) < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Broadcast over every non-batch dim so whole samples are dropped.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep:
            mask = mask.div(keep_prob)
        return x * mask

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"

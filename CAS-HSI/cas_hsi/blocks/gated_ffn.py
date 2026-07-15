"""Gated convolutional feed-forward network (specification section 4.7).

``T = DWConv3x3(Conv1x1_{2rC}(X))``, split into halves, multiplicatively gated,
projected back to C.  The gate is what makes this more than an MLP: it lets the
network suppress a channel's contribution *conditionally on the local content*,
which is how it separates metameric surfaces that share the same RGB.

The default gate is the multiplicative SimpleGate (``x1 * x2``, no activation).
A terminal sigmoid gate is explicitly not the default (spec 4.7.2): sigmoid
saturates and caps the branch's dynamic range, which hurts a task whose targets
carry real high-frequency spectral structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GatedConvFFN"]

_GATES = ("simple", "silu", "gelu")


class GatedConvFFN(nn.Module):
    """Gated conv FFN. ``expansion`` is r in the spec (default 2.0)."""

    def __init__(
        self,
        channels: int,
        expansion: float = 2.0,
        use_activation: bool = False,
        gate: str = "simple",
    ) -> None:
        super().__init__()

        self.channels = int(channels)
        self.expansion = float(expansion)
        self.hidden = int(round(self.channels * self.expansion))
        if self.hidden < 1:
            raise ValueError(
                f"FFN hidden width rounded to {self.hidden}; "
                f"channels={channels} expansion={expansion} is degenerate."
            )

        gate_key = str(gate).strip().lower()
        if gate_key not in _GATES:
            raise ValueError(f"gate must be one of {_GATES}, got {gate!r}")
        self.gate = gate_key
        self.use_activation = bool(use_activation)

        self.input_projection = nn.Conv2d(
            self.channels, 2 * self.hidden, kernel_size=1, bias=False
        )
        self.depthwise = nn.Conv2d(
            2 * self.hidden,
            2 * self.hidden,
            kernel_size=3,
            padding=1,
            groups=2 * self.hidden,
            bias=False,
        )
        self.output_projection = nn.Conv2d(
            self.hidden, self.channels, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.depthwise(x)

        x1, x2 = x.chunk(2, dim=1)

        if self.use_activation:
            if self.gate == "silu":
                x1 = F.silu(x1)
            elif self.gate == "gelu":
                x1 = F.gelu(x1)

        x = x1 * x2
        return self.output_projection(x)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, hidden={self.hidden}, "
            f"expansion={self.expansion}, gate={self.gate}, "
            f"use_activation={self.use_activation}"
        )

"""Cross-channel self-attention (specification section 4.6).

This is *not* a pooled sigmoid gate (SE-style).  It computes a genuine attention
matrix over the latent channel axis, with each channel's query/key summarizing
evidence from every spatial location:

    A_c = softmax( normalize(Q) normalize(K)^T * temperature )   in R^(h x d_h x d_h)
    O_c = A_c V

Cost is O(C^2 * HW) rather than O((HW)^2 * C), so it is affordable at *full*
resolution -- which is precisely why CAS-Lite can afford to keep it while
dropping spatial attention.

For RGB-to-HSI this is the module that carries the spectral prior: mixing latent
channels globally is how band-to-band structure is modelled before the final
31-band projection.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CrossChannelAttention", "fp32_matmul_guard"]


def fp32_matmul_guard(enabled: bool, device_type: str):
    """Disable autocast around a region so hand-cast fp32 operands stay fp32.

    ``torch.autocast`` re-downcasts matmul operands even when they were cast to fp32
    explicitly, so ``q.to(torch.float32)`` alone does NOT keep the attention logits in
    fp32 under AMP -- the ``torch.matmul`` that follows silently accumulates in bf16/fp16
    (spec 9.5 requires fp32 logits + softmax). Entering ``autocast(enabled=False)`` for
    the logits/softmax region is what actually honors the policy. When ``enabled`` is
    False, or when no autocast is active, this is a cheap no-op.
    """
    if enabled:
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


class CrossChannelAttention(nn.Module):
    """Channel-covariance attention with depthwise-conditioned Q, K, V."""

    def __init__(
        self,
        channels: int,
        head_dim: int = 32,
        softplus_temperature: bool = False,
        fp32_attention: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if channels % head_dim != 0:
            raise ValueError(
                f"channels must be divisible by head_dim: {channels} % {head_dim} != 0"
            )

        self.channels = int(channels)
        self.head_dim = int(head_dim)
        self.num_heads = self.channels // self.head_dim
        self.softplus_temperature = bool(softplus_temperature)
        self.fp32_attention = bool(fp32_attention)
        self.eps = float(eps)

        self.qkv = nn.Sequential(
            nn.Conv2d(self.channels, 3 * self.channels, kernel_size=1, bias=False),
            nn.Conv2d(
                3 * self.channels,
                3 * self.channels,
                kernel_size=3,
                padding=1,
                groups=3 * self.channels,
                bias=False,
            ),
        )

        # Restormer convention: the logits are *multiplied* by a learnable
        # per-head temperature (an inverse softmax temperature).
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        self.output_projection = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=False
        )

    def _temperature(self, dtype: torch.dtype) -> torch.Tensor:
        if self.softplus_temperature:
            # Keeps the scale strictly positive if raw training drives it negative.
            return (F.softplus(self.temperature) + 1e-4).to(dtype)
        return self.temperature.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {channels}"
            )

        q, k, v = self.qkv(x).chunk(3, dim=1)

        # Spatial dims are read from the tensor, never stored (spec 8.8).
        tokens = height * width
        q = q.reshape(batch, self.num_heads, self.head_dim, tokens)
        k = k.reshape(batch, self.num_heads, self.head_dim, tokens)
        v = v.reshape(batch, self.num_heads, self.head_dim, tokens)

        # Attention logits and softmax in fp32 even under autocast (spec 9.5). The
        # .to(fp32) on q/k is necessary but NOT sufficient: torch.matmul is on
        # autocast's downcast list, so the logit accumulation needs the guard too.
        # The A_c @ V matmul below is deliberately left under autocast -- spec 9.5 pins
        # only the logits and softmax to fp32; value aggregation may run reduced.
        attention_dtype = torch.float32 if self.fp32_attention else q.dtype
        q = F.normalize(q.to(attention_dtype), dim=-1, eps=self.eps)
        k = F.normalize(k.to(attention_dtype), dim=-1, eps=self.eps)

        with fp32_matmul_guard(self.fp32_attention, q.device.type):
            attention = torch.matmul(q, k.transpose(-2, -1))
            attention = attention * self._temperature(attention.dtype)
            attention = attention.softmax(dim=-1)

        output = torch.matmul(attention.to(v.dtype), v)
        output = output.reshape(batch, channels, height, width)
        return self.output_projection(output)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, head_dim={self.head_dim}, "
            f"num_heads={self.num_heads}, softplus_temperature={self.softplus_temperature}"
        )

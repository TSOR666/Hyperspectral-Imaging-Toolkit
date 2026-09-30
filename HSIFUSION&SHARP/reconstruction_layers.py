"""Explicit reconstruction controls; legacy models do not enable these implicitly."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def unpack_grouped_projection(value, groups, parts, layout="legacy"):
    """Return Q/K/V (or K/V) in channel order from a grouped 1x1 convolution.

    Conv2d emits [group, part, channels_in_part], whereas chunk(parts) assumes
    [part, group, channels_in_part]. Both layouts share parameter shapes, so the
    choice must be persisted in checkpoints rather than inferred from weights.
    """
    if layout not in {"legacy", "group_major"}:
        raise ValueError("projection layout must be legacy or group_major")
    if layout == "legacy":
        return value.chunk(parts, dim=1)
    b, channels, h, w = value.shape
    if groups <= 0 or channels % (groups * parts):
        raise ValueError("Grouped projection must divide evenly into groups and parts")
    value = value.reshape(b, groups, parts, channels // (groups * parts), h, w)
    value = value.permute(0, 2, 1, 3, 4, 5).reshape(b, parts, channels // parts, h, w)
    return value.unbind(1)


class FullResolutionSpectralBlock(nn.Module):
    """Channel attention and gated FFN without pooling the value maps.

    Attention costs O(HW*C*C/heads), never O((HW)^2). Branch normalization
    leaves the unnormalized residual stream intact. Nonzero residual gates let
    both branches learn on the first optimizer update.
    """
    def __init__(self, dim, heads=4, expansion=2, gate_init=0.01):
        super().__init__()
        if dim % heads or heads < 1:
            raise ValueError("spectral heads must divide feature channels")
        if not math.isfinite(gate_init) or gate_init <= 0:
            raise ValueError("spectral gate_init must be finite and positive")
        self.heads = heads
        self.norm1 = nn.GroupNorm(1, dim)
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.spatial = nn.Conv2d(3 * dim, 3 * dim, 3, padding=1, groups=3 * dim)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm2 = nn.GroupNorm(1, dim)
        hidden = int(dim * expansion)
        self.ffn_in = nn.Conv2d(dim, 2 * hidden, 1)
        self.ffn_spatial = nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1, groups=2 * hidden)
        self.ffn_out = nn.Conv2d(hidden, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(gate_init)))
        self.gamma2 = nn.Parameter(torch.full((1, dim, 1, 1), float(gate_init)))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.spatial(self.qkv(self.norm1(x))).chunk(3, dim=1)
        q, k, v = [t.reshape(b, self.heads, c // self.heads, h * w) for t in (q, k, v)]
        # FP32 scores and softmax also under external AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
            attn = (q @ k.transpose(-2, -1) * self.temperature.float()).softmax(dim=-1)
            attended = (attn @ v.float()).to(x.dtype).reshape(b, c, h, w)
        x = x + self.gamma * self.proj(attended)
        a, gate = self.ffn_spatial(self.ffn_in(self.norm2(x))).chunk(2, dim=1)
        return x + self.gamma2 * self.ffn_out(F.gelu(a) * gate)

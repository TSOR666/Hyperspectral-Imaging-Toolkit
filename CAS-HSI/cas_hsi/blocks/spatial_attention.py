"""Spatial token mixers (specification sections 4.5, 5.2, 5.4, 9.3).

Four families live here, all interchangeable behind :func:`build_spatial_mixer`
so that the block shell (Norm -> mixer -> residual) never changes:

Research backend
    ``dilated_local_attention``        3x3 neighborhood attention, heads split
                                       across dilations (spec 4.5.3)
    ``hybrid_local_stripe_attention``  some heads local, some attending along
                                       horizontal / vertical stripes (spec 4.5.4)

Edge backend (spec 9.3)
    ``depthwise_7x7``                  large-kernel depthwise mixer (CAS-Lite)
    ``dilated_depthwise_conv``         parallel depthwise dilated convs
    ``large_kernel_depthwise_conv``    wider depthwise kernel, stands in for stripes

Implementation note on neighborhood attention
---------------------------------------------
The spec's reference implementation (4.5.3) materializes ``[B, h, d, H, W, K]``
patch tensors via ``unfold``.  That is O(K) memory in the *head-dim* axis and is
the single largest activation in the network.  :func:`neighborhood_attention`
computes the identical function with a shift-and-gather over the K=9 offsets, so
the head-dim axis is never expanded -- only the ``[B, h, H, W, K]`` logits are.
The literal unfold reference is kept as :func:`local_attention_reference` and
``tests/test_equivalence.py`` asserts the two agree to fp32 tolerance.

The one deliberate deviation from the reference is ``mask_padding=True``: the
reference zero-pads K and V, so border pixels attend to fabricated zero-valued
neighbours that still receive softmax mass.  Masking those positions out instead
means a border pixel attends only to real neighbours.  Set ``mask_padding=False``
to recover the reference behaviour exactly.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .channel_attention import fp32_matmul_guard

__all__ = [
    "HeadGroup",
    "allocate_head_groups",
    "unfold_heads",
    "local_attention_reference",
    "neighborhood_attention",
    "stripe_attention",
    "DilatedLocalAttention",
    "HybridLocalStripeAttention",
    "ConvSpatialMixer",
    "ReparamConvSpatialMixer",
    "MultiDilationDepthwiseMixer",
    "build_spatial_mixer",
    "ATTENTION_MIXERS",
    "CONV_MIXERS",
]

# A very negative finite value rather than -inf: softmax rows are never fully
# masked here (the centre pixel is always valid, and every stripe holds at least
# one real row), but a finite sentinel keeps the graph free of inf/NaN under any
# autocast or export path.
#
# It must be dtype-aware. A hard-coded -1e9 overflows fp16 (max 65504) and becomes
# -inf, which is harmless today only because fp32_attention=True promotes the logits
# anyway -- but it would silently reintroduce inf the moment someone sets
# fp32_attention=False on an fp16 device. finfo(dtype).min / 4 is safely
# representable in every float dtype and still underflows exp() to exactly zero.
def _mask_value(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min / 4.0

ATTENTION_MIXERS = ("dilated_local_attention", "hybrid_local_stripe_attention")
CONV_MIXERS = (
    "depthwise_7x7",
    "dilated_depthwise_conv",
    "large_kernel_depthwise_conv",
    "reparam_depthwise",
)


class HeadGroup:
    """One contiguous slice of heads and the mixing rule applied to it."""

    __slots__ = ("kind", "heads", "dilation", "orientation")

    def __init__(
        self,
        kind: str,
        heads: int,
        dilation: int = 1,
        orientation: str = "horizontal",
    ) -> None:
        if kind not in {"local", "stripe"}:
            raise ValueError(f"kind must be 'local' or 'stripe', got {kind!r}")
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")
        self.kind = kind
        self.heads = int(heads)
        self.dilation = int(dilation)
        self.orientation = orientation

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        if self.kind == "local":
            return f"HeadGroup(local, heads={self.heads}, dilation={self.dilation})"
        return f"HeadGroup(stripe, heads={self.heads}, orientation={self.orientation})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HeadGroup):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.heads == other.heads
            and self.dilation == other.dilation
            and self.orientation == other.orientation
        )


def _split_local_heads(num_local: int, dilations: Sequence[int]) -> List[int]:
    """Distribute ``num_local`` heads over the dilation groups.

    Fewer heads than dilations  -> one head each to the *smallest* dilations.
    More heads than dilations   -> even split, remainder to the *largest*
                                   dilations (they have more area to cover).

    Reproduces the spec's worked example (4.5.2): C=128, d_h=32 -> h=4 heads over
    dilations (1, 2, 3) yields [1, 1, 2], i.e. heads 0/1/2/3 -> dilations 1/2/3/3.
    """
    groups = len(dilations)
    if groups == 0:
        raise ValueError("dilations must be non-empty")
    if num_local <= 0:
        return [0] * groups

    if num_local <= groups:
        return [1 if index < num_local else 0 for index in range(groups)]

    base, remainder = divmod(num_local, groups)
    return [base + (1 if index >= groups - remainder else 0) for index in range(groups)]


def allocate_head_groups(
    num_heads: int,
    dilations: Sequence[int],
    use_stripe: bool = False,
) -> List[HeadGroup]:
    """Assign every head to exactly one mixing rule (spec 4.5.2).

    The returned group head-counts always sum to ``num_heads``.

    Hybrid blocks reserve one horizontal + one vertical stripe head per four
    heads (minimum one pair), leaving the rest local.  For the spec's worked
    example (h=4) that is: dilation 1, dilation 2, horizontal, vertical.
    """
    if num_heads < 1:
        raise ValueError(f"num_heads must be >= 1, got {num_heads}")
    if not dilations:
        raise ValueError("dilations must be non-empty")

    if not use_stripe:
        sizes = _split_local_heads(num_heads, dilations)
        groups = [
            HeadGroup("local", heads, dilation=dilation)
            for heads, dilation in zip(sizes, dilations)
            if heads > 0
        ]
    else:
        if num_heads < 3:
            raise ValueError(
                f"hybrid_local_stripe_attention needs at least 3 heads "
                f"(one local + one horizontal + one vertical), got {num_heads}. "
                "Increase base_width or decrease head_dim."
            )
        stripe_pairs = max(1, num_heads // 4)
        stripe_heads = 2 * stripe_pairs
        local_heads = num_heads - stripe_heads
        if local_heads < 1:  # pragma: no cover - guarded by the num_heads check
            stripe_pairs = 1
            stripe_heads = 2
            local_heads = num_heads - stripe_heads

        sizes = _split_local_heads(local_heads, dilations)
        groups = [
            HeadGroup("local", heads, dilation=dilation)
            for heads, dilation in zip(sizes, dilations)
            if heads > 0
        ]
        groups.append(HeadGroup("stripe", stripe_pairs, orientation="horizontal"))
        groups.append(HeadGroup("stripe", stripe_pairs, orientation="vertical"))

    allocated = sum(group.heads for group in groups)
    if allocated != num_heads:  # pragma: no cover - invariant
        raise AssertionError(
            f"head allocation must sum to num_heads: {allocated} != {num_heads}"
        )
    return groups


# ---------------------------------------------------------------------------
# Neighborhood (dilated local) attention
# ---------------------------------------------------------------------------


def unfold_heads(
    x: torch.Tensor,
    kernel_size: int,
    dilation: int,
    padding: int,
) -> torch.Tensor:
    """``[B, h, d, H, W] -> [B, h, d, H, W, K]`` neighbourhood patches (zero-padded)."""
    batch, heads, head_dim, height, width = x.shape
    kernel_area = kernel_size * kernel_size

    patches = F.unfold(
        x.reshape(batch, heads * head_dim, height, width),
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
    )
    # F.unfold packs the second axis as (channel-major, kernel-minor).
    patches = patches.reshape(batch, heads, head_dim, kernel_area, height, width)
    return patches.permute(0, 1, 2, 4, 5, 3)


def local_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """The spec's literal unfold reference (4.5.3). Kept for equivalence testing.

    Zero-pads K and V at the borders, exactly as written in the specification.
    """
    _, _, head_dim, _, _ = q.shape
    padding = dilation * (kernel_size // 2)

    k_patches = unfold_heads(k, kernel_size=kernel_size, dilation=dilation, padding=padding)
    v_patches = unfold_heads(v, kernel_size=kernel_size, dilation=dilation, padding=padding)

    q = q.unsqueeze(-1)

    logits = (q * k_patches).sum(dim=2)
    logits = logits / math.sqrt(head_dim)

    weights = logits.softmax(dim=-1)

    return (weights.unsqueeze(2) * v_patches).sum(dim=-1)


def _offsets(kernel_size: int) -> List[Tuple[int, int]]:
    radius = kernel_size // 2
    return [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]


def _shift(x: torch.Tensor, padding: int, dy: int, dx: int, dilation: int) -> torch.Tensor:
    """Gather the neighbour at offset ``dilation * (dy, dx)`` for every pixel.

    ``x`` is the *padded* tensor, so the output extent is its size minus the pad on
    both sides -- reading H/W straight off ``x`` would slice the padded extent and
    silently return a window of the wrong size.
    """
    height = x.shape[-2] - 2 * padding
    width = x.shape[-1] - 2 * padding
    start_y = padding + dilation * dy
    start_x = padding + dilation * dx
    return x[..., start_y : start_y + height, start_x : start_x + width]


def neighborhood_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int,
    relative_bias: torch.Tensor | None = None,
    mask_padding: bool = True,
    fp32_attention: bool = True,
) -> torch.Tensor:
    """Memory-lean dilated 3x3 neighborhood attention.

    Args:
        q, k, v: ``[B, heads, head_dim, H, W]``.
        relative_bias: ``[heads, K]`` bias indexed by relative offset (spec 4.5.3).
        mask_padding: exclude out-of-image neighbours from the softmax.
        fp32_attention: accumulate logits and take the softmax in fp32 (spec 9.5).

    Returns:
        ``[B, heads, head_dim, H, W]``.
    """
    batch, heads, head_dim, height, width = q.shape
    padding = dilation * (kernel_size // 2)
    offsets = _offsets(kernel_size)

    work_dtype = torch.float32 if fp32_attention else q.dtype
    scale = 1.0 / math.sqrt(head_dim)

    q_work = q.to(work_dtype)
    k_pad = F.pad(k.to(work_dtype), (padding, padding, padding, padding))

    logits = []
    for index, (dy, dx) in enumerate(offsets):
        shifted_k = _shift(k_pad, padding, dy, dx, dilation)
        logit = (q_work * shifted_k).sum(dim=2) * scale  # [B, heads, H, W]

        if relative_bias is not None:
            logit = logit + relative_bias[:, index].to(work_dtype).view(1, heads, 1, 1)

        if mask_padding:
            valid = _validity(height, width, dy, dx, dilation, q.device, work_dtype)
            logit = torch.where(
                valid.bool(), logit, logit.new_full((), _mask_value(logit.dtype))
            )

        logits.append(logit)

    weights = torch.stack(logits, dim=-1).softmax(dim=-1)  # [B, heads, H, W, K]
    weights = weights.to(v.dtype)

    v_pad = F.pad(v, (padding, padding, padding, padding))
    output = torch.zeros_like(v)
    for index, (dy, dx) in enumerate(offsets):
        shifted_v = _shift(v_pad, padding, dy, dx, dilation)
        output = output + weights[..., index].unsqueeze(2) * shifted_v

    return output


def _validity(
    height: int,
    width: int,
    dy: int,
    dx: int,
    dilation: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[1, 1, H, W]`` mask: 1 where the neighbour at this offset is inside the image."""
    rows = torch.arange(height, device=device).view(1, 1, height, 1) + dilation * dy
    cols = torch.arange(width, device=device).view(1, 1, 1, width) + dilation * dx
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return valid.to(dtype)


# ---------------------------------------------------------------------------
# Stripe (axial) attention
# ---------------------------------------------------------------------------


def stripe_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    stripe_width: int,
    orientation: str = "horizontal",
    fp32_attention: bool = True,
) -> torch.Tensor:
    """CSWin-style stripe attention (spec 4.5.4).

    A horizontal stripe is ``stripe_width`` consecutive *rows* spanning the full
    width; a vertical stripe is ``stripe_width`` consecutive *columns* spanning
    the full height.  Every token attends to all tokens in its own stripe, which
    buys a long-range receptive field along one axis at a fraction of the cost of
    global attention.

    Dimensions need not be divisible by ``stripe_width``: the map is zero-padded
    internally, the padded keys are masked out of the softmax, and the result is
    cropped back (spec 4.5.4).
    """
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")

    if orientation == "vertical":
        # Transpose H and W, run the horizontal path, transpose back.
        out = stripe_attention(
            q.transpose(-2, -1),
            k.transpose(-2, -1),
            v.transpose(-2, -1),
            stripe_width=stripe_width,
            orientation="horizontal",
            fp32_attention=fp32_attention,
        )
        return out.transpose(-2, -1)

    batch, heads, head_dim, height, width = q.shape
    width_stripe = max(1, min(int(stripe_width), height))  # spec: min(8, H, W)

    pad_h = (width_stripe - height % width_stripe) % width_stripe
    if pad_h:
        q = F.pad(q, (0, 0, 0, pad_h))
        k = F.pad(k, (0, 0, 0, pad_h))
        v = F.pad(v, (0, 0, 0, pad_h))
    padded_height = height + pad_h
    num_stripes = padded_height // width_stripe
    tokens = width_stripe * width

    def to_stripes(x: torch.Tensor) -> torch.Tensor:
        # [B, heads, d, H', W] -> [B, heads, n_stripes, tokens, d]
        x = x.reshape(batch, heads, head_dim, num_stripes, width_stripe, width)
        x = x.permute(0, 1, 3, 4, 5, 2)  # B, heads, n, sw, W, d
        return x.reshape(batch, heads, num_stripes, tokens, head_dim)

    work_dtype = torch.float32 if fp32_attention else q.dtype
    q_s = to_stripes(q).to(work_dtype)
    k_s = to_stripes(k).to(work_dtype)
    v_s = to_stripes(v)

    # Logits + softmax in fp32 under autocast (spec 9.5): the guard is required because
    # torch.matmul would otherwise re-downcast the fp32 operands. The value matmul below
    # is left under autocast (only logits/softmax are pinned to fp32).
    with fp32_matmul_guard(fp32_attention, q.device.type):
        logits = torch.matmul(q_s, k_s.transpose(-2, -1)) / math.sqrt(head_dim)

        if pad_h:
            # Mask the key tokens that came from the zero padding.
            row_index = torch.arange(padded_height, device=q.device).reshape(num_stripes, width_stripe)
            valid_rows = row_index < height  # [n_stripes, sw]
            valid_tokens = valid_rows.unsqueeze(-1).expand(num_stripes, width_stripe, width)
            valid_tokens = valid_tokens.reshape(1, 1, num_stripes, 1, tokens)
            logits = torch.where(
                valid_tokens, logits, logits.new_full((), _mask_value(logits.dtype))
            )

        weights = logits.softmax(dim=-1)

    weights = weights.to(v_s.dtype)
    output = torch.matmul(weights, v_s)  # [B, heads, n, tokens, d]

    output = output.reshape(batch, heads, num_stripes, width_stripe, width, head_dim)
    output = output.permute(0, 1, 5, 2, 3, 4)  # B, heads, d, n, sw, W
    output = output.reshape(batch, heads, head_dim, padded_height, width)

    if pad_h:
        output = output[..., :height, :]
    return output


# ---------------------------------------------------------------------------
# Attention mixers
# ---------------------------------------------------------------------------


class _QKVProjection(nn.Module):
    """``[Q, K, V] = DWConv3x3(Conv1x1_{3C}(X))`` (spec 4.5.1).

    The depthwise convolution is what gives Q/K/V local positional context, which
    is why no learned absolute positional embedding is needed and why the model
    stays valid at any resolution (spec 8.5).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(channels, 3 * channels, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(
            3 * channels,
            3 * channels,
            kernel_size=3,
            padding=1,
            groups=3 * channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise(self.project(x))


class _BaseSpatialAttention(nn.Module):
    """Shared machinery: QKV, head reshaping, per-group dispatch, output projection."""

    def __init__(
        self,
        channels: int,
        head_dim: int = 32,
        dilations: Sequence[int] = (1, 2),
        kernel_size: int = 3,
        use_stripe: bool = False,
        stripe_width: int = 8,
        relative_position_bias: bool = True,
        mask_padding: bool = True,
        fp32_attention: bool = True,
    ) -> None:
        super().__init__()

        if channels % head_dim != 0:
            raise ValueError(
                f"channels must be divisible by head_dim: {channels} % {head_dim} != 0"
            )
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.channels = int(channels)
        self.head_dim = int(head_dim)
        self.num_heads = self.channels // self.head_dim
        self.kernel_size = int(kernel_size)
        self.kernel_area = self.kernel_size * self.kernel_size
        self.dilations = tuple(int(d) for d in dilations)
        self.stripe_width = int(stripe_width)
        self.mask_padding = bool(mask_padding)
        self.fp32_attention = bool(fp32_attention)

        self.head_groups = allocate_head_groups(
            self.num_heads, self.dilations, use_stripe=use_stripe
        )

        self.qkv = _QKVProjection(self.channels)
        self.output_projection = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=False
        )

        # Relative position bias b_g(p - q), one table per head over the K offsets
        # (spec 4.5.3). Zero-init keeps the block an exact identity when LayerScale
        # is zero and adds no prior before training.
        if relative_position_bias:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_heads, self.kernel_area)
            )
        else:
            self.register_parameter("relative_position_bias_table", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch, self.num_heads, self.head_dim, height, width)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        outputs = []
        head_start = 0
        for group in self.head_groups:
            head_end = head_start + group.heads
            q_g = q[:, head_start:head_end]
            k_g = k[:, head_start:head_end]
            v_g = v[:, head_start:head_end]

            if group.kind == "local":
                bias = None
                if self.relative_position_bias_table is not None:
                    bias = self.relative_position_bias_table[head_start:head_end]
                out_g = neighborhood_attention(
                    q_g,
                    k_g,
                    v_g,
                    kernel_size=self.kernel_size,
                    dilation=group.dilation,
                    relative_bias=bias,
                    mask_padding=self.mask_padding,
                    fp32_attention=self.fp32_attention,
                )
            else:
                out_g = stripe_attention(
                    q_g,
                    k_g,
                    v_g,
                    stripe_width=self.stripe_width,
                    orientation=group.orientation,
                    fp32_attention=self.fp32_attention,
                )

            outputs.append(out_g)
            head_start = head_end

        output = torch.cat(outputs, dim=1)  # concat over heads (spec 4.5.5)
        output = output.reshape(batch, self.channels, height, width)
        return self.output_projection(output)

    def extra_repr(self) -> str:
        groups = ", ".join(repr(group) for group in self.head_groups)
        return (
            f"channels={self.channels}, head_dim={self.head_dim}, "
            f"num_heads={self.num_heads}, groups=[{groups}]"
        )


class DilatedLocalAttention(_BaseSpatialAttention):
    """3x3 neighborhood attention with heads split across dilations (spec 4.5.3)."""

    def __init__(
        self,
        channels: int,
        head_dim: int = 32,
        dilations: Sequence[int] = (1, 2),
        kernel_size: int = 3,
        relative_position_bias: bool = True,
        mask_padding: bool = True,
        fp32_attention: bool = True,
        **_: object,
    ) -> None:
        super().__init__(
            channels=channels,
            head_dim=head_dim,
            dilations=dilations,
            kernel_size=kernel_size,
            use_stripe=False,
            relative_position_bias=relative_position_bias,
            mask_padding=mask_padding,
            fp32_attention=fp32_attention,
        )


class HybridLocalStripeAttention(_BaseSpatialAttention):
    """Local heads + horizontal/vertical stripe heads (spec 4.5.4)."""

    def __init__(
        self,
        channels: int,
        head_dim: int = 32,
        dilations: Sequence[int] = (1, 2, 3),
        kernel_size: int = 3,
        stripe_width: int = 8,
        relative_position_bias: bool = True,
        mask_padding: bool = True,
        fp32_attention: bool = True,
        **_: object,
    ) -> None:
        super().__init__(
            channels=channels,
            head_dim=head_dim,
            dilations=dilations,
            kernel_size=kernel_size,
            use_stripe=True,
            stripe_width=stripe_width,
            relative_position_bias=relative_position_bias,
            mask_padding=mask_padding,
            fp32_attention=fp32_attention,
        )


# ---------------------------------------------------------------------------
# Convolutional mixers (CAS-Lite and the edge backend)
# ---------------------------------------------------------------------------


class ConvSpatialMixer(nn.Module):
    """``Conv1x1(DWConv_kxk(X))`` -- the CAS-Lite token mixer (spec 5.2/5.3).

    At full resolution this replaces explicit attention: a 7x7 depthwise kernel
    gives a comparable receptive field per layer at a fraction of the memory, and
    exports to any runtime without a neighborhood-attention operator.
    """

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        padding = self.kernel_size // 2

        self.depthwise = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            padding=padding,
            groups=self.channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

    def extra_repr(self) -> str:
        return f"channels={self.channels}, kernel_size={self.kernel_size}"


class ReparamConvSpatialMixer(nn.Module):
    """Multi-branch depthwise mixer that fuses to one kernel at deploy (spec 5.4).

    Trains as ``X + D3(X) + D5(X) + D7(X)``; :meth:`fuse` folds the three
    depthwise kernels and the identity into a single ``k_max x k_max`` depthwise
    convolution, which is numerically equivalent and strictly cheaper.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        use_identity: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        if any(k % 2 == 0 for k in self.kernel_sizes):
            raise ValueError(f"kernel sizes must be odd, got {self.kernel_sizes}")
        self.use_identity = bool(use_identity)
        self.max_kernel = max(self.kernel_sizes)

        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=self.channels,
                    bias=False,
                )
                for k in self.kernel_sizes
            ]
        )
        self.pointwise = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.fused: nn.Conv2d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused is not None:
            return self.pointwise(self.fused(x))

        mixed = x if self.use_identity else torch.zeros_like(x)
        for branch in self.branches:
            mixed = mixed + branch(x)
        return self.pointwise(mixed)

    @torch.no_grad()
    def fuse(self) -> "ReparamConvSpatialMixer":
        """Collapse the parallel branches into a single depthwise convolution."""
        reference = self.pointwise.weight
        radius = self.max_kernel // 2
        fused_weight = torch.zeros(
            self.channels,
            1,
            self.max_kernel,
            self.max_kernel,
            device=reference.device,
            dtype=reference.dtype,
        )

        for branch in self.branches:
            k = branch.kernel_size[0]
            offset = radius - k // 2
            fused_weight[:, :, offset : offset + k, offset : offset + k] += branch.weight

        if self.use_identity:
            fused_weight[:, :, radius, radius] += 1.0

        fused = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=self.max_kernel,
            padding=radius,
            groups=self.channels,
            bias=False,
        )
        # The new module must join the model it lives in: a bare nn.Conv2d is created
        # on CPU in the default dtype, which silently breaks a .double() or .cuda() model.
        fused = fused.to(device=reference.device, dtype=reference.dtype)
        fused.weight.copy_(fused_weight)
        self.fused = fused
        return self

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, kernel_sizes={self.kernel_sizes}, "
            f"use_identity={self.use_identity}, fused={self.fused is not None}"
        )


class MultiDilationDepthwiseMixer(nn.Module):
    """Parallel depthwise dilated convs -- the edge stand-in for local attention (spec 9.3).

    The spec's reference requires ``channels % len(dilations) == 0``.  That check
    is *relaxed* here because it contradicts the spec's own recommended
    configuration: CAS-HSI-Tiny's quarter-resolution width is 128 and the
    quarter-stage dilations are (1, 2, 3), and 128 % 3 != 0.  Channels are instead
    split as evenly as possible with the remainder going to the larger dilations,
    mirroring the head-allocation rule of spec 4.5.2.  Pass ``strict_split=True``
    to restore the reference's hard failure.
    """

    def __init__(
        self,
        channels: int,
        dilations: Sequence[int] = (1, 2, 3),
        kernel_size: int = 3,
        strict_split: bool = False,
        **_: object,
    ) -> None:
        super().__init__()

        self.channels = int(channels)
        self.dilations = tuple(int(d) for d in dilations)
        self.kernel_size = int(kernel_size)
        groups = len(self.dilations)
        if groups == 0:
            raise ValueError("dilations must be non-empty")

        if self.channels % groups != 0:
            if strict_split:
                raise ValueError(
                    "channels must be divisible by number of dilations "
                    f"({self.channels} % {groups} != 0)"
                )
            base, remainder = divmod(self.channels, groups)
            self.group_channels = [
                base + (1 if index >= groups - remainder else 0) for index in range(groups)
            ]
        else:
            self.group_channels = [self.channels // groups] * groups

        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    group_channels,
                    group_channels,
                    kernel_size=self.kernel_size,
                    padding=dilation * (self.kernel_size // 2),
                    dilation=dilation,
                    groups=group_channels,
                    bias=False,
                )
                for group_channels, dilation in zip(self.group_channels, self.dilations)
            ]
        )
        self.output_projection = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups = x.split(self.group_channels, dim=1)
        outputs = [branch(group) for branch, group in zip(self.branches, groups)]
        return self.output_projection(torch.cat(outputs, dim=1))

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, dilations={self.dilations}, "
            f"group_channels={self.group_channels}"
        )


def build_spatial_mixer(
    name: str,
    channels: int,
    head_dim: int = 32,
    dilations: Sequence[int] = (1, 2),
    kernel_size: int = 3,
    spatial_kernel: int = 7,
    large_kernel: int = 11,
    stripe_width: int = 8,
    relative_position_bias: bool = True,
    mask_padding: bool = True,
    fp32_attention: bool = True,
    reparam_kernels: Sequence[int] = (3, 5, 7),
) -> nn.Module:
    """Factory shared by the research and edge backends (spec 9.2)."""
    key = str(name).strip().lower()

    if key == "dilated_local_attention":
        return DilatedLocalAttention(
            channels=channels,
            head_dim=head_dim,
            dilations=dilations,
            kernel_size=kernel_size,
            relative_position_bias=relative_position_bias,
            mask_padding=mask_padding,
            fp32_attention=fp32_attention,
        )

    if key == "hybrid_local_stripe_attention":
        return HybridLocalStripeAttention(
            channels=channels,
            head_dim=head_dim,
            dilations=dilations,
            kernel_size=kernel_size,
            stripe_width=stripe_width,
            relative_position_bias=relative_position_bias,
            mask_padding=mask_padding,
            fp32_attention=fp32_attention,
        )

    if key in {"depthwise_7x7", "conv", "conv_spatial_mixer"}:
        return ConvSpatialMixer(channels, kernel_size=spatial_kernel)

    if key == "large_kernel_depthwise_conv":
        return ConvSpatialMixer(channels, kernel_size=large_kernel)

    if key == "dilated_depthwise_conv":
        return MultiDilationDepthwiseMixer(
            channels, dilations=dilations, kernel_size=kernel_size
        )

    if key == "reparam_depthwise":
        return ReparamConvSpatialMixer(channels, kernel_sizes=reparam_kernels)

    raise ValueError(
        f"Unknown spatial mixer {name!r}. "
        f"Attention mixers: {ATTENTION_MIXERS}. Convolutional mixers: {CONV_MIXERS}."
    )

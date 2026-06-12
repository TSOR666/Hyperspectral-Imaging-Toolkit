from __future__ import annotations

from typing import Literal

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

RPEMode = Literal["none", "legacy_post_softmax", "pre_softmax"]


def img2windows(
    image: torch.Tensor,
    window_height: int,
    window_width: int,
) -> torch.Tensor:
    batch, channels, height, width = image.shape
    if height % window_height or width % window_width:
        raise ValueError(
            f"Image {(height, width)} is not divisible by window "
            f"{(window_height, window_width)}."
        )
    image = image.view(
        batch,
        channels,
        height // window_height,
        window_height,
        width // window_width,
        window_width,
    )
    return (
        image.permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .reshape(-1, window_height * window_width, channels)
    )


def windows2img(
    windows: torch.Tensor,
    window_height: int,
    window_width: int,
    height: int,
    width: int,
) -> torch.Tensor:
    batch = int(
        windows.shape[0]
        / (height * width / window_height / window_width)
    )
    image = windows.view(
        batch,
        height // window_height,
        width // window_width,
        window_height,
        window_width,
        -1,
    )
    return (
        image.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(batch, height, width, -1)
    )


class GDFN(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_expansion_factor: float = 4.0,
        bias: bool = False,
        act_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            dim,
            hidden_features * 2,
            kernel_size=1,
            bias=bias,
        )
        self.norm = nn.LayerNorm(hidden_features)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            hidden_features,
            dim,
            kernel_size=1,
            bias=bias,
        )
        self.act_fn = act_fn or nn.GELU()

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = rearrange(
            x,
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        batch, channels, _, _ = x1.shape
        x1 = x1.view(batch, channels, height * width).permute(0, 2, 1)
        x1 = self.norm(x1).permute(0, 2, 1).view(
            batch,
            channels,
            height,
            width,
        )
        x = self.act_fn(self.dwconv(x1)) * x2
        return rearrange(self.project_out(x), "b c h w -> b (h w) c")


class SpatialGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        batch, _, channels = x.shape
        x2 = (
            self.conv(
                self.norm(x2)
                .transpose(1, 2)
                .contiguous()
                .view(batch, channels // 2, height, width)
            )
            .flatten(2)
            .transpose(-1, -2)
            .contiguous()
        )
        return x1 * x2


class SGFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.sg(x, height, width))
        return self.drop(self.fc2(x))


class Spectral_MSA(nn.Module):
    """Cosine spectral attention with configurable relative-position behavior."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool,
        rpe_mode: RPEMode = "legacy_post_softmax",
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"Dimension {dim} must divide {num_heads} heads.")
        if rpe_mode not in {"none", "legacy_post_softmax", "pre_softmax"}:
            raise ValueError(f"Unknown spectral RPE mode: {rpe_mode}")

        self.window_size = [4, 4]
        self.num_heads = num_heads
        self.rpe_mode = rpe_mode
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        channels_per_head = dim // num_heads
        if rpe_mode != "none":
            offsets = torch.arange(channels_per_head)
            coords = (
                torch.meshgrid(offsets, -offsets, indexing="ij")[0]
                + torch.meshgrid(offsets, -offsets, indexing="ij")[1]
                + channels_per_head
                - 1
            )
            self.register_buffer("coords", coords, persistent=False)
            self.relative_bias = nn.Parameter(
                torch.zeros(channels_per_head * 2 - 1)
            )

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=dim,
            ),
            nn.GELU(),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=dim,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        q, k, value_map = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(
            q,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )
        k = rearrange(
            k,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )
        value = rearrange(
            value_map,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )
        logits = (
            F.normalize(q, dim=-1)
            @ F.normalize(k, dim=-1).transpose(-2, -1)
        ) * self.temperature

        bias = None
        if self.rpe_mode != "none":
            bias = self.relative_bias[self.coords].unsqueeze(0).unsqueeze(0)
        if self.rpe_mode == "pre_softmax":
            logits = logits + bias

        attention = logits.softmax(dim=-1)
        if self.rpe_mode == "legacy_post_softmax":
            attention = attention + bias

        out = attention @ value
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=height,
            w=width,
        )
        return self.project_out(out) + self.pos_emb(value_map)


class LePEAttentionCross(nn.Module):
    """Self-attention inside one orientation of a cross-shaped window."""

    def __init__(
        self,
        dim: int,
        resolution: int | tuple[int, int],
        idx: int,
        split_size: int = 7,
        dim_out: int | None = None,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_scale: float | None = None,
    ) -> None:
        super().__init__()
        del proj_drop, qk_scale
        if idx not in {-1, 0, 1}:
            raise ValueError(f"Unsupported stripe index: {idx}")
        if dim % num_heads:
            raise ValueError(f"Dimension {dim} must divide {num_heads} heads.")

        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = _resolution_tuple(resolution)
        self.split_size = split_size
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.idx = idx
        self.H_sp, self.W_sp = self._window_shape(self.resolution)
        self.get_v = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
        )
        self.attn_drop = nn.Dropout(attn_drop)

    def _window_shape(
        self,
        resolution: tuple[int, int],
        *,
        crossed: bool = False,
    ) -> tuple[int, int]:
        height, width = resolution
        if self.idx == -1:
            return height, width
        horizontal = (height, self.split_size)
        vertical = (self.split_size, width)
        if self.idx == 0:
            return vertical if crossed else horizontal
        return horizontal if crossed else vertical

    def _tokens_to_windows(
        self,
        x: torch.Tensor,
        resolution: tuple[int, int],
        *,
        crossed: bool = False,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        height, width = resolution
        if tokens != height * width:
            raise ValueError("Flattened token count does not match feature resolution.")
        window_height, window_width = self._window_shape(
            resolution,
            crossed=crossed,
        )
        x = x.transpose(-2, -1).contiguous().view(
            batch,
            channels,
            height,
            width,
        )
        x = img2windows(x, window_height, window_width)
        return (
            x.reshape(
                -1,
                window_height * window_width,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _value_windows(
        self,
        x: torch.Tensor,
        resolution: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels = x.shape
        height, width = resolution
        window_height, window_width = self._window_shape(resolution)
        x = x.transpose(-2, -1).contiguous().view(
            batch,
            channels,
            height,
            width,
        )
        x = x.view(
            batch,
            channels,
            height // window_height,
            window_height,
            width // window_width,
            window_width,
        )
        x = (
            x.permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .reshape(-1, channels, window_height, window_width)
        )
        lepe = (
            self.get_v(x)
            .reshape(
                -1,
                self.num_heads,
                channels // self.num_heads,
                window_height * window_width,
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        value = (
            x.reshape(
                -1,
                self.num_heads,
                channels // self.num_heads,
                window_height * window_width,
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        return value, lepe

    def forward(
        self,
        qkv: torch.Tensor,
        cross: bool = False,
        resolution: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        resolution = resolution or self.resolution
        q, k, v = qkv.unbind(0)
        q = self._tokens_to_windows(q, resolution, crossed=cross)
        k = self._tokens_to_windows(k, resolution)
        v, lepe = self._value_windows(v, resolution)
        attention = (
            F.normalize(q, dim=-1)
            @ F.normalize(k, dim=-1).transpose(-2, -1)
        ) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = attention @ v + lepe

        window_height, window_width = self._window_shape(resolution)
        batch, _, channels = qkv[0].shape
        x = x.transpose(1, 2).reshape(
            -1,
            window_height * window_width,
            channels,
        )
        height, width = resolution
        return windows2img(
            x,
            window_height,
            window_width,
            height,
            width,
        ).view(batch, -1, channels)

    def change_resol(self, new_resolution: int | tuple[int, int]) -> None:
        self.resolution = _resolution_tuple(new_resolution)
        self.H_sp, self.W_sp = self._window_shape(self.resolution)


class CSWinCrossAttention(nn.Module):
    """Cross-attention between horizontal and vertical stripe features."""

    def __init__(
        self,
        dim: int,
        resolution: int | tuple[int, int],
        idx: int,
        split_size: int = 7,
        dim_out: int | None = None,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_scale: float | None = None,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        del dim_out, qk_scale
        if idx not in {0, 1}:
            raise ValueError(f"Unsupported stripe index: {idx}")
        if dim % num_heads:
            raise ValueError(f"Dimension {dim} must divide {num_heads} heads.")

        self.num_heads = num_heads
        self.resolution = _resolution_tuple(resolution)
        self.split_size = split_size
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.idx = idx
        self.H_sp, self.W_sp = self._window_shape(self.resolution)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.get_v = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _window_shape(
        self,
        resolution: tuple[int, int],
        *,
        crossed: bool = False,
    ) -> tuple[int, int]:
        height, width = resolution
        horizontal = (height, self.split_size)
        vertical = (self.split_size, width)
        if self.idx == 0:
            return vertical if crossed else horizontal
        return horizontal if crossed else vertical

    def _windows(
        self,
        x: torch.Tensor,
        resolution: tuple[int, int],
        *,
        crossed: bool = False,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        height, width = resolution
        if tokens != height * width:
            raise ValueError("Flattened token count does not match feature resolution.")
        window_height, window_width = self._window_shape(
            resolution,
            crossed=crossed,
        )
        x = x.transpose(-2, -1).contiguous().view(
            batch,
            channels,
            height,
            width,
        )
        x = img2windows(x, window_height, window_width)
        return (
            x.reshape(
                -1,
                window_height * window_width,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _value_windows(
        self,
        x: torch.Tensor,
        resolution: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels = x.shape
        height, width = resolution
        window_height, window_width = self._window_shape(resolution)
        x = x.transpose(-2, -1).contiguous().view(
            batch,
            channels,
            height,
            width,
        )
        x = x.view(
            batch,
            channels,
            height // window_height,
            window_height,
            width // window_width,
            window_width,
        )
        x = (
            x.permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .reshape(-1, channels, window_height, window_width)
        )
        lepe = (
            self.get_v(x)
            .reshape(
                -1,
                self.num_heads,
                channels // self.num_heads,
                window_height * window_width,
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        value = (
            x.reshape(
                -1,
                self.num_heads,
                channels // self.num_heads,
                window_height * window_width,
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        return value, lepe

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        resolution: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        resolution = resolution or self.resolution
        height, width = resolution
        if height != width:
            raise ValueError(
                "Cross-shaped cross-attention currently requires square feature maps."
            )

        q = self._windows(self.to_q(x1), resolution, crossed=True)
        # These names intentionally preserve the trained legacy projection order.
        value, lepe = self._value_windows(self.to_k(x2), resolution)
        k = self._windows(self.to_v(x2), resolution)
        attention = (
            F.normalize(q, dim=-1)
            @ F.normalize(k, dim=-1).transpose(-2, -1)
        ) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))

        batch, tokens, channels = x1.shape
        x = (attention @ value + lepe).transpose(1, 2).reshape(
            batch,
            tokens,
            channels,
        )
        return self.proj_drop(self.proj(x))

    def change_resol(self, new_resolution: int | tuple[int, int]) -> None:
        self.resolution = _resolution_tuple(new_resolution)
        self.H_sp, self.W_sp = self._window_shape(self.resolution)


def _resolution_tuple(
    resolution: int | tuple[int, int] | list[int],
) -> tuple[int, int]:
    if isinstance(resolution, int):
        return (resolution, resolution)
    return (int(resolution[0]), int(resolution[1]))


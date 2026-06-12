from __future__ import annotations

import torch
from torch import nn

from ._compat import DropPath, to_2tuple
from .attention import _scaled_cosine_attention


def partition(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Partition a BHWC tensor into non-overlapping square patches."""
    batch, height, width, channels = x.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Feature map {(height, width)} must be divisible by patch size {patch_size}."
        )
    x = x.view(
        batch,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, patch_size, patch_size, channels)
    )


def reverse(
    patches: torch.Tensor,
    patch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reverse :func:`partition` and return a BHWC tensor."""
    patches_per_image = (height // patch_size) * (width // patch_size)
    batch = patches.shape[0] // patches_per_image
    x = patches.view(
        batch,
        height // patch_size,
        width // patch_size,
        patch_size,
        patch_size,
        -1,
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(batch, height, width, -1)
    )


class Mlp(nn.Module):
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
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    """Inner-patch or cross-patch cosine attention used by CAT."""

    def __init__(
        self,
        dim: int,
        patch_size: int | tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rpe: bool = True,
    ) -> None:
        super().__init__()
        del qk_scale
        if dim % num_heads:
            raise ValueError(f"Attention dimension {dim} must divide {num_heads} heads.")

        self.dim = dim
        self.patch_size = to_2tuple(patch_size)
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.rpe = rpe

        if rpe:
            height, width = self.patch_size
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * height - 1) * (2 * width - 1), num_heads)
            )
            coords_h = torch.arange(height)
            coords_w = torch.arange(width)
            coords = torch.stack(
                torch.meshgrid(coords_h, coords_w, indexing="ij")
            )
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += height - 1
            relative_coords[:, :, 1] += width - 1
            relative_coords[:, :, 0] *= 2 * width - 1
            self.register_buffer(
                "relative_position_index",
                relative_coords.sum(-1),
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch,
                tokens,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attention_bias = None
        if self.rpe:
            height, width = self.patch_size
            bias = self.relative_position_bias_table[
                self.relative_position_index.reshape(-1)
            ].view(height * width, height * width, -1)
            attention_bias = bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        x = _scaled_cosine_attention(
            q,
            k,
            v,
            self.scale,
            attention_bias=attention_bias,
            dropout_p=self.attn_drop.p,
            training=self.training,
        )
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class CATBlock(nn.Module):
    """Cross-aggregation transformer block with dynamic resolution support."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        patch_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn_type: str = "ipsa",
        rpe: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.num_heads = num_heads
        self.patch_size = min(patch_size, min(self.input_resolution))
        self.mlp_ratio = mlp_ratio
        self.attn_type = attn_type

        attention_dim = dim if attn_type == "ipsa" else self.patch_size**2
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=attention_dim,
            patch_size=self.patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            rpe=rpe,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        resolution: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        height, width = resolution or self.input_resolution
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError(
                f"Token count {tokens} does not match resolution {(height, width)}."
            )

        shortcut = x
        patches = partition(
            self.norm1(x).view(batch, height, width, channels),
            self.patch_size,
        )
        patches = patches.view(-1, self.patch_size**2, channels)

        if self.attn_type == "ipsa":
            attended = self.attn(patches)
        elif self.attn_type == "cpsa":
            patch_count = (height // self.patch_size) * (
                width // self.patch_size
            )
            patches = (
                patches.view(
                    batch,
                    patch_count,
                    self.patch_size**2,
                    channels,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(-1, patch_count, self.patch_size**2)
            )
            attended = (
                self.attn(patches)
                .view(batch, channels, patch_count, self.patch_size**2)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(-1, self.patch_size**2, channels)
            )
        else:
            raise ValueError(f"Unknown CAT attention type: {self.attn_type}")

        attended = attended.view(
            -1,
            self.patch_size,
            self.patch_size,
            channels,
        )
        x = reverse(attended, self.patch_size, height, width)
        x = x.view(batch, height * width, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))

    def change_resolution(self, new_resolution: tuple[int, int]) -> None:
        """Compatibility shim for older callers; forward no longer mutates state."""
        self.input_resolution = tuple(new_resolution)

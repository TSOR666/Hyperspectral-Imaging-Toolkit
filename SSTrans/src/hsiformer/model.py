from __future__ import annotations

import math
from typing import Literal

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .attention import (
    CSWinCrossAttention,
    GDFN,
    LePEAttentionCross,
    RPEMode,
    SGFN,
    Spectral_MSA,
)
from .cat import CATBlock

ResidualMode = Literal["legacy", "paper"]


class Spectral_MSAB(nn.Module):
    def __init__(
        self,
        dim: int,
        head: int,
        *,
        rpe_mode: RPEMode = "legacy_post_softmax",
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.norm1 = nn.LayerNorm(dim)
        self.s_msa = Spectral_MSA(
            dim,
            dim // head,
            False,
            rpe_mode=rpe_mode,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.gdfn = GDFN(dim, ffn_expansion_factor=4)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if not self.enabled:
            return x
        x_mid = rearrange(
            self.norm1(x),
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )
        x_mid = rearrange(
            self.s_msa(x_mid),
            "b c h w -> b (h w) c",
        )
        x = x + x_mid
        return x + self.gdfn(self.norm2(x), height, width)


class CSWinB_CrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        reso: int | tuple[int, int],
        num_heads: int,
        split_size: int = 7,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        last_stage: bool = False,
    ) -> None:
        super().__init__()
        del drop_path
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = _resolution_tuple(reso)
        self.split_size = split_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if min(self.patches_resolution) <= split_size:
            last_stage = True
        self.branch_num = 1 if last_stage else 2
        self.proj = nn.Linear(2 * dim, dim)

        if self.branch_num != 2:
            raise ValueError(
                "HSIFormer cross-shaped attention needs two stripe branches. "
                "Increase the feature resolution or reduce split_size."
            )

        self.attns = nn.ModuleList(
            [
                LePEAttentionCross(
                    dim // 2,
                    resolution=self.patches_resolution,
                    idx=index,
                    split_size=split_size,
                    num_heads=num_heads // 2,
                    dim_out=dim // 2,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                )
                for index in range(self.branch_num)
            ]
        )
        self.crossattns = nn.ModuleList(
            [
                CSWinCrossAttention(
                    dim // 2,
                    resolution=self.patches_resolution,
                    idx=index,
                    split_size=split_size,
                    num_heads=num_heads // 2,
                    dim_out=dim // 2,
                    qk_scale=qk_scale,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                )
                for index in range(self.branch_num)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        resolution: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        resolution = resolution or self.patches_resolution
        batch, tokens, channels = x.shape
        if tokens != resolution[0] * resolution[1]:
            raise ValueError("Flattened token count does not match feature resolution.")

        qkv = (
            self.qkv(x)
            .reshape(batch, -1, 3, channels)
            .permute(2, 0, 1, 3)
        )
        x1 = self.attns[0](
            qkv[:, :, :, : channels // 2],
            cross=False,
            resolution=resolution,
        )
        x2 = self.attns[1](
            qkv[:, :, :, channels // 2 :],
            cross=False,
            resolution=resolution,
        )
        x3 = self.crossattns[0](x1, x2, resolution=resolution)
        x4 = self.crossattns[1](x2, x1, resolution=resolution)
        return self.proj(torch.cat([x1, x2, x3, x4], dim=2))

    def change_resolution(
        self,
        new_resolution: int | tuple[int, int],
    ) -> None:
        """Compatibility shim; cleaned forwards pass resolution explicitly."""
        self.patches_resolution = _resolution_tuple(new_resolution)


class Spatial_MSAB(nn.Module):
    def __init__(
        self,
        dim: int,
        head: int,
        resolution: tuple[int, int],
        split_size: int,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.resolution = tuple(resolution)
        self.norm1 = nn.LayerNorm(dim)
        self.cswin = CSWinB_CrossAttn(
            dim,
            reso=resolution,
            num_heads=head,
            split_size=split_size,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.sgfn = SGFN(dim, dim * 4, dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if not self.enabled:
            return x
        resolution = (height, width)
        x = x + self.cswin(self.norm1(x), resolution=resolution)
        return x + self.sgfn(self.norm2(x), height, width)


class SST(nn.Module):
    def __init__(
        self,
        dim: int,
        head: int,
        resolution: tuple[int, int],
        split_size: int,
        *,
        spectral_rpe: RPEMode = "legacy_post_softmax",
        use_spectral_attention: bool = True,
        use_spatial_attention: bool = True,
    ) -> None:
        super().__init__()
        self.resolution = tuple(resolution)
        self.spectral_msa = Spectral_MSAB(
            dim,
            head,
            rpe_mode=spectral_rpe,
            enabled=use_spectral_attention,
        )
        self.spatial_msa = Spatial_MSAB(
            dim,
            head,
            resolution,
            split_size,
            enabled=use_spatial_attention,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.spectral_msa(x, height, width)
        x = self.spatial_msa(x, height, width)
        return rearrange(
            x,
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )


class ChannelAttn(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden_dim = max(1, dim // 8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([self.maxpool(x), self.avgpool(x)], dim=1)
        return x * torch.sigmoid(self.channel_interaction(pooled))


class SSTB(nn.Module):
    def __init__(
        self,
        dim: int,
        head: int,
        resolution: tuple[int, int],
        split_size: int,
        *,
        spectral_rpe: RPEMode = "legacy_post_softmax",
        residual_mode: ResidualMode = "legacy",
        use_spectral_attention: bool = True,
        use_spatial_attention: bool = True,
    ) -> None:
        super().__init__()
        self.residual_mode = residual_mode
        self.channel_attn = ChannelAttn(dim)
        self.sst = SST(
            dim,
            head,
            resolution,
            split_size,
            spectral_rpe=spectral_rpe,
            use_spectral_attention=use_spectral_attention,
            use_spatial_attention=use_spatial_attention,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.channel_attn(x)
        if self.residual_mode == "legacy":
            return attended + self.sst(attended)
        return x + self.sst(attended)


class SSTLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        head: int,
        resolution: tuple[int, int],
        split_size: int,
        num_blocks: int,
        *,
        spectral_rpe: RPEMode = "legacy_post_softmax",
        residual_mode: ResidualMode = "legacy",
        use_spectral_attention: bool = True,
        use_spatial_attention: bool = True,
    ) -> None:
        super().__init__()
        self.residual_mode = residual_mode
        self.model = nn.ModuleList(
            [
                SSTB(
                    dim,
                    head,
                    resolution,
                    split_size,
                    spectral_rpe=spectral_rpe,
                    residual_mode=residual_mode,
                    use_spectral_attention=use_spectral_attention,
                    use_spatial_attention=use_spatial_attention,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model:
            if self.residual_mode == "legacy":
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class CATLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        ipsa_attn_drop: float = 0.0,
        cpsa_attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        *,
        spectral_rpe: RPEMode = "legacy_post_softmax",
        cat_rpe: bool = True,
        use_spectral_attention: bool = True,
        use_spatial_attention: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_spatial_attention = use_spatial_attention

        self.pre_ipsa_blocks = nn.ModuleList()
        self.cpsa_blocks = nn.ModuleList()
        self.post_ipsa_blocks = nn.ModuleList()
        self.spectral_blocks = nn.ModuleList()
        for _ in range(depth):
            self.pre_ipsa_blocks.append(
                CATBlock(
                    dim=dim,
                    input_resolution=self.input_resolution,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=ipsa_attn_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    attn_type="ipsa",
                    rpe=cat_rpe,
                )
            )
            self.cpsa_blocks.append(
                CATBlock(
                    dim=dim,
                    input_resolution=self.input_resolution,
                    num_heads=1,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=cpsa_attn_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    attn_type="cpsa",
                    rpe=False,
                )
            )
            self.post_ipsa_blocks.append(
                CATBlock(
                    dim=dim,
                    input_resolution=self.input_resolution,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=ipsa_attn_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    attn_type="ipsa",
                    rpe=cat_rpe,
                )
            )
            self.spectral_blocks.append(
                Spectral_MSAB(
                    dim,
                    num_heads,
                    rpe_mode=spectral_rpe,
                    enabled=use_spectral_attention,
                )
            )
        self.downsample = downsample

    def _run_cat(
        self,
        block: CATBlock,
        x: torch.Tensor,
        resolution: tuple[int, int],
    ) -> torch.Tensor:
        if not self.use_spatial_attention:
            return x
        if self.use_checkpoint and self.training:
            return checkpoint(
                lambda value: block(value, resolution),
                x,
                use_reentrant=False,
            )
        return block(x, resolution)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        resolution = (height, width)
        x = rearrange(x, "b c h w -> b (h w) c")
        for pre, cross, post, spectral in zip(
            self.pre_ipsa_blocks,
            self.cpsa_blocks,
            self.post_ipsa_blocks,
            self.spectral_blocks,
            strict=True,
        ):
            x = self._run_cat(pre, x, resolution)
            x = self._run_cat(cross, x, resolution)
            x = self._run_cat(post, x, resolution)
            x = spectral(x, height, width)
        if self.downsample is not None:
            x = self.downsample(x)
        return rearrange(
            x,
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )

    def change_resolution(self, new_resolution: tuple[int, int]) -> None:
        self.input_resolution = tuple(new_resolution)


class DownSample(nn.Module):
    def __init__(
        self,
        inchannel: int,
        outchannel: int,
        pixel_shuffle: bool = True,
    ) -> None:
        super().__init__()
        if pixel_shuffle:
            self.model = nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1),
                nn.PixelUnshuffle(2),
                nn.Conv2d(
                    inchannel * 4,
                    outchannel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(
                    inchannel,
                    outchannel,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UpSample(nn.Module):
    def __init__(
        self,
        inchannel: int,
        outchannel: int,
        pixel_shuffle: bool = True,
    ) -> None:
        super().__init__()
        if pixel_shuffle:
            self.model = nn.Sequential(
                nn.Conv2d(
                    inchannel,
                    outchannel * 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.PixelShuffle(2),
            )
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(
                    inchannel,
                    outchannel,
                    stride=2,
                    kernel_size=2,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SSTransformer(nn.Module):
    """Clean, checkpoint-compatible implementation of the published model."""

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 31,
        hidden_dim: int = 32,
        split_size: int = 1,
        input_resolution: tuple[int, int] | list[int] = (128, 128),
        n_blocks: tuple[int, ...] | list[int] = (1, 2, 3),
        bottle_depth: int = 4,
        n_refine: int = 2,
        patch_size: int = 8,
        *,
        spectral_rpe: RPEMode = "legacy_post_softmax",
        cat_rpe: bool = True,
        residual_mode: ResidualMode = "legacy",
        use_spectral_attention: bool = True,
        use_spatial_attention: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if residual_mode not in {"legacy", "paper"}:
            raise ValueError(f"Unknown residual mode: {residual_mode}")

        resolution = _resolution_tuple(input_resolution)
        encoder_depths = tuple(n_blocks)
        self.embed = nn.Conv2d(in_dim, hidden_dim, 3, 1, 1)
        self.head = 2
        self.split_size = split_size
        self.input_resolution = resolution
        self.stage = len(encoder_depths)
        deepest_split = split_size * 2 ** max(0, self.stage - 1)
        self.min_window = math.lcm(
            patch_size * 2**self.stage,
            deepest_split * 2**self.stage,
        )

        new_channels = hidden_dim
        current_resolution = resolution
        current_split = split_size
        current_heads = self.head
        self.downblocks = nn.ModuleList()
        for block_count in encoder_depths:
            previous_channels = new_channels
            new_channels *= 2
            self.downblocks.append(
                nn.ModuleList(
                    [
                        SSTLayer(
                            previous_channels,
                            current_heads,
                            current_resolution,
                            current_split,
                            block_count,
                            spectral_rpe=spectral_rpe,
                            residual_mode=residual_mode,
                            use_spectral_attention=use_spectral_attention,
                            use_spatial_attention=use_spatial_attention,
                        ),
                        DownSample(previous_channels, new_channels),
                    ]
                )
            )
            current_resolution = (
                current_resolution[0] // 2,
                current_resolution[1] // 2,
            )
            current_split *= 2
            current_heads *= 2

        self.bottle_layer = CATLayer(
            new_channels,
            current_resolution,
            bottle_depth,
            current_heads,
            patch_size,
            use_checkpoint=use_checkpoint,
            spectral_rpe=spectral_rpe,
            cat_rpe=cat_rpe,
            use_spectral_attention=use_spectral_attention,
            use_spatial_attention=use_spatial_attention,
        )

        self.upblocks = nn.ModuleList()
        for block_count in reversed(encoder_depths):
            previous_channels = new_channels
            new_channels //= 2
            current_resolution = (
                current_resolution[0] * 2,
                current_resolution[1] * 2,
            )
            current_split //= 2
            current_heads //= 2
            self.upblocks.append(
                nn.ModuleList(
                    [
                        UpSample(previous_channels, new_channels),
                        nn.Conv2d(
                            new_channels * 2,
                            new_channels,
                            1,
                            1,
                            bias=False,
                        ),
                        SSTLayer(
                            new_channels,
                            current_heads,
                            current_resolution,
                            current_split,
                            block_count,
                            spectral_rpe=spectral_rpe,
                            residual_mode=residual_mode,
                            use_spectral_attention=use_spectral_attention,
                            use_spatial_attention=use_spatial_attention,
                        ),
                    ]
                )
            )

        self.refine_sst = SSTLayer(
            new_channels,
            current_heads,
            current_resolution,
            current_split,
            n_refine,
            spectral_rpe=spectral_rpe,
            residual_mode=residual_mode,
            use_spectral_attention=use_spectral_attention,
            use_spatial_attention=use_spatial_attention,
        )
        self.to_out = nn.Conv2d(new_channels, out_dim, 3, 1, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, input_height, input_width = x.shape
        x = _pad_to_square_multiple(x, self.min_window)
        x = self.embed(x)
        residual = x

        encoder_features: list[torch.Tensor] = []
        for encoder, downsample in self.downblocks:
            x = encoder(x)
            encoder_features.append(x)
            x = downsample(x)

        x = self.bottle_layer(x)
        for (upsample, fusion, decoder), skip in zip(
            self.upblocks,
            reversed(encoder_features),
            strict=True,
        ):
            x = upsample(x)
            x = fusion(torch.cat([x, skip], dim=1))
            x = decoder(x)

        x = self.refine_sst(x) + residual
        return self.to_out(x)[:, :, :input_height, :input_width]


class HSIFormer(SSTransformer):
    """Paper name for :class:`SSTransformer`."""


class SST_Multi_Stage(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 31,
        hidden_dim: int = 32,
        split_size: int = 1,
        input_resolution: tuple[int, int] | list[int] = (128, 128),
        patch_size: int = 8,
        stage: int = 3,
        **model_kwargs: object,
    ) -> None:
        super().__init__()
        self.embed = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.body = nn.ModuleList(
            [
                SSTransformer(
                    out_dim,
                    out_dim,
                    hidden_dim,
                    split_size=split_size,
                    input_resolution=input_resolution,
                    patch_size=patch_size,
                    n_blocks=(1, 1),
                    bottle_depth=1,
                    n_refine=1,
                    **model_kwargs,
                )
                for _ in range(stage)
            ]
        )
        self.min_window = patch_size * 4
        self.out = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, input_height, input_width = x.shape
        x = _pad_to_square_multiple(x, self.min_window)
        x = self.embed(x)
        residual = x
        for layer in self.body:
            x = layer(x)
        x = residual + self.out(x)
        return x[:, :, :input_height, :input_width]


def _pad_to_square_multiple(
    x: torch.Tensor,
    multiple: int,
) -> torch.Tensor:
    height, width = x.shape[-2:]
    side = max(height, width)
    padded_side = math.ceil(side / multiple) * multiple
    pad_height = padded_side - height
    pad_width = padded_side - width
    if not pad_height and not pad_width:
        return x
    mode = "reflect"
    if pad_height >= height or pad_width >= width:
        mode = "replicate"
    return F.pad(x, (0, pad_width, 0, pad_height), mode=mode)


def _resolution_tuple(
    resolution: int | tuple[int, int] | list[int],
) -> tuple[int, int]:
    if isinstance(resolution, int):
        return (resolution, resolution)
    return (int(resolution[0]), int(resolution[1]))


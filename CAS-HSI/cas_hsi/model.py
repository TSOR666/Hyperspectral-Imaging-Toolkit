"""CAS-HSI: three-resolution encoder-decoder for RGB -> HSI (specification section 3).

    RGB --> [ linear prior ]------------------------------------------+
        |                                                             |
        +-> stem -> CAS-Lite -> down -> CAS -> down -> CAS bottleneck |
                       |                 |                            |
                       S0                S1                           |
                       |                 |                            |
                    CAS-Lite <- up <-- CAS <- up <---------------+    |
                       |                                              |
                  refinement -> Conv3x3 -> spectral residual ---------+
                                                                      |
                                                                      v
                                                            prior + residual

Spatial attention is confined to H/2 and H/4; full resolution uses CAS-Lite. The
prediction is a learned linear RGB->HSI prior plus a near-zero-initialized deep
residual. The residual formulation exposes a direct, trainable colour-to-spectrum
path; it is not a calibrated physical prior until trained on a camera/dataset pair.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint

from .blocks.cas_block import CASBlock, build_bottleneck
from .blocks.cas_lite_block import CASLiteBlock
from .config import CASHSIConfig, variant_config
from .layers.padding import PadInfo, crop_to_original, pad_to_multiple
from .layers.spectral_head import RGBPrior, build_spectral_head

__all__ = [
    "CASHSI",
    "build_cas_hsi",
    "build_edge_model",
    "create_cas_hsi_tiny",
    "create_cas_hsi_base",
]

FEATURE_STAGES = ("encoder_full", "encoder_half", "bottleneck", "decoder_half", "decoder_full")


class _Stack(nn.Module):
    """A run of blocks, optionally gradient-checkpointed.

    ``nn.Sequential`` would be enough, but routing through one module keeps
    checkpointing (a pure memory/compute trade, never a semantic change) in one
    place instead of sprinkling it through the forward pass.
    """

    def __init__(self, blocks: nn.ModuleList | nn.Sequential, use_checkpoint: bool = False):
        super().__init__()
        self.blocks = blocks
        self.use_checkpoint = bool(use_checkpoint)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = torch_checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class CASHSI(nn.Module):
    """Convolutional Attention Stack for hyperspectral reconstruction."""

    def __init__(self, config: CASHSIConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()

        if config is None:
            config = CASHSIConfig()
        elif isinstance(config, Mapping):
            config = CASHSIConfig.from_dict(config)
        # Own the config outright. Sharing the caller's object means anything that
        # mutates ours -- replace_attention_mixers() flips `backend` to "edge" -- would
        # reach back and silently rewrite the caller's config, and every model later
        # built from it. (Caught exactly that way: it corrupted a shared test fixture.)
        self.config = copy.deepcopy(config)
        config = self.config

        c = config.base_width
        bands = config.output_bands

        # --- entry points -----------------------------------------------------
        self.stem = nn.Conv2d(
            config.input_channels, c, kernel_size=3, stride=1, padding=1
        )
        self.rgb_prior = RGBPrior(config.input_channels, bands)

        # --- encoder ----------------------------------------------------------
        self.encoder_full = self._lite_stack(c, config.depths.encoder_full)
        self.down_1 = _import_downsample()(c, 2 * c)
        self.encoder_half = self._cas_stack(
            2 * c, config.depths.encoder_half, config.half_mixer(), config.dilations_half
        )

        self.down_2 = _import_downsample()(2 * c, 4 * c)

        # --- bottleneck -------------------------------------------------------
        self.bottleneck = _Stack(
            build_bottleneck(config, channels=4 * c), use_checkpoint=config.use_checkpoint
        )

        # --- decoder ----------------------------------------------------------
        self.up_1 = _import_upsample()(4 * c, 2 * c)
        self.skip_fusion_1 = nn.Conv2d(4 * c, 2 * c, kernel_size=1)
        self.decoder_half = self._cas_stack(
            2 * c, config.depths.decoder_half, config.half_mixer(), config.dilations_half
        )

        self.up_2 = _import_upsample()(2 * c, c)
        self.skip_fusion_0 = nn.Conv2d(2 * c, c, kernel_size=1)
        self.decoder_full = self._lite_stack(c, config.depths.decoder_full)

        self.refinement = self._lite_stack(c, config.depths.refinement)

        # --- head -------------------------------------------------------------
        self.spectral_head = build_spectral_head(
            config.spectral_head,
            feature_channels=c,
            output_bands=bands,
            rank=config.spectral_rank,
            residual_scale=config.spectral_residual_scale,
            init_std=config.head_init_std,
        )

        # Deliberately NO blanket re-initialization pass here.
        #
        # The spec prescribes exactly two custom inits (7.3): Xavier on the RGB
        # prior, near-zero on the residual spectral head. Both modules do that
        # themselves. A blanket `self.apply(kaiming)` would run *after* them and
        # silently overwrite both -- destroying the "network starts equal to the
        # linear prior" property that the whole residual formulation rests on.
        # Every other conv keeps PyTorch's default init, which is well-behaved for
        # the depthwise/pointwise mix used here; the residual branches are already
        # near-zero via LayerScale.

    # ------------------------------------------------------------------ build --
    def _lite_stack(self, channels: int, depth: int) -> _Stack:
        config = self.config
        blocks = nn.ModuleList(
            [
                CASLiteBlock(
                    channels=channels,
                    head_dim=config.head_dim,
                    ffn_expansion=config.ffn_expansion,
                    layer_scale_init=config.layer_scale_init,
                    spatial_kernel=config.spatial_kernel,
                    drop_path=config.drop_path,
                    norm=config.norm,
                    norm_eps=config.norm_eps,
                    reparam=config.lite_reparam,
                    fp32_attention=config.fp32_attention,
                    ffn_use_activation=config.ffn_use_activation,
                    ffn_gate=config.ffn_gate,
                    softplus_temperature=config.softplus_temperature,
                )
                for _ in range(depth)
            ]
        )
        return _Stack(blocks, use_checkpoint=config.use_checkpoint)

    def _cas_stack(
        self,
        channels: int,
        depth: int,
        mixer: str,
        dilations: Tuple[int, ...],
    ) -> _Stack:
        config = self.config
        blocks = nn.ModuleList(
            [
                CASBlock(
                    channels=channels,
                    spatial_mixer=mixer,
                    head_dim=config.head_dim,
                    dilations=dilations,
                    ffn_expansion=config.ffn_expansion,
                    layer_scale_init=config.layer_scale_init,
                    drop_path=config.drop_path,
                    norm=config.norm,
                    norm_eps=config.norm_eps,
                    kernel_size=config.attention_kernel_size,
                    spatial_kernel=config.spatial_kernel,
                    large_kernel=config.large_kernel,
                    stripe_width=config.stripe_width,
                    relative_position_bias=config.relative_position_bias,
                    mask_padding=config.mask_padding,
                    fp32_attention=config.fp32_attention,
                    ffn_use_activation=config.ffn_use_activation,
                    ffn_gate=config.ffn_gate,
                    softplus_temperature=config.softplus_temperature,
                )
                for _ in range(depth)
            ]
        )
        return _Stack(blocks, use_checkpoint=config.use_checkpoint)

    # ---------------------------------------------------------------- forward --
    def forward_features(
        self, rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], PadInfo, torch.Tensor]:
        """Run the backbone and return the final feature map plus stage features.

        The padded RGB is returned alongside so the caller can feed the *same*
        padded tensor to the linear prior -- padding twice would be wasted work and
        an opportunity for the two paths to disagree.

        The stage features are what feature distillation consumes (spec 9.4).
        """
        rgb_padded, pad_info = pad_to_multiple(rgb, multiple=self.config.size_multiple)

        features, stage_features = self._forward_padded_features(rgb_padded)
        return features, stage_features, pad_info, rgb_padded

    def _forward_padded_features(
        self, rgb_padded: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Backbone on an input whose spatial axes are already multiples of four.

        This is intentionally private: ordinary callers must use :meth:`forward`,
        which applies the reflect/replicate boundary policy and crops back to their
        original extent. The ONNX export wrapper is the one exception because a
        portable ONNX graph cannot express arbitrary-size reflect-pad-to-modulo
        preprocessing with PyTorch's current symbolic-shape exporter.
        """

        x0 = self.stem(rgb_padded)
        s0 = self.encoder_full(x0)

        x1 = self.down_1(s0)
        s1 = self.encoder_half(x1)

        x2 = self.down_2(s1)
        x2 = self.bottleneck(x2)

        y1 = self.up_1(x2)
        y1 = self.skip_fusion_1(torch.cat([y1, s1], dim=1))
        y1 = self.decoder_half(y1)

        y0 = self.up_2(y1)
        y0 = self.skip_fusion_0(torch.cat([y0, s0], dim=1))
        y0 = self.decoder_full(y0)
        y0 = self.refinement(y0)

        features = {
            "encoder_full": s0,
            "encoder_half": s1,
            "bottleneck": x2,
            "decoder_half": y1,
            "decoder_full": y0,
        }
        return y0, features

    def forward_padded(self, rgb_padded: torch.Tensor) -> torch.Tensor:
        """Run a pre-padded image without internal padding or final cropping.

        ``rgb_padded`` must be NCHW with both spatial dimensions divisible by four.
        It is an export primitive, not the general inference API; use
        :meth:`forward` for arbitrary-size eager inference.
        """
        features, _ = self._forward_padded_features(rgb_padded)
        return self.rgb_prior(rgb_padded) + self.spectral_head(features)

    def forward(
        self,
        rgb: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if rgb.dim() != 4:
            raise ValueError(f"Expected a 4-D NCHW RGB tensor, got shape {tuple(rgb.shape)}")
        if rgb.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected {self.config.input_channels} input channels, got {rgb.shape[1]}"
            )

        features, stage_features, pad_info, rgb_padded = self.forward_features(rgb)

        # The prior sees the *padded* RGB so it lines up with the residual; both are
        # cropped together at the end.
        prior = self.rgb_prior(rgb_padded)
        residual = self.spectral_head(features)

        # No activation after the 31-band projection (spec 7.4).
        output = prior + residual
        output = crop_to_original(output, pad_info)

        if return_features:
            return output, stage_features
        return output

    # ------------------------------------------------------------------ info ---
    def get_model_info(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        return {
            "name": self.config.name,
            "architecture": "CAS-HSI",
            "backend": self.config.backend,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "total_memory_mb": (param_bytes + buffer_bytes) / (1024 ** 2),
            "base_width": self.config.base_width,
            "widths": self.config.widths,
            "output_bands": self.config.output_bands,
            "depths": self.config.depths.to_dict(),
            "bottleneck_mixers": self.config.bottleneck_mixers(),
            "half_mixer": self.config.half_mixer(),
        }


def _import_downsample():
    from .layers.downsample import PixelUnshuffleDownsample

    return PixelUnshuffleDownsample


def _import_upsample():
    from .layers.upsample import PixelShuffleUpsample

    return PixelShuffleUpsample


def build_cas_hsi(
    variant: str | CASHSIConfig | Mapping[str, Any] = "tiny",
    **overrides: Any,
) -> CASHSI:
    """Build a model from a variant name, a config object, or a mapping."""
    if isinstance(variant, CASHSIConfig):
        config = variant
        if overrides:
            data = config.to_dict()
            data.pop("channels", None)
            data.update(overrides)
            config = CASHSIConfig.from_dict(data)
    elif isinstance(variant, Mapping):
        data = dict(variant)
        data.update(overrides)
        config = CASHSIConfig.from_dict(data)
    else:
        config = variant_config(variant, **overrides)
    return CASHSI(config)


def build_edge_model(
    variant: str | CASHSIConfig | Mapping[str, Any] = "tiny",
    **overrides: Any,
) -> CASHSI:
    """Build the deployment model: same topology, no attention spatial mixers (spec 9.2).

    This is a *fresh* model with convolutional mixers. To convert an already
    trained research model in place, use
    :func:`cas_hsi.deployment.replace_attention.replace_attention_mixers`.
    """
    if isinstance(variant, CASHSIConfig):
        data = variant.to_dict()
        data.pop("channels", None)
        data.update(overrides)
        config = CASHSIConfig.from_dict(data)
    elif isinstance(variant, Mapping):
        data = dict(variant)
        data.update(overrides)
        config = CASHSIConfig.from_dict(data)
    else:
        config = variant_config(variant, **overrides)
    edge_config = config.as_edge()
    return CASHSI(edge_config)


def create_cas_hsi_tiny(**overrides: Any) -> CASHSI:
    return build_cas_hsi("tiny", **overrides)


def create_cas_hsi_base(**overrides: Any) -> CASHSI:
    return build_cas_hsi("base", **overrides)

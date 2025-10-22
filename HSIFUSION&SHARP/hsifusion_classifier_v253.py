from __future__ import annotations

"""Classifier-focused variant of HSIFusionNet v2.5.3."""

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from common_utils_v32 import validate_model_config
from hsifusion_v252_complete import HSIFusionNetV25LightningPro, LightningProConfig


_TV = version.parse(torch.__version__)
_T20 = _TV >= version.parse("2.0.0")
_T21 = _TV >= version.parse("2.1.0")


@dataclass
class LightningProClassifierHeadConfig:
    """Configuration for the classification head attached to HSIFusion encoder."""

    num_classes: int = 10
    pooling: str = "avg"
    use_multi_scale: bool = True
    hidden_dim: Optional[int] = None
    dropout: float = 0.0
    use_layernorm: bool = True
    selected_scales: Optional[Sequence[int]] = None


class HSIFusionNetV25LightningProClassifier(HSIFusionNetV25LightningPro):
    """HSIFusionNet encoder repurposed for hyperspectral classification."""

    def __init__(
        self,
        backbone_config: LightningProConfig,
        head_config: Optional[LightningProClassifierHeadConfig] = None,
    ) -> None:
        head_config = head_config or LightningProClassifierHeadConfig()
        backbone_config = copy.deepcopy(backbone_config)
        backbone_config.estimate_uncertainty = False
        backbone_config.out_channels = head_config.num_classes

        super().__init__(backbone_config)

        self.head_config = head_config
        self.pooling_mode = head_config.pooling.lower()
        if self.pooling_mode not in {"avg", "max", "avgmax"}:
            raise ValueError(
                "Invalid pooling mode. Choose from 'avg', 'max', or 'avgmax'."
            )

        self.selected_scales = self._resolve_scales(head_config)
        pooling_multiplier = 2 if self.pooling_mode == "avgmax" else 1
        feature_dims = [self.dims[idx] for idx in self.selected_scales]
        self.classifier_in_dim = sum(feature_dims) * pooling_multiplier

        self.normalizer = (
            nn.LayerNorm(self.classifier_in_dim)
            if head_config.use_layernorm
            else nn.Identity()
        )

        if head_config.hidden_dim is not None and head_config.hidden_dim <= 0:
            raise ValueError('hidden_dim must be a positive integer when provided')

        self._uses_hidden = (
            head_config.hidden_dim is not None and head_config.hidden_dim > 0
        )
        if self._uses_hidden:
            hidden_dim = head_config.hidden_dim  # type: ignore[assignment]
            layers: List[nn.Module] = [
                nn.Linear(self.classifier_in_dim, hidden_dim),
                nn.GELU(),
            ]
            if head_config.dropout > 0:
                layers.append(nn.Dropout(head_config.dropout))
            layers.append(nn.Linear(hidden_dim, head_config.num_classes))
            self.head = nn.Sequential(*layers)
            self.feature_dropout = nn.Identity()
        else:
            self.head = nn.Linear(self.classifier_in_dim, head_config.num_classes)
            self.feature_dropout = (
                nn.Dropout(head_config.dropout)
                if head_config.dropout > 0
                else nn.Identity()
            )

        self._strip_reconstruction_branches()
        self.head.apply(self._init_weights)

        self.num_classes = head_config.num_classes
        self.embedding_dim = self.classifier_in_dim

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _resolve_scales(
        self, head_config: LightningProClassifierHeadConfig
    ) -> Tuple[int, ...]:
        if head_config.selected_scales is not None:
            if not isinstance(head_config.selected_scales, Iterable):
                raise TypeError("selected_scales must be an iterable of integers")
            selected = sorted({int(idx) for idx in head_config.selected_scales})
        elif head_config.use_multi_scale:
            selected = list(range(len(self.dims)))
        else:
            selected = [len(self.dims) - 1]

        if not selected:
            raise ValueError("At least one scale must be selected for the classifier head")

        max_index = len(self.dims) - 1
        for idx in selected:
            if idx < 0 or idx > max_index:
                raise ValueError(
                    f"Scale index {idx} is out of range for encoder with {len(self.dims)} stages"
                )

        return tuple(selected)

    def _strip_reconstruction_branches(self) -> None:
        for name in ("decoder_stages", "upsample_layers", "cross_attns", "output_head", "uncertainty_head"):
            if hasattr(self, name):
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    self._modules.pop(name, None)
                setattr(self, name, None)

    def forward_decoder(self, x: torch.Tensor, encoder_features: List[torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        raise RuntimeError(
            "HSIFusionNetV25LightningProClassifier removes the reconstruction decoder."
        )

    def _encode_input(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (NCHW), got {x.ndim}D")

        _, _, H, W = x.shape
        if H < self.config.min_input_size or W < self.config.min_input_size:
            raise ValueError(
                f"Input size {H}x{W} below minimum {self.config.min_input_size}"
            )

        if self.config.use_channels_last:
            x = x.to(memory_format=torch.channels_last)

        _, encoder_features = self.forward_encoder(x)
        return encoder_features

    def _pool_feature_map(self, feat: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == "avg":
            pooled = F.adaptive_avg_pool2d(feat, output_size=1)
        elif self.pooling_mode == "max":
            pooled = F.adaptive_max_pool2d(feat, output_size=1)
        else:  # avgmax
            pooled = torch.cat(
                [
                    F.adaptive_avg_pool2d(feat, output_size=1),
                    F.adaptive_max_pool2d(feat, output_size=1),
                ],
                dim=1,
            )
        return pooled.flatten(1)

    def _aggregate_features(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        selected = [encoder_features[idx] for idx in self.selected_scales]
        pooled = [self._pool_feature_map(feat) for feat in selected]
        if len(pooled) == 1:
            return pooled[0]
        return torch.cat(pooled, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
        return_selected_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        if hasattr(self, "_lazy_compile_config") and not hasattr(self, "_compiled"):
            self._compile_forward()

        encoder_features = self._encode_input(x)
        aggregated = self._aggregate_features(encoder_features)
        embedding = self.normalizer(aggregated)
        head_input = self.feature_dropout(embedding)
        logits = self.head(head_input)

        outputs: List[Any] = [logits]
        if return_embeddings:
            outputs.append(embedding)
        if return_selected_features:
            outputs.append([encoder_features[idx] for idx in self.selected_scales])

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)  # type: ignore[return-value]

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        encoder_features = self._encode_input(x)
        aggregated = self._aggregate_features(encoder_features)
        return self.normalizer(aggregated)


def create_hsifusion_lightning_classifier(
    model_size: str = "base",
    in_channels: int = 3,
    num_classes: int = 10,
    compile_mode: Optional[str] = None,
    rank: int = 0,
    skip_compile_small_inputs: bool = True,
    expected_min_size: Optional[int] = None,
    lazy_compile: bool = False,
    force_compile: bool = False,
    classifier_head: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> HSIFusionNetV25LightningProClassifier:
    size_configs: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "base_channels": 64,
            "depths": [2, 2, 4, 2],
            "num_heads": 4,
            "num_experts": 2,
            "use_moe": False,
        },
        "small": {
            "base_channels": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": 6,
            "num_experts": 4,
            "use_moe": False,
        },
        "base": {
            "base_channels": 128,
            "depths": [2, 2, 8, 2],
            "num_heads": 8,
            "num_experts": 4,
            "use_moe": True,
        },
        "large": {
            "base_channels": 192,
            "depths": [2, 2, 12, 2],
            "num_heads": 12,
            "num_experts": 6,
            "use_moe": True,
        },
        "xlarge": {
            "base_channels": 256,
            "depths": [2, 4, 16, 2],
            "num_heads": 16,
            "num_experts": 8,
            "use_moe": True,
        },
    }

    if model_size not in size_configs:
        raise ValueError(
            f"Unknown model size: {model_size}. Choose from {list(size_configs.keys())}"
        )

    config_kwargs = {**size_configs[model_size], **kwargs}
    config_kwargs["in_channels"] = in_channels
    config_kwargs.setdefault("out_channels", num_classes)

    backbone_config = LightningProConfig(**config_kwargs)

    try:
        validate_model_config(backbone_config, f"HSIFusion-Classifier-{model_size}")
    except ValueError as exc:
        raise ValueError(f"Configuration validation failed: {exc}") from exc

    head_kwargs: Dict[str, Any] = dict(classifier_head or {})
    head_kwargs.setdefault("num_classes", num_classes)
    head_config = LightningProClassifierHeadConfig(**head_kwargs)

    model = HSIFusionNetV25LightningProClassifier(backbone_config, head_config=head_config)

    compile_threshold = expected_min_size or backbone_config.min_input_size
    if skip_compile_small_inputs and compile_threshold < 128 and not force_compile:
        if rank == 0:
            print(f"Skipping compilation for expected size < 128 (got {compile_threshold})")
        compile_mode = None
    elif force_compile and rank == 0:
        print("Force compilation enabled - will compile regardless of input size")

    if compile_mode is None:
        compile_mode = "default" if backbone_config.compile_model else None

    if compile_mode and _T20 and not lazy_compile:
        try:
            backend = (
                backbone_config.compile_backend
                if backbone_config.compile_backend in {"inductor", "aot_eager"}
                else "inductor"
            )
            if _T21:
                compile_options = {"mode": compile_mode, "backend": backend, "dynamic": False}
            else:
                compile_options = {"mode": compile_mode, "backend": backend, "fullgraph": True}
            model = torch.compile(model, **compile_options)
            if rank == 0:
                print(
                    f"OK Classifier compiled with {compile_mode} mode, backend={backend}"
                )
        except Exception as exc:  # pragma: no cover - compile failure path
            if rank == 0:
                warnings.warn(f"Compilation failed: {exc}")
    elif lazy_compile and compile_mode:
        model._lazy_compile_config = {
            "mode": compile_mode,
            "backend": backbone_config.compile_backend
            if backbone_config.compile_backend in {"inductor", "aot_eager"}
            else "inductor",
            "rank": rank,
        }
        if rank == 0:
            print("OK Lazy compilation enabled for classifier (will compile on first forward)")

    if rank == 0:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        selected_scales = ",".join(str(idx) for idx in model.selected_scales)
        print(f"\n HSIFusionNet v2.5.3 Lightning Pro Classifier ({model_size})")
        print(f"-  Parameters: {params / 1e6:.2f}M")
        print(
            f"-  Multi-scale aggregation: {'ON' if len(model.selected_scales) > 1 else 'OFF'}"
        )
        print(f"-  Selected scales: {selected_scales}")
        print(f"-  Pooling mode: {model.pooling_mode}")
        print(f"-  Classifier input dim: {model.classifier_in_dim}")
        hidden_info = head_config.hidden_dim if head_config.hidden_dim else "None"
        print(f"-  Hidden dim: {hidden_info}")
        print(f"-  Num classes: {head_config.num_classes}")

    return model


__all__ = [
    "LightningProClassifierHeadConfig",
    "HSIFusionNetV25LightningProClassifier",
    "create_hsifusion_lightning_classifier",
]

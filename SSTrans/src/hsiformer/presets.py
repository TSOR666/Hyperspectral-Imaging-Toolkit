from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .attention import RPEMode
from .model import HSIFormer, ResidualMode

PresetName = Literal[
    "legacy",
    "ablation_no_rpe",
    "corrected_rpe",
    "optimized_candidate",
]


@dataclass(frozen=True)
class HSIFormerConfig:
    in_dim: int = 3
    out_dim: int = 31
    hidden_dim: int = 32
    split_size: int = 1
    input_resolution: tuple[int, int] = (128, 128)
    n_blocks: tuple[int, ...] = (1, 2, 3)
    bottle_depth: int = 4
    n_refine: int = 2
    patch_size: int = 8
    spectral_rpe: RPEMode = "legacy_post_softmax"
    cat_rpe: bool = True
    residual_mode: ResidualMode = "legacy"
    use_spectral_attention: bool = True
    use_spatial_attention: bool = True
    use_checkpoint: bool = False

    def build(self) -> HSIFormer:
        return HSIFormer(**asdict(self))


def get_config(name: PresetName = "legacy") -> HSIFormerConfig:
    if name == "legacy":
        return HSIFormerConfig()
    if name == "ablation_no_rpe":
        return HSIFormerConfig(spectral_rpe="none")
    if name == "corrected_rpe":
        return HSIFormerConfig(spectral_rpe="pre_softmax")
    if name == "optimized_candidate":
        return HSIFormerConfig(
            spectral_rpe="none",
            cat_rpe=False,
            residual_mode="paper",
            use_checkpoint=True,
        )
    raise ValueError(f"Unknown preset: {name}")


def build_model(
    preset: PresetName = "legacy",
    **overrides: Any,
) -> HSIFormer:
    values = asdict(get_config(preset))
    values.update(overrides)
    return HSIFormer(**values)


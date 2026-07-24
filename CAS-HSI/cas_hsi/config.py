"""CAS-HSI configuration and construction-time validation (specification 3.2, 3.6).

Configuration is a plain dataclass so it can be round-tripped through YAML, saved
into checkpoints, and diffed on resume.  Every inconsistency is rejected at
construction time rather than surfacing as a shape error five layers deep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

__all__ = [
    "Depths",
    "CASHSIConfig",
    "validate_config",
    "TINY",
    "BASE",
    "VARIANTS",
]

_RESEARCH_BACKEND = "research"
_EDGE_BACKEND = "edge"
_BACKENDS = (_RESEARCH_BACKEND, _EDGE_BACKEND)


@dataclass
class Depths:
    """Blocks per stage (spec 3.2)."""

    encoder_full: int = 2
    encoder_half: int = 3
    bottleneck: int = 5
    decoder_half: int = 3
    decoder_full: int = 2
    refinement: int = 2

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Depths":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"Unknown depth key(s): {', '.join(unknown)}. Valid keys: {sorted(known)}"
            )
        return cls(**{key: int(value) for key, value in data.items()})


@dataclass
class CASHSIConfig:
    """Full model configuration.

    The three latent widths are derived from ``base_width`` (C, 2C, 4C).  A YAML
    file may still spell out ``channels: {full, half, quarter}``; those values are
    checked against the derived ones rather than silently overriding them, because
    a mismatch there would produce a network that does not match its own config.
    """

    name: str = "cas_hsi_tiny"

    # --- topology ---
    input_channels: int = 3
    output_bands: int = 31
    base_width: int = 32
    depths: Depths = field(default_factory=Depths)

    # --- block internals ---
    head_dim: int = 32
    ffn_expansion: float = 2.0
    # DEVIATES from spec 4.4's 1e-3, deliberately, and this default matters more than it
    # looks. LayerScale at 1e-3 gates EVERY residual branch of all ~17 blocks, and the
    # spectral head is separately initialized near zero (head_init_std). The two near-zero
    # gates multiply: measured on the tiny/research model, the fresh network's deviation
    # from an exactly affine map (||f(a+b) - (f(a)+f(b)-f(0))|| / ||f(a+b)-f(0)||) is
    # 4e-5 at 1e-3 versus 3.4e-2 at 1.0 -- i.e. at 1e-3 the model IS a linear RGB->31 map
    # to four decimal places, and its whole nonlinear capacity has to be discovered from
    # scratch by the optimizer. A linear map's MRAE floor on ARAD-1K is ~0.6-0.7, which is
    # exactly where a 46k-step run plateaued.
    #
    # 1e-3 is a VERY-DEEP-net (100+ layer) stabilizer; it over-damps this depth. MST++,
    # which this project is measured against, uses no LayerScale at all = effectively 1.0.
    # All three shipped configs already override this to 1.0; the dataclass now agrees with
    # them so that a run launched WITHOUT --config cannot silently build a crippled network.
    # Lower toward 0.1 only if you observe early-training divergence.
    layer_scale_init: float = 1.0
    drop_path: float = 0.0
    norm: str = "layernorm"
    norm_eps: float = 1.0e-6

    # --- spatial mixing ---
    backend: str = _RESEARCH_BACKEND
    dilations_half: Tuple[int, ...] = (1, 2)
    dilations_quarter: Tuple[int, ...] = (1, 2, 3)
    attention_kernel_size: int = 3
    spatial_kernel: int = 7          # CAS-Lite depthwise kernel (spec 5.2)
    large_kernel: int = 11           # edge stand-in for stripe attention (spec 9.3)
    enable_stripe_attention: bool = True
    stripe_frequency: int = 3        # one hybrid block every N bottleneck blocks
    stripe_width: int = 8
    relative_position_bias: bool = True
    mask_padding: bool = True
    fp32_attention: bool = True      # fp32 logits + softmax under autocast (spec 9.5)

    # --- FFN / channel attention options ---
    ffn_use_activation: bool = False
    ffn_gate: str = "simple"
    softplus_temperature: bool = False

    # --- spectral head (spec 7) ---
    spectral_head: str = "residual"  # 'residual' | 'low_rank'
    spectral_rank: int = 10
    spectral_residual_scale: float = 0.1
    head_init_std: float = 1.0e-3

    # --- misc ---
    lite_reparam: bool = False       # spec 5.4, optional
    use_checkpoint: bool = False     # gradient checkpointing (memory, not semantics)
    size_multiple: int = 4           # spec 3.6 requires exactly 4

    # ---------------------------------------------------------------- widths --
    @property
    def channels_full(self) -> int:
        return self.base_width

    @property
    def channels_half(self) -> int:
        return 2 * self.base_width

    @property
    def channels_quarter(self) -> int:
        return 4 * self.base_width

    @property
    def widths(self) -> List[int]:
        return [self.channels_full, self.channels_half, self.channels_quarter]

    # --------------------------------------------------------------- mixers ---
    def half_mixer(self) -> str:
        """Spatial mixer for the H/2 encoder and decoder stages (spec 9.2)."""
        if self.backend == _EDGE_BACKEND:
            return "dilated_depthwise_conv"
        return "dilated_local_attention"

    def full_mixer(self) -> str:
        """Spatial mixer at full resolution -- depthwise in both backends (spec 9.2)."""
        return "depthwise_7x7"

    def bottleneck_mixers(self) -> List[str]:
        """One mixer name per bottleneck block (spec 3.5).

        With ``stripe_frequency=3`` and 5 blocks: local, local, hybrid, local, local.
        """
        edge = self.backend == _EDGE_BACKEND
        plain = "dilated_depthwise_conv" if edge else "dilated_local_attention"
        stripe = "large_kernel_depthwise_conv" if edge else "hybrid_local_stripe_attention"

        mixers: List[str] = []
        for index in range(self.depths.bottleneck):
            use_stripe = (
                self.enable_stripe_attention
                and (index + 1) % self.stripe_frequency == 0
            )
            mixers.append(stripe if use_stripe else plain)
        return mixers

    # -------------------------------------------------------------- (de)ser ---
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["depths"] = self.depths.to_dict()
        data["dilations_half"] = list(self.dilations_half)
        data["dilations_quarter"] = list(self.dilations_quarter)
        data["channels"] = {
            "full": self.channels_full,
            "half": self.channels_half,
            "quarter": self.channels_quarter,
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CASHSIConfig":
        """Build from a mapping, accepting the YAML form of spec 3.2.

        Tolerates (and verifies) the redundant ``channels:`` block; rejects any
        key the model does not understand, so a typo in a config file fails loudly
        instead of silently doing nothing.
        """
        payload = dict(data)

        # A config file may nest everything under a top-level `model:` key.
        if "model" in payload and isinstance(payload["model"], Mapping):
            payload = dict(payload["model"])

        declared_channels = payload.pop("channels", None)

        depths = payload.pop("depths", None)
        if depths is not None:
            depths = Depths.from_dict(depths) if isinstance(depths, Mapping) else depths

        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"Unknown config key(s): {', '.join(unknown)}. Valid keys: {sorted(known)}"
            )

        for key in ("dilations_half", "dilations_quarter"):
            if key in payload and payload[key] is not None:
                payload[key] = tuple(int(v) for v in payload[key])

        if depths is not None:
            payload["depths"] = depths

        config = cls(**payload)

        if declared_channels is not None:
            expected = {
                "full": config.channels_full,
                "half": config.channels_half,
                "quarter": config.channels_quarter,
            }
            mismatched = {
                key: (int(value), expected[key])
                for key, value in declared_channels.items()
                if key in expected and int(value) != expected[key]
            }
            if mismatched:
                detail = "; ".join(
                    f"{key}: config says {declared}, base_width={config.base_width} implies {derived}"
                    for key, (declared, derived) in mismatched.items()
                )
                raise ValueError(
                    "channels block contradicts base_width (widths are C, 2C, 4C): " + detail
                )
            unknown_channels = sorted(set(declared_channels) - set(expected))
            if unknown_channels:
                raise ValueError(
                    f"Unknown channels key(s): {', '.join(unknown_channels)}"
                )

        validate_config(config)
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CASHSIConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} must contain a YAML mapping")
        return cls.from_dict(payload)

    # -------------------------------------------------------------- helpers ---
    def as_edge(self) -> "CASHSIConfig":
        """Same topology, convolutional spatial mixers (spec 9.2 edge backend)."""
        data = self.to_dict()
        data.pop("channels", None)
        data["backend"] = _EDGE_BACKEND
        return CASHSIConfig.from_dict(data)


def validate_config(config: CASHSIConfig) -> None:
    """Fail immediately on an inconsistent configuration (spec 3.6)."""
    if config.base_width <= 0:
        raise ValueError(f"base_width must be positive, got {config.base_width}")

    if config.output_bands <= 0:
        raise ValueError(f"output_bands must be positive, got {config.output_bands}")

    if config.input_channels <= 0:
        raise ValueError(f"input_channels must be positive, got {config.input_channels}")

    if config.head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {config.head_dim}")

    for width in config.widths:
        if width % config.head_dim != 0:
            divisors = [d for d in range(1, config.base_width + 1) if all(w % d == 0 for w in config.widths)]
            raise ValueError(
                f"Channel width {width} must be divisible by head_dim={config.head_dim}. "
                f"With base_width={config.base_width} the widths are {config.widths}; "
                f"valid head_dim values are {divisors}."
            )

    if config.ffn_expansion <= 1.0:
        raise ValueError(
            f"FFN expansion must be greater than 1, got {config.ffn_expansion}"
        )

    if config.size_multiple != 4:
        raise ValueError("The two-stage encoder requires size_multiple=4.")

    if config.backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {config.backend!r}")

    # Definition of Done: "latent features remain wider than 31 channels until the
    # final head". A narrower stem would force the network through a spectral
    # bottleneck before the head has a chance to model the 31 bands.
    if config.channels_full <= config.output_bands:
        raise ValueError(
            f"base_width={config.base_width} must exceed output_bands={config.output_bands}: "
            "latent features must stay wider than the spectral output until the final head."
        )

    for name, value in config.depths.to_dict().items():
        if value < 0:
            raise ValueError(f"depths.{name} must be non-negative, got {value}")
    if config.depths.bottleneck < 1:
        raise ValueError("depths.bottleneck must be at least 1.")

    if config.stripe_frequency < 1:
        raise ValueError(
            f"stripe_frequency must be >= 1, got {config.stripe_frequency}"
        )
    if config.stripe_width < 1:
        raise ValueError(f"stripe_width must be >= 1, got {config.stripe_width}")

    if config.attention_kernel_size % 2 != 1:
        raise ValueError(
            f"attention_kernel_size must be odd, got {config.attention_kernel_size}"
        )
    if config.spatial_kernel % 2 != 1:
        raise ValueError(f"spatial_kernel must be odd, got {config.spatial_kernel}")
    if config.large_kernel % 2 != 1:
        raise ValueError(f"large_kernel must be odd, got {config.large_kernel}")

    if not config.dilations_half:
        raise ValueError("dilations_half must be non-empty")
    if not config.dilations_quarter:
        raise ValueError("dilations_quarter must be non-empty")
    if any(d < 1 for d in config.dilations_half + config.dilations_quarter):
        raise ValueError("dilations must all be >= 1")

    # Hybrid stripe blocks need at least one local + one horizontal + one vertical
    # head, i.e. >= 3 heads at the bottleneck width.
    quarter_heads = config.channels_quarter // config.head_dim
    uses_hybrid = (
        config.backend == _RESEARCH_BACKEND
        and config.enable_stripe_attention
        and any(
            mixer == "hybrid_local_stripe_attention" for mixer in config.bottleneck_mixers()
        )
    )
    if uses_hybrid and quarter_heads < 3:
        raise ValueError(
            f"hybrid stripe attention needs >= 3 heads at the bottleneck, but "
            f"channels_quarter={config.channels_quarter} / head_dim={config.head_dim} "
            f"= {quarter_heads}. Raise base_width, lower head_dim, or set "
            "enable_stripe_attention=false."
        )

    if config.spectral_head not in {"residual", "low_rank"}:
        raise ValueError(
            f"spectral_head must be 'residual' or 'low_rank', got {config.spectral_head!r}"
        )
    if config.spectral_head == "low_rank" and not 1 <= config.spectral_rank <= config.output_bands:
        raise ValueError(
            f"spectral_rank must be in [1, {config.output_bands}], got {config.spectral_rank}"
        )

    if not 0.0 <= config.drop_path < 1.0:
        raise ValueError(f"drop_path must be in [0, 1), got {config.drop_path}")


def _variant(name: str, base_width: int, depths: Depths, **overrides: Any) -> CASHSIConfig:
    config = CASHSIConfig(name=name, base_width=base_width, depths=depths, **overrides)
    validate_config(config)
    return config


TINY = _variant(
    "cas_hsi_tiny",
    base_width=32,
    depths=Depths(
        encoder_full=2,
        encoder_half=3,
        bottleneck=5,
        decoder_half=3,
        decoder_full=2,
        refinement=2,
    ),
)

# DEVIATION FROM THE SPEC, and a necessary one.
#
# Spec 3.2 gives CAS-HSI-Base `base_width: 48` with `head_dim: 32`. That
# combination is impossible: spec 3.6's own validator requires every width
# (48, 96, 192) to be divisible by head_dim, and 48 % 32 = 16. It is not merely a
# validator quibble -- CrossChannelAttention(48, head_dim=32) cannot split 48
# channels into 32-wide heads either, so the Base variant as written cannot be
# constructed at all.
#
# The minimal correction keeps the specified capacity (base_width=48, i.e. the
# Restormer width the design clearly descends from) and moves head_dim to the
# largest divisor of 48 below 32: 24. Widths 48/96/192 then give 2/4/8 heads, and
# the bottleneck's 8 heads split cleanly for the hybrid stripe block (4 local
# across dilations (1,2,3), 2 horizontal, 2 vertical) -- the same structure as the
# spec's own worked example in 4.5.2.
#
# The alternative (base_width 48 -> 64 to keep head_dim=32) was rejected: it
# inflates the model ~1.8x and changes the variant's identity, whereas head_dim is
# an internal partitioning choice. Override `head_dim` in a config file if you
# prefer a different resolution of the contradiction.
BASE = _variant(
    "cas_hsi_base",
    base_width=48,
    depths=Depths(
        encoder_full=2,
        encoder_half=4,
        bottleneck=6,
        decoder_half=4,
        decoder_full=2,
        refinement=2,
    ),
    head_dim=24,
)

VARIANTS: Dict[str, CASHSIConfig] = {
    "tiny": TINY,
    "cas_hsi_tiny": TINY,
    "base": BASE,
    "cas_hsi_base": BASE,
}


def variant_config(name: str, **overrides: Any) -> CASHSIConfig:
    """Return a fresh copy of a named variant with optional field overrides."""
    key = str(name).strip().lower()
    if key not in VARIANTS:
        raise ValueError(
            f"Unknown variant {name!r}; expected one of {sorted(set(VARIANTS))}"
        )
    data = VARIANTS[key].to_dict()
    data.pop("channels", None)
    data.update(overrides)
    return CASHSIConfig.from_dict(data)

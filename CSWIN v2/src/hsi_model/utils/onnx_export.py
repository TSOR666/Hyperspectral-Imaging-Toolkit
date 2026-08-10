"""Architecture-independent ONNX export for CSWIN v2 generators.

Why this module exists
----------------------
``hsi_model.utils.inference.load_generator`` rebuilds the generator from the
config stored in the checkpoint. That only works while the *code* still builds
the same module tree from that config. It does not, for checkpoints trained
before later refactors, because:

* defaults changed (``spectral_attention_type`` ``efficient`` -> ``s_msa``,
  ``cswin_bias_mode`` ``window_cyclic`` -> ``long_axis``, ...), so the same
  config now yields different tensors;
* the transformer block itself was replaced (commit ``17c847c``): the old
  ``norm1/norm2/norm3 + ffn + noise_block`` layout became the CBAM-gated SSTB
  (``gate/norm1..4/gdfn/sgfn``). No config value can express the old block, so
  ``load_state_dict`` fails outright;
* ``iteration_count`` changed from a 0-d buffer to shape ``(1,)``.

The functions here recover the architecture from the *tensors themselves*
(keys and shapes are ground truth; the embedded config is only a hint), rebuild
a matching generator — including the legacy block via
``block_variant: legacy_dtb`` — and freeze the result into an ONNX graph. Once
exported, inference no longer depends on this repository at all: any ONNX
Runtime can run the ``.onnx``.

Typical use::

    from hsi_model.utils.onnx_export import export_checkpoint_to_onnx

    result = export_checkpoint_to_onnx(
        "artifacts/checkpoints/old_best.pth",
        "artifacts/onnx/old_best.onnx",
        height=128, width=128, precision="fp16",
    )

Limits that are reported, not hidden:

* The spatial size is baked into the graph (the model's reflect-padding to a
  multiple of ``split_size`` / the downsample factor is data-dependent Python
  control flow, which tracing resolves to constants). Batch is dynamic. Export
  at your inference tile size, or use ``dynamic_hw=True`` only when you have
  verified parity at every size you will feed it.
* ``norm_groups``, ``output_activation``, ``cascade_stages`` and a few other
  knobs leave no trace in the weights. They are taken from the embedded config
  when present, otherwise defaulted, and always listed in ``assumptions``.
"""
from __future__ import annotations

import copy
import json
import logging
import platform
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..models.generator_v3 import NoiseRobustCSWinGenerator
from .inference import _extract_state_dict, _normalize_generator_state

logger = logging.getLogger(__name__)

ONNX_MANIFEST_FORMAT = "cswin_onnx_manifest_v1"

STAGES: Tuple[str, ...] = (
    "encoder1",
    "encoder2",
    "bottleneck",
    "decoder1",
    "decoder2",
)

#: Config keys the generator and its submodules actually read. Training
#: checkpoints embed the whole run config (data paths, optimizer, ...); only
#: this subset is carried into the exported manifest.
ARCHITECTURE_CONFIG_KEYS = frozenset(
    {
        # generator body
        "in_channels",
        "out_channels",
        "base_channels",
        "split_sizes",
        "num_heads",
        "norm_groups",
        "block_variant",
        "blocks_per_stage",
        "stage_depths",
        "stage_num_heads",
        "stage_heads",
        "sampling",
        "decoder1_compress_first",
        "use_input_denoising",
        "use_feature_norm",
        "thick_output_head",
        "output_head_init_scale",
        "use_spectral_input_skip",
        "spectral_input_skip_hidden",
        "spectral_input_skip_init",
        "refinement_blocks",
        "refinement_channels",
        "refinement_ffn_expansion",
        "cascade_stages",
        # output behaviour
        "output_activation",
        "activation_delay_iters",
        "generator_clamp_range",
        "clamp_after_iters",
        # transformer block
        "ffn_expansion",
        "cbam_reduction",
        "sstb_outer_residual_scale",
        "spectral_attention_type",
        "activation_checkpointing",
        "activation_checkpoint_min_tokens",
        # legacy block
        "use_noise_block",
        "legacy_ffn_expansion_factor",
        # spatial attention
        "cswin_attention_mode",
        "cswin_bias_mode",
        "cswin_max_long_axis",
        "cswin_global_tokens",
        "ckpt_min_tokens",
        "use_fp16_bias",
        # spectral attention
        "smsa_output_norm",
        "spectral_attention_force_fp32",
        "spectral_attention_finite_clamp",
        "spectral_attention_sanitize_nonfinite",
    }
)

#: Knobs that change the forward pass but leave no fingerprint in the weights.
#: If the checkpoint has no embedded config these fall back to code defaults and
#: are reported so a wrong guess is visible rather than silent.
UNRECOVERABLE_KEYS: Tuple[Tuple[str, Any, str], ...] = (
    ("norm_groups", 8, "GroupNorm group count is not stored in the weights"),
    (
        "output_activation",
        "none",
        "the output head activation ('none'/'sigmoid'/'tanh'/'delayed_sigmoid') "
        "is applied functionally and leaves no parameter",
    ),
    (
        "sstb_outer_residual_scale",
        1.0,
        "the SSTB outer residual scale is a plain float multiplier",
    ),
    (
        "cswin_global_tokens",
        1024,
        "only used by cswin_attention_mode='local_global'",
    ),
)

PRECISIONS: Tuple[str, ...] = ("fp32", "fp16")

#: Relative-L2 budget for the eager-vs-ONNX parity check, per precision. fp32 is
#: pure graph-lowering noise; fp16 is dominated by the storage rounding.
DEFAULT_PARITY_TOLERANCE: Dict[str, float] = {"fp32": 1e-4, "fp16": 2e-2}


# ---------------------------------------------------------------------------
# architecture recovery
# ---------------------------------------------------------------------------


@dataclass
class ArchitectureRecovery:
    """Result of reading an architecture back out of a checkpoint."""

    config: Dict[str, Any]
    #: config key -> "tensors" | "checkpoint" | "override" | "default"
    evidence: Dict[str, str] = field(default_factory=dict)
    #: keys where the embedded config disagreed with the tensors (tensors win)
    conflicts: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    #: human-readable warnings about values that could not be recovered
    assumptions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"block_variant={self.config.get('block_variant')}"]
        for key in (
            "base_channels",
            "in_channels",
            "out_channels",
            "stage_depths",
            "split_sizes",
            "stage_num_heads",
            "spectral_attention_type",
            "cswin_attention_mode",
            "sampling",
            "output_activation",
        ):
            if key in self.config:
                lines.append(f"{key}={self.config[key]!r}")
        return ", ".join(lines)


def _shape(state: Mapping[str, torch.Tensor], key: str) -> Optional[Tuple[int, ...]]:
    tensor = state.get(key)
    return tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None


def _stage_block_indices(state: Mapping[str, torch.Tensor], stage: str) -> List[int]:
    prefix = f"{stage}."
    indices = set()
    for key in state:
        if not key.startswith(prefix):
            continue
        head = key[len(prefix):].split(".", 1)[0]
        if head.isdigit():
            indices.add(int(head))
    return sorted(indices)


def _has_prefix(state: Mapping[str, torch.Tensor], needle: str) -> bool:
    return any(needle in key for key in state)


def _solve_floor_ratio(
    pairs: Sequence[Tuple[int, int]],
    preferred: Sequence[float],
) -> Optional[float]:
    """Recover ``x`` from observations ``h == max(1, int(C * x))``.

    ``GDFN``/``SGFN`` store ``2 * max(1, int(channels * ffn_expansion))`` in
    ``project_in``. A float is not uniquely determined by a floor, so try the
    documented constants first and otherwise return the midpoint of the feasible
    interval, which reproduces every observation exactly.
    """
    if not pairs:
        return None

    def matches(value: float) -> bool:
        return all(max(1, int(channels * value)) == hidden for channels, hidden in pairs)

    for candidate in preferred:
        if matches(candidate):
            return float(candidate)

    low = max(hidden / channels for channels, hidden in pairs)
    high = min((hidden + 1) / channels for channels, hidden in pairs)
    if low < high:
        candidate = (low + high) / 2.0
        if matches(candidate):
            return float(candidate)
    return None


def _solve_integer_reduction(pairs: Sequence[Tuple[int, int]]) -> Optional[int]:
    """Recover ``r`` from observations ``h == max(1, C // r)`` (CBAM gate)."""
    if not pairs:
        return None
    limit = max(channels for channels, _ in pairs)
    for candidate in [4] + [value for value in range(1, limit + 1) if value != 4]:
        if all(max(1, channels // candidate) == hidden for channels, hidden in pairs):
            return int(candidate)
    return None


def _block_channels(
    state: Mapping[str, torch.Tensor], stage: str, variant: str
) -> Optional[int]:
    key = (
        f"{stage}.0.norm1.norm.weight" if variant == "sstb" else f"{stage}.0.norm1.weight"
    )
    shape = _shape(state, key)
    return int(shape[0]) if shape else None


def recover_structural_config(
    state: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, Any], List[str]]:
    """Read every architecture knob that the tensors themselves determine.

    Returns ``(config, notes)``. Anything absent from ``config`` could not be
    decided from the weights and must come from the embedded config or a
    default.
    """
    notes: List[str] = []
    config: Dict[str, Any] = {}

    embed = _shape(state, "embedding.0.weight")
    if embed is None:
        raise ValueError(
            "Checkpoint has no 'embedding.0.weight'; this does not look like a "
            "NoiseRobustCSWinGenerator state dict. Available keys (first 10): "
            f"{list(state)[:10]}"
        )
    base_channels, in_channels = int(embed[0]), int(embed[1])
    config["base_channels"] = base_channels
    config["in_channels"] = in_channels

    thick = _shape(state, "to_spectral.2.weight")
    thin = _shape(state, "to_spectral.weight")
    if thick is not None:
        config["thick_output_head"] = True
        config["out_channels"] = int(thick[0])
    elif thin is not None:
        config["thick_output_head"] = False
        config["out_channels"] = int(thin[0])
    else:
        raise ValueError("Checkpoint has no output head ('to_spectral.*') tensors.")

    config["use_input_denoising"] = "denoising.0.weight" in state
    config["use_feature_norm"] = "embedding.1.weight" in state

    if "up1.expand.weight" in state:
        config["sampling"] = "pixelshuffle"
    else:
        config["sampling"] = "bilinear"
    down1 = _shape(state, "down1.conv.weight")
    if down1 is not None:
        expected = base_channels * (4 if config["sampling"] == "pixelshuffle" else 1)
        if int(down1[1]) != expected:
            notes.append(
                f"down1.conv expects {down1[1]} input channels but sampling="
                f"{config['sampling']!r} implies {expected}; the recovered "
                "sampling mode may be wrong."
            )

    if _has_prefix(state, ".gdfn.project_in.weight"):
        config["block_variant"] = "sstb"
    elif _has_prefix(state, ".ffn.net.0.weight"):
        config["block_variant"] = "legacy_dtb"
    else:
        raise ValueError(
            "Could not identify the transformer block layout: neither "
            "'*.gdfn.project_in.weight' (SSTB) nor '*.ffn.net.0.weight' "
            "(legacy) is present."
        )
    variant = config["block_variant"]

    depths = [len(_stage_block_indices(state, stage)) for stage in STAGES]
    if any(depth == 0 for depth in depths):
        missing = [stage for stage, depth in zip(STAGES, depths) if depth == 0]
        raise ValueError(f"Checkpoint has no transformer blocks for stages: {missing}")
    config["stage_depths"] = depths

    channels = {stage: _block_channels(state, stage, variant) for stage in STAGES}
    decoder1_channels = channels["decoder1"]
    if decoder1_channels is not None:
        config["decoder1_compress_first"] = decoder1_channels == base_channels * 2
        if decoder1_channels not in (base_channels * 2, base_channels * 4):
            notes.append(
                f"decoder1 block width {decoder1_channels} matches neither "
                f"{base_channels * 2} (compress-first) nor {base_channels * 4}."
            )

    splits: Dict[str, int] = {}
    heads: Dict[str, int] = {}
    bias_dtype: Optional[torch.dtype] = None
    for stage in STAGES:
        table = state.get(
            f"{stage}.0.spatial_attn.attention.relative_position_bias_table_h"
        )
        if not isinstance(table, torch.Tensor) or table.ndim != 3:
            continue
        splits[stage] = (int(table.shape[0]) + 1) // 2
        heads[stage] = int(table.shape[2])
        bias_dtype = table.dtype
    if len(splits) != len(STAGES):
        raise ValueError(
            "Missing relative_position_bias_table_h for stages "
            f"{[stage for stage in STAGES if stage not in splits]}."
        )

    config["split_sizes"] = [
        splits["encoder1"],
        splits["encoder2"],
        splits["bottleneck"],
    ]
    if splits["decoder1"] != splits["encoder2"]:
        notes.append(
            f"decoder1 split_size {splits['decoder1']} != encoder2 "
            f"{splits['encoder2']}; the generator ties them, so the rebuilt "
            "model cannot reproduce this checkpoint exactly."
        )
    if splits["decoder2"] != splits["encoder1"]:
        notes.append(
            f"decoder2 split_size {splits['decoder2']} != encoder1 "
            f"{splits['encoder1']}; the generator ties them, so the rebuilt "
            "model cannot reproduce this checkpoint exactly."
        )
    config["stage_num_heads"] = [heads[stage] for stage in STAGES]
    config["num_heads"] = heads["encoder1"]
    if bias_dtype is not None:
        config["use_fp16_bias"] = bias_dtype == torch.float16

    if _has_prefix(state, ".spectral_attn.attention.qkv.weight"):
        config["spectral_attention_type"] = "s_msa"
        config["smsa_output_norm"] = _has_prefix(
            state, ".spectral_attn.attention.norm.weight"
        )
    elif _has_prefix(state, ".spectral_attn.attention.to_q.0.weight"):
        config["spectral_attention_type"] = "efficient"
    else:
        notes.append(
            "Could not identify the spectral attention type; falling back to "
            "the embedded config or the code default."
        )

    if _has_prefix(state, ".spatial_attn.attention.qkv_cswin.weight"):
        config["cswin_attention_mode"] = "cswin"
    elif _has_prefix(state, ".spatial_attn.attention.relative_position_bias_table_h_long"):
        # The long-axis tables are only allocated for axial + long_axis.
        config["cswin_attention_mode"] = "axial"
        config["cswin_bias_mode"] = "long_axis"
        long_shape = _shape(
            state,
            "encoder1.0.spatial_attn.attention.relative_position_bias_table_h_long",
        )
        if long_shape:
            config["cswin_max_long_axis"] = (int(long_shape[0]) + 1) // 2
    else:
        # 'axial'+'window_cyclic' and 'local_global' produce identical tensors.
        config["cswin_bias_mode"] = "window_cyclic"
        notes.append(
            "No long-axis bias tables: the checkpoint is either "
            "cswin_attention_mode='axial' with cswin_bias_mode='window_cyclic' "
            "or 'local_global'. These are indistinguishable from the weights; "
            "the embedded config (or --set cswin_attention_mode=...) decides."
        )

    if variant == "sstb":
        gdfn_pairs = [
            (channels[stage], int(shape[0]) // 2)
            for stage in STAGES
            if channels[stage]
            and (shape := _shape(state, f"{stage}.0.gdfn.project_in.weight"))
        ]
        expansion = _solve_floor_ratio(gdfn_pairs, preferred=(2.66, 2.0, 4.0, 3.0, 1.0))
        if expansion is not None:
            config["ffn_expansion"] = expansion
        else:
            notes.append(
                f"Could not solve ffn_expansion from GDFN widths {gdfn_pairs}."
            )

        gate_pairs = [
            (channels[stage], int(shape[0]))
            for stage in STAGES
            if channels[stage]
            and (shape := _shape(state, f"{stage}.0.gate.mlp.0.weight"))
        ]
        reduction = _solve_integer_reduction(gate_pairs)
        if reduction is not None:
            config["cbam_reduction"] = reduction
        else:
            notes.append(
                f"Could not solve cbam_reduction from gate widths {gate_pairs}."
            )
    else:
        ffn_pairs = [
            (channels[stage], int(shape[0]))
            for stage in STAGES
            if channels[stage]
            and (shape := _shape(state, f"{stage}.0.ffn.net.0.weight"))
        ]
        factors = {hidden // width for width, hidden in ffn_pairs if width}
        if len(factors) == 1:
            config["legacy_ffn_expansion_factor"] = int(factors.pop())
        elif factors:
            notes.append(
                f"Inconsistent legacy FFN expansion factors across stages: {factors}."
            )
        config["use_noise_block"] = _has_prefix(state, ".noise_block.")

    if "spectral_input_skip.net.0.weight" in state:
        config["use_spectral_input_skip"] = True
        hidden = _shape(state, "spectral_input_skip.net.0.weight")
        config["spectral_input_skip_hidden"] = int(hidden[0]) if hidden else 0
    elif "spectral_input_skip.net.weight" in state:
        config["use_spectral_input_skip"] = True
        config["spectral_input_skip_hidden"] = 0
    else:
        config["use_spectral_input_skip"] = False

    refinement_indices = _stage_block_indices(state, "refinement")
    config["refinement_blocks"] = len(refinement_indices)
    if refinement_indices:
        in_proj = _shape(state, "refinement.0.in_proj.0.weight")
        if in_proj:
            config["refinement_channels"] = int(in_proj[0])
            project_in = _shape(state, "refinement.0.body.1.project_in.weight")
            if project_in:
                expansion = _solve_floor_ratio(
                    [(int(in_proj[0]), int(project_in[0]) // 2)],
                    preferred=(2.0, 2.66, 4.0),
                )
                if expansion is not None:
                    config["refinement_ffn_expansion"] = expansion

    config["_has_cascade_modules"] = "cascade_feedback.weight" in state

    return config, notes


def _values_differ(left: Any, right: Any) -> bool:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return list(left) != list(right)
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) > 1e-9
        except (TypeError, ValueError):
            return left != right
    return left != right


def recover_architecture(
    state: Mapping[str, torch.Tensor],
    embedded_config: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ArchitectureRecovery:
    """Build a generator config that reproduces ``state`` exactly.

    Precedence is ``overrides`` > tensors > embedded config > code defaults.
    The tensors outrank the embedded config because default values in the code
    have changed since older configs were written.
    """
    structural, notes = recover_structural_config(state)
    has_cascade_modules = bool(structural.pop("_has_cascade_modules", False))

    config: Dict[str, Any] = {}
    evidence: Dict[str, str] = {}
    conflicts: Dict[str, Tuple[Any, Any]] = {}
    # keyed so an explicit override can retract the warning
    assumed: Dict[str, str] = {}

    if embedded_config:
        for key, value in embedded_config.items():
            if key in ARCHITECTURE_CONFIG_KEYS:
                config[key] = copy.deepcopy(value)
                evidence[key] = "checkpoint"

    for key, value in structural.items():
        if key in config and _values_differ(config[key], value):
            conflicts[key] = (config[key], value)
        config[key] = value
        evidence[key] = "tensors"

    # cascade_stages only exists as a loop count; the weights just show whether
    # the feedback/gate modules were built at all.
    if has_cascade_modules:
        embedded_stages = int(config.get("cascade_stages", 0) or 0)
        if embedded_stages < 2:
            config["cascade_stages"] = 2
            evidence["cascade_stages"] = "default"
            assumed["cascade_stages"] = (
                "cascade_stages=2: the checkpoint has cascade_feedback/cascade_gate "
                "weights but the loop count is not stored. Override with "
                "--set cascade_stages=N if the run used more."
            )
    else:
        config["cascade_stages"] = 1
        evidence["cascade_stages"] = "tensors"

    for key, default, reason in UNRECOVERABLE_KEYS:
        if key in config:
            continue
        config[key] = default
        evidence[key] = "default"
        # Only warn about knobs this architecture actually reads, so the
        # assumption list stays a signal rather than boilerplate.
        if key == "sstb_outer_residual_scale" and config.get("block_variant") != "sstb":
            continue
        if key == "cswin_global_tokens" and config.get("cswin_attention_mode") in (
            "cswin",
            "axial",
        ):
            continue
        assumed[key] = f"{key}={default!r} (code default): {reason}."

    if evidence.get("output_activation") == "checkpoint":
        notes.append(
            f"output_activation={config['output_activation']!r} came from the "
            "embedded config; it is baked into the exported graph."
        )

    # Activation checkpointing is a training-memory tactic and only fires in
    # train mode, but tracing is cleaner without it.
    config["activation_checkpointing"] = False
    evidence["activation_checkpointing"] = "override"

    if overrides:
        for key, value in overrides.items():
            config[key] = value
            evidence[key] = "override"
            assumed.pop(key, None)

    config = {
        key: value
        for key, value in config.items()
        if key in ARCHITECTURE_CONFIG_KEYS
    }

    return ArchitectureRecovery(
        config=config,
        evidence=evidence,
        conflicts=conflicts,
        assumptions=[assumed[key] for key in sorted(assumed)],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# rebuild + load
# ---------------------------------------------------------------------------


@dataclass
class LoadReport:
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    mismatched: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = field(
        default_factory=list
    )
    adapted: List[str] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return not (self.missing or self.unexpected or self.mismatched)

    def describe(self, limit: int = 10) -> str:
        if self.exact:
            return "exact match"
        parts = []
        if self.missing:
            parts.append(f"{len(self.missing)} missing: {self.missing[:limit]}")
        if self.unexpected:
            parts.append(f"{len(self.unexpected)} unexpected: {self.unexpected[:limit]}")
        if self.mismatched:
            parts.append(
                f"{len(self.mismatched)} shape mismatches: {self.mismatched[:limit]}"
            )
        return "; ".join(parts)


def _adapt_state_to_model(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], LoadReport]:
    """Align a historical state dict with the rebuilt model's tensors."""
    model_state = model.state_dict()
    report = LoadReport()
    adapted: Dict[str, torch.Tensor] = {}

    for key, tensor in state.items():
        if key not in model_state:
            # Derived/private buffers (``_relative_position_index`` and friends)
            # are non-persistent today; older saves may still carry them.
            if key.rsplit(".", 1)[-1].startswith("_"):
                report.adapted.append(f"dropped derived buffer {key}")
            else:
                report.unexpected.append(key)
            continue

        target = model_state[key]
        if tuple(tensor.shape) != tuple(target.shape):
            if tensor.numel() == target.numel():
                # ``iteration_count`` went from a 0-d buffer to shape (1,).
                adapted[key] = tensor.reshape(target.shape).to(target.dtype)
                report.adapted.append(
                    f"reshaped {key} {tuple(tensor.shape)} -> {tuple(target.shape)}"
                )
                continue
            report.mismatched.append(
                (key, tuple(tensor.shape), tuple(target.shape))
            )
            continue

        adapted[key] = tensor.to(target.dtype) if tensor.dtype != target.dtype else tensor

    report.missing = sorted(set(model_state) - set(adapted))
    return adapted, report


def load_checkpoint_payload(
    checkpoint_path: Union[str, Path],
    prefer_ema: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
    """Read a checkpoint and return ``(state_dict, embedded_config, metadata)``.

    Handles every layout the trainers have produced: bare state dicts, GAN-era
    full-model saves with ``generator.``/``discriminator.`` prefixes, DDP and
    ``torch.compile`` wrappers, and EMA shadows.
    """
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"Unexpected checkpoint object in {checkpoint_path}: {type(checkpoint)}"
        )

    state = _normalize_generator_state(_extract_state_dict(checkpoint))
    ema_applied = bool(checkpoint.get("ema_applied", False))

    if prefer_ema and not ema_applied:
        shadow: Any = {}
        ema_payload = checkpoint.get("ema")
        if isinstance(ema_payload, Mapping):
            shadow = ema_payload.get("shadow", {})
        if not shadow:
            shadow = checkpoint.get("ema_state_dict", {})
        if shadow:
            try:
                shadow_state = _normalize_generator_state(shadow)
            except ValueError:
                shadow_state = {}
            usable = {
                key: tensor
                for key, tensor in shadow_state.items()
                if key in state and tuple(tensor.shape) == tuple(state[key].shape)
            }
            # Only trust a shadow that covers every float parameter; a partial
            # one would silently mix EMA and raw weights.
            float_keys = {
                key
                for key, tensor in state.items()
                if tensor.is_floating_point()
            }
            if float_keys and float_keys.issubset(usable):
                state = {**state, **usable}
                ema_applied = True
            else:
                logger.warning(
                    "EMA shadow covers %d/%d float tensors; using raw weights.",
                    len(usable),
                    len(float_keys),
                )

    embedded_config: Dict[str, Any] = {}
    for key in ("config", "architecture_config", "model_config"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            embedded_config = dict(candidate)
            break

    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "resolution": checkpoint.get("resolution"),
        "val_metrics": checkpoint.get("val_metrics"),
        "ema_applied": ema_applied,
        "embedded_config_present": bool(embedded_config),
        "state_dict_tensors": len(state),
    }
    return state, embedded_config, metadata


def rebuild_generator(
    state: Mapping[str, torch.Tensor],
    embedded_config: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
) -> Tuple[NoiseRobustCSWinGenerator, ArchitectureRecovery, LoadReport]:
    """Recover the architecture, build the generator and load ``state`` into it.

    With ``strict=True`` (the default) any leftover missing/unexpected/mismatched
    tensor raises, because a partially loaded generator produces plausible-looking
    but meaningless spectra.
    """
    recovery = recover_architecture(state, embedded_config, overrides)
    generator = NoiseRobustCSWinGenerator(recovery.config)
    adapted, report = _adapt_state_to_model(generator, state)

    if strict and not report.exact:
        raise RuntimeError(
            "Recovered architecture does not match the checkpoint tensors: "
            f"{report.describe()}. Recovered config: {recovery.summary()}. "
            "Use --set KEY=VALUE to correct a knob, or --allow-partial to "
            "export anyway (the unmatched tensors keep their random init)."
        )

    generator.load_state_dict(adapted, strict=False)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return generator, recovery, report


def load_generator_from_any_checkpoint(
    checkpoint_path: Union[str, Path],
    overrides: Optional[Mapping[str, Any]] = None,
    prefer_ema: bool = True,
    strict: bool = True,
    architecture_config_hint: Optional[Mapping[str, Any]] = None,
) -> Tuple[NoiseRobustCSWinGenerator, ArchitectureRecovery, LoadReport, Dict[str, Any]]:
    """Load any historical CSWIN checkpoint into an eval-mode generator.

    ``architecture_config_hint`` fills gaps for checkpoints that embed no config;
    the checkpoint's own config outranks it, and tensor evidence outranks both.
    """
    state, embedded_config, metadata = load_checkpoint_payload(
        checkpoint_path, prefer_ema=prefer_ema
    )
    if architecture_config_hint:
        embedded_config = {**dict(architecture_config_hint), **embedded_config}
        metadata["architecture_config_hint_used"] = True
    generator, recovery, report = rebuild_generator(
        state, embedded_config, overrides=overrides, strict=strict
    )
    metadata["parameters"] = int(
        sum(parameter.numel() for parameter in generator.parameters())
    )
    return generator, recovery, report, metadata


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class _ExportWrapper(nn.Module):
    """Fixes dtype at the graph boundary and optionally clamps to reflectance.

    Keeping the ONNX inputs/outputs in fp32 while the body runs in fp16 means a
    downstream consumer never has to know which precision was chosen.
    """

    def __init__(
        self,
        generator: nn.Module,
        compute_dtype: torch.dtype,
        io_dtype: torch.dtype,
        clamp_output: bool,
    ) -> None:
        super().__init__()
        self.model = generator
        self.compute_dtype = compute_dtype
        self.io_dtype = io_dtype
        self.clamp_output = clamp_output

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        out = self.model(rgb.to(self.compute_dtype))
        out = out.to(self.io_dtype)
        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out


def _resolve_precision(precision: str) -> str:
    key = str(precision).strip().lower()
    aliases = {
        "fp32": "fp32",
        "float32": "fp32",
        "float": "fp32",
        "32": "fp32",
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "16": "fp16",
    }
    if key not in aliases:
        raise ValueError(
            f"precision must be one of {PRECISIONS} (or an alias), got {precision!r}"
        )
    return aliases[key]


def _fp16_converters() -> List[Tuple[str, Any]]:
    """Return the available fp32->fp16 graph converters, best first.

    ``onnxruntime.transformers.float16`` is a maintained fork of
    ``onnxconverter_common.float16`` and is tried first: released
    onnxconverter-common 1.16 crashes in ``remove_unnecessary_cast_node`` against
    recent onnx versions.
    """
    converters: List[Tuple[str, Any]] = []
    try:
        from onnxruntime.transformers import float16 as ort_float16

        converters.append(("onnxruntime.transformers.float16", ort_float16))
    except ImportError:
        pass
    try:
        from onnxconverter_common import float16 as occ_float16

        converters.append(("onnxconverter_common.float16", occ_float16))
    except ImportError:
        pass
    return converters


def _convert_onnx_to_fp16(onnx_path: Path, keep_io_fp32: bool) -> str:
    """Convert an fp32 ONNX graph to fp16 in place.

    Graph conversion beats ``model.half()`` here: half convolutions are not
    implemented in PyTorch's CPU backend (so tracing a half model would require a
    GPU), and the converters keep numerically sensitive ops in fp32 and insert
    the Cast nodes for you.
    """
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "precision='fp16' needs the 'onnx' package: pip install onnx"
        ) from exc

    converters = _fp16_converters()
    if not converters:
        raise ImportError(
            "precision='fp16' needs a float16 graph converter. Install either "
            "onnxruntime (provides onnxruntime.transformers.float16) or "
            "onnxconverter-common."
        )

    model = onnx.load(str(onnx_path))
    errors: List[str] = []
    for name, module in converters:
        try:
            converted = module.convert_float_to_float16(
                model,
                keep_io_types=keep_io_fp32,
                disable_shape_infer=True,
            )
        except Exception as exc:  # pragma: no cover - depends on installed versions
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            model = onnx.load(str(onnx_path))  # converters mutate in place
            continue
        onnx.save(converted, str(onnx_path))
        return name

    raise RuntimeError(
        "Every available float16 converter failed: " + " | ".join(errors)
    )


def convert_onnx_fp16_to_fp32(onnx_path: Union[str, Path]) -> Tuple[bytes, int]:
    """Return an in-memory FP32 version of an ONNX graph containing FP16.

    Some CPU ONNX Runtime builds execute converted FP16 convolution/attention
    subgraphs with FP16 intermediates. Large activations can then overflow even
    when the graph input and output types are both ``tensor(float)``. This
    recovery path restores FP32 tensor types, initializers, and ``Cast`` targets
    without modifying the source file, allowing an existing FP16 artifact to be
    evaluated safely on CPU. The original FP16 rounding cannot be recovered, but
    the arithmetic no longer overflows solely because of the reduced exponent
    range.

    Returns ``(serialized_model, converted_item_count)``. A zero count means the
    graph did not contain FP16 tensors or casts and should not be retried.
    """
    try:
        import onnx
        from onnx import AttributeProto, TensorProto, numpy_helper
    except ImportError as exc:  # pragma: no cover - optional inference fallback
        raise ImportError(
            "CPU FP32 fallback needs the 'onnx' package; re-export with "
            "--precision fp32 or install onnx."
        ) from exc

    model = onnx.load(str(onnx_path), load_external_data=True)
    converted = 0

    def convert_tensor(tensor: Any) -> None:
        nonlocal converted
        if tensor.data_type != TensorProto.FLOAT16:
            return
        array = numpy_helper.to_array(tensor).astype("float32", copy=False)
        replacement = numpy_helper.from_array(array, name=tensor.name)
        tensor.CopyFrom(replacement)
        converted += 1

    def convert_value_info(value_info: Any) -> None:
        nonlocal converted
        if not value_info.type.HasField("tensor_type"):
            return
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type == TensorProto.FLOAT16:
            tensor_type.elem_type = TensorProto.FLOAT
            converted += 1

    def convert_graph(graph: Any) -> None:
        nonlocal converted
        for initializer in graph.initializer:
            convert_tensor(initializer)
        for sparse in getattr(graph, "sparse_initializer", ()):
            convert_tensor(sparse.values)
        for value_info in (
            list(graph.input) + list(graph.output) + list(graph.value_info)
        ):
            convert_value_info(value_info)

        for node in graph.node:
            if node.op_type == "Cast":
                for attribute in node.attribute:
                    if (
                        attribute.name == "to"
                        and attribute.type == AttributeProto.INT
                        and attribute.i == TensorProto.FLOAT16
                    ):
                        attribute.i = TensorProto.FLOAT
                        converted += 1
            for attribute in node.attribute:
                if attribute.type == AttributeProto.TENSOR:
                    convert_tensor(attribute.t)
                elif attribute.type == AttributeProto.TENSORS:
                    for tensor in attribute.tensors:
                        convert_tensor(tensor)
                elif attribute.type == AttributeProto.GRAPH:
                    convert_graph(attribute.g)
                elif attribute.type == AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        convert_graph(nested)

    convert_graph(model.graph)
    return model.SerializeToString(), converted


@dataclass
class OnnxExportResult:
    onnx_path: Path
    manifest_path: Optional[Path]
    manifest: Dict[str, Any]
    recovery: ArchitectureRecovery
    load_report: LoadReport
    parity: Optional[Dict[str, float]]


def export_generator_to_onnx(
    generator: nn.Module,
    output_path: Union[str, Path],
    *,
    height: int,
    width: int,
    in_channels: int = 3,
    batch_size: int = 1,
    precision: str = "fp32",
    keep_io_fp32: bool = True,
    clamp_output: bool = False,
    opset: int = 17,
    dynamic_batch: bool = True,
    dynamic_hw: bool = False,
    exporter: str = "torchscript",
    device: Optional[torch.device] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Freeze ``generator`` into an ONNX graph. Returns ``(path, export_info)``.

    ``exporter='torchscript'`` (the default) uses the tracing exporter. It is
    deprecated upstream but is the right tool here: the generator's padding and
    finite-value guards are data-dependent Python branches that tracing resolves
    to the eval-path constants, which is exactly the intent. ``'dynamo'`` selects
    the newer ``torch.export`` path and additionally requires ``onnxscript``.
    """
    precision = _resolve_precision(precision)
    exporter = str(exporter).strip().lower()
    if exporter not in ("torchscript", "dynamo"):
        raise ValueError(
            f"exporter must be 'torchscript' or 'dynamo', got {exporter!r}"
        )
    device = device or torch.device("cpu")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    generator = generator.to(device).eval()
    wrapper = _ExportWrapper(
        generator,
        compute_dtype=torch.float32,
        io_dtype=torch.float32,
        clamp_output=clamp_output,
    ).to(device).eval()

    example = torch.rand(
        batch_size, in_channels, height, width, dtype=torch.float32, device=device
    )

    dynamic_axes: Dict[str, Dict[int, str]] = {}
    if dynamic_batch:
        dynamic_axes.setdefault("rgb", {})[0] = "batch"
        dynamic_axes.setdefault("hsi", {})[0] = "batch"
    if dynamic_hw:
        dynamic_axes.setdefault("rgb", {}).update({2: "height", 3: "width"})
        dynamic_axes.setdefault("hsi", {}).update({2: "height", 3: "width"})

    with torch.inference_mode(), warnings.catch_warnings():
        # Tracing reports the data-dependent finite-value guards and padding
        # branches; both are resolved to the eval-path constants on purpose.
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        # The TorchScript exporter is deprecated upstream but chosen here
        # deliberately; its deprecation notice would read as a fault.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            wrapper,
            (example,),
            str(destination),
            input_names=["rgb"],
            output_names=["hsi"],
            opset_version=opset,
            dynamic_axes=dynamic_axes or None,
            do_constant_folding=True,
            dynamo=(exporter == "dynamo"),
        )

    fp16_method: Optional[str] = None
    if precision == "fp16":
        fp16_method = _convert_onnx_to_fp16(destination, keep_io_fp32=keep_io_fp32)

    io_dtype = "float32" if (precision == "fp32" or keep_io_fp32) else "float16"
    export_info = {
        "precision": precision,
        "exporter": exporter,
        "fp16_conversion": fp16_method,
        "io_dtype": io_dtype,
        "opset": opset,
        "input_name": "rgb",
        "output_name": "hsi",
        "input_shape": [batch_size, in_channels, height, width],
        "dynamic_batch": bool(dynamic_batch),
        "dynamic_hw": bool(dynamic_hw),
        "clamp_output": bool(clamp_output),
        "file_bytes": destination.stat().st_size,
    }
    return destination, export_info


def verify_onnx_against_torch(
    onnx_path: Union[str, Path],
    generator: nn.Module,
    *,
    height: int,
    width: int,
    in_channels: int = 3,
    batch_size: int = 1,
    clamp_output: bool = False,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Compare the exported graph with the eager model on random input."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parity verification needs onnxruntime: pip install onnxruntime"
        ) from exc

    generator = generator.eval()
    torch.manual_seed(seed)
    example = torch.rand(batch_size, in_channels, height, width, dtype=torch.float32)

    with torch.inference_mode():
        reference = generator(example).float()
        if clamp_output:
            reference = reference.clamp(0.0, 1.0)
    if not torch.isfinite(reference).all():
        raise RuntimeError(
            "The eager generator produced non-finite output during ONNX parity "
            "verification; check the checkpoint before exporting."
        )

    session = ort.InferenceSession(
        str(onnx_path),
        providers=list(providers) if providers else ["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    input_dtype = session.get_inputs()[0].type
    feed = example.numpy()
    if "float16" in input_dtype:
        feed = feed.astype("float16")
    actual = torch.from_numpy(session.run(None, {input_name: feed})[0]).float()

    if not torch.isfinite(actual).all():
        raise RuntimeError(
            "The exported ONNX graph produced non-finite output during parity "
            "verification. Re-export with --precision fp32 or use a provider "
            "with reliable FP16 support."
        )

    if actual.shape != reference.shape:
        raise RuntimeError(
            f"ONNX output shape {tuple(actual.shape)} != torch "
            f"{tuple(reference.shape)}"
        )

    difference = (actual - reference).abs()
    # Relative L2 rather than max element-wise relative error: a near-zero
    # reference element makes the latter enormous while the cube is fine.
    reference_norm = float(reference.norm())
    return {
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "rel_l2": float(difference.norm() / max(reference_norm, 1e-12)),
        "reference_mean": float(reference.mean()),
        "reference_std": float(reference.std()),
        "reference_abs_max": float(reference.abs().max()),
    }


def export_checkpoint_to_onnx(
    checkpoint_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    height: int = 128,
    width: int = 128,
    batch_size: int = 1,
    precision: str = "fp32",
    keep_io_fp32: bool = True,
    clamp_output: bool = False,
    opset: int = 17,
    dynamic_batch: bool = True,
    dynamic_hw: bool = False,
    exporter: str = "torchscript",
    overrides: Optional[Mapping[str, Any]] = None,
    architecture_config_hint: Optional[Mapping[str, Any]] = None,
    prefer_ema: bool = True,
    strict: bool = True,
    verify: bool = True,
    parity_tolerance: Optional[float] = None,
    write_manifest: bool = True,
) -> OnnxExportResult:
    """Recover, rebuild, load and freeze a checkpoint into ONNX.

    The manifest written next to the ``.onnx`` records the recovered config,
    where each value came from, the assumptions that could not be recovered from
    the weights, and the eager-vs-ONNX parity numbers.
    """
    generator, recovery, report, metadata = load_generator_from_any_checkpoint(
        checkpoint_path,
        overrides=overrides,
        prefer_ema=prefer_ema,
        strict=strict,
        architecture_config_hint=architecture_config_hint,
    )

    in_channels = int(recovery.config.get("in_channels", 3))
    onnx_path, export_info = export_generator_to_onnx(
        generator,
        output_path,
        height=height,
        width=width,
        in_channels=in_channels,
        batch_size=batch_size,
        precision=precision,
        keep_io_fp32=keep_io_fp32,
        clamp_output=clamp_output,
        opset=opset,
        dynamic_batch=dynamic_batch,
        dynamic_hw=dynamic_hw,
        exporter=exporter,
    )

    parity: Optional[Dict[str, float]] = None
    if verify:
        parity = verify_onnx_against_torch(
            onnx_path,
            generator,
            height=height,
            width=width,
            in_channels=in_channels,
            batch_size=1,
            clamp_output=clamp_output,
        )
        tolerance = (
            parity_tolerance
            if parity_tolerance is not None
            else DEFAULT_PARITY_TOLERANCE[_resolve_precision(precision)]
        )
        parity["tolerance"] = float(tolerance)
        parity["within_tolerance"] = bool(parity["rel_l2"] <= tolerance)
        if not parity["within_tolerance"]:
            logger.warning(
                "ONNX/torch parity rel_l2=%.3g exceeds tolerance %.3g for %s.",
                parity["rel_l2"],
                tolerance,
                onnx_path,
            )

    manifest = {
        "format": ONNX_MANIFEST_FORMAT,
        "onnx_file": onnx_path.name,
        "checkpoint": metadata,
        "architecture": recovery.config,
        "evidence": recovery.evidence,
        "conflicts": {
            key: {"embedded_config": before, "tensors": after}
            for key, (before, after) in recovery.conflicts.items()
        },
        "assumptions": recovery.assumptions,
        "notes": recovery.notes,
        "load_report": {
            "exact": report.exact,
            "missing": report.missing,
            "unexpected": report.unexpected,
            "mismatched": [
                {"key": key, "checkpoint": list(a), "model": list(b)}
                for key, a, b in report.mismatched
            ],
            "adapted": report.adapted,
        },
        "export": export_info,
        "parity": parity,
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "output_semantics": {
            "output_activation": recovery.config.get("output_activation"),
            "clamped_in_graph": bool(clamp_output),
            "note": (
                "The graph returns the generator output as trained. Clamp to "
                "[0, 1] before reflectance-domain metrics unless "
                "clamped_in_graph is true."
            ),
        },
    }

    manifest_path: Optional[Path] = None
    if write_manifest:
        manifest_path = onnx_path.with_suffix(".json")
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=str)

    return OnnxExportResult(
        onnx_path=onnx_path,
        manifest_path=manifest_path,
        manifest=manifest,
        recovery=recovery,
        load_report=report,
        parity=parity,
    )


__all__ = [
    "ARCHITECTURE_CONFIG_KEYS",
    "DEFAULT_PARITY_TOLERANCE",
    "ONNX_MANIFEST_FORMAT",
    "PRECISIONS",
    "ArchitectureRecovery",
    "LoadReport",
    "OnnxExportResult",
    "export_checkpoint_to_onnx",
    "export_generator_to_onnx",
    "convert_onnx_fp16_to_fp32",
    "load_checkpoint_payload",
    "load_generator_from_any_checkpoint",
    "rebuild_generator",
    "recover_architecture",
    "recover_structural_config",
    "verify_onnx_against_torch",
]

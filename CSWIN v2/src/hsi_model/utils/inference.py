# src/hsi_model/utils/inference.py
"""
Inference helpers for the generator-only (post-GAN) CSWin RGB->HSI model.

`train_generator.py` saves checkpoints whose ``state_dict`` is the BARE generator
(plus the ``config`` used to build it). These helpers rebuild and load it for
inference, and also accept the legacy GAN checkpoint format (full model with
``generator.``/``discriminator.`` prefixed keys).

Public API:
    load_generator(checkpoint_or_weights) -> (NoiseRobustCSWinGenerator, info)
    load_generator_from_weights(weights)  -> (NoiseRobustCSWinGenerator, info)
    convert_checkpoint_to_weights(...)    -> info
    build_patch_inference(checkpoint) -> PatchInference wrapping the generator
    geometric_self_ensemble(fn, img)  -> x8 flip/rotate test-time augmentation

best_model.pth already holds EMA weights (``ema_applied=True``); for a raw
checkpoint (latest_checkpoint.pth) the EMA shadow is applied when
``prefer_ema=True`` so you always evaluate the smoothed weights.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from ..models.generator_v3 import NoiseRobustCSWinGenerator
from .patch_inference import PatchInference

logger = logging.getLogger(__name__)

ArchitectureConfig = Union[Mapping[str, Any], str, Path]
GENERATOR_WEIGHTS_FORMAT = "cswin_generator_weights_v1"


def load_architecture_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a generator architecture config from JSON or YAML."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Architecture config does not exist: {config_path}")

    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            config = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - project dependency
                raise ImportError(
                    "PyYAML is required to load YAML architecture configs."
                ) from exc
            config = yaml.safe_load(handle)
        else:
            raise ValueError(
                f"Unsupported architecture config format {config_path.suffix!r}; "
                "use .json, .yaml, or .yml."
            )

    if not isinstance(config, Mapping):
        raise ValueError(
            f"Architecture config {config_path} must contain a mapping at its root."
        )
    return dict(config)


def _resolve_architecture_config(
    checkpoint: Mapping[str, Any],
    config: Optional[ArchitectureConfig],
    architecture_config: Optional[ArchitectureConfig],
    checkpoint_path: Union[str, Path],
) -> Tuple[Mapping[str, Any], str]:
    """Resolve an explicit config or the config embedded in a checkpoint."""
    if config is not None and architecture_config is not None:
        raise ValueError("Pass only one of config= and architecture_config=.")

    supplied = config if config is not None else architecture_config
    if supplied is not None:
        if isinstance(supplied, (str, Path)):
            return load_architecture_config(supplied), "external_file"
        if isinstance(supplied, Mapping):
            return supplied, "explicit_mapping"
        raise TypeError(
            "Architecture config must be a mapping or a JSON/YAML path; "
            f"received {type(supplied)!r}."
        )

    for key in ("config", "architecture_config", "model_config"):
        embedded = checkpoint.get(key)
        if isinstance(embedded, Mapping):
            return embedded, f"checkpoint:{key}"

    raise ValueError(
        f"Weights file {checkpoint_path} has no embedded architecture config; "
        "pass architecture_config=... (a mapping or JSON/YAML path) so the "
        "known CSWIN architecture can be rebuilt exactly."
    )


def _tensor_state_dict(candidate: Any) -> Optional[Dict[str, torch.Tensor]]:
    """Return tensor entries from a state-dict-like mapping, if present."""
    if not isinstance(candidate, Mapping):
        return None
    state = {
        str(key): value
        for key, value in candidate.items()
        if isinstance(value, torch.Tensor)
    }
    return state or None


def _extract_state_dict(checkpoint: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Find a tensor state dict in common checkpoint and weights layouts."""
    for key in (
        "state_dict",
        "model_state_dict",
        "generator_state_dict",
        "weights",
        "generator_weights",
        "ema_state_dict",
        "ema_weights",
    ):
        state = _tensor_state_dict(checkpoint.get(key))
        if state is not None:
            return state

    # A few older exports nested the state dict under ``model`` or
    # ``generator`` instead of naming it explicitly.
    for key in ("model", "generator", "netG"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            state = _tensor_state_dict(nested)
            if state is not None:
                return state

    # EMA-only exports are also useful inference artifacts.
    ema_payload = checkpoint.get("ema")
    if isinstance(ema_payload, Mapping):
        for key in ("shadow", "state_dict", "weights"):
            state = _tensor_state_dict(ema_payload.get(key))
            if state is not None:
                return state

    # A bare torch.save(model.state_dict(), path) is itself a mapping.
    state = _tensor_state_dict(checkpoint)
    if state is not None:
        return state

    available = list(checkpoint.keys())[:20]
    raise ValueError(
        "No tensor state_dict found in the weights/checkpoint. Expected a bare "
        "state dict or one of state_dict, model_state_dict, generator_state_dict, "
        f"or weights; available keys: {available}"
    )


def _normalize_generator_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Reduce any checkpoint state_dict to bare generator keys.

    Handles DDP ``module.`` wrappers and the legacy full-model layout where the
    generator weights are prefixed with ``generator.`` (and a discriminator is
    also present).
    """
    state = _tensor_state_dict(state_dict)
    if state is None:
        raise ValueError("The selected state dict contains no tensor weights.")

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        normalized_key = key
        # DDP and torch.compile wrappers can be stacked.
        while normalized_key.startswith(("module.", "_orig_mod.")):
            if normalized_key.startswith("module."):
                normalized_key = normalized_key[len("module."):]
            else:
                normalized_key = normalized_key[len("_orig_mod."):]
        # Full-model exports sometimes have model.generator.* keys.
        if normalized_key.startswith("model."):
            normalized_key = normalized_key[len("model."):]
        cleaned[normalized_key] = value

    # Legacy full-model checkpoints use a generator/netG prefix and may also
    # contain discriminator weights.  Select only the generator branch.
    for prefix in ("generator.", "netG.", "G."):
        prefixed = {
            key[len(prefix):]: value
            for key, value in cleaned.items()
            if key.startswith(prefix)
        }
        if prefixed:
            return prefixed
    return cleaned


def _plain_config(value: Any) -> Any:
    """Convert common config container types into torch-save-friendly values."""
    if isinstance(value, Mapping):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


def load_generator(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    config: Optional[ArchitectureConfig] = None,
    prefer_ema: bool = True,
    strict: bool = True,
    architecture_config: Optional[ArchitectureConfig] = None,
) -> Tuple[NoiseRobustCSWinGenerator, Dict[str, Any]]:
    """Rebuild and load the generator from a checkpoint.

    Args:
        checkpoint_path: path to best_model.pth / latest_checkpoint.pth / net_*epoch.pth.
        device: target device (defaults to CUDA if available).
        config: architecture config override; if None, uses the checkpoint's
            saved ``config`` (required — the generator must be built with the
            exact architecture the weights were trained with).
        prefer_ema: if the checkpoint carries a raw state_dict plus an EMA
            shadow (latest_checkpoint), apply the EMA shadow.
        strict: strict state_dict loading; falls back to non-strict with a
            warning if it fails.

    Returns:
        (generator in eval mode on ``device``, info dict).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ck, Mapping):
        raise ValueError(f"Unexpected checkpoint object in {checkpoint_path}: {type(ck)}")

    cfg, config_source = _resolve_architecture_config(
        ck,
        config=config,
        architecture_config=architecture_config,
        checkpoint_path=checkpoint_path,
    )

    generator = NoiseRobustCSWinGenerator(cfg).to(device)

    state = _normalize_generator_state(_extract_state_dict(ck))

    if strict:
        generator.load_state_dict(state, strict=True)
    else:
        result = generator.load_state_dict(state, strict=False)
        if getattr(result, "missing_keys", None):
            logger.warning("Missing keys (first 8): %s", list(result.missing_keys)[:8])
        if getattr(result, "unexpected_keys", None):
            logger.warning("Unexpected keys (first 8): %s", list(result.unexpected_keys)[:8])
        model_keys = set(generator.state_dict())
        matched_keys = model_keys.intersection(state)
        if not matched_keys:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} matched no generator state keys."
            )

    # Apply EMA shadow for raw checkpoints that carry one (best_model already has
    # EMA baked into state_dict and is flagged ema_applied=True).
    applied_ema = bool(ck.get("ema_applied", False))
    ema_payload = ck.get("ema")
    shadow: Any = {}
    if isinstance(ema_payload, Mapping):
        shadow = ema_payload.get("shadow", {})
    if not shadow:
        shadow = ck.get("ema_state_dict", {})
    if prefer_ema and not applied_ema and shadow:
        try:
            shadow = _normalize_generator_state(shadow)
        except ValueError:
            shadow = {}
        own = {
            name: param
            for name, param in generator.named_parameters()
            if param.requires_grad and param.dtype.is_floating_point
        }
        valid_shadow = {
            name: tensor
            for name, tensor in shadow.items()
            if (
                name in own
                and isinstance(tensor, torch.Tensor)
                and tuple(tensor.shape) == tuple(own[name].shape)
            )
        }
        missing = sorted(set(own) - set(valid_shadow))
        if missing:
            logger.warning(
                "EMA shadow is incomplete (%d/%d trainable params matched); "
                "using the complete raw checkpoint weights instead. Missing "
                "keys (first 8): %s",
                len(valid_shadow),
                len(own),
                missing[:8],
            )
        else:
            with torch.no_grad():
                for name, tensor in valid_shadow.items():
                    own[name].copy_(
                        tensor.to(
                            device=own[name].device,
                            dtype=own[name].dtype,
                        )
                    )
            applied_ema = bool(valid_shadow)
            logger.info("Applied EMA shadow to %d generator params.", len(valid_shadow))

    generator.eval()
    out_act = str(cfg.get("output_activation", "none")).lower()
    info = {
        "config": cfg,
        "output_activation": out_act,
        # The generator already maps to [0,1] (sigmoid head) or was trained to
        # output reflectance directly (linear head) — inference should NOT apply
        # another sigmoid in either case.
        "applies_own_activation": out_act in ("sigmoid", "delayed_sigmoid", "tanh"),
        "ema_applied": applied_ema,
        "epoch": ck.get("epoch"),
        "val_metrics": ck.get("val_metrics"),
        "config_source": config_source,
        "state_dict_keys": len(state),
        "source_format": ck.get("format", "raw_state_dict"),
    }
    logger.info(
        "Loaded generator from %s (ema_applied=%s, output_activation=%s).",
        checkpoint_path, applied_ema, out_act,
    )
    return generator, info


def load_generator_from_weights(
    weights_path: str,
    architecture_config: Optional[ArchitectureConfig] = None,
    device: Optional[torch.device] = None,
    prefer_ema: bool = True,
    strict: bool = True,
) -> Tuple[NoiseRobustCSWinGenerator, Dict[str, Any]]:
    """Load a generator directly from raw or converted weights.

    ``architecture_config`` is required for a bare state dict unless the
    weights bundle already embeds a config. This explicit entry point is useful
    for older CSWIN architectures whose file contains tensors but no config.
    """
    return load_generator(
        weights_path,
        device=device,
        architecture_config=architecture_config,
        prefer_ema=prefer_ema,
        strict=strict,
    )


def convert_checkpoint_to_weights(
    checkpoint_path: str,
    output_path: str,
    *,
    architecture_config: Optional[ArchitectureConfig] = None,
    config: Optional[ArchitectureConfig] = None,
    device: Optional[torch.device] = None,
    prefer_ema: bool = True,
    strict: bool = True,
    embed_config: bool = True,
) -> Dict[str, Any]:
    """Convert a CSWIN checkpoint into a generator-only weights artifact.

    The default output is a compact, self-contained bundle containing only
    generator ``state_dict`` tensors and the architecture config. Set
    ``embed_config=False`` to write a bare state dict instead; that file must
    be loaded with ``architecture_config=...`` later.
    """
    load_device = device or torch.device("cpu")
    generator, info = load_generator(
        checkpoint_path,
        device=load_device,
        config=config,
        architecture_config=architecture_config,
        prefer_ema=prefer_ema,
        strict=strict,
    )
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in generator.state_dict().items()
        if isinstance(tensor, torch.Tensor)
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if embed_config:
        payload: Any = {
            "format": GENERATOR_WEIGHTS_FORMAT,
            "state_dict": state_dict,
            "config": _plain_config(info["config"]),
            "ema_applied": bool(info.get("ema_applied", False)),
            "metadata": {
                "source_checkpoint": str(checkpoint_path),
                "state_dict_keys": len(state_dict),
                "ema_applied": bool(info.get("ema_applied", False)),
            },
        }
    else:
        payload = state_dict
    torch.save(payload, destination)

    result = dict(info)
    result.update(
        {
            "output_path": str(destination),
            "weights_format": (
                GENERATOR_WEIGHTS_FORMAT if embed_config else "raw_state_dict"
            ),
            "weights_keys": len(state_dict),
        }
    )
    logger.info(
        "Converted %s to generator weights at %s (%d tensors, embedded_config=%s).",
        checkpoint_path,
        destination,
        len(state_dict),
        embed_config,
    )
    return result


def build_patch_inference(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    patch_size: int = 128,
    overlap: int = 16,
    batch_size: int = 4,
    use_fp16: bool = False,
    config: Optional[ArchitectureConfig] = None,
    prefer_ema: bool = True,
    amp_dtype: Optional[torch.dtype] = None,
    architecture_config: Optional[ArchitectureConfig] = None,
    strict: bool = True,
) -> PatchInference:
    """Convenience: load the generator and wrap it in :class:`PatchInference`.

    ``apply_sigmoid`` is forced False — both the new sigmoid head and the legacy
    linear head output reflectance directly, so no extra activation is applied.
    """
    generator, _info = load_generator(
        checkpoint_path,
        device=device,
        config=config,
        architecture_config=architecture_config,
        prefer_ema=prefer_ema,
        strict=strict,
    )
    return PatchInference(
        model=generator,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        use_fp16=use_fp16,
        apply_sigmoid=False,
        amp_dtype=amp_dtype,
    )


def _apply_d4(img: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        img = torch.flip(img, dims=[-1])
    if k:
        img = torch.rot90(img, k, dims=[-2, -1])
    return img


def _invert_d4(img: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    # Inverse of (rot90^k after hflip) is (hflip after rot90^-k).
    if k:
        img = torch.rot90(img, -k, dims=[-2, -1])
    if flip:
        img = torch.flip(img, dims=[-1])
    return img


def geometric_self_ensemble(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    img: torch.Tensor,
) -> torch.Tensor:
    """x8 geometric self-ensemble (the D4 dihedral group).

    Averages predictions over the 8 flip/rotation symmetries, each transformed
    back to canonical orientation. Standard ARAD-1K leaderboard trick — typically
    a few thousandths of MRAE and a few tenths of a dB PSNR for free, no retrain.

    The transforms are spatial-only, so they are valid for HSI (spectral channels
    are untouched). ``predict_fn`` maps ``(1, Cin, H, W) -> (1, Cout, H, W)`` and
    must tolerate swapped H/W (rot90 transposes non-square inputs); the generator
    does, via its internal pad/crop.

    Note: when ``predict_fn`` is a patch-tiling inferencer, odd-``k`` (transposed)
    members are re-tiled on a different patch grid than the even members, so the
    8 reconstructions are not produced by a strictly identical tiling. The effect
    is confined to tile seams / the padded outer ring (sub-mdB); each member is
    still inverted back to canonical orientation before averaging, so the result
    is geometrically correct. For exact grid parity, pad to a common size before
    tiling or apply D4 per tile.
    """
    outputs = []
    for flip in (False, True):
        for k in range(4):
            transformed = _apply_d4(img, k, flip)
            pred = predict_fn(transformed)
            outputs.append(_invert_d4(pred, k, flip))
    return torch.stack(outputs, dim=0).mean(dim=0)


__all__ = [
    "GENERATOR_WEIGHTS_FORMAT",
    "load_architecture_config",
    "load_generator",
    "load_generator_from_weights",
    "convert_checkpoint_to_weights",
    "build_patch_inference",
    "geometric_self_ensemble",
]

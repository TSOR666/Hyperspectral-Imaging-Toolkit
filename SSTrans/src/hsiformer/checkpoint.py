from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

_RGB_NORMALIZATIONS = {"scale_255", "per_image"}


def _unwrap_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping of parameter names to tensors.")
    for key in ("state_dict", "model", "model_state_dict", "generator", "G"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break
    return checkpoint


def _strip_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not state_dict or not all(key.startswith(prefix) for key in state_dict):
        return dict(state_dict)
    return {key[len(prefix) :]: value for key, value in state_dict.items()}


def _model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract model weights from local, Lightning, and upstream wrappers."""
    state_dict = dict(_unwrap_state_dict(checkpoint))
    for prefix in ("module.", "_orig_mod."):
        state_dict = _strip_prefix(state_dict, prefix)

    # The upstream Lightning checkpoint also stores the frozen
    # ``deltaE_criterion`` convolution in ``state_dict``. Select just the
    # generator namespace instead of requiring every key to share ``model.``.
    model_weights = {
        key[len("model.") :]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if model_weights:
        state_dict = model_weights
    else:
        for prefix in ("generator.", "G."):
            state_dict = _strip_prefix(state_dict, prefix)

    for prefix in ("module.", "_orig_mod."):
        state_dict = _strip_prefix(state_dict, prefix)
    return state_dict


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> nn.modules.module._IncompatibleKeys:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    state_dict = _model_state_dict(checkpoint)
    return model.load_state_dict(state_dict, strict=strict)


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def checkpoint_rgb_normalization(
    payload: Any,
    *,
    default: str | None = None,
) -> str:
    """Recover the RGB preprocessing contract stored by the trainer.

    Raw/legacy weight files have no training metadata, so their caller must
    either provide an explicit override or pass a deliberate ``default``.
    """
    if default is not None and default not in _RGB_NORMALIZATIONS:
        raise ValueError(f"Unknown RGB normalization mode: {default}")
    training_config = (
        payload.get("training_config")
        if isinstance(payload, Mapping)
        else None
    )
    if not isinstance(training_config, Mapping):
        if default is not None:
            return default
        raise ValueError(
            "Checkpoint has no RGB preprocessing metadata. Pass an explicit "
            "--rgb-normalization matching the checkpoint's training run."
        )
    normalization = training_config.get("rgb_normalization")
    if normalization is None:
        if default is not None:
            return default
        raise ValueError(
            "Checkpoint training metadata has no rgb_normalization value. "
            "Pass an explicit --rgb-normalization."
        )
    normalization = str(normalization)
    if normalization not in _RGB_NORMALIZATIONS:
        raise ValueError(
            "Checkpoint contains an unsupported RGB normalization mode: "
            f"{normalization}"
        )
    return normalization


def build_model_from_checkpoint(
    path: str | Path,
    *,
    preset: str | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, Any]:
    """Build a model from trainer metadata, with a preset fallback for raw weights."""
    from .model import HSIFormer
    from .presets import build_model

    payload = load_checkpoint_payload(path, map_location=map_location)
    model_config = payload.get("model_config") if isinstance(payload, Mapping) else None
    if isinstance(model_config, Mapping):
        model = HSIFormer(**dict(model_config))
    elif preset is not None:
        model = build_model(preset)
    else:
        raise ValueError(
            "Checkpoint has no model_config metadata. Pass the architecture "
            "preset explicitly."
        )

    state_dict = _model_state_dict(payload)
    model.load_state_dict(state_dict, strict=strict)
    return model, payload

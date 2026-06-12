from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _unwrap_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping of parameter names to tensors.")
    for key in ("state_dict", "model", "model_state_dict", "generator"):
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


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> nn.modules.module._IncompatibleKeys:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    state_dict = _unwrap_state_dict(checkpoint)
    for prefix in ("module.", "model.", "generator."):
        state_dict = _strip_prefix(state_dict, prefix)
    return model.load_state_dict(state_dict, strict=strict)


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


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

    state_dict = _unwrap_state_dict(payload)
    for prefix in ("module.", "model.", "generator.", "_orig_mod."):
        state_dict = _strip_prefix(state_dict, prefix)
    model.load_state_dict(state_dict, strict=strict)
    return model, payload

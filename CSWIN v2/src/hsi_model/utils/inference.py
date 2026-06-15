# src/hsi_model/utils/inference.py
"""
Inference helpers for the generator-only (post-GAN) CSWin RGB->HSI model.

`train_generator.py` saves checkpoints whose ``state_dict`` is the BARE generator
(plus the ``config`` used to build it). These helpers rebuild and load it for
inference, and also accept the legacy GAN checkpoint format (full model with
``generator.``/``discriminator.`` prefixed keys).

Public API:
    load_generator(checkpoint)        -> (NoiseRobustCSWinGenerator, info)
    build_patch_inference(checkpoint) -> PatchInference wrapping the generator
    geometric_self_ensemble(fn, img)  -> x8 flip/rotate test-time augmentation

best_model.pth already holds EMA weights (``ema_applied=True``); for a raw
checkpoint (latest_checkpoint.pth) the EMA shadow is applied when
``prefer_ema=True`` so you always evaluate the smoothed weights.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ..models.generator_v3 import NoiseRobustCSWinGenerator
from .patch_inference import PatchInference

logger = logging.getLogger(__name__)


def _normalize_generator_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Reduce any checkpoint state_dict to bare generator keys.

    Handles DDP ``module.`` wrappers and the legacy full-model layout where the
    generator weights are prefixed with ``generator.`` (and a discriminator is
    also present).
    """
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
    gen_keys = [k for k in state_dict if k.startswith("generator.")]
    if gen_keys:
        return {k[len("generator."):]: v for k, v in state_dict.items() if k.startswith("generator.")}
    return dict(state_dict)


def load_generator(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    config: Optional[Dict[str, Any]] = None,
    prefer_ema: bool = True,
    strict: bool = True,
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
    if not isinstance(ck, dict):
        raise ValueError(f"Unexpected checkpoint object in {checkpoint_path}: {type(ck)}")

    cfg = config if config is not None else ck.get("config")
    if cfg is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'config'; pass config=... so the "
            "generator architecture can be rebuilt to match the weights."
        )

    generator = NoiseRobustCSWinGenerator(cfg).to(device)

    raw_state = ck.get("state_dict") or ck.get("model_state_dict") or ck
    state = _normalize_generator_state(raw_state)

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
    if prefer_ema and not applied_ema and ck.get("ema"):
        shadow = (ck["ema"] or {}).get("shadow", {})
        own = dict(generator.named_parameters())
        n = 0
        with torch.no_grad():
            for name, tensor in shadow.items():
                if name in own and isinstance(tensor, torch.Tensor):
                    own[name].copy_(tensor.to(own[name].device))
                    n += 1
        applied_ema = n > 0
        logger.info("Applied EMA shadow to %d generator params.", n)

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
    }
    logger.info(
        "Loaded generator from %s (ema_applied=%s, output_activation=%s).",
        checkpoint_path, applied_ema, out_act,
    )
    return generator, info


def build_patch_inference(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    patch_size: int = 128,
    overlap: int = 16,
    batch_size: int = 4,
    use_fp16: bool = False,
    config: Optional[Dict[str, Any]] = None,
    prefer_ema: bool = True,
) -> PatchInference:
    """Convenience: load the generator and wrap it in :class:`PatchInference`.

    ``apply_sigmoid`` is forced False — both the new sigmoid head and the legacy
    linear head output reflectance directly, so no extra activation is applied.
    """
    generator, _info = load_generator(
        checkpoint_path, device=device, config=config, prefer_ema=prefer_ema
    )
    return PatchInference(
        model=generator,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        use_fp16=use_fp16,
        apply_sigmoid=False,
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
    "load_generator",
    "build_patch_inference",
    "geometric_self_ensemble",
]

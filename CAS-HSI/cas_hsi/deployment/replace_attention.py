"""Swap attention spatial mixers for convolutional ones (specification section 9.2).

The CAS block shell -- Norm -> mixer -> residual, Norm -> cross-channel attention ->
residual, Norm -> gated FFN -> residual -- is deliberately identical in both
backends, so exactly one submodule per block has to change:

    DilatedLocalAttention        ->  dilated_depthwise_conv       (MultiDilationDepthwiseMixer)
    HybridLocalStripeAttention   ->  large_kernel_depthwise_conv  (ConvSpatialMixer, wide kernel)

Cross-channel attention is **not** touched. Its cost is O(C^2 * HW), independent of
resolution, and it is what carries the spectral prior; removing it would gut the
model rather than speed it up.

*** THE NEW CONVOLUTIONS HAVE RANDOM WEIGHTS. ***

This does not hand you a trained edge model. Everything the replaced attention had
learned about *where* to look is discarded. The output of this function is a
starting point for distillation (spec 9.4) or QAT (spec 9.6) -- run it, then train
the student against the research teacher. If you want an edge model without
distillation, train `configs/cas_hsi_edge.yaml` from scratch instead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch.nn as nn

from ..blocks.cas_block import CASBlock
from ..blocks.spatial_attention import (
    DilatedLocalAttention,
    HybridLocalStripeAttention,
    build_spatial_mixer,
)

__all__ = ["has_attention_mixers", "attention_mixer_paths", "replace_attention_mixers"]

_ATTENTION_TYPES = (DilatedLocalAttention, HybridLocalStripeAttention)


def attention_mixer_paths(model: nn.Module) -> List[str]:
    """Dotted paths of every attention spatial mixer still in the graph."""
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, _ATTENTION_TYPES)
    ]


def has_attention_mixers(model: nn.Module) -> bool:
    """True if the model still contains an operator with no portable export (spec 9.7)."""
    return bool(attention_mixer_paths(model))


def _edge_mixer_for(old: nn.Module, block: CASBlock) -> Tuple[nn.Module, str]:
    """Build the convolutional stand-in, reading its shape from the module it replaces."""
    channels = old.channels
    dilations = old.dilations

    if isinstance(old, HybridLocalStripeAttention):
        # Stripe attention buys a long-range receptive field along one axis; the
        # closest convolutional analogue is simply a much wider depthwise kernel.
        name = "large_kernel_depthwise_conv"
    elif isinstance(old, DilatedLocalAttention):
        name = "dilated_depthwise_conv"
    else:  # pragma: no cover - guarded by the caller
        raise TypeError(f"Not an attention mixer: {type(old).__name__}")

    mixer = build_spatial_mixer(
        name=name,
        channels=channels,
        head_dim=old.head_dim,
        dilations=dilations,
        kernel_size=old.kernel_size,
        large_kernel=getattr(block, "_large_kernel", 11),
    )
    return mixer, name


def replace_attention_mixers(
    model: nn.Module,
    *,
    large_kernel: int = 11,
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """Replace every attention spatial mixer in ``model`` with its edge equivalent.

    Mutates ``model`` in place and returns it, together with a report describing
    each swap (path, old class, new class, parameter delta).

    ``model.config.backend`` is flipped to ``"edge"`` so ``get_model_info()`` does
    not go on claiming an architecture the model no longer has.
    """
    report: List[Dict[str, Any]] = []

    for block_name, block in model.named_modules():
        if not isinstance(block, CASBlock):
            continue
        old = block.spatial_mixer
        if not isinstance(old, _ATTENTION_TYPES):
            continue

        block._large_kernel = large_kernel  # consulted by _edge_mixer_for
        new, mixer_name = _edge_mixer_for(old, block)

        old_params = sum(p.numel() for p in old.parameters())
        new_params = sum(p.numel() for p in new.parameters())

        # Land the replacement on the same device/dtype as the block it joins.
        reference = next(block.parameters())
        block.spatial_mixer = new.to(device=reference.device, dtype=reference.dtype)
        block.spatial_mixer_name = mixer_name

        report.append(
            {
                "path": f"{block_name}.spatial_mixer",
                "old": type(old).__name__,
                "new": mixer_name,
                "channels": old.channels,
                "old_parameters": old_params,
                "new_parameters": new_params,
                "parameter_delta": new_params - old_params,
            }
        )

    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "backend"):
        config.backend = "edge"

    return model, report

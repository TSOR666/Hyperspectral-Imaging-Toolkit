"""Arbitrary-image-size support (specification section 8).

The network downsamples twice by a factor of two, so the *padded* spatial dims
must be divisible by ``size_multiple`` (4).  The caller must never be asked to
resize: we pad on the right/bottom, run the network, and crop back.

Cropping is a slice, never an interpolation (spec 8.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = ["PadInfo", "required_padding", "pad_to_multiple", "crop_to_original"]


def _is_compiling() -> bool:
    """Whether the current call is being captured by ``torch.export``/Dynamo."""
    compiler = getattr(torch, "compiler", None)
    return bool(compiler is not None and compiler.is_compiling())


@dataclass(frozen=True)
class PadInfo:
    """Bookkeeping for a right/bottom pad so the crop is exact."""

    original_height: int
    original_width: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int


def required_padding(size: int, multiple: int) -> int:
    """Padding needed to raise ``size`` to the next multiple of ``multiple``."""
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return (multiple - size % multiple) % multiple


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int = 4,
    mode: str = "reflect",
) -> Tuple[torch.Tensor, PadInfo]:
    """Pad ``x`` on the right/bottom so H and W are divisible by ``multiple``.

    ``reflect`` needs the pad width to be strictly smaller than the input extent,
    and needs at least 2 rows/columns to reflect from.  Both conditions fail on
    tiny inputs (the spec's own test matrix includes 1x1), so we fall back to
    ``replicate``, which has no such constraint.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected a 4-D NCHW tensor, got shape {tuple(x.shape)}")

    # Keep these as SymInt values while torch.export captures the model. Coercing
    # them with int() makes H/W constants in the exported graph, even if ONNX input
    # axes are later named "height" and "width".
    height, width = x.shape[-2:]

    pad_h = required_padding(height, multiple)
    pad_w = required_padding(width, multiple)

    pad_info = PadInfo(
        original_height=height,
        original_width=width,
        pad_left=0,
        pad_right=pad_w,
        pad_top=0,
        pad_bottom=pad_h,
    )

    if not _is_compiling():
        if pad_h == 0 and pad_w == 0:
            return x, pad_info

        selected_mode = mode
        if mode == "reflect" and (pad_h >= height or pad_w >= width or height <= 1 or width <= 1):
            selected_mode = "replicate"
    else:
        # ``pad_h``/``pad_w`` must remain symbolic. The production model's normal
        # reflect boundary condition is preserved for ONNX inputs with axes >= 4;
        # this is the smallest extent for which the maximum three-pixel reflect pad
        # is well-defined. Eager inference retains its replicate fallback for 1--3.
        selected_mode = mode

    x = F.pad(
        x,
        (
            pad_info.pad_left,
            pad_info.pad_right,
            pad_info.pad_top,
            pad_info.pad_bottom,
        ),
        mode=selected_mode,
    )

    return x, pad_info


def crop_to_original(x: torch.Tensor, pad_info: PadInfo) -> torch.Tensor:
    """Undo :func:`pad_to_multiple` by slicing (never by interpolation)."""
    return x[
        :,
        :,
        pad_info.pad_top : pad_info.pad_top + pad_info.original_height,
        pad_info.pad_left : pad_info.pad_left + pad_info.original_width,
    ]

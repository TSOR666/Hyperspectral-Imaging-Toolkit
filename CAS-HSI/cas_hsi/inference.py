"""Tiled inference for large images (specification section 8.7).

Arbitrary-size support does not remove GPU-memory limits: a 482x512 scene at
full resolution holds far larger activations than a 128x128 training patch.
Overlapping tiles with a Hann blend keep peak memory bounded and leave no visible
seams, because the window goes to zero at the tile border so no single tile
dominates the boundary pixels.

    Y = sum_i (W_i * Y_i) / (sum_i W_i + eps)
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn

__all__ = ["tiled_inference", "hann_window_2d"]

BlendMode = Literal["hann", "uniform"]


def hann_window_2d(
    height: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Separable Hann window, strictly positive inside the tile.

    ``torch.hann_window(periodic=False)`` is zero at both endpoints; a tile whose
    every border pixel has zero weight contributes nothing there, and a corner
    pixel covered by only one tile would divide 0/0.  Clamping to a small floor
    keeps the partition of unity well-posed.
    """
    win_h = torch.hann_window(max(height, 2), periodic=False, device=device, dtype=dtype)[:height]
    win_w = torch.hann_window(max(width, 2), periodic=False, device=device, dtype=dtype)[:width]
    window = torch.outer(win_h, win_w).clamp_min(1e-3)
    return window.reshape(1, 1, height, width)


@torch.no_grad()
def tiled_inference(
    model: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    rgb: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 32,
    blend_mode: BlendMode = "hann",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reconstruct a large scene tile-by-tile with overlap blending.

    Args:
        model: anything callable with ``[B, 3, h, w] -> [B, bands, h, w]``.
        rgb: ``[B, 3, H, W]``.
        tile_size: tile side in pixels. Tiles larger than the image fall back to
            a single full-image forward pass.
        overlap: overlap between neighbouring tiles, in pixels.
        blend_mode: ``hann`` (tapered, seamless) or ``uniform`` (box average).

    Returns:
        ``[B, bands, H, W]``.
    """
    if rgb.dim() != 4:
        raise ValueError(f"Expected a 4-D NCHW tensor, got shape {tuple(rgb.shape)}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if tile_size <= overlap:
        raise ValueError(
            f"tile_size ({tile_size}) must exceed overlap ({overlap}); "
            "otherwise the tile grid never advances."
        )

    batch, _, height, width = rgb.shape

    if tile_size >= height and tile_size >= width:
        return model(rgb)

    stride = tile_size - overlap

    def starts(extent: int) -> list[int]:
        if extent <= tile_size:
            return [0]
        positions = list(range(0, extent - tile_size + 1, stride))
        if positions[-1] != extent - tile_size:
            positions.append(extent - tile_size)
        return positions

    row_starts = starts(height)
    col_starts = starts(width)

    accumulator: torch.Tensor | None = None
    weights: torch.Tensor | None = None

    for top in row_starts:
        for left in col_starts:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = rgb[:, :, top:bottom, left:right]

            prediction = model(tile)
            if isinstance(prediction, tuple):
                prediction = prediction[0]

            tile_h, tile_w = prediction.shape[-2], prediction.shape[-1]

            if blend_mode == "hann":
                window = hann_window_2d(
                    tile_h, tile_w, device=prediction.device, dtype=prediction.dtype
                )
            elif blend_mode == "uniform":
                window = torch.ones(
                    1, 1, tile_h, tile_w, device=prediction.device, dtype=prediction.dtype
                )
            else:  # pragma: no cover - guarded by the Literal
                raise ValueError(f"Unknown blend_mode {blend_mode!r}")

            if accumulator is None:
                bands = prediction.shape[1]
                accumulator = torch.zeros(
                    batch, bands, height, width,
                    device=prediction.device, dtype=prediction.dtype,
                )
                weights = torch.zeros(
                    1, 1, height, width,
                    device=prediction.device, dtype=prediction.dtype,
                )

            accumulator[:, :, top:bottom, left:right] += prediction * window
            weights[:, :, top:bottom, left:right] += window

    assert accumulator is not None and weights is not None  # non-empty tile grid
    return accumulator / (weights + eps)

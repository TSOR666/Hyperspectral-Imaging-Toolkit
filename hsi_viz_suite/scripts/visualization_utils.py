
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.colors as mcolors
from scipy import ndimage  # type: ignore[import-untyped]

EPS = 1e-8


def to_chw(x: np.ndarray, expected_bands: int | None = None) -> np.ndarray:
    """
    Ensure an HSI array is channel-first (C,H,W).

    Args:
        x: Input array of shape (H,W,C) or (C,H,W)
        expected_bands: Expected number of spectral bands. If None, uses heuristic
            (last dim smaller than spatial dims indicates HWC format).

    Returns:
        Array in (C,H,W) format.
    """
    if x.ndim != 3:
        return x

    # If expected_bands provided, use it to determine format
    if expected_bands is not None:
        if x.shape[-1] == expected_bands and x.shape[0] != expected_bands:
            return x.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
        return x

    # Heuristic: if last dim is much smaller than first two, assume HWC
    # This handles various band counts (31, 32, etc.)
    h, w, c = x.shape[0], x.shape[1], x.shape[2]
    if c < min(h, w) and c <= 256:  # Reasonable band count upper limit
        return x.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
    return x


def _ensure_bchw(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
    if target.dim() == 3:
        target = target.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if target.device != pred.device:
        target = target.to(pred.device)
    if target.dtype != pred.dtype:
        target = target.to(pred.dtype)
    return pred, target


def compute_mrae_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = EPS,
    signal_threshold: float = 1e-3,
) -> np.ndarray:
    """
    Compute robust MRAE (Mean Relative Absolute Error) map per pixel.

    Uses masking to avoid numerical instability for near-zero target values.
    For pixels where target is below threshold, uses absolute error instead.

    Args:
        pred: Predicted HSI, shape (B,C,H,W) or (C,H,W)
        target: Target HSI, same shape as pred
        epsilon: Small constant for numerical stability
        signal_threshold: Minimum target value for computing relative error

    Input shapes: (B,C,H,W) or (C,H,W). Output: (H,W) if B=1 else (B,H,W).
    """
    pred, target = _ensure_bchw(pred, target)
    pred_f = pred.float()
    target_f = target.float()

    abs_error = torch.abs(pred_f - target_f)
    target_abs = target_f.abs()

    # Create mask for valid (non-near-zero) target values
    valid_mask = target_abs > signal_threshold

    # Compute relative error where valid, absolute error elsewhere
    relative_error = abs_error / (target_abs + epsilon)

    # Use relative error where target is strong, cap otherwise to avoid explosion
    # Cap at a reasonable maximum (e.g., 10x = 1000% error)
    max_relative_error = 10.0
    capped_error = torch.where(
        valid_mask,
        relative_error,
        torch.clamp(abs_error / signal_threshold, max=max_relative_error),
    )

    mrae = torch.mean(capped_error, dim=1)  # (B,C,H,W) -> (B,H,W)
    return mrae.squeeze(0).detach().cpu().numpy()  # (H,W) or (B,H,W)


def compute_sam_map(pred: torch.Tensor, target: torch.Tensor, epsilon: float = EPS) -> np.ndarray:
    """
    Compute Spectral Angle Mapper (SAM) in degrees per pixel.

    Input shapes: (B,C,H,W) or (C,H,W). Output: (H,W) if B=1 else (B,H,W).
    """
    pred, target = _ensure_bchw(pred, target)
    pred_f = pred.float()
    target_f = target.float()
    pred_n = F.normalize(pred_f, dim=1, eps=epsilon)  # (B,C,H,W) -> (B,C,H,W)
    targ_n = F.normalize(target_f, dim=1, eps=epsilon)  # (B,C,H,W) -> (B,C,H,W)
    dot = (pred_n * targ_n).sum(dim=1).clamp(-1 + epsilon, 1 - epsilon)  # (B,H,W)
    sam = torch.acos(dot) * 180.0 / torch.pi  # (B,H,W)
    return sam.squeeze(0).detach().cpu().numpy()


def compute_rmse_map(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """
    Compute RMSE map per pixel.

    Input shapes: (B,C,H,W) or (C,H,W). Output: (H,W) if B=1 else (B,H,W).
    """
    pred, target = _ensure_bchw(pred, target)
    pred_f = pred.float()
    target_f = target.float()
    mse = ((pred_f - target_f) ** 2).mean(dim=1)  # (B,H,W)
    rmse = torch.sqrt(mse.clamp_min(0.0))  # (B,H,W)
    return rmse.squeeze(0).detach().cpu().numpy()


def create_error_colormap() -> mcolors.LinearSegmentedColormap:
    colors = ["#0000FF", "#00FF00", "#FFFF00", "#FF0000"]
    return mcolors.LinearSegmentedColormap.from_list("error_cmap", colors, N=256)


def apply_gaussian_smoothing(error_map: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    return ndimage.gaussian_filter(error_map, sigma=sigma)

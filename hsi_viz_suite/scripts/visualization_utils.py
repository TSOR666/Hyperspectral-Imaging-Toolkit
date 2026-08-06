
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy import ndimage  # type: ignore[import-untyped]
from pathlib import Path

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
        if x.shape[0] == expected_bands:
            return x
        if x.shape[-1] == expected_bands:
            return x.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
        if x.shape[1] == expected_bands:
            return x.transpose(1, 0, 2)  # (H,C,W) -> (C,H,W)
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


def setup_publication_style(style: str = "paper", dpi: int = 300) -> None:
    """Apply a compact, vector-friendly style shared by all figure scripts."""
    style_name = "seaborn-v0_8-paper" if style == "paper" else "seaborn-v0_8-talk"
    plt.style.use(style_name)
    base_size = 8 if style == "paper" else 12
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size + 1,
            "axes.linewidth": 0.7,
            "xtick.labelsize": max(base_size - 1, 7),
            "ytick.labelsize": max(base_size - 1, 7),
            "legend.fontsize": max(base_size - 1, 7),
            "image.interpolation": "nearest",
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    """Save a figure as both editable PDF and high-resolution PNG."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stem = destination.with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def robust_limits(
    values: np.ndarray,
    *,
    lower: float = 2.0,
    upper: float = 98.0,
    floor: float | None = None,
) -> tuple[float, float]:
    """Return finite percentile limits suitable for a comparable heatmap."""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, lower))
    vmax = float(np.percentile(finite, upper))
    if floor is not None:
        vmin = max(vmin, floor)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        center = float(np.nanmean(finite))
        spread = max(abs(center) * 0.05, 1e-6)
        vmin, vmax = center - spread, center + spread
    return vmin, vmax


def compute_bandwise_errors(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    signal_threshold: float = 1e-3,
    max_relative_error: float = 10.0,
) -> dict[str, np.ndarray]:
    """Compute publication-friendly per-band error summaries for CHW cubes."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f"Expected equal CHW cubes, got {pred.shape} and {target.shape}."
        )
    absolute = np.abs(pred - target)
    target_abs = np.abs(target)
    valid = target_abs > signal_threshold
    relative = absolute / (target_abs + EPS)
    relative = np.where(
        valid,
        relative,
        np.minimum(absolute / signal_threshold, max_relative_error),
    )
    return {
        "mae": absolute.mean(axis=(1, 2)),
        "rmse": np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2))),
        "mrae": relative.mean(axis=(1, 2)),
    }


def finite_subsample(values: np.ndarray, max_points: int = 20_000) -> np.ndarray:
    """Flatten and deterministically subsample finite values for scatter plots."""
    flattened = np.asarray(values).reshape(-1)
    flattened = flattened[np.isfinite(flattened)]
    if flattened.size <= max_points:
        return flattened
    indices = np.linspace(0, flattened.size - 1, max_points, dtype=np.int64)
    return flattened[indices]

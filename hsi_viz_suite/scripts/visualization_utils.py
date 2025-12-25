
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.colors as mcolors
from scipy import ndimage

EPS = 1e-8


def to_chw(x: np.ndarray) -> np.ndarray:
    """Ensure an HSI array is channel-first if the last dim is the band axis."""
    if x.ndim == 3 and x.shape[-1] == 31:
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


def compute_mrae_map(pred: torch.Tensor, target: torch.Tensor, epsilon: float = EPS) -> np.ndarray:
    """
    Compute MRAE map per pixel.

    Input shapes: (B,C,H,W) or (C,H,W). Output: (H,W) if B=1 else (B,H,W).
    """
    pred, target = _ensure_bchw(pred, target)
    pred_f = pred.float()
    target_f = target.float()
    ratio = torch.abs(pred_f - target_f) / (target_f.abs() + epsilon)  # (B,C,H,W) broadcast
    mrae = torch.mean(ratio, dim=1)  # (B,C,H,W) -> (B,H,W)
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

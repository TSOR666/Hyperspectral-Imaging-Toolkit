
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.colors as mcolors
from scipy import ndimage

def to_chw(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[-1] == 31:
        return x.transpose(2,0,1)
    return x

def compute_mrae_map(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8):
    if pred.dim() == 3:
        pred = pred.unsqueeze(0); target = target.unsqueeze(0)
    mrae = torch.mean(torch.abs(pred - target) / (target.abs() + epsilon), dim=1)
    return mrae.squeeze(0).detach().cpu().numpy()

def compute_sam_map(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8):
    if pred.dim() == 3:
        pred = pred.unsqueeze(0); target = target.unsqueeze(0)
    pred_n = F.normalize(pred, dim=1, eps=epsilon)
    targ_n = F.normalize(target, dim=1, eps=epsilon)
    dot = (pred_n * targ_n).sum(dim=1).clamp(-1+epsilon, 1-epsilon)
    sam = torch.acos(dot) * 180.0 / torch.pi
    return sam.squeeze(0).detach().cpu().numpy()

def compute_rmse_map(pred: torch.Tensor, target: torch.Tensor):
    if pred.dim() == 3:
        pred = pred.unsqueeze(0); target = target.unsqueeze(0)
    rmse = torch.sqrt(((pred - target) ** 2).mean(dim=1))
    return rmse.squeeze(0).detach().cpu().numpy()

def create_error_colormap() -> mcolors.LinearSegmentedColormap:
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    return mcolors.LinearSegmentedColormap.from_list('error_cmap', colors, N=256)

def apply_gaussian_smoothing(error_map, sigma: float = 0.8):
    return ndimage.gaussian_filter(error_map, sigma=sigma)


from __future__ import annotations

import torch

EPS = 1e-8


def _gaussian(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Return a Gaussian evaluated at x."""
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)  # (N,) -> (N,)


def get_cached_cmf(n_bands: int = 31, device: torch.device | None = None) -> torch.Tensor:
    """Return a (3, C) color matching function over C wavelength bands."""
    device = device or torch.device("cpu")
    wl = torch.linspace(400.0, 700.0, n_bands, device=device)  # (C,)
    B = _gaussian(wl, 450.0, 25.0)  # (C,)
    G = _gaussian(wl, 550.0, 30.0)  # (C,)
    R = _gaussian(wl, 610.0, 35.0)  # (C,)
    cmf = torch.stack([R, G, B], dim=0)  # (3,C)
    cmf = cmf / (cmf.amax(dim=1, keepdim=True) + EPS)  # (3,C) / (3,1) -> (3,C)
    return cmf


def hsi_to_rgb(hsi: torch.Tensor, cmf: torch.Tensor | None = None, clamp: bool = True) -> torch.Tensor:
    """
    Convert HSI cube to RGB using a 3xC color matching function.

    Expected input shape: (B,C,H,W) or (C,H,W).
    Output shape: (B,3,H,W).
    """
    if hsi.dim() == 3:
        hsi = hsi.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
    if hsi.dim() != 4:
        raise ValueError(f"hsi must be 3D or 4D, got shape {tuple(hsi.shape)}")
    _, C, _, _ = hsi.shape
    device = hsi.device
    if cmf is None:
        cmf = get_cached_cmf(C, device=device)  # (3,C)
    if cmf.device != device or cmf.dtype != hsi.dtype:
        cmf = cmf.to(device=device, dtype=hsi.dtype)
    if cmf.shape[1] != C:
        cmf = torch.nn.functional.interpolate(
            cmf.unsqueeze(0), size=C, mode="linear", align_corners=True
        ).squeeze(0)  # (1,3,C0)->(1,3,C)->(3,C)
    orig_is_float = torch.is_floating_point(hsi)
    hsi_f = hsi.float()
    cmf_f = cmf.float()
    rgb = torch.einsum("bcHW,rc->brHW", hsi_f, cmf_f)  # (B,C,H,W)@(3,C)->(B,3,H,W)
    rgb_min = rgb.amin(dim=(2, 3), keepdim=True)  # (B,3,1,1)
    rgb = rgb - rgb_min  # (B,3,H,W) - (B,3,1,1) -> (B,3,H,W) broadcast
    rgb_max = rgb.amax(dim=(2, 3), keepdim=True)  # (B,3,1,1)
    rgb = rgb / (rgb_max + EPS)  # (B,3,H,W) / (B,3,1,1) -> (B,3,H,W) broadcast
    if clamp:
        rgb = rgb.clamp(0, 1)
    if orig_is_float and rgb.dtype != hsi.dtype:
        rgb = rgb.to(dtype=hsi.dtype)
    return rgb


def crop_center_arad1k(x: torch.Tensor) -> torch.Tensor:
    """Center-crop a (B,C,H,W) or (C,H,W) tensor to 256x256 if possible."""
    if x.dim() == 3:
        x = x.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
    if x.dim() != 4:
        raise ValueError(f"x must be 3D or 4D, got shape {tuple(x.shape)}")
    _, _, H, W = x.shape
    target = 256
    if H < target or W < target:
        return x
    y0 = (H - target) // 2
    x0 = (W - target) // 2
    return x[:, :, y0 : y0 + target, x0 : x0 + target]  # (B,C,256,256)

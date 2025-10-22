
from __future__ import annotations
import torch

def _gaussian(x, mu, sigma):
    return torch.exp(-0.5*((x-mu)/sigma)**2)

def get_cached_cmf(n_bands: int = 31, device: torch.device | None = None) -> torch.Tensor:
    device = device or torch.device('cpu')
    wl = torch.linspace(400.0, 700.0, n_bands, device=device)
    B = _gaussian(wl, 450.0, 25.0)
    G = _gaussian(wl, 550.0, 30.0)
    R = _gaussian(wl, 610.0, 35.0)
    cmf = torch.stack([R, G, B], dim=0)
    cmf = cmf / (cmf.amax(dim=1, keepdim=True) + 1e-8)
    return cmf

def hsi_to_rgb(hsi: torch.Tensor, cmf: torch.Tensor | None = None, clamp: bool = True) -> torch.Tensor:
    if hsi.dim() == 3:
        hsi = hsi.unsqueeze(0)
    B, C, H, W = hsi.shape
    device = hsi.device
    if cmf is None:
        cmf = get_cached_cmf(C, device=device)
    if cmf.shape[1] != C:
        cmf = torch.nn.functional.interpolate(cmf.unsqueeze(0), size=C, mode='linear', align_corners=True).squeeze(0)
    rgb = torch.einsum('bcHW,rc->brHW', hsi, cmf)
    rgb = rgb - rgb.amin(dim=(2,3), keepdim=True)
    rgb = rgb / (rgb.amax(dim=(2,3), keepdim=True) + 1e-8)
    if clamp:
        rgb = rgb.clamp(0,1)
    return rgb

def crop_center_arad1k(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    target = 256
    if H < target or W < target:
        return x
    y0 = (H - target)//2
    x0 = (W - target)//2
    return x[:, :, y0:y0+target, x0:x0+target]

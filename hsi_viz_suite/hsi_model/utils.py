
from __future__ import annotations

import warnings

import torch

EPS = 1e-8


def _gaussian(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Return a Gaussian evaluated at x."""
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)  # (N,) -> (N,)


def get_cached_cmf(n_bands: int = 31, device: torch.device | None = None) -> torch.Tensor:
    """
    Return a (3, C) color matching function over C wavelength bands.

    Args:
        n_bands: Number of spectral bands (default: 31 for ARAD1K)
        device: Target device for the tensor

    Returns:
        CMF tensor of shape (3, n_bands) with R, G, B response functions.
    """
    device = device or torch.device("cpu")
    wl = torch.linspace(400.0, 700.0, n_bands, device=device)  # (C,)
    B = _gaussian(wl, 450.0, 25.0)  # (C,)
    G = _gaussian(wl, 550.0, 30.0)  # (C,)
    R = _gaussian(wl, 610.0, 35.0)  # (C,)
    cmf = torch.stack([R, G, B], dim=0)  # (3,C)

    # Validate CMF has meaningful values before normalizing
    cmf_max = cmf.amax(dim=1, keepdim=True)
    if (cmf_max < EPS).any():
        warnings.warn(
            "CMF contains near-zero channels, normalization may be unstable",
            RuntimeWarning,
        )
    cmf = cmf / (cmf_max + EPS)  # (3,C) / (3,1) -> (3,C)
    return cmf


def hsi_to_rgb(
    hsi: torch.Tensor,
    cmf: torch.Tensor | None = None,
    clamp: bool = True,
    preserve_dtype: bool = False,
) -> torch.Tensor:
    """
    Convert HSI cube to RGB using a 3xC color matching function.

    Args:
        hsi: Hyperspectral image tensor, shape (B,C,H,W) or (C,H,W)
        cmf: Color matching function, shape (3, C). If None, uses default Gaussian CMF.
        clamp: Whether to clamp output to [0, 1] range
        preserve_dtype: If True and input is float64, computation uses float64.
            Default False uses float32 for efficiency.

    Returns:
        RGB tensor of shape (B, 3, H, W), normalized to [0, 1] range.

    Note:
        For scientific computing requiring high precision, set preserve_dtype=True.
        The default float32 computation provides ~7 decimal digits of precision.
    """
    if hsi.dim() == 3:
        hsi = hsi.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
    if hsi.dim() != 4:
        raise ValueError(f"hsi must be 3D or 4D, got shape {tuple(hsi.shape)}")

    # Warn about non-floating point input
    if not torch.is_floating_point(hsi):
        warnings.warn(
            f"Input HSI has non-floating dtype {hsi.dtype}. "
            "Converting to float32. Consider providing float input for accuracy.",
            RuntimeWarning,
        )

    _, C, _, _ = hsi.shape
    device = hsi.device
    orig_dtype = hsi.dtype

    if cmf is None:
        cmf = get_cached_cmf(C, device=device)  # (3,C)
    if cmf.device != device or cmf.dtype != hsi.dtype:
        cmf = cmf.to(device=device, dtype=hsi.dtype)
    if cmf.shape[1] != C:
        cmf = torch.nn.functional.interpolate(
            cmf.unsqueeze(0), size=C, mode="linear", align_corners=True
        ).squeeze(0)  # (1,3,C0)->(1,3,C)->(3,C)

    # Choose computation dtype based on preserve_dtype flag
    if preserve_dtype and hsi.dtype == torch.float64:
        compute_dtype = torch.float64
    else:
        compute_dtype = torch.float32

    hsi_f = hsi.to(dtype=compute_dtype)
    cmf_f = cmf.to(dtype=compute_dtype)

    rgb = torch.einsum("bcHW,rc->brHW", hsi_f, cmf_f)  # (B,C,H,W)@(3,C)->(B,3,H,W)

    # Min-max normalization per channel
    rgb_min = rgb.amin(dim=(2, 3), keepdim=True)  # (B,3,1,1)
    rgb = rgb - rgb_min  # (B,3,H,W) - (B,3,1,1) -> (B,3,H,W) broadcast
    rgb_max = rgb.amax(dim=(2, 3), keepdim=True)  # (B,3,1,1)

    # Use torch.where for safer division (avoid explosion when rgb_max is very small)
    rgb = torch.where(
        rgb_max > EPS,
        rgb / rgb_max,
        torch.zeros_like(rgb),
    )

    if clamp:
        rgb = rgb.clamp(0, 1)

    # Restore original dtype if it was floating point
    if torch.is_floating_point(torch.empty(0, dtype=orig_dtype)):
        if preserve_dtype and rgb.dtype != orig_dtype:
            rgb = rgb.to(dtype=orig_dtype)

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

"""Adaptive wavelet thresholding and noise estimation."""
import math

import torch
import torch.nn as nn

# Numerical stability constant
_EPS = 1e-8


class AdaptiveWaveletThresholding(nn.Module):
    """Adaptive thresholding for wavelet coefficients."""

    def __init__(
        self,
        channels: int,
        method: str = 'soft',
        trainable: bool = True,
        init_threshold: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.method = method

        if trainable:
            self.ll_threshold = nn.Parameter(torch.ones(1) * init_threshold * 0.5)
            self.detail_thresholds = nn.Parameter(torch.ones(3) * init_threshold)
        else:
            self.register_buffer('ll_threshold', torch.ones(1) * init_threshold * 0.5)
            self.register_buffer('detail_thresholds', torch.ones(3) * init_threshold)

    def forward(self, coeffs, noise_level=None):
        """Apply thresholding to wavelet coefficients."""
        B, C, _, H, W = coeffs.shape

        components = [coeffs[:, :, i] for i in range(4)]

        device = coeffs.device
        dtype = coeffs.dtype

        # CRITICAL FIX: Properly handle threshold scaling with noise level
        ll_thresh = self.ll_threshold.to(device=device, dtype=dtype)
        detail_threshs = self.detail_thresholds.to(device=device, dtype=dtype)

        # Initialize scale to None; will be set if noise_level is provided
        scale: torch.Tensor | None = None

        if noise_level is not None:
            # Compute universal threshold scale
            n_elements = max(H * W, 2)
            universal_scale_val = math.sqrt(2.0 * math.log(n_elements))
            universal_scale_t = torch.as_tensor(universal_scale_val, device=device, dtype=dtype)

            # noise_level shape: [B, C, 1, 1]
            # Scale thresholds per batch and channel
            scale = noise_level.to(device=device, dtype=dtype) * universal_scale_t  # [B, C, 1, 1]

            # ll_thresh: [1] -> broadcast to [B, C, 1, 1]
            ll_thresh = ll_thresh.view(1, 1, 1, 1) * scale

            # detail_threshs: [3] -> need to expand properly for each component
            # We'll apply each threshold separately in the loop below

        # Apply thresholding to each detail component
        for i in range(3):
            # Get threshold for this component
            if noise_level is not None and scale is not None:
                # detail_threshs[i]: scalar -> expand to [B, C, 1, 1]
                thresh = detail_threshs[i].view(1, 1, 1, 1) * scale
            else:
                thresh = detail_threshs[i]

            if self.method == 'hard':
                components[i + 1] = self._hard_threshold(components[i + 1], thresh)
            elif self.method == 'soft':
                components[i + 1] = self._soft_threshold(components[i + 1], thresh)
            else:
                components[i + 1] = self._garrote_threshold(components[i + 1], thresh)

        return torch.stack(components, dim=2)

    def _hard_threshold(self, x, threshold):
        return x * (torch.abs(x) > threshold).float()

    def _soft_threshold(self, x, threshold):
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)

    def _garrote_threshold(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """
        Non-negative garrote thresholding with numerical stability.

        Garrote: x - threshold^2 / x for |x| > threshold, else 0

        Args:
            x: Input tensor [B, C, H, W]
            threshold: Threshold value (scalar or broadcastable tensor)

        Returns:
            Thresholded tensor [B, C, H, W]
        """
        abs_x = torch.abs(x)

        # Ensure threshold broadcasts correctly
        if threshold.dim() == 0:
            thresh = threshold
        else:
            thresh = threshold  # Already [B, C, 1, 1] or broadcastable

        # Create mask for values above threshold
        mask = (abs_x > thresh).float()

        # Safe computation: use clamped abs_x to avoid division by small values
        safe_abs_x = torch.clamp(abs_x, min=_EPS)

        # Compute garrote: sign(x) * (|x| - threshold^2 / |x|)
        # For values below threshold, result is 0
        thresh_sq = thresh ** 2
        correction = thresh_sq / safe_abs_x

        # Clamp correction to prevent sign flips (correction should be <= |x|)
        correction = torch.clamp(correction, max=abs_x)

        # Apply garrote formula
        garrote_value = torch.sign(x) * (abs_x - correction)

        # Apply mask: zero out values below threshold
        result = garrote_value * mask

        return result


class WaveletNoiseEstimator(nn.Module):
    """Noise level estimation using wavelet coefficients."""

    def __init__(self, robust_scale=1.4826):
        super().__init__()
        self.robust_scale = robust_scale

    def forward(self, coeffs):
        B, C, _, _, _ = coeffs.shape
        hh_coeffs = coeffs[:, :, 3]

        median = torch.median(hh_coeffs.abs().view(B, C, -1), dim=2)[0]
        mad = self.robust_scale * median / 0.6745

        return mad.view(B, C, 1, 1)

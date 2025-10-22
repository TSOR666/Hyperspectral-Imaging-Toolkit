import math

import torch
import torch.nn as nn


class AdaptiveWaveletThresholding(nn.Module):
    """Adaptive thresholding for wavelet coefficients."""

    def __init__(self, channels, method='soft', trainable=True, init_threshold=0.1):
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

        if noise_level is not None:
            # Compute universal threshold scale
            n_elements = max(H * W, 2)
            universal_scale = math.sqrt(2.0 * math.log(n_elements))
            universal_scale = torch.as_tensor(universal_scale, device=device, dtype=dtype)

            # noise_level shape: [B, C, 1, 1]
            # Scale thresholds per batch and channel
            scale = noise_level.to(device=device, dtype=dtype) * universal_scale  # [B, C, 1, 1]

            # ll_thresh: [1] -> broadcast to [B, C, 1, 1]
            ll_thresh = ll_thresh.view(1, 1, 1, 1) * scale

            # detail_threshs: [3] -> need to expand properly for each component
            # We'll apply each threshold separately in the loop below

        # Apply thresholding to each detail component
        for i in range(3):
            # Get threshold for this component
            if noise_level is not None:
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

    def _garrote_threshold(self, x, threshold):
        mask = torch.abs(x) > threshold
        result = torch.zeros_like(x)

        if not torch.any(mask):
            return result

        x_masked = x[mask]
        eps = torch.finfo(x.dtype).eps
        safe_denominator = torch.where(
            x_masked >= 0,
            torch.clamp(x_masked, min=eps),
            torch.clamp(x_masked, max=-eps)
        )

        result[mask] = x_masked - (threshold ** 2 / safe_denominator)
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

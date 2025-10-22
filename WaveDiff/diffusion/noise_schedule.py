import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transforms.haar_wavelet import HaarWaveletTransform


class BaseNoiseSchedule(nn.Module):
    """Base class for diffusion noise schedules with useful cached statistics."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.ones_like(alphas_cumprod)
        alphas_cumprod_prev[1:] = alphas_cumprod[:-1]

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recipm1_alphas', torch.sqrt(1.0 / alphas - 1.0))

        # CRITICAL FIX: Ensure numerical stability in posterior variance calculation
        # Avoid division by zero when alphas_cumprod approaches 1
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-8)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer('posterior_variance', posterior_variance)
        # Safe log computation with clamped values
        self.register_buffer('posterior_log_variance', torch.log(torch.clamp(posterior_variance, min=1e-20)))

    def extract(self, name: str, t: torch.Tensor, x_shape: Optional[torch.Size] = None) -> torch.Tensor:
        """Extract values from a cached buffer for the provided timesteps."""
        buffer = getattr(self, name)
        values = buffer[t]
        if x_shape is None:
            return values
        return values.view(-1, *[1] * (len(x_shape) - 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.betas[t]
        return beta_t.view(-1, 1, 1, 1)


class SpectralNoiseSchedule(BaseNoiseSchedule):
    """Adaptive noise schedule that analyzes spectral properties of the data."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2,
                 num_freq_bands: int = 8):
        super().__init__(timesteps, beta_start, beta_end)
        self.num_freq_bands = num_freq_bands

        self.spectral_weights = nn.Parameter(torch.ones(num_freq_bands))

    def compute_band_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the energy contained in logarithmically spaced frequency bands."""
        _, _, h, w = x.shape

        fft_x = torch.fft.rfft2(x, norm="ortho")
        power_spectrum = torch.abs(fft_x) ** 2

        freq_h = torch.fft.fftfreq(h, device=x.device)[:, None]
        freq_w = torch.fft.rfftfreq(w, device=x.device)[None, :]
        freq_distance = torch.sqrt(freq_h ** 2 + freq_w ** 2)

        max_freq = torch.clamp(torch.max(freq_distance), min=1e-6)
        band_boundaries = torch.logspace(
            math.log10(1e-5), math.log10(float(max_freq)), self.num_freq_bands + 1, device=x.device
        )

        band_energies = []
        for i in range(self.num_freq_bands):
            low, high = band_boundaries[i], band_boundaries[i + 1]
            mask = ((freq_distance >= low) & (freq_distance < high))[None, None, :, :]
            masked_power = power_spectrum * mask
            denom = mask.sum().clamp(min=1.0)
            band_energy = masked_power.sum(dim=(2, 3)) / denom
            band_energies.append(band_energy)

        return torch.stack(band_energies, dim=2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        base_beta = self.betas[t]

        band_energies = self.compute_band_energy(x)
        energy_sum = band_energies.sum(dim=2, keepdim=True).clamp(min=1e-8)
        normalized_energies = band_energies / energy_sum

        weights = F.softmax(self.spectral_weights, dim=0)
        weighted_beta = base_beta.view(-1, 1, 1) * (normalized_energies @ weights)

        beta_adapted = weighted_beta.mean(dim=1).view(-1, 1, 1, 1)
        beta_adapted = torch.clamp(beta_adapted, min=1e-5, max=5e-2)
        return beta_adapted


class WaveletSpectralNoiseSchedule(BaseNoiseSchedule):
    """Wavelet-enhanced spectral noise schedule using Haar decomposition."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2,
                 num_freq_bands: int = 8, latent_dim: int = 64):
        super().__init__(timesteps, beta_start, beta_end)
        self.num_freq_bands = num_freq_bands
        self.latent_dim = latent_dim

        self.wavelet = HaarWaveletTransform(latent_dim)

        self.ll_weight = nn.Parameter(torch.ones(1))
        self.detail_weights = nn.Parameter(torch.ones(3))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        base_beta = self.betas[t]

        wavelet_coeffs = self.wavelet(x)

        energies = []
        for i in range(4):
            component_energy = torch.mean(torch.abs(wavelet_coeffs[:, :, i]), dim=[1, 2, 3])
            energies.append(component_energy)

        energies = torch.stack(energies, dim=1)
        energies = energies / torch.sum(energies, dim=1, keepdim=True).clamp(min=1e-8)

        weights = torch.cat([self.ll_weight.view(-1), self.detail_weights.view(-1)])
        weights = F.softmax(weights, dim=0)

        adaptive_beta = base_beta.unsqueeze(1) * (energies * weights.unsqueeze(0)).sum(dim=1)
        beta_adapted = adaptive_beta.view(-1, 1, 1, 1)
        beta_adapted = torch.clamp(beta_adapted, min=1e-5, max=5e-2)
        return beta_adapted


class AdaptiveNoiseSchedule(BaseNoiseSchedule):
    """Content-aware adaptive noise schedule predicted by a lightweight CNN."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2,
                 channels: int = 64):
        super().__init__(timesteps, beta_start, beta_end)
        self.channels = channels

        self.adaptation_net = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        base_beta = self.betas[t]

        adaptation = self.adaptation_net(x)
        adaptation = 0.5 + adaptation / 2.0

        adapted_beta = base_beta.unsqueeze(1) * adaptation[:, 0]
        beta_adapted = adapted_beta.view(-1, 1, 1, 1)
        beta_adapted = torch.clamp(beta_adapted, min=1e-5, max=5e-2)
        return beta_adapted

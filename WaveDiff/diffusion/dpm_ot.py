"""DPM-OT: Diffusion Probabilistic Models with Optimal Transport sampling."""
from __future__ import annotations

from typing import Optional, List, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from diffusion.noise_schedule import BaseNoiseSchedule

if TYPE_CHECKING:
    pass

# Numerical stability constant
_EPS = 1e-8


class DPMOT(nn.Module):
    """
    DPM-OT (Diffusion Probabilistic Models with Optimal Transport)

    A diffusion model implementation that incorporates optimal transport
    for improved sampling efficiency.
    """

    def __init__(
        self,
        denoiser,
        spectral_schedule=None,
        timesteps=1000,
        conditional=False,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        self.conditional = bool(conditional)

        # Base schedule is always available as a fallback
        self.base_schedule = BaseNoiseSchedule(timesteps=timesteps)

        # Use provided spectral schedule or fall back to base schedule
        self.spectral_schedule = spectral_schedule or self.base_schedule

        # Precompute timesteps
        self.register_buffer('timestep_indices', torch.arange(0, timesteps, dtype=torch.long))

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process with spectral-aware noise scheduling."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar_t = self.spectral_schedule.extract('sqrt_alphas_cumprod', t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = self.spectral_schedule.extract(
            'sqrt_one_minus_alphas_cumprod', t, x_0.shape
        )
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def p_losses(self, x_0, t=None, conditioning=None):
        """
        Training losses for DPM-OT

        Args:
            x_0: Clean input data [B, C, H, W]
            t: Optional timesteps (randomly sampled if None)

        Returns:
            loss, predicted noise, true noise
        """
        b, _, _, _ = x_0.shape

        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=x_0.device, dtype=torch.long)

        # Add noise with spectral-aware schedule
        x_t, noise = self.q_sample(x_0, t)

        # Predict noise
        noise_pred = self._predict_noise(x_t, t, conditioning)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        return loss, noise_pred, noise

    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device | str,
        conditioning: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        use_dpm_solver: bool = False,
        steps: Optional[int] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Sample from the model.

        ``use_dpm_solver=True`` selects a deterministic, reduced-step DDIM
        sampler for the epsilon-prediction parameterization. The legacy method
        name is retained for checkpoint and caller compatibility.

        Args:
            shape: Shape of sample to generate [B, C, H, W]
            device: Device to generate on
            return_intermediates: Whether to return intermediate steps
            use_dpm_solver: Whether to use the (currently experimental) DPM
                Solver path. Defaults to False.
            steps: Number of steps for DPM Solver (default: 20)

        Returns:
            Generated sample
        """
        if use_dpm_solver:
            return self.sample_dpm_solver(shape, device, conditioning=conditioning, steps=steps or 20)
        return self.sample_ddpm(shape, device, conditioning=conditioning, return_intermediates=return_intermediates)

    @torch.no_grad()
    def sample_ddpm(
        self,
        shape: Tuple[int, ...],
        device: torch.device | str,
        conditioning: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Standard DDPM sampling.

        Wrapped in ``@torch.no_grad()``: without it, the autograd graph grew
        linearly across all ``self.timesteps`` denoiser calls, causing OOM on
        long schedules even when the caller only wanted a sample.
        """
        b = shape[0]

        # Start from pure noise
        x = self._prepare_starting_point(shape, device, conditioning)

        intermediates: List[torch.Tensor] = [x] if return_intermediates else []

        # Iteratively denoise
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling time step', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)

            # Forward noising uses the cached schedule statistics. Reverse
            # sampling must use the same beta sequence.
            beta_t = self.spectral_schedule.extract('betas', t, x.shape)

            sqrt_one_minus_alpha_bar_t = self.spectral_schedule.extract(
                'sqrt_one_minus_alphas_cumprod', t, x.shape
            )
            sqrt_recip_alpha_t = self.spectral_schedule.extract('sqrt_recip_alphas', t, x.shape)
            sigma_t = torch.sqrt(self.spectral_schedule.extract('posterior_variance', t, x.shape))

            # Predict noise
            noise_pred = self._predict_noise(x, t, conditioning)

            # Update sample
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            model_mean = sqrt_recip_alpha_t * (
                x - (beta_t / torch.clamp(sqrt_one_minus_alpha_bar_t, min=1e-12)) * noise_pred
            )
            x = model_mean + sigma_t * noise

            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x

    @torch.no_grad()
    def sample_dpm_solver(
        self,
        shape: Tuple[int, ...],
        device: torch.device | str,
        conditioning: Optional[torch.Tensor] = None,
        steps: int = 20,
    ) -> torch.Tensor:
        """
        Deterministic reduced-step DDIM sampling.

        Args:
            shape: Output shape [B, C, H, W]
            device: Device
            steps: Number of solver steps

        Returns:
            Generated sample tensor
        """
        if steps < 1 or steps > self.timesteps:
            raise ValueError(
                f"steps must be in [1, {self.timesteps}], got {steps}"
            )

        x = self._prepare_starting_point(shape, device, conditioning)
        batch_size = shape[0]
        timestep_values = torch.linspace(
            self.timesteps - 1,
            0,
            steps,
            device=device,
        ).round().long()
        terminal = timestep_values.new_full((1,), -1)
        next_values = torch.cat([timestep_values[1:], terminal])

        for t_value, next_value in tqdm(
            zip(timestep_values, next_values),
            total=steps,
            desc="DDIM sampling",
        ):
            t = t_value.expand(batch_size)
            next_t = next_value.expand(batch_size)
            pred_noise = self._predict_noise(x, t, conditioning)
            x = self._ddim_update(x, pred_noise, t, next_t)

        return x

    def _ddim_update(
        self,
        x: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        next_t: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one deterministic epsilon-prediction DDIM update."""
        alpha_t = self._gather_alphas(t).view(-1, 1, 1, 1)
        next_indices = next_t.clamp(min=0)
        alpha_next = self._gather_alphas(next_indices).view(-1, 1, 1, 1)
        alpha_next = torch.where(
            (next_t < 0).view(-1, 1, 1, 1),
            torch.ones_like(alpha_next),
            alpha_next,
        )

        sqrt_alpha_t = torch.sqrt(torch.clamp(alpha_t, min=_EPS))
        sigma_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=_EPS))
        pred_x0 = (x - sigma_t * noise_pred) / sqrt_alpha_t

        return (
            torch.sqrt(torch.clamp(alpha_next, min=_EPS)) * pred_x0
            + torch.sqrt(torch.clamp(1.0 - alpha_next, min=0.0)) * noise_pred
        )

    def _prepare_starting_point(
        self,
        shape: Tuple[int, ...],
        device: torch.device | str,
        conditioning: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return the sampler's initial noisy state."""
        if conditioning is None or self.conditional:
            return torch.randn(shape, device=device)

        if list(conditioning.shape) != list(shape):
            raise ValueError(f"Conditioning shape {tuple(conditioning.shape)} does not match requested {shape}.")

        conditioning = conditioning.to(device)
        timesteps = torch.full(
            (conditioning.shape[0],), self.timesteps - 1, device=device, dtype=torch.long
        )
        noisy_latent, _ = self.q_sample(conditioning, timesteps)
        return noisy_latent

    def _predict_noise(self, x, t, conditioning=None):
        """Call conditional denoisers without changing legacy checkpoint behavior."""
        if self.conditional:
            if conditioning is None:
                raise ValueError("Conditional diffusion requires a conditioning latent")
            return self.denoiser(x, t, conditioning=conditioning)
        return self.denoiser(x, t)

    def _dpm_solver_update(
        self, x: torch.Tensor, noise_pred: torch.Tensor, t: torch.Tensor, next_t: torch.Tensor
    ) -> torch.Tensor:
        """
        First-order DPM-Solver update (DDIM / deterministic DDPM step).

        Correct formula for ε-prediction parameterisation:

            x_next = √(ᾱ_next/ᾱ_t) · x
                   + [√(1−ᾱ_next) − √(ᾱ_next/ᾱ_t) · √(1−ᾱ_t)] · ε_θ

        The previous implementation used (√ᾱ_next − √ᾱ_t) as the noise
        coefficient, which omits the σ_bar terms and underestimates the
        correction by ~30-50× at high-noise timesteps.

        Args:
            x: Current sample [B, C, H, W]
            noise_pred: Predicted noise [B, C, H, W]
            t: Current (noisier) timestep [B]
            next_t: Next (cleaner) timestep [B]

        Returns:
            Updated sample [B, C, H, W]
        """
        idx_t = self._time_to_index(t)
        idx_next = self._time_to_index(next_t)

        a_bar_t = self._gather_alphas(idx_t).view(-1, 1, 1, 1)
        a_bar_next = self._gather_alphas(idx_next).view(-1, 1, 1, 1)

        sqrt_a_t = torch.sqrt(torch.clamp(a_bar_t, min=_EPS))
        sqrt_a_next = torch.sqrt(torch.clamp(a_bar_next, min=_EPS))
        sigma_t = torch.sqrt(torch.clamp(1.0 - a_bar_t, min=_EPS))
        sigma_next = torch.sqrt(torch.clamp(1.0 - a_bar_next, min=_EPS))

        # √(ᾱ_next / ᾱ_t)
        sqrt_ratio = sqrt_a_next / sqrt_a_t
        # σ_next − √(ᾱ_next/ᾱ_t) · σ_t  (negative: denoising direction)
        coeff_eps = sigma_next - sqrt_ratio * sigma_t

        return sqrt_ratio * x + coeff_eps * noise_pred

    def _dpm_solver_second_order_update(
        self,
        x: torch.Tensor,
        x_mid: torch.Tensor,
        noise_t: torch.Tensor,
        noise_mid: torch.Tensor,
        t: torch.Tensor,
        mid_t: torch.Tensor,
        next_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Second-order DPM-Solver update (midpoint method).

        Applies the same DDIM formula as _dpm_solver_update, but uses the
        noise prediction at the midpoint (noise_mid) as a second-order
        corrector.  This is equivalent to the DPM-Solver-2 (midpoint variant):

            x_next = √(ᾱ_next/ᾱ_t) · x
                   + [√(1−ᾱ_next) − √(ᾱ_next/ᾱ_t) · √(1−ᾱ_t)] · ε_mid

        The previous implementation used (√ᾱ_next − √ᾱ_t) as the noise
        coefficient (same σ-omission bug as the first-order method).

        Args:
            x: Current sample [B, C, H, W]
            x_mid: Intermediate first-order estimate (retained for API compat)
            noise_t: Noise at t (retained for API compat, not used here)
            noise_mid: Noise prediction at midpoint [B, C, H, W]
            t, mid_t, next_t: Timestep tensors

        Returns:
            Updated sample [B, C, H, W]
        """
        idx_t = self._time_to_index(t)
        idx_next = self._time_to_index(next_t)

        a_bar_t = self._gather_alphas(idx_t).view(-1, 1, 1, 1)
        a_bar_next = self._gather_alphas(idx_next).view(-1, 1, 1, 1)

        sqrt_a_t = torch.sqrt(torch.clamp(a_bar_t, min=_EPS))
        sqrt_a_next = torch.sqrt(torch.clamp(a_bar_next, min=_EPS))
        sigma_t = torch.sqrt(torch.clamp(1.0 - a_bar_t, min=_EPS))
        sigma_next = torch.sqrt(torch.clamp(1.0 - a_bar_next, min=_EPS))

        sqrt_ratio = sqrt_a_next / sqrt_a_t
        coeff_eps = sigma_next - sqrt_ratio * sigma_t

        return sqrt_ratio * x + coeff_eps * noise_mid

    def _dpm_solver_third_order_update(
        self,
        x: torch.Tensor,
        x_mid: torch.Tensor,
        x_end: torch.Tensor,
        noise_t: torch.Tensor,
        noise_mid: torch.Tensor,
        noise_end: torch.Tensor,
        t: torch.Tensor,
        mid_t: torch.Tensor,
        next_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Third-order DPM-Solver update (Simpson's-rule noise integration).

        Combines three noise predictions using composite Simpson's 1/3 weights
        (1/6, 4/6, 1/6) over the interval [t, next_t].  The base DDIM formula
        is unchanged; only the effective noise estimate improves:

            ε_eff = (ε_t + 4·ε_mid + ε_next) / 6

            x_next = √(ᾱ_next/ᾱ_t) · x
                   + [√(1−ᾱ_next) − √(ᾱ_next/ᾱ_t) · √(1−ᾱ_t)] · ε_eff

        The previous implementation used (√ᾱ_next − √ᾱ_t) style coefficients
        which were both in the wrong variable (√ᾱ vs √(1−ᾱ)) and structured
        as a Taylor expansion that diverged from the DDIM formula.

        Args:
            x: Current sample [B, C, H, W]
            x_mid, x_end: Intermediate estimates (retained for API compat)
            noise_t: Noise at t [B, C, H, W]
            noise_mid: Noise at midpoint [B, C, H, W]
            noise_end: Noise at next_t [B, C, H, W]
            t, mid_t, next_t: Timestep tensors

        Returns:
            Updated sample [B, C, H, W]
        """
        idx_t = self._time_to_index(t)
        idx_next = self._time_to_index(next_t)

        a_bar_t = self._gather_alphas(idx_t).view(-1, 1, 1, 1)
        a_bar_next = self._gather_alphas(idx_next).view(-1, 1, 1, 1)

        sqrt_a_t = torch.sqrt(torch.clamp(a_bar_t, min=_EPS))
        sqrt_a_next = torch.sqrt(torch.clamp(a_bar_next, min=_EPS))
        sigma_t = torch.sqrt(torch.clamp(1.0 - a_bar_t, min=_EPS))
        sigma_next = torch.sqrt(torch.clamp(1.0 - a_bar_next, min=_EPS))

        sqrt_ratio = sqrt_a_next / sqrt_a_t
        coeff_eps = sigma_next - sqrt_ratio * sigma_t

        # Simpson's 1/3 composite weights over [ε_t, ε_mid, ε_next]
        noise_eff = (noise_t + 4.0 * noise_mid + noise_end) / 6.0

        return sqrt_ratio * x + coeff_eps * noise_eff

    def _integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute integral of beta schedule from 0 to t (negative log-alpha)."""
        idx = self._time_to_index(t)
        alphas_cumprod = self._gather_alphas(idx)
        return -torch.log(torch.clamp(alphas_cumprod, min=_EPS))

    def _gather_alphas(self, idx):
        """Safely gather alpha values from schedule using batched indices."""
        alphas_cumprod = self.spectral_schedule.alphas_cumprod
        # Use gather for batched indexing
        if idx.dim() == 0:
            return alphas_cumprod[idx]
        return torch.gather(alphas_cumprod, 0, idx.clamp(0, len(alphas_cumprod) - 1))

    def _time_to_index(self, t):
        """Convert a (possibly normalized) time value to discrete index."""
        if torch.is_floating_point(t):
            scaled = torch.clamp(t, min=0.0, max=1.0) * (self.timesteps - 1)
            idx = scaled.round().long()
        else:
            idx = torch.clamp(t.long(), min=0, max=self.timesteps - 1)
        return idx

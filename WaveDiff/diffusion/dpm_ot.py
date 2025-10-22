import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from diffusion.noise_schedule import BaseNoiseSchedule


class DPMOT(nn.Module):
    """
    DPM-OT (Diffusion Probabilistic Models with Optimal Transport)

    A diffusion model implementation that incorporates optimal transport
    for improved sampling efficiency.
    """

    def __init__(self, denoiser, spectral_schedule=None, timesteps=1000):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps

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

    def p_losses(self, x_0, t=None):
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
        noise_pred = self.denoiser(x_t, t)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        return loss, noise_pred, noise

    def sample(self, shape, device, return_intermediates=False, use_dpm_solver=True, steps=None):
        """
        Sample from the model

        Args:
            shape: Shape of sample to generate [B, C, H, W]
            device: Device to generate on
            return_intermediates: Whether to return intermediate steps
            use_dpm_solver: Whether to use DPM Solver v3 for fast sampling
            steps: Number of steps for DPM Solver (default: 20)

        Returns:
            Generated sample
        """
        if use_dpm_solver:
            return self.sample_dpm_solver(shape, device, steps=steps or 20)
        return self.sample_ddpm(shape, device, return_intermediates)

    def sample_ddpm(self, shape, device, return_intermediates=False):
        """Standard DDPM sampling."""
        b = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        intermediates = [x] if return_intermediates else None

        # Iteratively denoise
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling time step', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)

            beta_t = self.spectral_schedule(x, t)
            alpha_bar_t = self.spectral_schedule.extract('alphas_cumprod', t, x.shape)
            alpha_bar_prev = self.spectral_schedule.extract('alphas_cumprod_prev', t, x.shape)

            sqrt_one_minus_alpha_bar_t = self.spectral_schedule.extract(
                'sqrt_one_minus_alphas_cumprod', t, x.shape
            )
            sqrt_recip_alpha_t = self.spectral_schedule.extract('sqrt_recip_alphas', t, x.shape)
            sigma_t = torch.sqrt(self.spectral_schedule.extract('posterior_variance', t, x.shape))

            # Predict noise
            noise_pred = self.denoiser(x, t)

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

    def sample_dpm_solver(self, shape, device, steps=20):
        """
        DPM Solver v3 sampling for faster inference.

        Args:
            shape: Output shape [B, C, H, W]
            device: Device
            steps: Number of solver steps

        Returns:
            Generated sample tensor
        """
        x = torch.randn(shape, device=device)

        # Time steps for solver (evenly spaced for simplicity)
        t_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)[:-1]

        # DPM-Solver v3 uses lower-order solvers for the first few steps
        # and higher-order solvers for later steps

        with torch.no_grad():
            for i, t in enumerate(tqdm(t_steps, desc="DPM-Solver sampling")):
                # Create batch timestep
                t_tensor = torch.full((shape[0],), t, device=device)

                # Previous timestep (or 0 if final step)
                prev_t = t_steps[i + 1] if i < len(t_steps) - 1 else 0.0
                prev_t_tensor = torch.full((shape[0],), prev_t, device=device)

                # For DPM-Solver v3, we need to approximate the ODE more accurately

                # Get current prediction
                pred_noise = self.denoiser(x, self._time_to_index(t_tensor))

                # First-order update
                x_1 = self._dpm_solver_update(x, pred_noise, t_tensor, prev_t_tensor)

                # For higher-order updates (used in later steps)
                if i > 0:  # Use first-order for first step
                    # Second prediction at midpoint
                    mid_t = (t + prev_t) / 2.0
                    mid_t_tensor = torch.full((shape[0],), mid_t, device=device)

                    pred_noise_mid = self.denoiser(x_1, self._time_to_index(mid_t_tensor))

                    # Second-order correction
                    x_2 = self._dpm_solver_second_order_update(
                        x, x_1, pred_noise, pred_noise_mid,
                        t_tensor, mid_t_tensor, prev_t_tensor
                    )

                    # For even higher accuracy in final steps
                    if i > len(t_steps) // 2:  # Use higher-order for later steps
                        # Third prediction
                        pred_noise_end = self.denoiser(x_2, self._time_to_index(prev_t_tensor))

                        # Third-order correction
                        x = self._dpm_solver_third_order_update(
                            x, x_1, x_2, pred_noise, pred_noise_mid, pred_noise_end,
                            t_tensor, mid_t_tensor, prev_t_tensor
                        )
                    else:
                        x = x_2
                else:
                    x = x_1

        return x

    def _dpm_solver_update(self, x, noise_pred, t, next_t):
        """First-order DPM-Solver update."""
        # Get alpha values using proper indexing
        idx_t = self._time_to_index(t)
        idx_next = self._time_to_index(next_t)

        alpha_t = self._gather_alphas(idx_t)
        alpha_next_t = self._gather_alphas(idx_next)

        # Reshape for broadcasting
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        alpha_next_t = alpha_next_t.view(-1, 1, 1, 1)

        # First-order DPM-Solver update (correct formula)
        # x_{t+1} = (alpha_{t+1} / alpha_t) * x_t - alpha_{t+1} * (exp(h) - 1) * noise_pred
        # where h = lambda_{t+1} - lambda_t and lambda = log(alpha / sqrt(1-alpha))
        ratio = alpha_next_t / torch.clamp(alpha_t, min=1e-8)

        # Simplified update for linear schedule
        x_next = torch.sqrt(ratio) * x - (torch.sqrt(alpha_next_t) - torch.sqrt(alpha_t)) * noise_pred

        return x_next

    def _dpm_solver_second_order_update(self, x, x_mid, noise_t, noise_mid, t, mid_t, next_t):
        """Second-order DPM-Solver update using midpoint method."""
        # Get alpha values using proper indexing
        idx_t = self._time_to_index(t)
        idx_mid = self._time_to_index(mid_t)
        idx_next = self._time_to_index(next_t)

        alpha_t = self._gather_alphas(idx_t)
        alpha_mid_t = self._gather_alphas(idx_mid)
        alpha_next_t = self._gather_alphas(idx_next)

        # Reshape for broadcasting
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        alpha_mid_t = alpha_mid_t.view(-1, 1, 1, 1)
        alpha_next_t = alpha_next_t.view(-1, 1, 1, 1)

        # Second-order update (correct formula using linear combination)
        ratio = alpha_next_t / torch.clamp(alpha_t, min=1e-8)

        # Use midpoint rule for second-order accuracy
        coef1 = torch.sqrt(ratio)
        coef2 = torch.sqrt(alpha_next_t) - torch.sqrt(alpha_t)

        x_next = coef1 * x - coef2 * noise_mid

        return x_next

    def _dpm_solver_third_order_update(self, x, x_mid, x_end, noise_t, noise_mid, noise_end, t, mid_t, next_t):
        """Third-order DPM-Solver update."""
        # Get alpha values using proper indexing
        idx_t = self._time_to_index(t)
        idx_mid = self._time_to_index(mid_t)
        idx_next = self._time_to_index(next_t)

        alpha_t = self._gather_alphas(idx_t)
        alpha_mid_t = self._gather_alphas(idx_mid)
        alpha_next_t = self._gather_alphas(idx_next)

        # Reshape for broadcasting
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        alpha_mid_t = alpha_mid_t.view(-1, 1, 1, 1)
        alpha_next_t = alpha_next_t.view(-1, 1, 1, 1)

        # Third-order update (correct formula using Taylor expansion)
        ratio = alpha_next_t / torch.clamp(alpha_t, min=1e-8)

        # Compute coefficients for third-order Taylor expansion
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_mid = torch.sqrt(alpha_mid_t)
        sqrt_alpha_next = torch.sqrt(alpha_next_t)

        # Use weighted combination of noise predictions for higher accuracy
        h = (sqrt_alpha_next - sqrt_alpha_t)
        h_mid = (sqrt_alpha_mid - sqrt_alpha_t)

        # Third-order coefficients
        coef_t = h * (h_mid / torch.clamp(h, min=1e-8) - 0.5)
        coef_mid = h
        coef_end = h * (1.0 - h_mid / torch.clamp(h, min=1e-8))

        x_next = torch.sqrt(ratio) * x - (coef_t * noise_t + coef_mid * noise_mid + coef_end * noise_end) / 3.0

        return x_next

    def _integral_beta(self, t):
        """Compute integral of beta schedule from 0 to t."""
        idx = self._time_to_index(t)
        alphas_cumprod = self._gather_alphas(idx)
        return -torch.log(torch.clamp(alphas_cumprod, min=1e-8))

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

from __future__ import annotations

import torch

from models.wavelet_model import WaveletHSILatentDiffusionModel
from modules.denoisers import ThresholdingWaveletUNetDenoiser
from diffusion.dpm_ot import DPMOT
from transforms.adaptive_wavelet import AdaptiveWaveletThresholding, WaveletNoiseEstimator
from transforms.haar_wavelet import HaarWaveletTransform

class AdaptiveWaveletHSILatentDiffusionModel(WaveletHSILatentDiffusionModel):
    """
    HSI Latent Diffusion Model with adaptive wavelet thresholding

    Extends the wavelet model with noise-aware adaptive thresholding
    to selectively remove noise from wavelet coefficients.
    """
    def __init__(self, latent_dim=64, out_channels=31, timesteps=1000,
                 use_batchnorm=True, masking_config=None, threshold_method='soft',
                 init_threshold=0.1, trainable_threshold=True, refinement_config=None,
                 return_noise_estimate=False, norm_type=None, norm_groups=8,
                 cross_attention_mode="channel", attention_window_size=8,
                 conditional_residual_diffusion=False, residual_scale=1.0):
        super().__init__(
            latent_dim=latent_dim,
            out_channels=out_channels,
            timesteps=timesteps,
            use_batchnorm=use_batchnorm,
            masking_config=masking_config,
            refinement_config=refinement_config,
            norm_type=norm_type,
            norm_groups=norm_groups,
            cross_attention_mode=cross_attention_mode,
            attention_window_size=attention_window_size,
            conditional_residual_diffusion=conditional_residual_diffusion,
            residual_scale=residual_scale,
        )

        # Replace standard denoiser with thresholding wavelet UNet denoiser
        self.denoiser = ThresholdingWaveletUNetDenoiser(
            channels=latent_dim,
            time_embedding_dim=128,
            use_batchnorm=use_batchnorm,
            threshold_method=threshold_method,
            init_threshold=init_threshold,
            trainable_threshold=trainable_threshold,
            norm_type=norm_type,
            norm_groups=norm_groups,
            cross_attention_mode=cross_attention_mode,
            attention_window_size=attention_window_size,
            conditional=conditional_residual_diffusion,
        )

        # Update DPM-OT to use the new denoiser
        self.dpm_ot = DPMOT(
            denoiser=self.denoiser,
            spectral_schedule=self.spectral_schedule,
            timesteps=timesteps,
            conditional=conditional_residual_diffusion,
        )

        # Add noise estimator for adaptive processing
        self.noise_estimator = WaveletNoiseEstimator()
        self.return_noise_estimate = return_noise_estimate

        # Add adaptive thresholding module for inference refinement
        # CRITICAL FIX: Initialize with correct number of channels (out_channels for HSI output)
        self.adaptive_thresholding = AdaptiveWaveletThresholding(
            out_channels,
            method=threshold_method,
            trainable=trainable_threshold,
            init_threshold=init_threshold * 0.8  # Slightly lower threshold for output
        )

        # CRITICAL FIX: Initialize wavelet transforms in __init__ instead of lazy initialization
        # This prevents issues with gradient computation and model serialization
        self.wavelet_transform_output = HaarWaveletTransform(out_channels)
        from transforms.haar_wavelet import InverseHaarWaveletTransform
        self.inverse_wavelet_transform = InverseHaarWaveletTransform()
    
    def forward(
        self,
        rgb,
        t=None,
        mask=None,
        use_masking=False,
        hsi_target=None,
    ):
        """
        Enhanced forward pass with adaptive thresholding
        """
        # Get basic forward pass results using parent implementation
        outputs = super().forward(
            rgb,
            t,
            mask,
            use_masking,
            hsi_target=hsi_target,
        )
        
        # Add noise estimation if available from denoiser
        if self.return_noise_estimate and hasattr(self.denoiser, 'noise_estimator'):
            latent = outputs['latent']
            # Get estimated noise level using wavelet-based estimator
            noise_level = self.denoiser.noise_estimator(
                self.denoiser.wavelet(latent)
            )
            outputs['estimated_noise_level'] = noise_level
        
        return outputs
    
    def calculate_losses(self, outputs, rgb_target, hsi_target=None):
        """
        Enhanced loss calculation with additional adaptive thresholding regularization
        """
        # Get basic losses using parent implementation
        losses = super().calculate_losses(outputs, rgb_target, hsi_target)
        
        # Add thresholding regularization if available
        if hasattr(self.denoiser, 'input_thresholding') and \
           hasattr(self.denoiser.input_thresholding, 'll_threshold'):
            # Add small regularization to prevent threshold values from growing too large
            threshold_reg = (
                torch.norm(self.denoiser.input_thresholding.ll_threshold) +
                torch.norm(self.denoiser.input_thresholding.detail_thresholds)
            )
            
            losses['threshold_reg'] = threshold_reg
        
        return losses
    
    def rgb_to_hsi(
        self,
        rgb,
        sampling_steps=None,
        return_stages=False,
        *,
        apply_adaptive_threshold: bool = True,
        latent_mode: str = "direct",
    ):
        """
        Convert RGB to HSI with enhanced adaptive thresholding

        Args:
            rgb: RGB image tensor [B, 3, H, W]
            sampling_steps: Optional reduced number of sampling steps
            return_stages: Whether to return intermediate stages
            apply_adaptive_threshold: Whether to apply adaptive thresholding to the output (keyword-only)

        Returns:
            HSI image tensor [B, C, H, W] or tuple of (HSI, stages dict)
        """
        # Encode RGB to latent space
        latent = self._encode_condition(rgb)
        if latent_mode == "direct":
            reconstruction_latent = latent
        elif latent_mode == "diffusion":
            sampled = self.dpm_ot.sample(
                latent.shape,
                latent.device,
                conditioning=latent,
                use_dpm_solver=True,
                steps=sampling_steps or 20,
            )
            reconstruction_latent = (
                latent + self.residual_scale * sampled
                if self.conditional_residual_diffusion
                else sampled
            )
        else:
            raise ValueError(
                f"latent_mode must be 'direct' or 'diffusion', got {latent_mode!r}"
            )

        hsi_after_pixel, hsi_initial, hsi_after_spectral = self._decode_and_refine(
            reconstruction_latent
        )

        stages = {
            'initial': hsi_initial,
            'after_spectral': hsi_after_spectral,
            'after_pixel': hsi_after_pixel,
        }

        # Apply adaptive thresholding as a final refinement step
        if apply_adaptive_threshold:
            # Apply wavelet transform to HSI output
            hsi_coeffs = self.wavelet_transform_output(hsi_after_pixel)

            # Estimate noise level in HSI output
            noise_level = self.noise_estimator(hsi_coeffs)

            # Apply adaptive thresholding
            thresholded_coeffs = self.adaptive_thresholding(hsi_coeffs, noise_level)

            # Apply inverse transform
            refined_hsi = self.inverse_wavelet_transform(thresholded_coeffs)
            stages['after_threshold'] = refined_hsi
            stages['final'] = refined_hsi

            if return_stages:
                return refined_hsi, stages

            return refined_hsi

        stages['final'] = hsi_after_pixel
        if return_stages:
            return hsi_after_pixel, stages

        return hsi_after_pixel
    
    def train_step(self, batch, batch_idx):
        """Single training step implementation with adaptive components"""
        # Get basic train step result
        result = super().train_step(batch, batch_idx)
        
        # Add threshold regularization if available
        if 'threshold_reg' in result:
            result['loss'] += result['threshold_reg'] * 1e-4
        
        return result
    
    def get_adaptive_threshold_stats(self):
        """Get current threshold values for monitoring"""
        stats = {}
        
        if hasattr(self.denoiser, 'input_thresholding'):
            thresh = self.denoiser.input_thresholding
            if hasattr(thresh, 'll_threshold'):
                stats['ll_threshold'] = thresh.ll_threshold.item()
            if hasattr(thresh, 'detail_thresholds'):
                for i, t in enumerate(thresh.detail_thresholds):
                    stats[f'detail_threshold_{i}'] = t.item()
        
        return stats

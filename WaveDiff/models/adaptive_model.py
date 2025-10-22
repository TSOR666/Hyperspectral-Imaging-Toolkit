import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
                 init_threshold=0.1, trainable_threshold=True, refinement_config=None):
        super().__init__(
            latent_dim=latent_dim,
            out_channels=out_channels,
            timesteps=timesteps,
            use_batchnorm=use_batchnorm,
            masking_config=masking_config,
            refinement_config=refinement_config
        )

        # Replace standard denoiser with thresholding wavelet UNet denoiser
        self.denoiser = ThresholdingWaveletUNetDenoiser(
            channels=latent_dim,
            time_embedding_dim=128,
            use_batchnorm=use_batchnorm,
            threshold_method=threshold_method,
            init_threshold=init_threshold,
            trainable_threshold=trainable_threshold
        )

        # Update DPM-OT to use the new denoiser
        self.dpm_ot = DPMOT(
            denoiser=self.denoiser,
            spectral_schedule=self.spectral_schedule,
            timesteps=timesteps
        )

        # Add noise estimator for adaptive processing
        self.noise_estimator = WaveletNoiseEstimator()

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
    
    def forward(self, rgb, t=None, mask=None, use_masking=False):
        """
        Enhanced forward pass with adaptive thresholding
        """
        # Get basic forward pass results using parent implementation
        outputs = super().forward(rgb, t, mask, use_masking)
        
        # Add noise estimation if available from denoiser
        if hasattr(self.denoiser, 'noise_estimator'):
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
            ) * 1e-4
            
            losses['threshold_reg'] = threshold_reg
        
        return losses
    
    def rgb_to_hsi(self, rgb, sampling_steps=None, apply_adaptive_threshold=True, return_stages=False):
        """
        Convert RGB to HSI with enhanced adaptive thresholding

        Args:
            rgb: RGB image tensor [B, 3, H, W]
            sampling_steps: Optional reduced number of sampling steps
            apply_adaptive_threshold: Whether to apply adaptive thresholding to the output

        Returns:
            HSI image tensor [B, C, H, W]
        """
        # Encode RGB to latent space
        latent = self.encode(rgb)

        # Sample latent using DPM Solver v3 for efficient inference
        sampled_latent = self.dpm_ot.sample(
            latent.shape,
            latent.device,
            use_dpm_solver=True,
            steps=sampling_steps or 20
        )

        # Decode to HSI
        hsi_initial = self.decode(sampled_latent)
        hsi_after_spectral = self.refinement_head(hsi_initial)
        if getattr(self, 'pixel_refinement_head', None) is not None:
            hsi_after_pixel = self.pixel_refinement_head(hsi_after_spectral)
        else:
            hsi_after_pixel = hsi_after_spectral

        stages = {
            'initial': hsi_initial,
            'after_spectral': hsi_after_spectral,
            'after_pixel': hsi_after_pixel,
        }

        # Apply adaptive thresholding as a final refinement step
        if apply_adaptive_threshold:
            # CRITICAL FIX: Ensure transforms are on the correct device
            self.wavelet_transform_output = self.wavelet_transform_output.to(hsi_after_pixel.device)
            self.inverse_wavelet_transform = self.inverse_wavelet_transform.to(hsi_after_pixel.device)

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

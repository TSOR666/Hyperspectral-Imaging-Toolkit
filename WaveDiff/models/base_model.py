import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.encoders import RGBEncoder
from modules.decoders import HSIDecoder, HSI2RGBConverter
from modules.denoisers import UNetDenoiser
from modules.refinement import SpectralRefinementHead, PixelRefinementHead
from modules.attention import MultiHeadSpectralAttention, DomainAdaptiveAttention
from diffusion.dpm_ot import DPMOT
from diffusion.noise_schedule import BaseNoiseSchedule, SpectralNoiseSchedule
from utils.masking import MaskingManager

class HSILatentDiffusionModel(nn.Module):
    """
    Base Latent Diffusion Model for RGB to HSI conversion
    with DPM-OT training and spectral attention
    """
    def __init__(self, latent_dim=64, out_channels=31, timesteps=1000,
                 use_batchnorm=True, masking_config=None, refinement_config=None,
                 use_enhanced_attention=False, use_domain_adaptation=False,
                 dropout=0.1):
        super().__init__()

        # Store out_channels as instance variable for later reference
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        self.use_enhanced_attention = use_enhanced_attention
        self.use_domain_adaptation = use_domain_adaptation

        # RGB Encoder: Maps RGB images to latent space
        self.encoder = RGBEncoder(
            in_channels=3,
            latent_dim=latent_dim,
            use_batchnorm=use_batchnorm
        )

        # Latent Denoiser: For reverse diffusion process in latent space
        self.denoiser = UNetDenoiser(
            channels=latent_dim,
            time_embedding_dim=128,
            use_batchnorm=use_batchnorm
        )

        # HSI Decoder: Reconstructs HSI from latent space
        self.decoder = HSIDecoder(
            out_channels=out_channels,
            latent_dim=latent_dim,
            use_batchnorm=use_batchnorm
        )
        
        refinement_config = refinement_config or {}

        # Enhanced attention mechanisms for better generalization
        if use_enhanced_attention:
            self.enhanced_attention = MultiHeadSpectralAttention(
                channels=latent_dim,
                num_heads=8,
                reduction=4,
                dropout=dropout
            )
        else:
            self.enhanced_attention = None

        # Domain adaptation module for cross-dataset transfer
        if use_domain_adaptation:
            self.domain_attention = DomainAdaptiveAttention(
                channels=latent_dim,
                num_domains=4,
                reduction=8
            )
        else:
            self.domain_attention = None

        # Refinement head to enhance decoded spectra
        spectral_hidden = refinement_config.get('spectral_hidden_channels', max(out_channels * 2, latent_dim))
        spectral_blocks = refinement_config.get('spectral_blocks', 3)
        self.refinement_head = SpectralRefinementHead(
            in_channels=out_channels,
            hidden_channels=spectral_hidden,
            num_blocks=spectral_blocks,
            use_batchnorm=refinement_config.get('spectral_batchnorm', use_batchnorm)
        )

        pixel_config = refinement_config.get('pixel', {})
        self.pixel_refinement_head = None
        if pixel_config.get('enabled', False):
            self.pixel_refinement_head = PixelRefinementHead(
                in_channels=out_channels,
                hidden_channels=pixel_config.get('hidden_channels', out_channels),
                num_blocks=pixel_config.get('num_blocks', 2),
                use_batchnorm=pixel_config.get('use_batchnorm', use_batchnorm),
                expansion=pixel_config.get('expansion', 2),
            )
        
        # HSI to RGB converter for cycle consistency
        self.hsi2rgb = HSI2RGBConverter(
            hsi_channels=out_channels, 
            rgb_channels=3
        )
        
        # Spectral noise schedule
        self.spectral_schedule = SpectralNoiseSchedule(
            timesteps=timesteps,
            num_freq_bands=8
        )
        
        # DPM-OT diffusion process
        self.dpm_ot = DPMOT(
            denoiser=self.denoiser,
            spectral_schedule=self.spectral_schedule,
            timesteps=timesteps
        )
        
        # Advanced masking configuration
        self.masking_config = masking_config or {
            'mask_strategy': 'random',
            'mask_ratio': 0.5,
            'num_epochs': 100
        }
        
        self.masking_manager = MaskingManager(self.masking_config)
    
    def update_masking_epoch(self, epoch):
        """Update the current epoch for progressive masking strategies"""
        self.masking_manager.update_epoch(epoch)
    
    def generate_mask(self, batch_size, height, width, device, inputs=None, num_channels=None, **kwargs):
        """Generate mask using the configured masking strategy"""
        # Determine number of bands for masking (default to inputs if available)
        if num_channels is None:
            if inputs is not None:
                num_bands = inputs.shape[1]
            else:
                # Use stored out_channels (safe and reliable)
                num_bands = self.out_channels
        else:
            num_bands = num_channels

        # Validate num_bands is reasonable
        if num_bands <= 0 or num_bands > 1000:  # Sanity check
            raise ValueError(
                f"Invalid num_bands={num_bands}. Must be in range (0, 1000]. "
                f"Check your input dimensions or out_channels configuration."
            )

        return self.masking_manager.generate_mask(
            inputs,
            batch_size, num_bands, height, width, device,
            **kwargs
        )
    
    def _align_mask_channels(self, mask: torch.Tensor, channels: int) -> torch.Tensor:
        """Broadcast or reduce mask to match the requested channel count."""
        if mask is None:
            return None
        mask_base = mask.float() if mask.dtype == torch.bool else mask
        if mask_base.shape[1] == channels:
            return mask_base
        if channels == 1:
            return mask_base.mean(dim=1, keepdim=True)
        if mask_base.shape[1] == 1:
            return mask_base.expand(-1, channels, -1, -1)
        reduced = mask_base.mean(dim=1, keepdim=True)
        return reduced.expand(-1, channels, -1, -1)

    def apply_mask(self, x, mask):
        """Apply mask to input tensor"""
        if mask is None:
            return x
        mask_prepared = self._align_mask_channels(mask, x.shape[1])
        if mask_prepared.dtype != x.dtype:
            mask_prepared = mask_prepared.to(x.dtype)
        return x * mask_prepared
    
    def encode(self, rgb):
        """Encode RGB images to latent representations"""
        return self.encoder(rgb)
    
    def decode(self, latent):
        """Decode latent representations to HSI"""
        return self.decoder(latent)
    
    def rgb_to_hsi(self, rgb, sampling_steps=None, return_stages=False):
        """
        Convert RGB to HSI using the full model
        
        Args:
            rgb: RGB image tensor [B, 3, H, W]
            sampling_steps: Optional reduced number of sampling steps
            
        Returns:
            HSI image tensor [B, C, H, W]
        """
        # Encode RGB to latent space
        latent = self.encode(rgb)
        
        # Sample latent using DPM Solver v3 for efficient inference
        sampled_latent = self.dpm_ot.sample(
            latent.shape,
            latent.device,
            conditioning=latent,
            use_dpm_solver=True,
            steps=sampling_steps or 20
        )
        
        # Decode to HSI
        hsi_initial = self.decode(sampled_latent)
        hsi_after_spectral = self.refinement_head(hsi_initial)
        if self.pixel_refinement_head is not None:
            hsi_final = self.pixel_refinement_head(hsi_after_spectral)
        else:
            hsi_final = hsi_after_spectral

        if return_stages:
            stages = {
                'initial': hsi_initial,
                'after_spectral': hsi_after_spectral,
                'final': hsi_final
            }
            if self.pixel_refinement_head is not None:
                stages['after_pixel'] = hsi_final
            return hsi_final, stages

        return hsi_final
    
    def forward(self, rgb, t=None, mask=None, use_masking=False):
        """
        Forward pass for training
        
        Args:
            rgb: RGB image tensor [B, 3, H, W]
            t: Optional timestep (if None, one will be sampled)
            mask: Optional mask for self-supervised learning
            use_masking: Whether to use masking
            
        Returns:
            Dictionary of outputs
        """
        # Generate mask for self-supervised learning if requested
        if use_masking and mask is None:
            # Generate mask
            mask = self.generate_mask(
                rgb.shape[0],
                rgb.shape[2],
                rgb.shape[3],
                rgb.device,
                inputs=rgb,
                num_channels=rgb.shape[1]
            )
                
            # Apply mask to RGB
            rgb_masked = self.apply_mask(rgb, mask)
        elif use_masking:
            # Use provided mask
            rgb_masked = self.apply_mask(rgb, mask)
        else:
            rgb_masked = rgb
            mask = None
        
        # Encode RGB to latent space
        latent = self.encode(rgb_masked)

        # Apply enhanced attention if enabled (improves generalization)
        if self.enhanced_attention is not None:
            latent = self.enhanced_attention(latent)

        # Apply domain-adaptive attention if enabled (cross-dataset transfer)
        if self.domain_attention is not None:
            latent = self.domain_attention(latent)

        # Sample timestep if not provided
        batch_size = rgb.shape[0]
        if t is None:
            t = torch.randint(0, self.dpm_ot.timesteps, (batch_size,), device=rgb.device)

        # DPM-OT loss calculation
        diffusion_loss, pred_noise, noise = self.dpm_ot.p_losses(latent, t)

        # Get clean latent (for direct path)
        latent_clean = latent
        
        # Decode latent to HSI then refine spectrally
        hsi_initial = self.decode(latent_clean)
        hsi_after_spectral = self.refinement_head(hsi_initial)
        if self.pixel_refinement_head is not None:
            hsi_output = self.pixel_refinement_head(hsi_after_spectral)
        else:
            hsi_output = hsi_after_spectral

        # Convert HSI back to RGB for cycle consistency
        rgb_from_hsi = self.hsi2rgb(hsi_output)

        return {
            'latent': latent,
            'diffusion_loss': diffusion_loss,
            'pred_noise': pred_noise,
            'noise': noise,
            'hsi_initial': hsi_initial,
            'hsi_after_spectral': hsi_after_spectral,
            'hsi_output': hsi_output,
            'rgb_from_hsi': rgb_from_hsi,
            'mask': mask
        }
    
    def calculate_losses(self, outputs, rgb_target, hsi_target=None):
        """
        Calculate multiple losses:
        1. Diffusion loss from DPM-OT
        2. Cycle consistency loss between reconstructed RGB and original RGB
        3. L1 loss between predicted HSI and target HSI (if provided)
        """
        losses = {}
        
        # DPM-OT diffusion loss (already calculated)
        losses['diffusion_loss'] = outputs['diffusion_loss']
        
        # Predicted outputs
        hsi_output = outputs['hsi_output']
        rgb_from_hsi = outputs['rgb_from_hsi']
        mask = outputs['mask']

        # Cycle consistency loss (RGB → HSI → RGB)
        mask_for_loss = mask
        if mask_for_loss is not None:
            losses['cycle_loss'] = F.l1_loss(
                self.apply_mask(rgb_from_hsi, mask_for_loss),
                self.apply_mask(rgb_target, mask_for_loss)
            )
        else:
            losses['cycle_loss'] = F.l1_loss(rgb_from_hsi, rgb_target)
        
        # L1 loss for HSI reconstruction (if target available)
        if hsi_target is not None:
            if mask is not None:
                losses['l1_loss'] = F.l1_loss(
                    self.apply_mask(hsi_output, mask),
                    self.apply_mask(hsi_target, mask)
                )
            else:
                losses['l1_loss'] = F.l1_loss(hsi_output, hsi_target)

            # Track intermediate reconstruction losses for analysis
            hsi_initial = outputs.get('hsi_initial')
            if hsi_initial is not None:
                if mask is not None:
                    losses['pre_spectral_l1'] = F.l1_loss(
                        self.apply_mask(hsi_initial, mask),
                        self.apply_mask(hsi_target, mask)
                    )
                else:
                    losses['pre_spectral_l1'] = F.l1_loss(hsi_initial, hsi_target)

            hsi_after_spectral = outputs.get('hsi_after_spectral')
            if hsi_after_spectral is not None and hsi_after_spectral is not hsi_output:
                if mask is not None:
                    losses['pre_pixel_l1'] = F.l1_loss(
                        self.apply_mask(hsi_after_spectral, mask),
                        self.apply_mask(hsi_target, mask)
                    )
                else:
                    losses['pre_pixel_l1'] = F.l1_loss(hsi_after_spectral, hsi_target)
        else:
            # Default L1 loss if no target (just for code consistency)
            losses['l1_loss'] = torch.tensor(0.0, device=rgb_target.device)

        return losses
    
    def get_learning_rate(self):
        """Get current learning rate for logging purposes"""
        optimizer = getattr(self, 'optimizer', None)
        if optimizer is None:
            return None

        for param_group in optimizer.param_groups:
            return param_group['lr']
        return None

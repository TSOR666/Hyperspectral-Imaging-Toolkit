import torch
import torch.nn.functional as F

from models.base_model import HSILatentDiffusionModel
from modules.encoders import WaveletRGBEncoder
from modules.decoders import WaveletHSIDecoder, HSI2RGBConverter
from modules.denoisers import WaveletUNetDenoiser
from modules.refinement import SpectralRefinementHead, PixelRefinementHead
from diffusion.dpm_ot import DPMOT
from diffusion.noise_schedule import WaveletSpectralNoiseSchedule
from transforms.haar_wavelet import HaarWaveletTransform
from utils.masking import MaskingManager

class WaveletHSILatentDiffusionModel(HSILatentDiffusionModel):
    """
    HSI Latent Diffusion Model with wavelet transforms and DPM-OT
    
    Enhances the base model with wavelet-based processing for multi-scale
    frequency analysis and reconstruction.
    """
    def __init__(self, latent_dim=64, out_channels=31, timesteps=1000,
                 use_batchnorm=True, masking_config=None, refinement_config=None,
                 norm_type=None, norm_groups=8,
                 cross_attention_mode="channel", attention_window_size=8,
                 conditional_residual_diffusion=False, residual_scale=1.0):
        refinement_config = refinement_config or {}
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

        # RGB Encoder with wavelet transform
        self.encoder = WaveletRGBEncoder(
            in_channels=3,
            latent_dim=latent_dim,
            use_batchnorm=use_batchnorm,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )
        
        # Wavelet-enhanced denoiser
        self.denoiser = WaveletUNetDenoiser(
            channels=latent_dim,
            time_embedding_dim=128,
            use_batchnorm=use_batchnorm,
            norm_type=norm_type,
            norm_groups=norm_groups,
            cross_attention_mode=cross_attention_mode,
            attention_window_size=attention_window_size,
            conditional=conditional_residual_diffusion,
        )
        
        # HSI Decoder with wavelet reconstruction
        self.decoder = WaveletHSIDecoder(
            out_channels=out_channels, 
            latent_dim=latent_dim, 
            use_batchnorm=use_batchnorm,
            norm_type=norm_type,
            norm_groups=norm_groups,
            cross_attention_mode=cross_attention_mode,
            attention_window_size=attention_window_size,
        )

        # Spectral refinement head shared with base implementation
        refinement_hidden = refinement_config.get('spectral_hidden_channels', max(out_channels * 2, latent_dim))
        spectral_blocks = refinement_config.get('spectral_blocks', 3)
        self.refinement_head = SpectralRefinementHead(
            in_channels=out_channels,
            hidden_channels=refinement_hidden,
            num_blocks=spectral_blocks,
            use_batchnorm=refinement_config.get(
                'spectral_batchnorm', use_batchnorm
            ),
            norm_type=refinement_config.get('spectral_norm_type', norm_type),
            norm_groups=norm_groups,
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
                norm_type=pixel_config.get('norm_type', norm_type),
                norm_groups=norm_groups,
            )
        
        # HSI to RGB converter for cycle consistency
        self.hsi2rgb = HSI2RGBConverter(
            hsi_channels=out_channels, 
            rgb_channels=3
        )
        
        # Wavelet-enhanced spectral noise schedule
        self.spectral_schedule = WaveletSpectralNoiseSchedule(
            timesteps=timesteps,
            num_freq_bands=8,
            latent_dim=latent_dim
        )
        
        # DPM-OT diffusion process
        self.dpm_ot = DPMOT(
            denoiser=self.denoiser,
            spectral_schedule=self.spectral_schedule,
            timesteps=timesteps,
            conditional=conditional_residual_diffusion,
        )
        
        # Advanced masking configuration
        self.masking_config = masking_config or {
            'mask_strategy': 'curriculum',
            'mask_ratio': 0.5,
            'num_epochs': 100,
            'curriculum_strategies': ['random', 'block', 'spectral', 'combined']
        }
        
        self.masking_manager = MaskingManager(self.masking_config)
        
        # Fixed transform used by the supervised wavelet-domain objective.
        self.wavelet_transform = HaarWaveletTransform(out_channels)
    
    def calculate_losses(self, outputs, rgb_target, hsi_target=None):
        """
        Calculate wavelet-enhanced losses including:
        - Standard losses from parent class
        - Additional wavelet domain loss when target HSI is available
        """
        # Get regular losses first
        losses = super().calculate_losses(outputs, rgb_target, hsi_target)
        
        # Add wavelet-specific losses if HSI target is available
        if hsi_target is not None:
            # Predicted HSI
            hsi_output = outputs['hsi_output']
            
            output_coeffs = self.wavelet_transform(hsi_output)
            target_coeffs = self.wavelet_transform(hsi_target)
            
            # Calculate wavelet domain loss (MSE on wavelet coefficients)
            # Separate losses for approximation and detail coefficients
            ll_loss = F.mse_loss(output_coeffs[:,:,0], target_coeffs[:,:,0])
            detail_loss = F.mse_loss(output_coeffs[:,:,1:], target_coeffs[:,:,1:])
            
            # Combine with weights (emphasis on low-frequency approximation)
            wavelet_loss = 0.7 * ll_loss + 0.3 * detail_loss
            
            # Add to losses
            losses['wavelet_loss'] = wavelet_loss
        
        return losses
    
    def train_step(self, batch, batch_idx):
        """Single training step implementation"""
        # Extract batch data
        rgb = batch['rgb']
        hsi = batch.get('hsi', None)
        
        # Forward pass
        outputs = self.forward(rgb, use_masking=True, hsi_target=hsi)
        
        # Calculate losses
        losses = self.calculate_losses(outputs, rgb, hsi)
        
        # Combine losses with weighting
        total_loss = (
            losses['diffusion_loss'] * 1.0 +  # Diffusion loss
            losses['cycle_loss'] * 0.8 +       # Cycle consistency
            losses.get('l1_loss', 0) * 1.0     # L1 loss (if available)
        )
        
        # Add wavelet loss if available
        if 'wavelet_loss' in losses:
            total_loss += losses['wavelet_loss'] * 0.5
        
        # Update model 
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # Return losses for logging
        return {
            'loss': total_loss.item(),
            **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()},
            'lr': self.get_learning_rate()
        }
    
    def validation_step(self, batch, batch_idx):
        """Single validation step implementation"""
        # Extract batch data
        rgb = batch['rgb']
        hsi = batch.get('hsi', None)
        
        # Forward pass without masking
        outputs = self.forward(rgb, use_masking=False, hsi_target=hsi)
        
        # Calculate losses
        losses = self.calculate_losses(outputs, rgb, hsi)
        
        # Combine losses with weighting
        total_loss = (
            losses['diffusion_loss'] * 1.0 +  # Diffusion loss
            losses['cycle_loss'] * 0.8 +       # Cycle consistency
            losses.get('l1_loss', 0) * 1.0     # L1 loss (if available)
        )
        
        # Add wavelet loss if available
        if 'wavelet_loss' in losses:
            total_loss += losses['wavelet_loss'] * 0.5
        
        # Return losses for logging
        return {
            'val_loss': total_loss.item(),
            **{f"val_{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        }

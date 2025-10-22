import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.attention import SpectralAttention, CrossSpectralAttention, SpectralSpatialAttention
from modules.encoders import ResidualBlock
from transforms.haar_wavelet import HaarWaveletTransform, InverseHaarWaveletTransform
from transforms.adaptive_wavelet import AdaptiveWaveletThresholding, WaveletNoiseEstimator
from utils.timestep import get_timestep_embedding

class UNetDenoiser(nn.Module):
    """
    U-Net architecture with spectral attention for denoising in latent space
    """
    def __init__(self, channels=64, time_embedding_dim=128, use_batchnorm=True):
        super().__init__()
        
        # Time embedding
        self.time_dim = time_embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim),
        )
        
        # Initial convolution with time embedding
        self.init_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Conv2d(time_embedding_dim, channels, kernel_size=1)
        
        # Initial spectral attention
        self.init_attn = SpectralAttention(channels)
        
        # Encoder part of U-Net with spectral attention
        self.down1 = nn.Sequential(
            ResidualBlock(channels, use_batchnorm=use_batchnorm),
            SpectralAttention(channels),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU()
        )
        
        self.down2 = nn.Sequential(
            ResidualBlock(channels * 2, use_batchnorm=use_batchnorm),
            SpectralAttention(channels * 2),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU()
        )
        
        # Middle blocks with cross-spectral attention
        self.middle_attn = CrossSpectralAttention(channels * 2)
        self.middle = nn.Sequential(
            ResidualBlock(channels * 2, use_batchnorm=use_batchnorm),
            SpectralAttention(channels * 2),
            ResidualBlock(channels * 2, use_batchnorm=use_batchnorm)
        )
        
        # Decoder part of U-Net with spectral attention
        self.up1 = nn.Sequential(
            ResidualBlock(channels * 4, use_batchnorm=use_batchnorm),  # Doubled due to skip connection
            SpectralAttention(channels * 4),
            nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU()
        )
        
        self.up2 = nn.Sequential(
            ResidualBlock(channels * 3, use_batchnorm=use_batchnorm),  # Doubled due to skip connection
            SpectralAttention(channels * 3),
            nn.ConvTranspose2d(channels * 3, channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU()
        )
        
        # Final blocks with spectral-spatial attention
        self.final_attn = SpectralSpatialAttention(channels * 2)  # Doubled due to skip connection
        self.final_res = ResidualBlock(channels * 2, use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        
    def forward(self, x, t):
        # Embed timestep
        t_emb = get_timestep_embedding(t, self.time_dim, dtype=x.dtype)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.view(-1, self.time_dim, 1, 1)
        t_emb = self.time_proj(t_emb)
        t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])

        # Initial convolution and spectral attention
        h = self.init_conv(x) + t_emb
        h = self.init_attn(h)
        
        # Save skip connections
        skip1 = h
        
        # Downsampling with spectral attention
        h = self.down1(h)
        skip2 = h
        h = self.down2(h)
        
        # Middle with cross-spectral attention
        h = self.middle_attn(h)
        h = self.middle(h)
        
        # Upsampling with skip connections and spectral attention
        if skip2.shape[2:] != h.shape[2:]:
            skip2 = F.interpolate(
                skip2,
                size=h.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        h = torch.cat([h, skip2], dim=1)
        h = self.up1(h)

        if skip1.shape[2:] != h.shape[2:]:
            skip1 = F.interpolate(
                skip1,
                size=h.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        h = torch.cat([h, skip1], dim=1)
        h = self.up2(h)

        # Final processing with spectral-spatial attention
        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(
                h,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        h = torch.cat([h, x], dim=1)  # Skip connection from input
        h = self.final_attn(h)
        h = self.final_res(h)
        output = self.final_conv(h)
        
        return output


class WaveletUNetDenoiser(nn.Module):
    """
    U-Net denoiser with wavelet transform for multi-scale noise prediction
    """
    def __init__(self, channels=64, time_embedding_dim=128, use_batchnorm=True):
        super().__init__()
        
        # Time embedding
        self.time_dim = time_embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim),
        )
        
        # Initial wavelet transform
        self.wavelet = HaarWaveletTransform(channels)
        
        # Process different wavelet components separately
        self.process_ll = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        self.process_detail = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.SiLU()
            ) for _ in range(3)  # LH, HL, HH
        ])
        
        # Time projection
        self.time_proj = nn.Conv2d(time_embedding_dim, channels * 4, kernel_size=1)
        
        # Combine processed wavelet components
        self.combine_wavelet = nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1)
        
        # Middle blocks with spectral attention
        self.middle_attn = CrossSpectralAttention(channels * 2)
        self.middle = nn.Sequential(
            ResidualBlock(channels * 2, use_batchnorm=use_batchnorm),
            SpectralAttention(channels * 2),
            ResidualBlock(channels * 2, use_batchnorm=use_batchnorm)
        )
        
        # Inverse wavelet transform
        self.inverse_wavelet = InverseHaarWaveletTransform()
        
        # Upsampling with spectral attention
        self.up = nn.Sequential(
            ResidualBlock(channels * 2, use_batchnorm=use_batchnorm),
            SpectralAttention(channels * 2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        # Final blocks
        self.final_attn = SpectralSpatialAttention(channels * 2)  # Considers both input and processed features
        self.final_res = ResidualBlock(channels * 2, use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        
    def forward(self, x, t):
        # Original input for residual connection
        original_x = x
        
        # Embed timestep
        t_emb = get_timestep_embedding(t, self.time_dim, dtype=x.dtype)
        t_emb = self.time_mlp(t_emb).view(-1, self.time_dim, 1, 1)
        
        # Apply wavelet transform
        wavelet_coeffs = self.wavelet(x)
        B, C, _, H, W = wavelet_coeffs.shape
        
        # Process each component
        components = [wavelet_coeffs[:, :, i] for i in range(4)]  # LL, LH, HL, HH
        
        # Process approximation (LL) coefficient
        components[0] = self.process_ll(components[0])
        
        # Process detail coefficients
        for i in range(3):
            components[i+1] = self.process_detail[i](components[i+1])
            
        # Apply time embedding to each component
        # OPTIMIZATION: More efficient reshaping and broadcasting
        time_context = self.time_proj(t_emb)  # [B, C*4, 1, 1]
        t_emb_reshaped = time_context.view(B, C, 4, 1, 1)
        # Broadcast efficiently during addition instead of pre-expanding
        for i in range(4):
            components[i] = components[i] + t_emb_reshaped[:, :, i]
            
        # Reshape and combine
        components = torch.cat([c.unsqueeze(2) for c in components], dim=2)  # B, C, 4, H, W
        combined = components.view(B, C * 4, H, W)
        features = F.silu(self.combine_wavelet(combined))
        
        # Middle processing
        features = self.middle_attn(features)
        features = self.middle(features)
        
        # Prepare for inverse wavelet transform
        features_for_iwt = features.view(B, features.shape[1] // 4, 4, H, W)
        
        # Apply inverse wavelet transform
        features_upsampled = self.inverse_wavelet(features_for_iwt)
        
        # Final processing
        features_upsampled = self.up(features_upsampled)
        
        # Concatenate with original input
        output = torch.cat([features_upsampled, original_x], dim=1)
        
        # Final attention and convolution
        output = self.final_attn(output)
        output = self.final_res(output)
        output = self.final_conv(output)
        
        return output


class ThresholdingWaveletUNetDenoiser(nn.Module):
    """
    Wavelet UNet denoiser with adaptive thresholding
    """
    def __init__(self, channels=64, time_embedding_dim=128, use_batchnorm=True,
                threshold_method='soft', init_threshold=0.1, trainable_threshold=True):
        super().__init__()
        
        # Time embedding
        self.time_dim = time_embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim),
        )
        
        # Initial wavelet transform
        self.wavelet = HaarWaveletTransform(channels)
        
        # Noise estimator for adaptive thresholding
        self.noise_estimator = WaveletNoiseEstimator()
        
        # Thresholding modules (one before processing, one after)
        self.input_thresholding = AdaptiveWaveletThresholding(
            channels, method=threshold_method, trainable=trainable_threshold, init_threshold=init_threshold
        )
        
        self.output_thresholding = AdaptiveWaveletThresholding(
            channels * 2, method=threshold_method, trainable=trainable_threshold, init_threshold=init_threshold * 0.5
        )
        
        # Process different wavelet components separately
        self.process_ll = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        self.process_detail = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.SiLU()
            ) for _ in range(3)  # LH, HL, HH
        ])
        
        # Time projection
        self.time_proj = nn.Conv2d(time_embedding_dim, channels * 4, kernel_size=1)
        
        # Combine processed wavelet components
        self.combine_wavelet = nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1)
        
        # Middle blocks with spectral attention
        self.middle_attn = CrossSpectralAttention(channels * 2)
        self.middle = nn.Sequential(
            ResidualBlock(channels * 2, use_batchnorm=use_batchnorm),
            SpectralAttention(channels * 2),
            ResidualBlock(channels * 2, use_batchnorm=use_batchnorm)
        )
        
        # Inverse wavelet transform
        self.inverse_wavelet = InverseHaarWaveletTransform()
        
        # Upsampling with spectral attention
        self.up = nn.Sequential(
            ResidualBlock(channels * 2, use_batchnorm=use_batchnorm),
            SpectralAttention(channels * 2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        # Final blocks
        self.final_attn = SpectralSpatialAttention(channels * 2)  # Considers both input and processed features
        self.final_res = ResidualBlock(channels * 2, use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        
    def forward(self, x, t):
        # Original input for residual connection
        original_x = x
        
        # Embed timestep
        t_emb = get_timestep_embedding(t, self.time_dim, dtype=x.dtype)
        t_emb = self.time_mlp(t_emb).view(-1, self.time_dim, 1, 1)
        
        # Apply wavelet transform
        wavelet_coeffs = self.wavelet(x)
        B, C, _, H, W = wavelet_coeffs.shape
        
        # Estimate noise level
        noise_level = self.noise_estimator(wavelet_coeffs)
        
        # Apply thresholding to input coefficients
        wavelet_coeffs = self.input_thresholding(wavelet_coeffs, noise_level)
        
        # Process each component
        components = [wavelet_coeffs[:, :, i] for i in range(4)]  # LL, LH, HL, HH
        
        # Process approximation (LL) coefficient
        components[0] = self.process_ll(components[0])
        
        # Process detail coefficients
        for i in range(3):
            components[i+1] = self.process_detail[i](components[i+1])
            
        # Apply time embedding to each component
        # OPTIMIZATION: More efficient reshaping and broadcasting
        time_context = self.time_proj(t_emb)  # [B, C*4, 1, 1]
        t_emb_reshaped = time_context.view(B, C, 4, 1, 1)
        # Broadcast efficiently during addition instead of pre-expanding
        for i in range(4):
            components[i] = components[i] + t_emb_reshaped[:, :, i]
            
        # Reshape and combine
        components = torch.cat([c.unsqueeze(2) for c in components], dim=2)  # B, C, 4, H, W
        combined = components.view(B, C * 4, H, W)
        features = F.silu(self.combine_wavelet(combined))
        
        # Middle processing
        features = self.middle_attn(features)
        features = self.middle(features)
        
        # Generate wavelet coefficients for the output
        out_coeffs = features.view(B, features.shape[1] // 4, 4, H, W)
        
        # Apply output thresholding
        out_coeffs = self.output_thresholding(out_coeffs, noise_level)
        
        # Apply inverse wavelet transform
        features_upsampled = self.inverse_wavelet(out_coeffs)
        
        # Final processing
        features_upsampled = self.up(features_upsampled)
        
        # Concatenate with original input
        output = torch.cat([features_upsampled, original_x], dim=1)
        
        # Final attention and convolution
        output = self.final_attn(output)
        output = self.final_res(output)
        output = self.final_conv(output)
        
        return output
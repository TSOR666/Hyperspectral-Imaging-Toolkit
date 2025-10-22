import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.attention import SpectralAttention, SpectralSpatialAttention, CrossSpectralAttention
from transforms.haar_wavelet import HaarWaveletTransform

class ResidualBlock(nn.Module):
    """Residual block with optional batch normalization"""
    def __init__(self, channels, use_batchnorm=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        
        # First convolution block
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(channels)
            
        # Second convolution block
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        if self.use_batchnorm:
            out = self.bn1(out)
        out = F.silu(out)
        
        # Second conv block
        out = self.conv2(out)
        if self.use_batchnorm:
            out = self.bn2(out)
            
        # Add skip connection and activation
        return F.silu(out + identity)


class RGBEncoder(nn.Module):
    """
    Encoder network with spectral attention to map RGB images to latent space
    """
    def __init__(self, in_channels=3, latent_dim=64, use_batchnorm=True):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Initial convolution 
        self.init_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        
        # Downsampling blocks with spectral attention
        self.down1 = nn.Sequential(
            ResidualBlock(32, use_batchnorm=use_batchnorm),
            SpectralAttention(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU()
        )
        
        self.down2 = nn.Sequential(
            ResidualBlock(64, use_batchnorm=use_batchnorm),
            SpectralAttention(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU()
        )
        
        self.down3 = nn.Sequential(
            ResidualBlock(128, use_batchnorm=use_batchnorm),
            CrossSpectralAttention(128),
            nn.Conv2d(128, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )
        
        # Final spectral-spatial attention
        self.final_attn = SpectralSpatialAttention(latent_dim)
        self.final_res = ResidualBlock(latent_dim, use_batchnorm=use_batchnorm)
        
    def forward(self, x):
        # Initial features
        x = F.silu(self.init_conv(x))
        
        # Downsampling path with attention
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        # Final processing with spectral-spatial attention
        x = self.final_attn(x)
        latent = self.final_res(x)
        
        return latent


class WaveletRGBEncoder(nn.Module):
    """
    RGB encoder using wavelet transform for multi-scale analysis
    """
    def __init__(self, in_channels=3, latent_dim=64, use_batchnorm=True):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Haar wavelet transform
        self.wavelet = HaarWaveletTransform(in_channels)
        
        # Initial convolution on original input
        self.init_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        
        # Process wavelet components (4 coefficient types) with a shared conv
        self.wavelet_conv = nn.Conv2d(in_channels * 4, 32, kernel_size=3, padding=1)
        
        # Process and merge features
        self.merge_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Downsampling path with wavelet processing at each stage
        self.down1_wavelet = HaarWaveletTransform(64)
        self.down1_attn = self._create_wavelet_attention(64)
        self.down1_conv = nn.Conv2d(64 * 4, 128, kernel_size=3, padding=1)
        
        self.down2 = nn.Sequential(
            ResidualBlock(128, use_batchnorm=use_batchnorm),
            SpectralAttention(128),
            nn.Conv2d(128, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )
        
        # Final spectral-spatial attention
        self.final_attn = SpectralSpatialAttention(latent_dim)
        self.final_res = ResidualBlock(latent_dim, use_batchnorm=use_batchnorm)
    
    def _create_wavelet_attention(self, channels):
        """Helper to create attention modules for wavelet coefficients"""
        return nn.ModuleList([SpectralAttention(channels) for _ in range(4)])
        
    def forward(self, x):
        # Apply Haar wavelet transform
        wavelet_coeffs = self.wavelet(x)
        B, C, _, H, W = wavelet_coeffs.shape
        
        # Flatten wavelet coefficients along the channel dimension
        wavelet_features = wavelet_coeffs.reshape(B, C * 4, H, W)
        wavelet_features = F.silu(self.wavelet_conv(wavelet_features))
        
        # Process original input
        spatial_features = F.silu(self.init_conv(x))
        
        # Downsample spatial features to match wavelet resolution
        if spatial_features.shape[2] != wavelet_features.shape[2]:
            spatial_features = F.avg_pool2d(spatial_features, 2)
        
        # Merge features
        merged = torch.cat([spatial_features, wavelet_features], dim=1)
        merged = F.silu(self.merge_conv(merged))
        
        # First downsampling with wavelet attention
        wavelet_coeffs1 = self.down1_wavelet(merged)
        
        # Apply attention to each wavelet component
        components = [wavelet_coeffs1[:, :, i] for i in range(4)]
        for i in range(4):
            components[i] = self.down1_attn[i](components[i])
        
        # Recombine wavelet components
        wavelet_coeffs1 = torch.stack(components, dim=2)
        
        # Flatten and proceed with downsampling
        wavelet_features1 = wavelet_coeffs1.reshape(B, merged.shape[1] * 4, H // 2, W // 2)
        features = F.silu(self.down1_conv(wavelet_features1))
        
        # Second downsampling (conventional)
        features = self.down2(features)
        
        # Final attention
        features = self.final_attn(features)
        latent = self.final_res(features)
        
        return latent


class EnhancedRGBEncoder(nn.Module):
    """
    Enhanced RGB encoder with additional frequency domain processing
    """
    def __init__(self, in_channels=3, latent_dim=64, use_batchnorm=True):
        super().__init__()
        
        # Standard encoder path
        self.standard_encoder = RGBEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            use_batchnorm=use_batchnorm
        )
        
        # Wavelet encoder path
        self.wavelet_encoder = WaveletRGBEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            use_batchnorm=use_batchnorm
        )
        
        # Fusion of both paths
        self.fusion = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=1),
            nn.SiLU(),
            ResidualBlock(latent_dim, use_batchnorm=use_batchnorm)
        )
        
    def forward(self, x):
        # Process using standard encoder
        standard_features = self.standard_encoder(x)
        
        # Process using wavelet encoder
        wavelet_features = self.wavelet_encoder(x)
        
        # Fuse features
        fused = torch.cat([standard_features, wavelet_features], dim=1)
        latent = self.fusion(fused)
        
        return latent
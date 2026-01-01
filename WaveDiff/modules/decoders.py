import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention import SpectralAttention, CrossSpectralAttention
from transforms.haar_wavelet import InverseHaarWaveletTransform
from modules.encoders import ResidualBlock

class HSIDecoder(nn.Module):
    """
    Decoder network with spectral attention to map latent representation to HSI
    """
    def __init__(self, out_channels=31, latent_dim=64, use_batchnorm=True):
        super().__init__()
        
        # Initial processing in latent space with spectral attention
        self.init_attn = CrossSpectralAttention(latent_dim)
        self.init_res = ResidualBlock(latent_dim, use_batchnorm=use_batchnorm)
        
        # Upsampling blocks with spectral attention
        self.up1 = nn.Sequential(
            ResidualBlock(latent_dim, use_batchnorm=use_batchnorm),
            SpectralAttention(latent_dim),
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU()
        )
        
        self.up2 = nn.Sequential(
            ResidualBlock(128, use_batchnorm=use_batchnorm),
            SpectralAttention(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU()
        )
        
        self.up3 = nn.Sequential(
            ResidualBlock(64, use_batchnorm=use_batchnorm),
            SpectralAttention(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )
        
        # Final spectral-aware convolution to HSI bands
        self.final_attn = SpectralAttention(32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Initial processing with cross-spectral attention
        x = self.init_attn(x)
        x = self.init_res(x)
        
        # Upsampling path with spectral attention
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        
        # Final spectral attention and convolution to HSI bands
        x = self.final_attn(x)
        hsi = self.final_conv(x)
        
        return hsi


class WaveletHSIDecoder(nn.Module):
    """
    HSI decoder using inverse wavelet transform for multi-scale reconstruction
    """
    def __init__(self, out_channels=31, latent_dim=64, use_batchnorm=True):
        super().__init__()
        
        # Initial processing in latent space
        self.init_attn = CrossSpectralAttention(latent_dim)
        self.init_res = ResidualBlock(latent_dim, use_batchnorm=use_batchnorm)
        
        # First upsampling (conventional)
        self.up1 = nn.Sequential(
            ResidualBlock(latent_dim, use_batchnorm=use_batchnorm),
            SpectralAttention(latent_dim),
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU()
        )
        
        # Second upsampling with wavelet processing
        self.up2_conv = nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1)  # Prepare for wavelet coefficients
        self.inverse_wavelet = InverseHaarWaveletTransform()
        
        # Process reconstructed features
        self.up3 = nn.Sequential(
            ResidualBlock(128, use_batchnorm=use_batchnorm),
            SpectralAttention(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        # Final spectral-aware processing
        self.final_attn = SpectralAttention(64)
        self.final_res = ResidualBlock(64, use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Initial processing
        x = self.init_attn(x)
        x = self.init_res(x)
        
        # First upsampling
        x = self.up1(x)
        
        # Prepare for inverse wavelet transform
        B, C, H, W = x.shape
        x = F.silu(self.up2_conv(x))
        x = x.view(B, C, 4, H, W)  # Reshape to wavelet coefficients format
        
        # Apply inverse wavelet transform
        x = self.inverse_wavelet(x)
        
        # Final processing
        x = self.up3(x)
        x = self.final_attn(x)
        x = self.final_res(x)
        hsi = self.final_conv(x)
        
        return hsi


class MultiscaleHSIDecoder(nn.Module):
    """
    HSI decoder with multi-scale reconstruction at different resolutions
    """
    def __init__(self, out_channels=31, latent_dim=64, use_batchnorm=True):
        super().__init__()
        
        # Initial processing
        self.init_block = nn.Sequential(
            CrossSpectralAttention(latent_dim),
            ResidualBlock(latent_dim, use_batchnorm=use_batchnorm)
        )
        
        # Multi-scale path: create decoders at different scales
        # Scale 1: Original resolution (finest detail)
        self.decoder_scale1 = nn.Sequential(
            ResidualBlock(latent_dim, use_batchnorm=use_batchnorm),
            SpectralAttention(latent_dim),
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU()
        )
        
        # Scale 2: Medium resolution (mid-level detail)
        self.decoder_scale2 = nn.Sequential(
            ResidualBlock(latent_dim, use_batchnorm=use_batchnorm),
            SpectralAttention(latent_dim),
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU()
        )
        
        # Scale 3: Low resolution (coarse structure)
        self.decoder_scale3 = nn.Sequential(
            ResidualBlock(latent_dim, use_batchnorm=use_batchnorm),
            SpectralAttention(latent_dim)
        )
        
        # Output heads for each scale
        self.out_scale1 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.out_scale2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.out_scale3 = nn.Conv2d(latent_dim, out_channels, kernel_size=3, padding=1)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, 32, kernel_size=3, padding=1),
            SpectralAttention(32),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Initial processing
        x = self.init_block(x)
        
        # Process at different scales
        feat_scale1 = self.decoder_scale1(x)
        feat_scale2 = self.decoder_scale2(x)
        feat_scale3 = self.decoder_scale3(x)
        
        # Generate outputs at each scale
        out_scale1 = self.out_scale1(feat_scale1)
        out_scale2 = self.out_scale2(feat_scale2)
        out_scale3 = self.out_scale3(feat_scale3)
        
        # Upsample lower resolution outputs to match highest resolution
        out_scale2_up = F.interpolate(out_scale2, size=out_scale1.shape[2:], mode='bilinear', align_corners=False)
        out_scale3_up = F.interpolate(out_scale3, size=out_scale1.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine multi-scale outputs
        combined = torch.cat([out_scale1, out_scale2_up, out_scale3_up], dim=1)
        
        # Final fusion
        hsi = self.fusion(combined)
        
        return hsi


class HSI2RGBConverter(nn.Module):
    """
    Maps HSI to RGB using learnable 1x1 convolution (represents camera response)
    """
    def __init__(self, hsi_channels=31, rgb_channels=3):
        super().__init__()
        # 1x1 convolution for learning the mapping from HSI to RGB
        self.conv1x1 = nn.Conv2d(hsi_channels, rgb_channels, kernel_size=1)

    def forward(self, hsi):
        return self.conv1x1(hsi)
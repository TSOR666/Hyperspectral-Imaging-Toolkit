import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HaarWaveletTransform(nn.Module):
    """
    Haar wavelet transform implementation as a PyTorch module
    
    Decomposes an input tensor into four subbands:
    - LL: Low-frequency approximation (average)
    - LH: Horizontal details
    - HL: Vertical details
    - HH: Diagonal details
    """
    def __init__(self, in_channels):
        super().__init__()
        # Define Haar wavelet filter bank
        filters = torch.tensor([
            [[0.5, 0.5], [0.5, 0.5]],  # LL - Average
            [[0.5, 0.5], [-0.5, -0.5]],  # LH - Horizontal
            [[0.5, -0.5], [0.5, -0.5]],  # HL - Vertical
            [[0.5, -0.5], [-0.5, 0.5]]   # HH - Diagonal
        ])
        # Register filters as buffer (not trainable)
        self.register_buffer("weight", filters.repeat(in_channels, 1, 1, 1))
        self.groups = in_channels

    def forward(self, x):
        """
        Apply wavelet transform to input
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Wavelet coefficients [B, C, 4, H//2, W//2]
        """
        B, C, H, W = x.shape
        # Apply convolution with stride 2 to perform filtering and downsampling
        out = F.conv2d(x, weight=self.weight, stride=2, groups=self.groups)
        # Reshape to separate the wavelet components
        return out.view(B, C, 4, H // 2, W // 2)


class InverseHaarWaveletTransform(nn.Module):
    """
    Inverse Haar wavelet transform implementation as a PyTorch module
    
    Reconstructs an input tensor from its wavelet subbands:
    - LL: Low-frequency approximation
    - LH: Horizontal details
    - HL: Vertical details
    - HH: Diagonal details
    """
    def __init__(self):
        super().__init__()
        # No parameters needed for default Haar inverse
    
    def forward(self, coeffs):
        """
        Apply inverse wavelet transform
        
        Args:
            coeffs: Wavelet coefficients [B, C, 4, H, W]
            
        Returns:
            Reconstructed tensor [B, C, H*2, W*2]
        """
        B, C, num_coeff, H, W = coeffs.shape
        # Extract components
        LL, LH, HL, HH = [coeffs[:, :, i, :, :] for i in range(4)]
        
        # Initialize output tensor (2x larger in each spatial dimension)
        recon = torch.zeros(B, C, H * 2, W * 2, device=coeffs.device, dtype=coeffs.dtype)
        
        # Reconstruction formula for Haar wavelet
        # Each 2x2 block in the output is reconstructed from the 4 coefficients
        recon[:, :, 0::2, 0::2] = (LL + LH + HL + HH) / 2  # Top-left pixels
        recon[:, :, 0::2, 1::2] = (LL + LH - HL - HH) / 2  # Top-right pixels
        recon[:, :, 1::2, 0::2] = (LL - LH + HL - HH) / 2  # Bottom-left pixels
        recon[:, :, 1::2, 1::2] = (LL - LH - HL + HH) / 2  # Bottom-right pixels
        
        return recon
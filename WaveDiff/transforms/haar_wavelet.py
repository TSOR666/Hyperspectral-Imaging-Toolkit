"""Haar wavelet transform implementation for PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletTransform(nn.Module):
    """
    Haar wavelet transform implementation as a PyTorch module.

    Decomposes an input tensor into four subbands:
    - LL: Low-frequency approximation (average)
    - LH: Horizontal details
    - HL: Vertical details
    - HH: Diagonal details
    """

    # Explicit buffer type annotation for pyright
    weight: torch.Tensor

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # Define Haar wavelet filter bank: [4, 1, 2, 2]
        # Each filter is applied to each input channel independently
        filters = torch.tensor([
            [[[0.5, 0.5], [0.5, 0.5]]],    # LL - Average
            [[[0.5, 0.5], [-0.5, -0.5]]],  # LH - Horizontal
            [[[0.5, -0.5], [0.5, -0.5]]],  # HL - Vertical
            [[[0.5, -0.5], [-0.5, 0.5]]]   # HH - Diagonal
        ], dtype=torch.float32)  # Shape: [4, 1, 2, 2]

        # For grouped convolution with groups=in_channels:
        # Weight shape must be [out_channels, in_channels/groups, kH, kW]
        # = [in_channels*4, 1, 2, 2]
        # Repeat the 4 filters for each input channel
        weight = filters.repeat(in_channels, 1, 1, 1)  # [in_channels*4, 1, 2, 2]
        self.register_buffer("weight", weight)
        self.groups = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply wavelet transform to input.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Wavelet coefficients [B, C, 4, H//2, W//2]
        """
        B, C, H, W = x.shape

        # Apply grouped convolution: each channel gets 4 output coefficients
        # Input: [B, C, H, W]
        # Weight: [C*4, 1, 2, 2], groups=C
        # Output: [B, C*4, H//2, W//2]
        out = F.conv2d(x, weight=self.weight, stride=2, groups=self.groups)

        # Reshape to separate the wavelet components: [B, C, 4, H//2, W//2]
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
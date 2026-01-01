import torch
import torch.nn as nn
import torch.nn.functional as F

from transforms.haar_wavelet import HaarWaveletTransform


class WaveletLoss(nn.Module):
    """
    Loss function operating directly on wavelet coefficients

    Allows different weights for approximation and detail coefficients
    to emphasize different frequency components.
    """

    # Explicit buffer type annotations for pyright
    ll_weight: torch.Tensor
    lh_weight: torch.Tensor
    hl_weight: torch.Tensor
    hh_weight: torch.Tensor

    def __init__(self, in_channels=31, ll_weight=1.0, lh_weight=0.5, hl_weight=0.5, hh_weight=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.wavelet = HaarWaveletTransform(in_channels)

        # Weights for different coefficient types (can be learned or fixed)
        self.register_buffer('ll_weight', torch.tensor(ll_weight))  # Approximation (low frequency)
        self.register_buffer('lh_weight', torch.tensor(lh_weight))  # Horizontal detail
        self.register_buffer('hl_weight', torch.tensor(hl_weight))  # Vertical detail
        self.register_buffer('hh_weight', torch.tensor(hh_weight))  # Diagonal detail (highest frequency)
        
    def forward(self, pred, target):
        """
        Calculate wavelet domain loss
        
        Args:
            pred: Predicted HSI [B, C, H, W]
            target: Target HSI [B, C, H, W]
            
        Returns:
            Weighted loss in wavelet domain
        """
        # Transform to wavelet domain
        pred_coeffs = self.wavelet(pred)
        target_coeffs = self.wavelet(target)
        
        # Calculate loss for each type of coefficient
        ll_loss = F.mse_loss(pred_coeffs[:, :, 0], target_coeffs[:, :, 0])
        lh_loss = F.mse_loss(pred_coeffs[:, :, 1], target_coeffs[:, :, 1])
        hl_loss = F.mse_loss(pred_coeffs[:, :, 2], target_coeffs[:, :, 2])
        hh_loss = F.mse_loss(pred_coeffs[:, :, 3], target_coeffs[:, :, 3])
        
        # Combine with weights
        total_loss = (
            self.ll_weight * ll_loss + 
            self.lh_weight * lh_loss + 
            self.hl_weight * hl_loss + 
            self.hh_weight * hh_loss
        )
        
        return total_loss


class MultiscaleWaveletLoss(nn.Module):
    """
    Multi-scale wavelet loss that applies wavelet transform at multiple levels
    and calculates loss at each level.
    """

    # Explicit buffer type annotation for pyright
    level_weights: torch.Tensor

    def __init__(self, in_channels=31, num_levels=3, level_weights=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_levels = num_levels

        # Create wavelet transform for each input channel
        self.wavelet = HaarWaveletTransform(in_channels)

        # Weights for each decomposition level
        if level_weights is None:
            # Default: higher weight for lower levels (larger structures)
            level_weights = [2**(-i) for i in range(num_levels)]
            # Normalize weights
            level_weights = [w / sum(level_weights) for w in level_weights]

        self.register_buffer('level_weights', torch.tensor(level_weights))
        
    def forward(self, pred, target):
        """
        Calculate multi-scale wavelet loss
        
        Args:
            pred: Predicted HSI [B, C, H, W]
            target: Target HSI [B, C, H, W]
            
        Returns:
            Weighted multi-scale loss
        """
        total_loss = 0.0
        
        # Current level inputs
        curr_pred = pred
        curr_target = target
        
        for level in range(self.num_levels):
            # Calculate wavelet coefficients at this level
            pred_coeffs = self.wavelet(curr_pred)
            target_coeffs = self.wavelet(curr_target)
            
            # Calculate MSE loss at this level
            level_loss = F.mse_loss(pred_coeffs, target_coeffs)
            
            # Add weighted loss for this level
            total_loss += self.level_weights[level] * level_loss
            
            # For next level, use approximation (LL) coefficients
            # This creates a multi-scale decomposition
            curr_pred = pred_coeffs[:, :, 0]  # LL component
            curr_target = target_coeffs[:, :, 0]  # LL component
            
            # If the resolution becomes too small, stop
            if curr_pred.shape[2] < 4 or curr_pred.shape[3] < 4:
                break
        
        return total_loss


class CombinedWaveletLoss(nn.Module):
    """
    Combined loss using both wavelet and other domain-specific losses
    """
    def __init__(self, in_channels=31, wavelet_weight=0.5, l1_weight=0.3, cycle_weight=0.2):
        super().__init__()
        self.wavelet_loss = WaveletLoss(in_channels=in_channels)
        self.wavelet_weight = wavelet_weight
        self.l1_weight = l1_weight
        self.cycle_weight = cycle_weight
        
    def forward(self, pred_hsi, target_hsi, pred_rgb=None, target_rgb=None):
        """
        Calculate combined losses
        
        Args:
            pred_hsi: Predicted HSI [B, C, H, W]
            target_hsi: Target HSI [B, C, H, W]
            pred_rgb: Optional predicted RGB [B, 3, H, W]
            target_rgb: Optional target RGB [B, 3, H, W]
            
        Returns:
            Combined loss
        """
        # Wavelet domain loss
        w_loss = self.wavelet_loss(pred_hsi, target_hsi)
        
        # L1 loss in spatial domain
        l1_loss = F.l1_loss(pred_hsi, target_hsi)
        
        # Initialize combined loss
        combined_loss = self.wavelet_weight * w_loss + self.l1_weight * l1_loss
        
        # Add cycle consistency loss if RGB predictions are provided
        if pred_rgb is not None and target_rgb is not None:
            cycle_loss = F.l1_loss(pred_rgb, target_rgb)
            combined_loss += self.cycle_weight * cycle_loss
        
        return combined_loss
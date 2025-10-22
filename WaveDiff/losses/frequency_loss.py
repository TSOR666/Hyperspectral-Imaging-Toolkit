import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FrequencyDomainLoss(nn.Module):
    """
    Loss function operating in frequency domain using FFT
    
    Emphasizes low frequency components which are perceptually more important.
    """
    def __init__(self, low_freq_weight=1.0, high_freq_weight=0.5, freq_bands=5):
        super().__init__()
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight
        self.freq_bands = freq_bands
        
    def forward(self, pred, target):
        """
        Calculate frequency domain loss
        
        Args:
            pred: Predicted HSI [B, C, H, W]
            target: Target HSI [B, C, H, W]
            
        Returns:
            Weighted loss in frequency domain
        """
        # Transform to frequency domain
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        
        # Calculate magnitude and phase
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # Generate frequency weights
        B, C, H, half_W = pred_mag.shape
        
        # Create frequency grid
        freq_h = torch.fft.fftfreq(H, device=pred.device)[:, None]
        freq_w = torch.fft.rfftfreq(pred.shape[3], device=pred.device)[None, :]
        
        # Calculate distance from DC (center)
        freq_dist = torch.sqrt(freq_h**2 + freq_w**2)
        
        # Normalize distance to [0, 1]
        max_dist = torch.sqrt(torch.tensor(0.5**2 + 0.5**2, device=pred.device))
        norm_dist = freq_dist / max_dist
        
        # Create weights that emphasize low frequencies
        # Weights decay from low_freq_weight to high_freq_weight as frequency increases
        weights = self.low_freq_weight - (self.low_freq_weight - self.high_freq_weight) * norm_dist
        
        # Expand weights to match dimensions
        weights = weights.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        
        # Calculate weighted magnitude loss
        mag_loss = F.mse_loss(weights * pred_mag, weights * target_mag)
        
        # Calculate phase loss (only where magnitude is significant)
        # This prevents phase from dominating in regions with negligible energy
        magnitude_mask = (target_mag > target_mag.mean() * 0.1).float()
        phase_diff = 1.0 - torch.cos(pred_phase - target_phase)
        phase_loss = (phase_diff * magnitude_mask).mean()
        
        # Combine losses
        return mag_loss + 0.5 * phase_loss


class FrequencyBandLoss(nn.Module):
    """
    Loss function that divides frequency spectrum into bands
    and applies different weights to each band
    """
    def __init__(self, num_bands=5, band_weights=None):
        super().__init__()
        self.num_bands = num_bands
        
        # Default weights emphasize low and mid frequencies
        if band_weights is None:
            # Create band weights that emphasize low and mid frequencies
            # First band (lowest freq) gets highest weight, then gradually decrease
            band_weights = [1.0 - 0.7 * (i / (num_bands - 1)) for i in range(num_bands)]
        
        self.register_buffer('band_weights', torch.tensor(band_weights))
        
    def forward(self, pred, target):
        """
        Calculate band-wise frequency domain loss
        
        Args:
            pred: Predicted HSI [B, C, H, W]
            target: Target HSI [B, C, H, W]
            
        Returns:
            Weighted loss in frequency domain
        """
        # Transform to frequency domain
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        
        # Calculate magnitude
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Get dimensions
        B, C, H, half_W = pred_mag.shape
        
        # Create frequency grid
        freq_h = torch.fft.fftfreq(H, device=pred.device)[:, None]
        freq_w = torch.fft.rfftfreq(pred.shape[3], device=pred.device)[None, :]
        
        # Calculate distance from DC (center)
        freq_dist = torch.sqrt(freq_h**2 + freq_w**2)
        
        # Normalize distance to [0, 1]
        max_dist = torch.sqrt(torch.tensor(0.5**2 + 0.5**2, device=pred.device))
        norm_dist = freq_dist / max_dist
        
        # Calculate band boundaries (logarithmic spacing for better emphasis on low frequencies)
        log_bands = torch.logspace(-3, 0, self.num_bands + 1, device=pred.device)
        
        # Initialize total loss
        total_loss = 0.0
        
        # Calculate loss for each frequency band
        for i in range(self.num_bands):
            # Create band mask
            band_mask = ((norm_dist >= log_bands[i]) & (norm_dist < log_bands[i+1])).float()
            band_mask = band_mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
            
            # Mask magnitude
            pred_band = pred_mag * band_mask
            target_band = target_mag * band_mask
            
            # Calculate MSE for this band
            band_loss = F.mse_loss(pred_band, target_band)
            
            # Apply weight and add to total
            total_loss += self.band_weights[i] * band_loss
        
        return total_loss


class CombinedSpectralLoss(nn.Module):
    """
    Combined loss using both frequency and wavelet domain approaches
    """
    def __init__(self, in_channels=31, wavelet_weight=0.5, freq_weight=0.5):
        super().__init__()
        from losses.wavelet_loss import WaveletLoss
        
        self.wavelet_loss = WaveletLoss(in_channels=in_channels)
        self.frequency_loss = FrequencyDomainLoss()
        self.wavelet_weight = wavelet_weight
        self.freq_weight = freq_weight
        
    def forward(self, pred, target):
        """
        Calculate combined wavelet and frequency domain loss
        
        Args:
            pred: Predicted HSI [B, C, H, W]
            target: Target HSI [B, C, H, W]
            
        Returns:
            Combined loss
        """
        w_loss = self.wavelet_loss(pred, target)
        f_loss = self.frequency_loss(pred, target)
        
        return self.wavelet_weight * w_loss + self.freq_weight * f_loss
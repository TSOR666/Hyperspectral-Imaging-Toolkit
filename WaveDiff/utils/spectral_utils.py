import torch
import torch.nn.functional as F
import torch.fft
import numpy as np
from scipy.stats import wasserstein_distance

def cycle_consistency_loss(rgb_original, rgb_reconstructed, mask=None):
    """
    Calculate cycle consistency loss between original RGB and reconstructed RGB
    
    Args:
        rgb_original: Original RGB input [B, 3, H, W]
        rgb_reconstructed: Reconstructed RGB from HSI [B, 3, H, W]
        mask: Optional mask to ignore certain regions [B, 1, H, W] or [B, 3, H, W]
    
    Returns:
        Cycle consistency loss value
    """
    if mask is not None:
        # If mask has only one channel, expand to match RGB
        if mask.shape[1] == 1:
            mask = mask.expand(-1, 3, -1, -1)
        
        # Apply mask to inputs
        rgb_original = rgb_original * mask
        rgb_reconstructed = rgb_reconstructed * mask
        
        # Calculate L1 loss on masked regions
        loss = F.l1_loss(rgb_original, rgb_reconstructed, reduction='sum')
        # Normalize by number of valid pixels
        num_valid_pixels = torch.sum(mask) + 1e-8
        loss = loss / num_valid_pixels
    else:
        # Regular L1 loss
        loss = F.l1_loss(rgb_original, rgb_reconstructed)
    
    return loss

def spectral_angular_mapper(pred_hsi, target_hsi, mask=None, eps=1e-8):
    """
    Calculate Spectral Angular Mapper (SAM) metric between predicted and target HSI
    
    Args:
        pred_hsi: Predicted HSI [B, C, H, W]
        target_hsi: Target HSI [B, C, H, W]
        mask: Optional mask to ignore certain regions [B, 1, H, W] or [B, C, H, W]
        eps: Small value to prevent division by zero
    
    Returns:
        SAM value (lower is better)
    """
    # Reshape to [B, C, H*W] for easier processing
    B, C, H, W = pred_hsi.shape
    pred_flat = pred_hsi.reshape(B, C, -1)
    target_flat = target_hsi.reshape(B, C, -1)
    
    # Calculate dot product along spectral dimension
    dot_product = torch.sum(pred_flat * target_flat, dim=1)
    
    # Calculate L2 norms
    pred_norm = torch.norm(pred_flat, dim=1)
    target_norm = torch.norm(target_flat, dim=1)
    
    # Calculate cosine similarity
    cos_sim = dot_product / (pred_norm * target_norm + eps)
    # Clamp to avoid numerical instability
    cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    
    # Calculate angular distance
    sam = torch.acos(cos_sim)
    
    if mask is not None:
        # Reshape mask to match output shape
        mask_flat = mask.reshape(B, -1) if mask.shape[1] == 1 else mask.reshape(B, C, -1).mean(dim=1)
        
        # Apply mask to SAM values
        sam = sam * mask_flat
        # Normalize by number of valid pixels
        num_valid_pixels = torch.sum(mask_flat, dim=1) + eps
        sam = torch.sum(sam, dim=1) / num_valid_pixels
    else:
        # Average over spatial dimensions
        sam = torch.mean(sam, dim=1)
    
    # Return mean over batch
    return torch.mean(sam)

def root_mean_square_error(pred_hsi, target_hsi, mask=None):
    """
    Calculate Root Mean Square Error (RMSE) between predicted and target HSI
    
    Args:
        pred_hsi: Predicted HSI [B, C, H, W]
        target_hsi: Target HSI [B, C, H, W]
        mask: Optional mask to ignore certain regions [B, 1, H, W] or [B, C, H, W]
    
    Returns:
        RMSE value (lower is better)
    """
    if mask is not None:
        # If mask has only one channel, expand to match HSI
        if mask.shape[1] == 1:
            mask = mask.expand(-1, pred_hsi.shape[1], -1, -1)
        
        # Apply mask to inputs
        pred_hsi = pred_hsi * mask
        target_hsi = target_hsi * mask
        
        # Calculate MSE on masked regions
        sq_diff = (pred_hsi - target_hsi) ** 2
        # Sum across all dimensions except batch
        sum_sq_diff = torch.sum(sq_diff, dim=[1, 2, 3])
        # Count valid pixels
        num_valid_pixels = torch.sum(mask, dim=[1, 2, 3]) + 1e-8
        # Calculate MSE for each sample in batch
        mse = sum_sq_diff / num_valid_pixels
    else:
        # Regular MSE
        mse = F.mse_loss(pred_hsi, target_hsi, reduction='none').mean(dim=[1, 2, 3])
    
    # Take square root for RMSE
    rmse = torch.sqrt(mse)
    
    # Return mean over batch
    return torch.mean(rmse)

def peak_signal_to_noise_ratio(pred_hsi, target_hsi, mask=None, data_range=None, eps=1e-8):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between predicted and target HSI
    
    Args:
        pred_hsi: Predicted HSI [B, C, H, W]
        target_hsi: Target HSI [B, C, H, W]
        mask: Optional mask to ignore certain regions [B, 1, H, W] or [B, C, H, W]
        data_range: Optional explicit data range of the input image. When None, the
            range is inferred from the valid target pixels.
    
    Returns:
        PSNR value (higher is better)
    """
    # Determine appropriate data range when not provided
    if data_range is None:
        if mask is not None:
            expanded_mask = mask
            if mask.shape[1] == 1:
                expanded_mask = mask.expand(-1, target_hsi.shape[1], -1, -1)

            valid = expanded_mask > 0
            if torch.any(valid):
                target_vals = target_hsi[valid]
            else:
                target_vals = target_hsi.reshape(-1)
        else:
            target_vals = target_hsi.reshape(-1)

        target_max = torch.max(target_vals) if target_vals.numel() > 0 else torch.tensor(1.0, device=target_hsi.device, dtype=target_hsi.dtype)
        target_min = torch.min(target_vals) if target_vals.numel() > 0 else torch.tensor(0.0, device=target_hsi.device, dtype=target_hsi.dtype)

        dynamic_range = target_max - target_min
        if not torch.isfinite(dynamic_range) or dynamic_range.item() <= 0:
            data_range = torch.tensor(1.0, device=target_hsi.device, dtype=target_hsi.dtype)
        else:
            data_range = dynamic_range
    else:
        data_range = torch.as_tensor(data_range, device=target_hsi.device, dtype=target_hsi.dtype)

    data_range = torch.clamp(data_range, min=eps)

    # Calculate MSE
    if mask is not None:
        # If mask has only one channel, expand to match HSI
        if mask.shape[1] == 1:
            mask = mask.expand(-1, pred_hsi.shape[1], -1, -1)
        
        # Apply mask to inputs
        pred_masked = pred_hsi * mask
        target_masked = target_hsi * mask
        
        # Calculate MSE on masked regions
        sq_diff = (pred_masked - target_masked) ** 2
        # Sum across all dimensions except batch
        sum_sq_diff = torch.sum(sq_diff, dim=[1, 2, 3])
        # Count valid pixels
        num_valid_pixels = torch.sum(mask, dim=[1, 2, 3]) + 1e-8
        # Calculate MSE for each sample in batch
        mse = sum_sq_diff / num_valid_pixels
    else:
        # Regular MSE
        mse = F.mse_loss(pred_hsi, target_hsi, reduction='none').mean(dim=[1, 2, 3])
    
    # Calculate PSNR
    psnr = 20 * torch.log10(data_range / torch.sqrt(torch.clamp(mse, min=eps)))

    # Return mean over batch
    return torch.mean(psnr)

def spectral_information_divergence(pred_hsi, target_hsi, mask=None, eps=1e-8):
    """
    Calculate Spectral Information Divergence (SID) between predicted and target HSI
    
    Args:
        pred_hsi: Predicted HSI [B, C, H, W]
        target_hsi: Target HSI [B, C, H, W]
        mask: Optional mask to ignore certain regions [B, 1, H, W] or [B, C, H, W]
        eps: Small value to prevent numerical issues
    
    Returns:
        SID value (lower is better)
    """
    # Reshape to [B, C, H*W] for easier processing
    B, C, H, W = pred_hsi.shape
    pred_flat = pred_hsi.reshape(B, C, -1)
    target_flat = target_hsi.reshape(B, C, -1)
    
    # Ensure non-negative values and normalize along spectral dimension
    pred_flat = torch.clamp(pred_flat, min=eps)
    target_flat = torch.clamp(target_flat, min=eps)
    
    pred_sum = torch.sum(pred_flat, dim=1, keepdim=True)
    target_sum = torch.sum(target_flat, dim=1, keepdim=True)
    
    pred_norm = pred_flat / pred_sum
    target_norm = target_flat / target_sum
    
    # Calculate SID components
    term1 = pred_norm * torch.log(pred_norm / target_norm)
    term2 = target_norm * torch.log(target_norm / pred_norm)
    
    # Sum along spectral dimension
    sid = torch.sum(term1 + term2, dim=1)
    
    if mask is not None:
        # Reshape mask to match output shape
        mask_flat = mask.reshape(B, -1) if mask.shape[1] == 1 else mask.reshape(B, C, -1).mean(dim=1)
        
        # Apply mask to SID values
        sid = sid * mask_flat
        # Normalize by number of valid pixels
        num_valid_pixels = torch.sum(mask_flat, dim=1) + eps
        sid = torch.sum(sid, dim=1) / num_valid_pixels
    else:
        # Average over spatial dimensions
        sid = torch.mean(sid, dim=1)
    
    # Return mean over batch
    return torch.mean(sid)

def wasserstein_spectral_loss(pred_hsi, target_hsi, mask=None):
    """
    Calculate Wasserstein distance between predicted and target HSI spectra
    
    Args:
        pred_hsi: Predicted HSI [B, C, H, W]
        target_hsi: Target HSI [B, C, H, W]
        mask: Optional mask to ignore certain regions [B, 1, H, W]
    
    Returns:
        Wasserstein distance (lower is better)
    """
    # Convert to numpy for easier calculation with scipy
    # Note: This is not efficient for GPU training, a pure PyTorch implementation would be better
    pred_np = pred_hsi.detach().cpu().numpy()
    target_np = target_hsi.detach().cpu().numpy()
    
    B, C, H, W = pred_np.shape
    total_distance = 0.0
    
    # If mask is provided, convert to numpy
    mask_np = None
    if mask is not None:
        mask_np = mask.detach().cpu().numpy()
        if mask_np.shape[1] == 1:
            mask_np = np.repeat(mask_np, pred_np.shape[1], axis=1)
    
    # Calculate Wasserstein distance for each pixel and average
    for b in range(B):
        batch_distance = 0.0
        count = 0
        
        for h in range(H):
            for w in range(W):
                # Check mask if provided
                if mask_np is None or mask_np[b, 0, h, w] > 0.5:
                    # Get spectra at this pixel
                    pred_spectrum = pred_np[b, :, h, w]
                    target_spectrum = target_np[b, :, h, w]
                    
                    # Calculate Wasserstein distance
                    dist = wasserstein_distance(pred_spectrum, target_spectrum)
                    batch_distance += dist
                    count += 1
        
        # Average distance for this batch sample
        if count > 0:
            total_distance += batch_distance / count
    
    # Return average over batch
    return total_distance / B

def frequency_domain_loss(pred_hsi, target_hsi, low_freq_weight=1.0, high_freq_weight=0.5, mask=None):
    """
    Calculate loss in frequency domain using FFT
    
    Args:
        pred_hsi: Predicted HSI [B, C, H, W]
        target_hsi: Target HSI [B, C, H, W]
        low_freq_weight: Weight for low frequency components
        high_freq_weight: Weight for high frequency components
        mask: Optional mask for spatial domain [B, 1, H, W] or [B, C, H, W]
    
    Returns:
        Weighted loss in frequency domain
    """
    # Apply spatial mask if provided
    if mask is not None:
        # If mask has only one channel, expand to match HSI
        if mask.shape[1] == 1:
            mask = mask.expand(-1, pred_hsi.shape[1], -1, -1)
        
        # Apply mask to inputs
        pred_hsi = pred_hsi * mask
        target_hsi = target_hsi * mask
    
    # Transform to frequency domain
    pred_fft = torch.fft.rfft2(pred_hsi, norm="ortho")
    target_fft = torch.fft.rfft2(target_hsi, norm="ortho")
    
    # Calculate magnitude and phase
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    pred_phase = torch.angle(pred_fft)
    target_phase = torch.angle(target_fft)
    
    # Generate frequency weights
    B, C, H, half_W = pred_mag.shape
    
    # Create frequency grid
    freq_h = torch.fft.fftfreq(H, device=pred_hsi.device)[:, None]
    freq_w = torch.fft.rfftfreq(pred_hsi.shape[3], device=pred_hsi.device)[None, :]
    
    # Calculate distance from DC (center)
    freq_dist = torch.sqrt(freq_h**2 + freq_w**2)
    
    # Normalize distance to [0, 1]
    max_dist = torch.sqrt(torch.tensor(0.5**2 + 0.5**2, device=pred_hsi.device))
    norm_dist = freq_dist / max_dist
    
    # Create weights that emphasize low frequencies
    # Weights decay from low_freq_weight to high_freq_weight as frequency increases
    weights = low_freq_weight - (low_freq_weight - high_freq_weight) * norm_dist
    
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

def mean_relative_absolute_error(pred_hsi, target_hsi, mask=None, eps=1e-6):
    """
    Calculate Mean Relative Absolute Error (MRAE) between predicted and target HSI
    
    Args:
        pred_hsi: Predicted HSI [B, C, H, W]
        target_hsi: Target HSI [B, C, H, W]
        mask: Optional mask to ignore certain regions [B, 1, H, W] or [B, C, H, W]
        eps: Small constant to avoid division by zero
        
    Returns:
        MRAE value (lower is better)
    """
    # Calculate absolute difference
    abs_diff = torch.abs(pred_hsi - target_hsi)
    
    # Calculate absolute target values with epsilon for stability
    abs_target = torch.abs(target_hsi) + eps
    
    # Calculate relative absolute error
    rel_abs_error = abs_diff / abs_target
    
    if mask is not None:
        # If mask has only one channel, expand to match HSI
        if mask.shape[1] == 1:
            mask = mask.expand(-1, pred_hsi.shape[1], -1, -1)
        
        # Apply mask
        rel_abs_error = rel_abs_error * mask
        
        # Calculate mean over valid regions
        valid_pixels = torch.sum(mask, dim=[1, 2, 3])
        mrae_per_sample = torch.sum(rel_abs_error, dim=[1, 2, 3]) / (valid_pixels + eps)
    else:
        # Calculate mean over all dimensions except batch
        mrae_per_sample = torch.mean(rel_abs_error, dim=[1, 2, 3])
    
    # Return mean over batch
    return torch.mean(mrae_per_sample)

def calculate_metrics(pred_hsi, target_hsi, mask=None):
    """
    Calculate multiple metrics for HSI reconstruction evaluation
    
    Args:
        pred_hsi: Predicted HSI [B, C, H, W]
        target_hsi: Target HSI [B, C, H, W]
        mask: Optional mask to ignore certain regions [B, 1, H, W] or [B, C, H, W]
    
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        metrics = {}
        
        # Calculate RMSE
        metrics['rmse'] = root_mean_square_error(pred_hsi, target_hsi, mask).item()
        
        # Calculate PSNR
        metrics['psnr'] = peak_signal_to_noise_ratio(pred_hsi, target_hsi, mask).item()
        
        # Calculate SAM
        metrics['sam'] = spectral_angular_mapper(pred_hsi, target_hsi, mask).item()
        
        # Calculate SID
        metrics['sid'] = spectral_information_divergence(pred_hsi, target_hsi, mask).item()
        
        # Calculate MRAE (Mean Relative Absolute Error)
        metrics['mrae'] = mean_relative_absolute_error(pred_hsi, target_hsi, mask).item()
        
        # Note: Wasserstein is computed on CPU and can be slow for large datasets
        # Uncomment if needed
        # metrics['wasserstein'] = wasserstein_spectral_loss(pred_hsi, target_hsi, mask)
        
        return metrics
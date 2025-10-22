import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def visualize_rgb_hsi(rgb, hsi, save_path=None, title=None, figsize=(12, 8)):
    """
    Visualize RGB and selected HSI bands
    
    Args:
        rgb: RGB image tensor [C, H, W]
        hsi: HSI image tensor [C, H, W]
        save_path: Optional path to save visualization
        title: Optional title for the plot
        figsize: Figure size (width, height)
    """
    # Convert tensors to numpy
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if isinstance(hsi, torch.Tensor):
        hsi = hsi.detach().cpu().numpy()
    
    # Ensure channel-first format
    if rgb.shape[0] != 3:
        rgb = np.transpose(rgb, (2, 0, 1))
    
    # Normalize RGB to [0, 1]
    rgb = np.clip(rgb, 0, 1)
    
    # Transpose to channel-last for plotting (H, W, C)
    rgb = np.transpose(rgb, (1, 2, 0))
    
    # Number of HSI bands to display
    num_hsi_bands = min(5, hsi.shape[0])
    band_indices = np.linspace(0, hsi.shape[0]-1, num_hsi_bands, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(1, num_hsi_bands + 1, figsize=figsize)
    
    # Plot RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis('off')
    
    # Plot selected HSI bands
    for i, band_idx in enumerate(band_indices):
        band = hsi[band_idx]
        
        # Normalize band to [0, 1]
        band_min, band_max = band.min(), band.max()
        if band_max > band_min:
            band = (band - band_min) / (band_max - band_min)
        
        # Plot
        im = axes[i+1].imshow(band, cmap='viridis')
        axes[i+1].set_title(f"Band {band_idx}")
        axes[i+1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)
    
    # Set title if provided
    if title:
        plt.suptitle(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_wavelet_coeffs(coeffs, save_path=None, band_idx=0, title=None, figsize=(16, 4)):
    """
    Visualize wavelet coefficients for a specific band
    
    Args:
        coeffs: Wavelet coefficients tensor [B, C, 4, H, W]
        save_path: Optional path to save visualization
        band_idx: Index of spectral band to visualize
        title: Optional title for the plot
        figsize: Figure size (width, height)
    """
    # Convert to numpy if tensor
    if isinstance(coeffs, torch.Tensor):
        coeffs = coeffs.detach().cpu().numpy()
    
    # Extract components for the selected band
    ll = coeffs[0, band_idx, 0]  # Approximation
    lh = coeffs[0, band_idx, 1]  # Horizontal detail
    hl = coeffs[0, band_idx, 2]  # Vertical detail
    hh = coeffs[0, band_idx, 3]  # Diagonal detail
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Component names and data
    names = ['LL (Approximation)', 'LH (Horizontal)', 'HL (Vertical)', 'HH (Diagonal)']
    components = [ll, lh, hl, hh]
    
    # Custom colormaps for better visualization
    cmaps = ['viridis', 'coolwarm', 'coolwarm', 'coolwarm']
    
    # Plot each component
    for i, (component, name, cmap) in enumerate(zip(components, names, cmaps)):
        # Normalize for better visualization
        vmin, vmax = np.percentile(component, [2, 98])
        if i > 0:  # For detail coefficients, use symmetric range
            vlim = max(abs(vmin), abs(vmax))
            vmin, vmax = -vlim, vlim
        
        im = axes[i].imshow(component, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_title(name)
        axes[i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Set title if provided
    if title:
        plt.suptitle(title)
    else:
        plt.suptitle(f"Wavelet Coefficients for Band {band_idx}")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_thresholding(original, thresholded, save_path=None, band_idx=0, component_idx=1, 
                          title=None, figsize=(10, 4)):
    """
    Visualize effect of thresholding on wavelet coefficients
    
    Args:
        original: Original wavelet coefficients [B, C, 4, H, W]
        thresholded: Thresholded wavelet coefficients [B, C, 4, H, W]
        save_path: Optional path to save visualization
        band_idx: Index of spectral band to visualize
        component_idx: Index of wavelet component to visualize (0=LL, 1=LH, 2=HL, 3=HH)
        title: Optional title for the plot
        figsize: Figure size (width, height)
    """
    # Convert to numpy if tensor
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(thresholded, torch.Tensor):
        thresholded = thresholded.detach().cpu().numpy()
    
    # Extract component for the selected band
    orig_comp = original[0, band_idx, component_idx]
    thresh_comp = thresholded[0, band_idx, component_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Component name
    component_names = ['LL (Approximation)', 'LH (Horizontal)', 'HL (Vertical)', 'HH (Diagonal)']
    component_name = component_names[component_idx]
    
    # Determine color range for consistent visualization
    vmin = min(np.percentile(orig_comp, 2), np.percentile(thresh_comp, 2))
    vmax = max(np.percentile(orig_comp, 98), np.percentile(thresh_comp, 98))
    
    if component_idx > 0:  # For detail coefficients, use symmetric range
        vlim = max(abs(vmin), abs(vmax))
        vmin, vmax = -vlim, vlim
    
    # Plot original
    im1 = axes[0].imshow(orig_comp, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Original {component_name}")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot thresholded
    im2 = axes[1].imshow(thresh_comp, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Thresholded {component_name}")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot difference
    diff = orig_comp - thresh_comp
    im3 = axes[2].imshow(diff, cmap='RdBu_r')
    axes[2].set_title("Difference")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Set title if provided
    if title:
        plt.suptitle(title)
    else:
        plt.suptitle(f"Thresholding Effect on Band {band_idx}, {component_name}")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_spectral_signature(hsi, pixel_coords, wavelengths=None, save_path=None, title=None, figsize=(10, 6)):
    """
    Visualize spectral signature at specific pixel location(s)
    
    Args:
        hsi: HSI image tensor/array [C, H, W]
        pixel_coords: List of (row, col) coordinates or single tuple
        wavelengths: Optional wavelengths corresponding to bands
        save_path: Optional path to save visualization
        title: Optional title for the plot
        figsize: Figure size (width, height)
    """
    # Convert to numpy if tensor
    if isinstance(hsi, torch.Tensor):
        hsi = hsi.detach().cpu().numpy()
    
    # Ensure channel first format
    if hsi.ndim == 4:
        hsi = hsi[0]  # Take first item if batched
    
    # Convert single coordinate to list
    if isinstance(pixel_coords, tuple):
        pixel_coords = [pixel_coords]
    
    # Create default wavelengths if not provided
    if wavelengths is None:
        wavelengths = np.arange(hsi.shape[0])
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot spectral signature for each pixel
    for i, (row, col) in enumerate(pixel_coords):
        spectrum = hsi[:, row, col]
        plt.plot(wavelengths, spectrum, '-o', label=f"Pixel ({row}, {col})")
    
    # Set labels
    plt.xlabel("Wavelength")
    plt.ylabel("Intensity")
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title("Spectral Signatures")
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(alpha=0.3)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_reconstruction_comparison(rgb, original_hsi, predicted_hsi, save_path=None, 
                                      error_metric=None, figsize=(14, 10)):
    """
    Create comprehensive visualization comparing ground truth and predicted HSI
    
    Args:
        rgb: RGB image tensor [C, H, W]
        original_hsi: Ground truth HSI tensor [C, H, W]
        predicted_hsi: Predicted HSI tensor [C, H, W]
        save_path: Optional path to save visualization
        error_metric: Optional dictionary with error metrics (PSNR, SAM, RMSE, etc.)
        figsize: Figure size (width, height)
    """
    # Convert to numpy if tensor
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if isinstance(original_hsi, torch.Tensor):
        original_hsi = original_hsi.detach().cpu().numpy()
    if isinstance(predicted_hsi, torch.Tensor):
        predicted_hsi = predicted_hsi.detach().cpu().numpy()
    
    # Ensure correct shape (channel-first)
    if rgb.shape[0] != 3:
        rgb = np.transpose(rgb, (2, 0, 1))
    
    # Transpose RGB to channel-last for plotting
    rgb = np.transpose(rgb, (1, 2, 0))
    
    # Normalize RGB to [0, 1]
    rgb = np.clip(rgb, 0, 1)
    
    # Number of HSI bands to display
    num_bands = min(3, original_hsi.shape[0])
    band_indices = np.linspace(0, original_hsi.shape[0]-1, num_bands, dtype=int)
    
    # Create figure with 3 rows and num_bands+1 columns
    fig, axes = plt.subplots(3, num_bands+1, figsize=figsize)
    
    # Row 0: RGB image and original HSI bands
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB Input")
    axes[0, 0].axis('off')
    
    # Row 1: Will be the predicted HSI bands
    axes[1, 0].axis('off')
    
    # Row 2: Error maps
    axes[2, 0].axis('off')
    
    # Plot each selected band
    for i, band_idx in enumerate(band_indices):
        # Original HSI band
        orig_band = original_hsi[band_idx]
        orig_band_norm = (orig_band - np.min(orig_band)) / (np.max(orig_band) - np.min(orig_band) + 1e-8)
        axes[0, i+1].imshow(orig_band_norm, cmap='viridis')
        axes[0, i+1].set_title(f"GT Band {band_idx}")
        axes[0, i+1].axis('off')
        
        # Predicted HSI band
        pred_band = predicted_hsi[band_idx]
        pred_band_norm = (pred_band - np.min(pred_band)) / (np.max(pred_band) - np.min(pred_band) + 1e-8)
        axes[1, i+1].imshow(pred_band_norm, cmap='viridis')
        axes[1, i+1].set_title(f"Pred Band {band_idx}")
        axes[1, i+1].axis('off')
        
        # Error map
        error = np.abs(orig_band - pred_band)
        error_norm = error / np.max(error + 1e-8)
        im = axes[2, i+1].imshow(error_norm, cmap='hot')
        axes[2, i+1].set_title(f"Error Band {band_idx}")
        axes[2, i+1].axis('off')
        
        # Add colorbar to error map
        plt.colorbar(im, ax=axes[2, i+1], fraction=0.046, pad=0.04)
    
    # Display overall error if provided
    if error_metric is not None:
        error_text = "Error Metrics:\n"
        for k, v in error_metric.items():
            error_text += f"{k}: {v:.4f}\n"
        
        axes[1, 0].text(0.5, 0.5, error_text, 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[1, 0].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
    
    # Calculate mean error across all bands
    mean_error = np.mean(np.abs(original_hsi - predicted_hsi), axis=0)
    mean_error_norm = mean_error / np.max(mean_error + 1e-8)
    
    # Plot mean error in first column of row 2
    im = axes[2, 0].imshow(mean_error_norm, cmap='hot')
    axes[2, 0].set_title("Mean Error")
    axes[2, 0].axis('off')
    plt.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_training_progress(history, save_path=None, figsize=(12, 8)):
    """
    Visualize training progress from history dictionary
    
    Args:
        history: Dictionary containing loss history
        save_path: Optional path to save visualization
        figsize: Figure size (width, height)
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Determine which metrics are available
    available_metrics = []
    for key in history.keys():
        if 'loss' in key.lower() and len(history[key]) > 0:
            available_metrics.append(key)
    
    # Plot up to 4 metrics
    metrics_to_plot = available_metrics[:4]
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Get train and validation data if available
        train_data = history.get(metric, [])
        val_metric = f"val_{metric}" if f"val_{metric}" in history else None
        val_data = history.get(val_metric, []) if val_metric else []
        
        # Plot training data
        ax.plot(train_data, 'b-', label=f'Training {metric}')
        
        # Plot validation data if available
        if len(val_data) > 0:
            ax.plot(val_data, 'r-', label=f'Validation {metric}')
        
        # Set labels
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} over Epochs')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(metrics_to_plot), 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def create_false_color_image(hsi, band_indices=[0, 10, 20], gamma=1.0, save_path=None, figsize=(8, 8)):
    """
    Create false color image from HSI data by selecting 3 bands to display as RGB
    
    Args:
        hsi: HSI image tensor/array [C, H, W]
        band_indices: List of 3 band indices to use as [R, G, B]
        gamma: Optional gamma correction
        save_path: Optional path to save visualization
        figsize: Figure size (width, height)
    """
    # Convert to numpy if tensor
    if isinstance(hsi, torch.Tensor):
        hsi = hsi.detach().cpu().numpy()
    
    # Ensure channel first format
    if hsi.ndim == 4:
        hsi = hsi[0]  # Take first item if batched
    
    # Extract the three bands
    r_band = hsi[band_indices[0]]
    g_band = hsi[band_indices[1]]
    b_band = hsi[band_indices[2]]
    
    # Normalize each band to [0, 1]
    r_norm = (r_band - np.min(r_band)) / (np.max(r_band) - np.min(r_band) + 1e-8)
    g_norm = (g_band - np.min(g_band)) / (np.max(g_band) - np.min(g_band) + 1e-8)
    b_norm = (b_band - np.min(b_band)) / (np.max(b_band) - np.min(b_band) + 1e-8)
    
    # Apply gamma correction
    if gamma != 1.0:
        r_norm = np.power(r_norm, gamma)
        g_norm = np.power(g_norm, gamma)
        b_norm = np.power(b_norm, gamma)
    
    # Stack to create RGB image
    rgb = np.stack([r_norm, g_norm, b_norm], axis=2)
    
    # Clip to [0, 1]
    rgb = np.clip(rgb, 0, 1)
    
    # Create figure
    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.title(f"False Color Image (Bands {band_indices[0]}, {band_indices[1]}, {band_indices[2]})")
    plt.axis('off')
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    return rgb
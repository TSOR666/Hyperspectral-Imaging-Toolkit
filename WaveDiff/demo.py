#!/usr/bin/env python3
"""
Demo script to showcase the HSI Wavelet Diffusion model
Demonstrates inference on sample images and compares different model variants
"""

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from PIL import Image

# Import model types
from models.base_model import HSILatentDiffusionModel
from models.wavelet_model import WaveletHSILatentDiffusionModel
from models.adaptive_model import AdaptiveWaveletHSILatentDiffusionModel

# Import utilities


def load_sample_image(image_path, size=256):
    """Load and preprocess a sample image"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    # transform() returns a Tensor after ToTensor() is applied
    image_tensor: torch.Tensor = transform(image)  # type: ignore[assignment]
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    return image_tensor


def create_sample_models(device, latent_dim=64):
    """Create sample models for demonstration"""
    # Create Base LDM model
    base_model = HSILatentDiffusionModel(
        latent_dim=latent_dim,
        out_channels=31,
        timesteps=1000,
        use_batchnorm=True
    ).to(device)
    
    # Create Wavelet LDM model
    wavelet_model = WaveletHSILatentDiffusionModel(
        latent_dim=latent_dim,
        out_channels=31,
        timesteps=1000,
        use_batchnorm=True
    ).to(device)
    
    # Create Adaptive Wavelet LDM model
    adaptive_model = AdaptiveWaveletHSILatentDiffusionModel(
        latent_dim=latent_dim,
        out_channels=31,
        timesteps=1000,
        use_batchnorm=True,
        threshold_method='soft',
        init_threshold=0.1,
        trainable_threshold=True
    ).to(device)
    
    return {
        'base': base_model,
        'wavelet': wavelet_model,
        'adaptive_wavelet': adaptive_model
    }


def load_pretrained_models(checkpoint_dir, device):
    """Load pretrained models if available"""
    models = {}
    
    # Check for base model checkpoint
    base_path = os.path.join(checkpoint_dir, 'base_model.pt')
    if os.path.exists(base_path):
        print(f"Loading base model from {base_path}")
        checkpoint = torch.load(base_path, map_location=device)
        base_model = HSILatentDiffusionModel(
            latent_dim=64,
            out_channels=31,
            timesteps=1000,
            use_batchnorm=True
        ).to(device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        base_model.eval()
        models['base'] = base_model
    
    # Check for wavelet model checkpoint
    wavelet_path = os.path.join(checkpoint_dir, 'wavelet_model.pt')
    if os.path.exists(wavelet_path):
        print(f"Loading wavelet model from {wavelet_path}")
        checkpoint = torch.load(wavelet_path, map_location=device)
        wavelet_model = WaveletHSILatentDiffusionModel(
            latent_dim=64,
            out_channels=31,
            timesteps=1000,
            use_batchnorm=True
        ).to(device)
        wavelet_model.load_state_dict(checkpoint['model_state_dict'])
        wavelet_model.eval()
        models['wavelet'] = wavelet_model
    
    # Check for adaptive wavelet model checkpoint
    adaptive_path = os.path.join(checkpoint_dir, 'adaptive_wavelet_model.pt')
    if os.path.exists(adaptive_path):
        print(f"Loading adaptive wavelet model from {adaptive_path}")
        checkpoint = torch.load(adaptive_path, map_location=device)
        adaptive_model = AdaptiveWaveletHSILatentDiffusionModel(
            latent_dim=64,
            out_channels=31,
            timesteps=1000,
            use_batchnorm=True
        ).to(device)
        adaptive_model.load_state_dict(checkpoint['model_state_dict'])
        adaptive_model.eval()
        models['adaptive_wavelet'] = adaptive_model
    
    return models


def run_inference(model, rgb_tensor, device, sampling_steps=20, apply_adaptive_threshold=True):
    """Run inference with a model"""
    # Move tensor to device
    rgb_tensor = rgb_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        # Check if model has dedicated inference method
        if hasattr(model, 'rgb_to_hsi'):
            # For adaptive models, use the method that applies thresholding
            if isinstance(model, AdaptiveWaveletHSILatentDiffusionModel):
                hsi_output = model.rgb_to_hsi(
                    rgb_tensor, 
                    sampling_steps=sampling_steps,
                    apply_adaptive_threshold=apply_adaptive_threshold
                )
            else:
                hsi_output = model.rgb_to_hsi(rgb_tensor, sampling_steps=sampling_steps)
        else:
            # Manual inference for other models
            latent = model.encode(rgb_tensor)
            
            # Sample latent using DPM Solver
            sampled_latent = model.dpm_ot.sample(
                latent.shape,
                latent.device,
                conditioning=latent,
                use_dpm_solver=True,
                steps=sampling_steps
            )
            
            # Decode to HSI
            hsi_output = model.decode(sampled_latent)
    
    return hsi_output


def visualize_comparison(rgb_tensor, hsi_outputs, output_dir, filename='model_comparison'):
    """Visualize comparison between different models"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert RGB tensor to numpy
    rgb_np = rgb_tensor.squeeze(0).cpu().numpy()
    rgb_np = np.transpose(rgb_np, (1, 2, 0))  # Convert to HWC format
    rgb_np = (rgb_np * 0.5 + 0.5).clip(0, 1)  # Denormalize
    
    # Number of models
    num_models = len(hsi_outputs)
    model_names = list(hsi_outputs.keys())
    
    # Create figure for comparison
    fig, axes = plt.subplots(num_models + 1, 5, figsize=(20, 4 * (num_models + 1)))
    
    # Plot RGB input
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title("RGB Input")
    axes[0, 0].axis('off')
    
    # Hide other plots in first row
    for i in range(1, 5):
        axes[0, i].axis('off')
    
    # Select bands to visualize
    num_bands = hsi_outputs[model_names[0]].shape[1]
    band_indices = [0, num_bands//4, num_bands//2, 3*num_bands//4, num_bands-1]
    
    # Plot each model's output
    for i, model_name in enumerate(model_names):
        model_idx = i + 1
        hsi = hsi_outputs[model_name].squeeze(0).cpu().numpy()
        
        # Create false color image
        false_color = np.stack([
            normalize_band(hsi[0]),  # Red channel
            normalize_band(hsi[num_bands//2]),  # Green channel
            normalize_band(hsi[-1])  # Blue channel
        ], axis=2)
        
        # Plot false color
        axes[model_idx, 0].imshow(false_color)
        axes[model_idx, 0].set_title(f"{model_name}\nFalse Color")
        axes[model_idx, 0].axis('off')
        
        # Plot selected bands
        for j, band_idx in enumerate(band_indices[1:]):
            band = hsi[band_idx]
            band_norm = normalize_band(band)
            axes[model_idx, j+1].imshow(band_norm, cmap='viridis')
            axes[model_idx, j+1].set_title(f"Band {band_idx}")
            axes[model_idx, j+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=150)
    plt.close(fig)
    
    # Create spectral signature comparison
    compare_spectral_signatures(hsi_outputs, output_dir)


def normalize_band(band):
    """Normalize a band to [0, 1] range"""
    band_min, band_max = band.min(), band.max()
    if band_max > band_min:
        return (band - band_min) / (band_max - band_min)
    return band


def compare_spectral_signatures(hsi_outputs, output_dir, filename='spectral_signatures'):
    """Compare spectral signatures from different models"""
    # Get center pixel coordinates
    model_names = list(hsi_outputs.keys())
    hsi = hsi_outputs[model_names[0]].squeeze(0).cpu().numpy()
    h, w = hsi.shape[1:]
    center_point = (h//2, w//2)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Approximate wavelengths
    wavelengths = np.linspace(400, 700, hsi.shape[0])
    
    # Plot spectral signature for each model
    for model_name, hsi_output in hsi_outputs.items():
        hsi = hsi_output.squeeze(0).cpu().numpy()
        spectrum = hsi[:, center_point[0], center_point[1]]
        plt.plot(wavelengths, spectrum, '-o', label=model_name, linewidth=2, markersize=6)
    
    plt.xlabel("Wavelength (nm)", fontsize=14)
    plt.ylabel("Reflectance", fontsize=14)
    plt.title("Spectral Signature Comparison at Center Pixel", fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="HSI Wavelet Diffusion Demo")
    
    # Add arguments
    parser.add_argument('--image', type=str, default=None,
                        help='Path to sample image (or a directory containing sample images)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str, default='demo_results',
                        help='Directory to save results')
    parser.add_argument('--sampling_steps', type=int, default=20,
                        help='Number of sampling steps for inference')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='Use pretrained models if available')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for sample images
    sample_images = []
    if args.image:
        if os.path.isdir(args.image):
            # Get all image files in the directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for ext in image_extensions:
                sample_images.extend(list(Path(args.image).glob(f"*{ext}")))
        else:
            sample_images = [Path(args.image)]
    
    # Use default sample image if none provided
    if not sample_images:
        print("No sample images provided. Using a red square for demonstration.")
        # Create a simple red square image
        red_square = np.zeros((256, 256, 3), dtype=np.float32)
        red_square[:, :, 0] = 1.0  # Red channel
        
        # Convert to tensor
        sample_tensor = torch.from_numpy(np.transpose(red_square, (2, 0, 1)))
        sample_tensor = (sample_tensor * 2) - 1  # Scale to [-1, 1]
        sample_tensor = sample_tensor.unsqueeze(0)  # Add batch dimension
        
        sample_tensors = {'red_square': sample_tensor}
    else:
        # Load sample images
        sample_tensors = {}
        for img_path in sample_images:
            sample_tensors[img_path.stem] = load_sample_image(img_path)
    
    # Get models
    if args.use_pretrained:
        # Try to load pretrained models
        models = load_pretrained_models(args.checkpoint_dir, device)
        
        # If no pretrained models found, create sample models
        if not models:
            print("No pretrained models found. Creating sample models.")
            models = create_sample_models(device)
    else:
        # Create sample models
        models = create_sample_models(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference and visualize results for each sample
    for sample_name, rgb_tensor in sample_tensors.items():
        print(f"Processing sample: {sample_name}")
        
        # Run inference with each model
        hsi_outputs = {}
        for model_name, model in models.items():
            print(f"  Running inference with {model_name} model...")
            hsi_output = run_inference(
                model, rgb_tensor, device, 
                sampling_steps=args.sampling_steps,
                apply_adaptive_threshold=(model_name == 'adaptive_wavelet')
            )
            hsi_outputs[model_name] = hsi_output
        
        # Visualize comparison
        print(f"  Generating visualizations...")
        sample_output_dir = os.path.join(args.output_dir, sample_name)
        visualize_comparison(
            rgb_tensor, hsi_outputs, sample_output_dir, 
            filename=f"{sample_name}_comparison"
        )
    
    print(f"Demo completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Inference script for SHARP v3.2.2.
Loads a trained model checkpoint and performs inference on RGB images.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from sharp_v322_hardened import create_sharp_v32, SHARPv32Config


class SHARPInference:
    """SHARP v3.2.2 inference wrapper."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.model_size = config.model_size
            self.in_channels = config.in_channels
            self.out_channels = config.out_channels
            
            # Extract SHARP-specific parameters
            sharp_kwargs = {
                'sparse_sparsity_ratio': getattr(config, 'sparse_sparsity_ratio', 0.9),
                'rbf_centers_per_head': getattr(config, 'rbf_centers_per_head', 32),
                'sparse_k_cap': getattr(config, 'sparse_k_cap', 1024),
                'sparse_block_size': getattr(config, 'sparse_block_size', 2048),
                'sparse_q_block_size': getattr(config, 'sparse_q_block_size', 1024),
                'sparse_window_size': getattr(config, 'sparse_window_size', 49),
                'sparse_max_tokens': getattr(config, 'sparse_max_tokens', 8192),
                'key_rbf_mode': getattr(config, 'key_rbf_mode', 'mean'),
                'sparsemax_pad_value': getattr(config, 'sparsemax_pad_value', None),
                'ema_update_every': getattr(config, 'ema_update_every', 1),
            }
        else:
            # Default configuration
            print("No configuration found in checkpoint, using defaults")
            self.model_size = 'base'
            self.in_channels = 3
            self.out_channels = 31
            sharp_kwargs = {
                'sparse_sparsity_ratio': 0.9,
                'rbf_centers_per_head': 32,
                'sparse_k_cap': 1024,
                'sparse_block_size': 2048,
                'sparse_q_block_size': 1024,
                'sparse_window_size': 49,
                'sparse_max_tokens': 8192,
                'key_rbf_mode': 'mean',
                'sparsemax_pad_value': None,
                'ema_update_every': 1,
            }
        
        # Create model
        print(f"Creating SHARP {self.model_size} model...")
        self.model = create_sharp_v32(
            model_size=self.model_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            compile_model=False,  # Disable compilation for inference
            verbose=False,
            **sharp_kwargs
        )
        
        # Load weights
        model_state = checkpoint['model_state_dict']
        self.model.load_state_dict(model_state)
        
        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get model info
        if 'sharp_version' in checkpoint:
            print(f"SHARP version: {checkpoint['sharp_version']}")
        if 'best_mrae' in checkpoint:
            print(f"Best validation MRAE: {checkpoint['best_mrae']:.6f}")
        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch']}")
        
        print("Model loaded successfully!")
    
    @torch.no_grad()
    def predict(self, rgb_image: torch.Tensor, 
                patch_size: Optional[int] = None,
                overlap: int = 16) -> torch.Tensor:
        """
        Perform inference on RGB image
        
        Args:
            rgb_image: RGB image tensor of shape (3, H, W) or (B, 3, H, W)
            patch_size: If specified, process image in patches (for large images)
            overlap: Overlap between patches (only used if patch_size is specified)
            
        Returns:
            HSI prediction of shape (31, H, W) or (B, 31, H, W)
        """
        # Add batch dimension if needed
        single_image = rgb_image.dim() == 3
        if single_image:
            rgb_image = rgb_image.unsqueeze(0)
        
        B, C, H, W = rgb_image.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        
        # Move to device
        rgb_image = rgb_image.to(self.device)
        
        # Process full image or in patches
        if patch_size is None or (H <= patch_size and W <= patch_size):
            # Process full image
            hsi_pred = self.model(rgb_image)
        else:
            # Process in patches with overlap
            hsi_pred = self._predict_patches(rgb_image, patch_size, overlap)
        
        # Remove batch dimension if needed
        if single_image:
            hsi_pred = hsi_pred.squeeze(0)
        
        return hsi_pred
    
    def _predict_patches(self, rgb_image: torch.Tensor, 
                        patch_size: int, overlap: int) -> torch.Tensor:
        """Process large image in overlapping patches"""
        B, C, H, W = rgb_image.shape
        
        # Initialize output
        output = torch.zeros(B, self.out_channels, H, W, device=self.device)
        weight_map = torch.zeros(B, 1, H, W, device=self.device)
        
        # Create overlapping windows
        stride = patch_size - overlap
        
        for y in range(0, H - overlap, stride):
            for x in range(0, W - overlap, stride):
                # Define patch boundaries
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                y_start = y_end - patch_size if y_end == H else y
                x_start = x_end - patch_size if x_end == W else x
                
                # Extract patch
                patch = rgb_image[:, :, y_start:y_end, x_start:x_end]
                
                # Predict
                pred_patch = self.model(patch)
                
                # Add to output with blending weights
                weight = self._create_blend_weight(patch_size, patch_size).to(self.device)
                output[:, :, y_start:y_end, x_start:x_end] += pred_patch * weight
                weight_map[:, :, y_start:y_end, x_start:x_end] += weight
        
        # Normalize by weights
        output = output / weight_map.clamp(min=1e-8)
        
        return output
    
    def _create_blend_weight(self, h: int, w: int) -> torch.Tensor:
        """Create blending weight for smooth patch merging"""
        # Create 1D gradients
        h_grad = torch.linspace(0, 1, h // 2)
        h_grad = torch.cat([h_grad, torch.ones(h - 2 * (h // 2)), h_grad.flip(0)])
        
        w_grad = torch.linspace(0, 1, w // 2)
        w_grad = torch.cat([w_grad, torch.ones(w - 2 * (w // 2)), w_grad.flip(0)])
        
        # Create 2D weight
        weight = h_grad.unsqueeze(1) * w_grad.unsqueeze(0)
        return weight.unsqueeze(0).unsqueeze(0)
    
    def process_image_file(self, image_path: str, 
                          output_path: Optional[str] = None,
                          patch_size: Optional[int] = None) -> np.ndarray:
        """
        Process image file and optionally save results
        
        Args:
            image_path: Path to RGB image
            output_path: Optional path to save HSI output
            patch_size: Optional patch size for large images
            
        Returns:
            HSI prediction as numpy array of shape (31, H, W)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor (HWC -> CHW)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        
        # Predict
        hsi_pred = self.predict(image_tensor, patch_size=patch_size)
        
        # Convert to numpy
        hsi_np = hsi_pred.cpu().numpy()
        
        # Save if requested
        if output_path:
            # Save as .npy or .mat based on extension
            output_path = Path(output_path)
            if output_path.suffix == '.npy':
                np.save(output_path, hsi_np)
            elif output_path.suffix == '.mat':
                import scipy.io
                scipy.io.savemat(output_path, {'hsi': hsi_np})
            else:
                # Save as multi-channel TIFF
                from PIL import Image
                # Convert to uint16 for better precision
                hsi_uint16 = (hsi_np * 65535).clip(0, 65535).astype(np.uint16)
                # Save each channel
                for i in range(hsi_np.shape[0]):
                    channel_path = output_path.parent / f"{output_path.stem}_ch{i:02d}.tiff"
                    Image.fromarray(hsi_uint16[i]).save(channel_path)
                print(f"Saved {hsi_np.shape[0]} channels to {output_path.parent}")
            
            print(f"Saved output to: {output_path}")
        
        return hsi_np


def visualize_hsi(hsi: np.ndarray, rgb_bands: Tuple[int, int, int] = (13, 8, 3),
                  save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize HSI as RGB using specific bands
    
    Args:
        hsi: HSI data of shape (31, H, W)
        rgb_bands: Band indices to use for R, G, B channels
        save_path: Optional path to save visualization
        
    Returns:
        RGB visualization as numpy array
    """
    # Extract RGB bands
    r = hsi[rgb_bands[0]]
    g = hsi[rgb_bands[1]]
    b = hsi[rgb_bands[2]]
    
    # Stack and normalize
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    
    # Apply gamma correction for better visualization
    rgb = np.power(rgb, 0.7)
    
    # Save if requested
    if save_path:
        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb_uint8).save(save_path)
        print(f"Saved visualization to: {save_path}")
    
    return rgb


def main():
    parser = argparse.ArgumentParser(description='SHARP v3.2.2 Inference')
    
    parser.add_argument('checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('input', type=str,
                       help='Path to input RGB image or directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for HSI data')
    parser.add_argument('--patch_size', type=int, default=None,
                       help='Process in patches (for large images)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create RGB visualization of HSI')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Create inference engine
    print("Initializing SHARP v3.2.2 inference engine...")
    inference = SHARPInference(args.checkpoint, device=args.device)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        print(f"\nProcessing image: {input_path}")
        hsi = inference.process_image_file(
            str(input_path), 
            output_path=args.output,
            patch_size=args.patch_size
        )
        
        print(f"Output shape: {hsi.shape}")
        print(f"Value range: [{hsi.min():.3f}, {hsi.max():.3f}]")
        
        # Visualize if requested
        if args.visualize:
            vis_path = args.output.replace('.npy', '_vis.png') if args.output else 'hsi_visualization.png'
            visualize_hsi(hsi, save_path=vis_path)
    
    elif input_path.is_dir():
        # Process directory
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        print(f"\nFound {len(image_files)} images to process")
        
        output_dir = Path(args.output) if args.output else input_path / 'hsi_output'
        output_dir.mkdir(exist_ok=True)
        
        for img_path in image_files:
            print(f"\nProcessing: {img_path.name}")
            output_path = output_dir / f"{img_path.stem}.npy"
            
            hsi = inference.process_image_file(
                str(img_path),
                output_path=str(output_path),
                patch_size=args.patch_size
            )
            
            if args.visualize:
                vis_path = output_dir / f"{img_path.stem}_vis.png"
                visualize_hsi(hsi, save_path=str(vis_path))
        
        print(f"\nProcessed {len(image_files)} images")
        print(f"Results saved to: {output_dir}")
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


if __name__ == '__main__':
    main()

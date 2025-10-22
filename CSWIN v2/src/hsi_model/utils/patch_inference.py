# src/hsi_model/utils/patch_inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PatchInference:
    """Handles patch-based inference for large images to avoid OOM issues"""
    
    def __init__(
        self,
        model: nn.Module,
        patch_size: int = 128,
        overlap: int = 16,
        batch_size: int = 4,
        device: torch.device = None,
        use_fp16: bool = False,
        apply_sigmoid: bool = False
    ):
        """
        Args:
            model: The model to use for inference
            patch_size: Size of patches (assumes square patches)
            overlap: Overlap between patches to reduce edge artifacts
            batch_size: Number of patches to process at once
            device: Device to run on
            use_fp16: Whether to use mixed precision for inference
            apply_sigmoid: Whether to apply sigmoid to model output (for models that output logits)
        """
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.apply_sigmoid = apply_sigmoid
        
        # Validate parameters
        if overlap >= patch_size:
            raise ValueError(f"Overlap ({overlap}) must be less than patch_size ({patch_size})")
        if patch_size <= 0:
            raise ValueError(f"Patch size must be positive, got {patch_size}")
            
        logger.info(f"PatchInference initialized: patch_size={patch_size}, overlap={overlap}, "
                   f"stride={self.stride}, batch_size={batch_size}")
        
    def _extract_patches(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Extract overlapping patches from image"""
        B, C, H, W = img.shape
        if B != 1:
            raise ValueError(f"Batch size must be 1 for patch extraction, got {B}")
        
        patches = []
        positions = []
        
        # Calculate number of patches needed
        h_patches = max(1, (H - self.overlap) // self.stride)
        w_patches = max(1, (W - self.overlap) // self.stride)
        
        # Adjust to ensure full coverage
        if (h_patches - 1) * self.stride + self.patch_size < H:
            h_patches += 1
        if (w_patches - 1) * self.stride + self.patch_size < W:
            w_patches += 1
        
        logger.debug(f"Image size: {H}x{W}, will extract {h_patches}x{w_patches} patches")
        
        for i in range(h_patches):
            for j in range(w_patches):
                # Calculate position ensuring we don't go out of bounds
                h_start = min(i * self.stride, max(0, H - self.patch_size))
                w_start = min(j * self.stride, max(0, W - self.patch_size))
                h_end = min(h_start + self.patch_size, H)
                w_end = min(w_start + self.patch_size, W)
                
                # Handle edge case where image is smaller than patch size
                if h_end - h_start < self.patch_size or w_end - w_start < self.patch_size:
                    # Pad the patch
                    patch = F.pad(
                        img[:, :, h_start:h_end, w_start:w_end],
                        (0, self.patch_size - (w_end - w_start), 
                         0, self.patch_size - (h_end - h_start)),
                        mode='reflect'
                    )
                else:
                    patch = img[:, :, h_start:h_end, w_start:w_end]
                
                patches.append(patch)
                positions.append((h_start, w_start, h_end, w_end))
        
        # Stack patches
        patches = torch.cat(patches, dim=0)  # Shape: (n_patches, C, patch_size, patch_size)
        
        info = {
            'positions': positions,
            'h_patches': h_patches,
            'w_patches': w_patches,
            'original_shape': (H, W),
            'n_patches': len(patches)
        }
        
        return patches, info
    
    def _stitch_patches(
        self, 
        patches: torch.Tensor, 
        info: Dict[str, Any],
        out_channels: int
    ) -> torch.Tensor:
        """Stitch patches back together with weighted averaging in overlap regions"""
        H, W = info['original_shape']
        positions = info['positions']
        
        # Initialize output and weight tensors
        output = torch.zeros(1, out_channels, H, W, device=patches.device, dtype=patches.dtype)
        weights = torch.zeros(1, 1, H, W, device=patches.device, dtype=patches.dtype)
        
        # Create weight mask for blending
        patch_weight = self._create_weight_mask(self.patch_size, self.overlap)
        patch_weight = patch_weight.to(device=patches.device, dtype=patches.dtype)
        
        # Place each patch
        for idx, (h_start, w_start, h_end, w_end) in enumerate(positions):
            h_size = h_end - h_start
            w_size = w_end - w_start
            
            # Get the patch and weight mask, cropping if necessary
            patch = patches[idx:idx+1, :, :h_size, :w_size]
            weight = patch_weight[:, :, :h_size, :w_size]
            
            # Add to output with weighting
            output[:, :, h_start:h_end, w_start:w_end] += patch * weight
            weights[:, :, h_start:h_end, w_start:w_end] += weight
        
        # Normalize by weights
        output = output / (weights + 1e-8)
        
        return output
    
    def _create_weight_mask(self, patch_size: int, overlap: int) -> torch.Tensor:
        """Create weight mask for smooth blending"""
        if overlap == 0:
            return torch.ones(1, 1, patch_size, patch_size)
        
        # Create 1D weight profile with cosine blending
        ramp_size = min(overlap, patch_size // 4)  # Don't make ramp too large
        
        if ramp_size > 0:
            # Cosine ramp for smoother blending
            ramp = 0.5 * (1 + torch.cos(torch.linspace(np.pi, 0, ramp_size)))
            plateau_size = max(1, patch_size - 2 * ramp_size)
            plateau = torch.ones(plateau_size)
            weight_1d = torch.cat([ramp, plateau, ramp.flip(0)])
        else:
            weight_1d = torch.ones(patch_size)
        
        # Ensure weight_1d has correct size
        if len(weight_1d) > patch_size:
            weight_1d = weight_1d[:patch_size]
        elif len(weight_1d) < patch_size:
            # Pad with ones if needed
            pad_size = patch_size - len(weight_1d)
            weight_1d = F.pad(weight_1d, (pad_size // 2, pad_size - pad_size // 2), value=1.0)
        
        # Create 2D weight mask
        weight_2d = weight_1d.unsqueeze(0) * weight_1d.unsqueeze(1)
        weight_2d = weight_2d.unsqueeze(0).unsqueeze(0)
        
        return weight_2d
    
    def predict(self, img: torch.Tensor, show_progress: bool = True) -> torch.Tensor:
        """
        Perform patch-based inference on full image
        
        Args:
            img: Input image tensor of shape (1, C, H, W)
            show_progress: Whether to show progress bar
            
        Returns:
            Output tensor of shape (1, out_channels, H, W)
        """
        self.model.eval()
        
        # Check if image is small enough to process directly
        _, _, H, W = img.shape
        if H <= self.patch_size and W <= self.patch_size:
            logger.info(f"Image size {H}x{W} is smaller than patch size, processing directly")
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    if hasattr(self.model, 'generator'):
                        output = self.model.generator(img.to(self.device))
                    else:
                        output = self.model(img.to(self.device))
                    
                    # Apply sigmoid if requested
                    if self.apply_sigmoid:
                        output = torch.sigmoid(output)
                    
                    return output
        
        # Extract patches
        patches, info = self._extract_patches(img)
        n_patches = patches.shape[0]
        
        logger.info(f"Extracted {n_patches} patches from {info['original_shape']} image")
        
        # Process patches in batches
        outputs = []
        
        # Create progress bar if requested
        if show_progress:
            pbar = tqdm(range(0, n_patches, self.batch_size), 
                       desc="Processing patches",
                       total=(n_patches + self.batch_size - 1) // self.batch_size)
        else:
            pbar = range(0, n_patches, self.batch_size)
        
        with torch.no_grad():
            for i in pbar:
                batch = patches[i:i+self.batch_size].to(self.device)
                
                # Forward pass with mixed precision if enabled
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    if hasattr(self.model, 'generator'):
                        output = self.model.generator(batch)
                    else:
                        output = self.model(batch)
                    
                    # Apply sigmoid if requested (for models that output logits)
                    if self.apply_sigmoid:
                        output = torch.sigmoid(output)
                
                outputs.append(output.cpu())
                
                # Clear cache after each batch to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate outputs
        all_outputs = torch.cat(outputs, dim=0)
        
        # Determine output channels from first output
        out_channels = all_outputs.shape[1]
        
        # Stitch patches back together
        full_output = self._stitch_patches(all_outputs, info, out_channels)
        
        return full_output.to(self.device)
    
    def predict_dataset(
        self, 
        dataloader: torch.utils.data.DataLoader,
        return_targets: bool = False
    ) -> Tuple[list, list]:
        """
        Perform patch-based inference on entire dataset
        
        Args:
            dataloader: DataLoader containing the dataset
            return_targets: Whether to also return target HSI data
            
        Returns:
            Tuple of (predictions, targets) lists
        """
        predictions = []
        targets = []
        
        for i, batch in enumerate(tqdm(dataloader, desc="Processing dataset")):
            if len(batch) == 2:
                rgb, hsi = batch
            else:
                rgb = batch
                hsi = None
            
            # Ensure batch size is 1
            if rgb.shape[0] != 1:
                raise ValueError("DataLoader must have batch_size=1 for patch-based inference")
            
            # Predict
            pred = self.predict(rgb, show_progress=False)
            predictions.append(pred.cpu())
            
            if return_targets and hsi is not None:
                targets.append(hsi.cpu())
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataloader)} images")
        
        return predictions, targets
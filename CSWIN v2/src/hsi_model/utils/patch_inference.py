# src/hsi_model/utils/patch_inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from tqdm import tqdm

from .training_setup import autocast_context

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
        apply_sigmoid: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
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
            amp_dtype: Explicit autocast dtype. Overrides ``use_fp16`` and
                supports BF16 inference on Ampere-or-newer CUDA devices.
        """
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if amp_dtype not in (None, torch.float16, torch.bfloat16):
            raise ValueError(f"amp_dtype must be float16, bfloat16, or None, got {amp_dtype}")
        if amp_dtype is None and use_fp16:
            amp_dtype = torch.float16
        self.amp_dtype = amp_dtype if self.device.type == "cuda" else None
        self.use_amp = self.amp_dtype is not None
        self.use_fp16 = self.amp_dtype == torch.float16
        self.apply_sigmoid = apply_sigmoid
        
        # Validate parameters
        if overlap >= patch_size:
            raise ValueError(f"Overlap ({overlap}) must be less than patch_size ({patch_size})")
        if patch_size <= 0:
            raise ValueError(f"Patch size must be positive, got {patch_size}")
            
        logger.info(f"PatchInference initialized: patch_size={patch_size}, overlap={overlap}, "
                   f"stride={self.stride}, batch_size={batch_size}")
        
    def _patch_info(self, img: torch.Tensor) -> Dict[str, Any]:
        """Return patch positions without materializing every patch tensor."""
        B, C, H, W = img.shape
        if B != 1:
            raise ValueError(f"Batch size must be 1 for patch extraction, got {B}")

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
                positions.append((h_start, w_start, h_end, w_end))

        return {
            'positions': positions,
            'h_patches': h_patches,
            'w_patches': w_patches,
            'original_shape': (H, W),
            'n_patches': len(positions),
        }

    def _extract_patch(
        self,
        img: torch.Tensor,
        position: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Extract and edge-pad one patch."""
        h_start, w_start, h_end, w_end = position
        patch = img[:, :, h_start:h_end, w_start:w_end]
        patch_h, patch_w = patch.shape[-2:]
        if patch_h == self.patch_size and patch_w == self.patch_size:
            return patch

        pad_h = self.patch_size - patch_h
        pad_w = self.patch_size - patch_w
        pad_mode = "reflect"
        if patch_h <= pad_h or patch_w <= pad_w:
            pad_mode = "replicate"
        return F.pad(patch, (0, pad_w, 0, pad_h), mode=pad_mode)

    def _extract_patch_batch(
        self,
        img: torch.Tensor,
        positions: list[Tuple[int, int, int, int]],
    ) -> torch.Tensor:
        return torch.cat(
            [self._extract_patch(img, position) for position in positions],
            dim=0,
        )

    def _extract_patches(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Extract all patches for compatibility and testing.

        Production inference uses :meth:`_extract_patch_batch` so resident tile
        memory is bounded by ``batch_size``.
        """
        info = self._patch_info(img)
        patches = self._extract_patch_batch(img, info["positions"])
        return patches, info

    def _accumulate_patch_batch(
        self,
        output: torch.Tensor,
        positions: list[Tuple[int, int, int, int]],
        accumulator: torch.Tensor,
        weights: torch.Tensor,
        patch_weight: torch.Tensor,
    ) -> None:
        """Blend one output batch directly into the full-image accumulator."""
        for index, (h_start, w_start, h_end, w_end) in enumerate(positions):
            h_size = h_end - h_start
            w_size = w_end - w_start
            patch = output[index:index + 1, :, :h_size, :w_size].float()
            weight = patch_weight[:, :, :h_size, :w_size]
            accumulator[:, :, h_start:h_end, w_start:w_end] += patch * weight
            weights[:, :, h_start:h_end, w_start:w_end] += weight
    
    def _stitch_patches(
        self, 
        patches: torch.Tensor, 
        info: Dict[str, Any],
        out_channels: int
    ) -> torch.Tensor:
        """Stitch patches back together with weighted averaging in overlap regions"""
        H, W = info['original_shape']
        positions = info['positions']
        
        # Initialize output and weight tensors. Accumulate in fp32 regardless
        # of the patch dtype: fp16 accumulation across overlapping tiles adds
        # ~1e-3-relative rounding noise to the stitched cube, the same order
        # as the MRAE gaps being measured.
        output = torch.zeros(1, out_channels, H, W, device=patches.device, dtype=torch.float32)
        weights = torch.zeros(1, 1, H, W, device=patches.device, dtype=torch.float32)

        # Create weight mask for blending
        patch_weight = self._create_weight_mask(self.patch_size, self.overlap)
        patch_weight = patch_weight.to(device=patches.device, dtype=torch.float32)

        self._accumulate_patch_batch(
            patches,
            positions,
            output,
            weights,
            patch_weight,
        )

        # Normalize by weights. clamp_min (not +eps) so positions covered by a
        # single low-weight border tile divide by their TRUE weight - additive
        # eps biased the 1-px ring by eps/weight (~1% at the old corners).
        output = output / weights.clamp_min(1e-8)

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
            # Floor the profile: ramp[0] is exactly 0 (cos(pi)), and image-border
            # pixels are covered by only ONE tile whose mask is 0 there - the
            # epsilon-normalized result zeroed the outer 1-px ring of every
            # stitched prediction. Interior seams renormalize by the weight
            # sum, so the floor leaves them bit-identical. 1e-2 keeps the
            # corner weight (floor^2 = 1e-4) far above the normalizer guard.
            weight_1d = weight_1d.clamp_min(1e-2)
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
            with torch.inference_mode():
                with autocast_context(
                    self.device.type,
                    self.use_amp,
                    self.amp_dtype or torch.float16,
                ):
                    if hasattr(self.model, 'generator'):
                        output = self.model.generator(img.to(self.device))
                    else:
                        output = self.model(img.to(self.device))

                    # Apply sigmoid if requested
                    if self.apply_sigmoid:
                        output = torch.sigmoid(output)

                    return output.float()

        # Build patch coordinates only. Input/output patches are materialized one
        # batch at a time so memory no longer grows with the number of tiles.
        info = self._patch_info(img)
        positions = info["positions"]
        n_patches = len(positions)
        
        logger.info(f"Extracted {n_patches} patches from {info['original_shape']} image")

        # Create progress bar if requested
        if show_progress:
            pbar = tqdm(range(0, n_patches, self.batch_size), 
                       desc="Processing patches",
                       total=(n_patches + self.batch_size - 1) // self.batch_size)
        else:
            pbar = range(0, n_patches, self.batch_size)
        
        accumulator = None
        weights = None
        patch_weight = None
        with torch.inference_mode():
            for i in pbar:
                batch_positions = positions[i:i + self.batch_size]
                batch = self._extract_patch_batch(img, batch_positions).to(self.device)

                # Forward pass with mixed precision if enabled
                with autocast_context(
                    self.device.type,
                    self.use_amp,
                    self.amp_dtype or torch.float16,
                ):
                    if hasattr(self.model, 'generator'):
                        output = self.model.generator(batch)
                    else:
                        output = self.model(batch)

                    # Apply sigmoid if requested (for models that output logits)
                    if self.apply_sigmoid:
                        output = torch.sigmoid(output)

                if accumulator is None:
                    out_channels = int(output.shape[1])
                    accumulator = torch.zeros(
                        1,
                        out_channels,
                        H,
                        W,
                        device=output.device,
                        dtype=torch.float32,
                    )
                    weights = torch.zeros(
                        1,
                        1,
                        H,
                        W,
                        device=output.device,
                        dtype=torch.float32,
                    )
                    patch_weight = self._create_weight_mask(
                        self.patch_size,
                        self.overlap,
                    ).to(device=output.device, dtype=torch.float32)

                self._accumulate_patch_batch(
                    output,
                    batch_positions,
                    accumulator,
                    weights,
                    patch_weight,
                )

        if accumulator is None or weights is None:
            raise RuntimeError("Patch inference produced no output batches.")
        return accumulator / weights.clamp_min(1e-8)
    
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

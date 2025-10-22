"""
Data augmentation utilities for robust HSI reconstruction
Implements augmentations that preserve spectral-spatial consistency
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class RGBHSIAugmentation:
    """
    Synchronized augmentation for RGB-HSI pairs to improve generalization
    Ensures spectral-spatial consistency between RGB and HSI
    """

    def __init__(
        self,
        geometric_prob=0.5,
        photometric_prob=0.5,
        noise_prob=0.3,
        spectral_shift_prob=0.2,
        mixup_prob=0.0,
        training=True
    ):
        """
        Args:
            geometric_prob: Probability of applying geometric augmentations
            photometric_prob: Probability of applying photometric augmentations
            noise_prob: Probability of adding noise
            spectral_shift_prob: Probability of spectral shift simulation
            mixup_prob: Probability of applying mixup augmentation
            training: Whether in training mode
        """
        self.geometric_prob = geometric_prob
        self.photometric_prob = photometric_prob
        self.noise_prob = noise_prob
        self.spectral_shift_prob = spectral_shift_prob
        self.mixup_prob = mixup_prob
        self.training = training

    def __call__(self, rgb: torch.Tensor, hsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply synchronized augmentations to RGB-HSI pair

        Args:
            rgb: RGB tensor [B, 3, H, W] or [3, H, W]
            hsi: HSI tensor [B, C, H, W] or [C, H, W]

        Returns:
            Augmented (rgb, hsi) pair
        """
        if not self.training:
            return rgb, hsi

        # Handle both batched and unbatched inputs
        is_batched = rgb.dim() == 4
        if not is_batched:
            rgb = rgb.unsqueeze(0)
            hsi = hsi.unsqueeze(0)

        # Geometric augmentations (must be synchronized)
        if torch.rand(1).item() < self.geometric_prob:
            rgb, hsi = self._geometric_augment(rgb, hsi)

        # Photometric augmentations (affect both RGB and HSI)
        if torch.rand(1).item() < self.photometric_prob:
            rgb, hsi = self._photometric_augment(rgb, hsi)

        # Add realistic noise
        if torch.rand(1).item() < self.noise_prob:
            rgb, hsi = self._add_noise(rgb, hsi)

        # Spectral shift simulation (helps generalize across sensors)
        if torch.rand(1).item() < self.spectral_shift_prob:
            hsi = self._spectral_shift(hsi)

        # Mixup augmentation for better generalization
        if torch.rand(1).item() < self.mixup_prob and is_batched and rgb.shape[0] > 1:
            rgb, hsi = self._mixup(rgb, hsi)

        if not is_batched:
            rgb = rgb.squeeze(0)
            hsi = hsi.squeeze(0)

        return rgb, hsi

    def _geometric_augment(self, rgb: torch.Tensor, hsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply synchronized geometric transformations"""
        B, _, H, W = rgb.shape

        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            rgb = torch.flip(rgb, dims=[-1])
            hsi = torch.flip(hsi, dims=[-1])

        # Random vertical flip
        if torch.rand(1).item() < 0.5:
            rgb = torch.flip(rgb, dims=[-2])
            hsi = torch.flip(hsi, dims=[-2])

        # Random rotation (90, 180, 270 degrees)
        if torch.rand(1).item() < 0.5:
            k = torch.randint(1, 4, (1,)).item()  # 1, 2, or 3 rotations
            rgb = torch.rot90(rgb, k=k, dims=[-2, -1])
            hsi = torch.rot90(hsi, k=k, dims=[-2, -1])

        # Small random crops (helps with scale invariance)
        if torch.rand(1).item() < 0.3:
            crop_ratio = 0.9 + torch.rand(1).item() * 0.1  # 0.9 to 1.0
            new_h = int(H * crop_ratio)
            new_w = int(W * crop_ratio)

            top = torch.randint(0, H - new_h + 1, (1,)).item()
            left = torch.randint(0, W - new_w + 1, (1,)).item()

            rgb = rgb[:, :, top:top+new_h, left:left+new_w]
            hsi = hsi[:, :, top:top+new_h, left:left+new_w]

            # Resize back to original size
            rgb = F.interpolate(rgb, size=(H, W), mode='bilinear', align_corners=False)
            hsi = F.interpolate(hsi, size=(H, W), mode='bilinear', align_corners=False)

        return rgb, hsi

    def _photometric_augment(self, rgb: torch.Tensor, hsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply photometric transformations (brightness, contrast)"""
        # Random brightness adjustment
        if torch.rand(1).item() < 0.5:
            brightness_factor = 0.8 + torch.rand(1).item() * 0.4  # 0.8 to 1.2
            rgb = rgb * brightness_factor
            hsi = hsi * brightness_factor

        # Random contrast adjustment
        if torch.rand(1).item() < 0.5:
            contrast_factor = 0.8 + torch.rand(1).item() * 0.4  # 0.8 to 1.2
            rgb_mean = rgb.mean(dim=[-2, -1], keepdim=True)
            hsi_mean = hsi.mean(dim=[-2, -1], keepdim=True)

            rgb = rgb_mean + (rgb - rgb_mean) * contrast_factor
            hsi = hsi_mean + (hsi - hsi_mean) * contrast_factor

        # Clamp to valid range
        rgb = torch.clamp(rgb, -1.0, 1.0)
        hsi = torch.clamp(hsi, -1.0, 1.0)

        return rgb, hsi

    def _add_noise(self, rgb: torch.Tensor, hsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add realistic sensor noise"""
        # Gaussian noise for RGB
        rgb_noise_std = 0.01 + torch.rand(1).item() * 0.04  # 0.01 to 0.05
        rgb = rgb + torch.randn_like(rgb) * rgb_noise_std

        # Lower noise for HSI (typically better SNR)
        hsi_noise_std = 0.005 + torch.rand(1).item() * 0.015  # 0.005 to 0.02
        hsi = hsi + torch.randn_like(hsi) * hsi_noise_std

        # Clamp to valid range
        rgb = torch.clamp(rgb, -1.0, 1.0)
        hsi = torch.clamp(hsi, -1.0, 1.0)

        return rgb, hsi

    def _spectral_shift(self, hsi: torch.Tensor) -> torch.Tensor:
        """
        Simulate spectral shift across different sensors
        Helps model generalize to different HSI cameras
        """
        B, C, H, W = hsi.shape

        # Small random shift in spectral domain
        shift_amount = torch.randint(-2, 3, (B,)).to(hsi.device)  # -2 to 2 bands

        shifted_hsi = hsi.clone()
        for b in range(B):
            if shift_amount[b] != 0:
                if shift_amount[b] > 0:
                    # Shift right (red shift)
                    shifted_hsi[b, shift_amount[b]:] = hsi[b, :-shift_amount[b]]
                    shifted_hsi[b, :shift_amount[b]] = hsi[b, 0:1].expand(shift_amount[b], -1, -1)
                else:
                    # Shift left (blue shift)
                    shift = -shift_amount[b]
                    shifted_hsi[b, :-shift] = hsi[b, shift:]
                    shifted_hsi[b, -shift:] = hsi[b, -1:].expand(shift, -1, -1)

        return shifted_hsi

    def _mixup(self, rgb: torch.Tensor, hsi: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup augmentation for better generalization

        Args:
            rgb: RGB tensor [B, 3, H, W]
            hsi: HSI tensor [B, C, H, W]
            alpha: Mixup interpolation strength

        Returns:
            Mixed (rgb, hsi) pair
        """
        B = rgb.shape[0]

        # Sample mixing coefficient
        lam = np.random.beta(alpha, alpha)

        # Random permutation
        indices = torch.randperm(B).to(rgb.device)

        # Mix samples
        rgb_mixed = lam * rgb + (1 - lam) * rgb[indices]
        hsi_mixed = lam * hsi + (1 - lam) * hsi[indices]

        return rgb_mixed, hsi_mixed


class TestTimeAugmentation:
    """
    Test-time augmentation for robust inference
    Applies multiple augmentations and averages results
    """

    def __init__(
        self,
        num_augments: int = 5,
        use_flips: bool = True,
        use_rotations: bool = True
    ):
        """
        Args:
            num_augments: Number of augmented versions to generate
            use_flips: Whether to use flip augmentations
            use_rotations: Whether to use rotation augmentations
        """
        self.num_augments = num_augments
        self.use_flips = use_flips
        self.use_rotations = use_rotations

    def augment_batch(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Generate augmented versions of RGB input

        Args:
            rgb: RGB tensor [B, 3, H, W]

        Returns:
            augmented_rgb: [B*N, 3, H, W] where N is num_augments
            transforms: List of transforms applied to each augmentation
        """
        B, C, H, W = rgb.shape
        augmented_list = [rgb]  # Original
        transforms = [{'flip_h': False, 'flip_v': False, 'rot': 0}]

        if self.use_flips:
            # Horizontal flip
            augmented_list.append(torch.flip(rgb, dims=[-1]))
            transforms.append({'flip_h': True, 'flip_v': False, 'rot': 0})

            # Vertical flip
            augmented_list.append(torch.flip(rgb, dims=[-2]))
            transforms.append({'flip_h': False, 'flip_v': True, 'rot': 0})

        if self.use_rotations:
            # 180 degree rotation
            augmented_list.append(torch.rot90(rgb, k=2, dims=[-2, -1]))
            transforms.append({'flip_h': False, 'flip_v': False, 'rot': 2})

        # Concatenate all augmentations
        augmented_rgb = torch.cat(augmented_list[:self.num_augments], dim=0)

        return augmented_rgb, transforms[:self.num_augments]

    def merge_predictions(
        self,
        predictions: torch.Tensor,
        transforms: list,
        batch_size: int
    ) -> torch.Tensor:
        """
        Merge predictions from augmented inputs

        Args:
            predictions: HSI predictions [B*N, C, H, W]
            transforms: List of transforms that were applied
            batch_size: Original batch size B

        Returns:
            merged: Averaged predictions [B, C, H, W]
        """
        num_augments = len(transforms)
        predictions_list = torch.chunk(predictions, num_augments, dim=0)

        # Reverse transformations and collect
        reversed_preds = []
        for pred, transform in zip(predictions_list, transforms):
            # Reverse rotation
            if transform['rot'] > 0:
                pred = torch.rot90(pred, k=4-transform['rot'], dims=[-2, -1])

            # Reverse vertical flip
            if transform['flip_v']:
                pred = torch.flip(pred, dims=[-2])

            # Reverse horizontal flip
            if transform['flip_h']:
                pred = torch.flip(pred, dims=[-1])

            reversed_preds.append(pred)

        # Average predictions
        merged = torch.stack(reversed_preds, dim=0).mean(dim=0)

        return merged


class AdaptiveNormalization:
    """
    Adaptive normalization for cross-dataset generalization
    Automatically adjusts normalization parameters based on input statistics
    """

    def __init__(
        self,
        momentum: float = 0.1,
        eps: float = 1e-5,
        adaptive: bool = True
    ):
        """
        Args:
            momentum: Momentum for running statistics
            eps: Small constant for numerical stability
            adaptive: Whether to use adaptive normalization
        """
        self.momentum = momentum
        self.eps = eps
        self.adaptive = adaptive

        # Running statistics
        self.running_mean = None
        self.running_std = None
        self.num_batches_tracked = 0

    def normalize(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """
        Normalize input with adaptive statistics

        Args:
            x: Input tensor [B, C, H, W]
            update_stats: Whether to update running statistics

        Returns:
            Normalized tensor
        """
        if not self.adaptive:
            # Standard normalization to [-1, 1]
            x_min = x.amin(dim=[-3, -2, -1], keepdim=True)
            x_max = x.amax(dim=[-3, -2, -1], keepdim=True)
            return 2.0 * (x - x_min) / (x_max - x_min + self.eps) - 1.0

        # Compute batch statistics
        batch_mean = x.mean(dim=[0, 2, 3], keepdim=True)
        batch_std = x.std(dim=[0, 2, 3], keepdim=True)

        if update_stats:
            # Update running statistics
            if self.running_mean is None:
                self.running_mean = batch_mean.detach()
                self.running_std = batch_std.detach()
            else:
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean +
                    self.momentum * batch_mean.detach()
                )
                self.running_std = (
                    (1 - self.momentum) * self.running_std +
                    self.momentum * batch_std.detach()
                )

            self.num_batches_tracked += 1

        # Use running statistics if available, otherwise use batch statistics
        mean = self.running_mean if self.running_mean is not None else batch_mean
        std = self.running_std if self.running_std is not None else batch_std

        # Normalize
        normalized = (x - mean) / (std + self.eps)

        return normalized

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize using running statistics

        Args:
            x: Normalized tensor [B, C, H, W]

        Returns:
            Denormalized tensor
        """
        if self.running_mean is None or self.running_std is None:
            return x

        return x * (self.running_std + self.eps) + self.running_mean

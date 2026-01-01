"""
Spectral consistency and perceptual losses for robust HSI reconstruction
Helps ensure model generates physically plausible and generalizable results
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConsistencyLoss(nn.Module):
    """
    Enforces spectral consistency across adjacent bands
    Helps generate smooth and physically plausible spectra
    """
    def __init__(self, num_bands=31, smoothness_weight=1.0):
        super().__init__()
        self.num_bands = num_bands
        self.smoothness_weight = smoothness_weight

    def forward(self, pred_hsi, target_hsi=None):
        """
        Compute spectral consistency loss

        Args:
            pred_hsi: Predicted HSI [B, C, H, W]
            target_hsi: Optional target HSI for supervised consistency

        Returns:
            Spectral consistency loss
        """
        # Spectral gradient loss (smoothness across bands)
        spectral_grad = pred_hsi[:, 1:] - pred_hsi[:, :-1]
        smoothness_loss = torch.mean(spectral_grad ** 2)

        total_loss = smoothness_loss * self.smoothness_weight

        # If target is available, enforce similar spectral structure
        if target_hsi is not None:
            target_grad = target_hsi[:, 1:] - target_hsi[:, :-1]
            structure_loss = F.l1_loss(spectral_grad, target_grad)
            total_loss = total_loss + structure_loss

        return total_loss


class SpectralAngleLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) loss
    Measures similarity between spectral signatures independent of magnitude
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_hsi, target_hsi):
        """
        Compute SAM loss

        Args:
            pred_hsi: Predicted HSI [B, C, H, W]
            target_hsi: Target HSI [B, C, H, W]

        Returns:
            Mean spectral angle in radians
        """
        # Flatten spatial dimensions
        pred_flat = pred_hsi.reshape(pred_hsi.shape[0], pred_hsi.shape[1], -1)  # B, C, H*W
        target_flat = target_hsi.reshape(target_hsi.shape[0], target_hsi.shape[1], -1)  # B, C, H*W

        # Compute dot product along spectral dimension
        dot_product = (pred_flat * target_flat).sum(dim=1)  # B, H*W

        # Compute norms
        pred_norm = torch.sqrt((pred_flat ** 2).sum(dim=1) + self.eps)  # B, H*W
        target_norm = torch.sqrt((target_flat ** 2).sum(dim=1) + self.eps)  # B, H*W

        # Compute cosine similarity
        cos_angle = dot_product / (pred_norm * target_norm + self.eps)
        cos_angle = torch.clamp(cos_angle, -1.0 + self.eps, 1.0 - self.eps)

        # Compute angle
        angle = torch.acos(cos_angle)

        return angle.mean()


class SpectralGradientLoss(nn.Module):
    """
    Computes gradient loss in spectral domain
    Helps preserve spectral transitions and features
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_hsi, target_hsi):
        """
        Compute spectral gradient loss

        Args:
            pred_hsi: Predicted HSI [B, C, H, W]
            target_hsi: Target HSI [B, C, H, W]

        Returns:
            Spectral gradient loss
        """
        # Spectral gradients
        pred_grad = pred_hsi[:, 1:] - pred_hsi[:, :-1]
        target_grad = target_hsi[:, 1:] - target_hsi[:, :-1]

        return F.l1_loss(pred_grad, target_grad)


class FrequencyDomainLoss(nn.Module):
    """
    Loss in frequency domain using DCT
    Helps capture global spectral patterns
    """
    def __init__(self, num_components=8):
        super().__init__()
        self.num_components = num_components

    def forward(self, pred_hsi, target_hsi):
        """
        Compute frequency domain loss using DCT

        Args:
            pred_hsi: Predicted HSI [B, C, H, W]
            target_hsi: Target HSI [B, C, H, W]

        Returns:
            Frequency domain loss
        """
        # Apply DCT along spectral dimension
        pred_freq = torch.fft.rfft(pred_hsi, dim=1, norm='ortho')
        target_freq = torch.fft.rfft(target_hsi, dim=1, norm='ortho')

        # Focus on low-frequency components (most informative)
        k = min(self.num_components, pred_freq.shape[1])
        pred_freq_low = pred_freq[:, :k]
        target_freq_low = target_freq[:, :k]

        # Compute loss on magnitude and phase
        pred_mag = torch.abs(pred_freq_low)
        target_mag = torch.abs(target_freq_low)
        mag_loss = F.l1_loss(pred_mag, target_mag)

        pred_phase = torch.angle(pred_freq_low)
        target_phase = torch.angle(target_freq_low)
        phase_loss = F.l1_loss(pred_phase, target_phase)

        return mag_loss + 0.5 * phase_loss


class PerceptualSpectralLoss(nn.Module):
    """
    Perceptual loss adapted for spectral domain
    Uses a simple feature extractor to compare high-level spectral features
    """
    def __init__(self, num_bands=31, feature_channels=(16, 32, 64)):
        super().__init__()

        # Simple feature extractor for spectral features
        layers = []
        in_channels = num_bands

        for out_channels in feature_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)

        # Freeze feature extractor (we want fixed features)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred_hsi, target_hsi):
        """
        Compute perceptual loss in spectral domain

        Args:
            pred_hsi: Predicted HSI [B, C, H, W]
            target_hsi: Target HSI [B, C, H, W]

        Returns:
            Perceptual spectral loss
        """
        # Extract features
        pred_features = self.feature_extractor(pred_hsi)
        target_features = self.feature_extractor(target_hsi)

        # Compute L2 loss on features
        return F.mse_loss(pred_features, target_features)


class PhysicalConstraintLoss(nn.Module):
    """
    Enforces physical constraints on HSI data
    - Non-negativity (radiance must be >= 0)
    - Spectral smoothness
    - Valid intensity ranges
    """
    def __init__(self, lambda_neg=1.0, lambda_range=1.0):
        super().__init__()
        self.lambda_neg = lambda_neg
        self.lambda_range = lambda_range

    def forward(self, pred_hsi):
        """
        Compute physical constraint loss

        Args:
            pred_hsi: Predicted HSI [B, C, H, W] in normalized space [-1, 1]

        Returns:
            Physical constraint loss
        """
        loss = 0.0

        # Convert from [-1, 1] to [0, 1] for physical interpretation
        pred_physical = (pred_hsi + 1.0) / 2.0

        # Penalize negative values (after denormalization)
        neg_values = torch.relu(-pred_physical)
        loss += self.lambda_neg * neg_values.mean()

        # Penalize out-of-range values
        over_values = torch.relu(pred_physical - 1.0)
        loss += self.lambda_range * over_values.mean()

        return loss


class CombinedSpectralLoss(nn.Module):
    """
    Combined spectral loss for comprehensive training
    Combines multiple spectral consistency and perceptual losses
    """
    def __init__(
        self,
        num_bands=31,
        use_sam=True,
        use_spectral_grad=True,
        use_frequency=True,
        use_perceptual=True,
        use_physical=True,
        sam_weight=0.1,
        spectral_grad_weight=0.5,
        frequency_weight=0.3,
        perceptual_weight=0.2,
        physical_weight=0.1
    ):
        super().__init__()

        self.use_sam = use_sam
        self.use_spectral_grad = use_spectral_grad
        self.use_frequency = use_frequency
        self.use_perceptual = use_perceptual
        self.use_physical = use_physical

        self.sam_weight = sam_weight
        self.spectral_grad_weight = spectral_grad_weight
        self.frequency_weight = frequency_weight
        self.perceptual_weight = perceptual_weight
        self.physical_weight = physical_weight

        # Initialize loss modules
        if use_sam:
            self.sam_loss = SpectralAngleLoss()

        if use_spectral_grad:
            self.spectral_grad_loss = SpectralGradientLoss()

        if use_frequency:
            self.frequency_loss = FrequencyDomainLoss()

        if use_perceptual:
            self.perceptual_loss = PerceptualSpectralLoss(num_bands=num_bands)

        if use_physical:
            self.physical_loss = PhysicalConstraintLoss()

    def forward(self, pred_hsi, target_hsi=None, return_components=False):
        """
        Compute combined spectral loss

        Args:
            pred_hsi: Predicted HSI [B, C, H, W]
            target_hsi: Target HSI [B, C, H, W] (required for supervised losses)
            return_components: Whether to return individual loss components

        Returns:
            Combined loss (and optionally loss components dict)
        """
        total_loss = 0.0
        components = {}

        # Supervised losses (require target)
        if target_hsi is not None:
            if self.use_sam:
                sam = self.sam_loss(pred_hsi, target_hsi)
                total_loss += self.sam_weight * sam
                components['sam'] = sam.item()

            if self.use_spectral_grad:
                spec_grad = self.spectral_grad_loss(pred_hsi, target_hsi)
                total_loss += self.spectral_grad_weight * spec_grad
                components['spectral_grad'] = spec_grad.item()

            if self.use_frequency:
                freq = self.frequency_loss(pred_hsi, target_hsi)
                total_loss += self.frequency_weight * freq
                components['frequency'] = freq.item()

            if self.use_perceptual:
                perc = self.perceptual_loss(pred_hsi, target_hsi)
                total_loss += self.perceptual_weight * perc
                components['perceptual'] = perc.item()

        # Unsupervised losses (only require prediction)
        if self.use_physical:
            phys = self.physical_loss(pred_hsi)
            total_loss += self.physical_weight * phys
            components['physical'] = phys.item()

        if return_components:
            return total_loss, components

        return total_loss

"""
Domain adaptation utilities for cross-dataset generalization
Helps model transfer from source (ARAD-1K) to target datasets
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DomainAdversarialLoss(nn.Module):
    """
    Domain adversarial training loss for unsupervised domain adaptation
    Encourages domain-invariant features
    """
    def __init__(self, feature_dim, num_domains=2, hidden_dim=256):
        super().__init__()

        # Domain discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_domains)
        )

    def forward(self, features, domain_labels, alpha=1.0):
        """
        Compute domain adversarial loss

        Args:
            features: Feature tensor [B, C, H, W] or [B, C]
            domain_labels: Domain labels [B] (0 for source, 1 for target, etc.)
            alpha: Gradient reversal strength (controls trade-off)

        Returns:
            Domain adversarial loss
        """
        # Flatten features if needed
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)

        # Apply gradient reversal (conceptually)
        # In practice, we'll use negative loss for discriminator training
        features_reversed = GradientReversalFunction.apply(features, alpha)

        # Predict domain
        domain_pred = self.discriminator(features_reversed)

        # Compute classification loss
        loss = F.cross_entropy(domain_pred, domain_labels)

        return loss


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal layer for domain adversarial training
    Forward: identity
    Backward: negates gradient
    """
    @staticmethod
    def forward(ctx, x, alpha):  # type: ignore[override]
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper for gradient reversal"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class MaximumMeanDiscrepancy(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss
    Measures distribution distance between source and target domains
    """
    def __init__(self, kernel='rbf', num_kernels=5):
        super().__init__()
        self.kernel = kernel
        self.num_kernels = num_kernels

    def _gaussian_kernel(self, x, y, gamma):
        """Compute Gaussian RBF kernel"""
        x_size = x.shape[0]
        y_size = y.shape[0]

        # Compute pairwise distances
        x_tiled = x.unsqueeze(1).expand(x_size, y_size, -1)
        y_tiled = y.unsqueeze(0).expand(x_size, y_size, -1)

        distances = torch.sum((x_tiled - y_tiled) ** 2, dim=2)

        return torch.exp(-gamma * distances)

    def forward(self, source_features, target_features):
        """
        Compute MMD loss

        Args:
            source_features: Source domain features [B1, C, H, W] or [B1, C]
            target_features: Target domain features [B2, C, H, W] or [B2, C]

        Returns:
            MMD loss
        """
        # Flatten features if needed
        if source_features.dim() > 2:
            source_features = F.adaptive_avg_pool2d(source_features, 1).flatten(1)
        if target_features.dim() > 2:
            target_features = F.adaptive_avg_pool2d(target_features, 1).flatten(1)

        # Use multiple kernel scales
        gammas = [2 ** i for i in range(-self.num_kernels // 2, self.num_kernels // 2 + 1)]
        total_loss = 0.0

        for gamma in gammas:
            # Compute kernel matrices
            K_ss = self._gaussian_kernel(source_features, source_features, gamma)
            K_tt = self._gaussian_kernel(target_features, target_features, gamma)
            K_st = self._gaussian_kernel(source_features, target_features, gamma)

            # Compute MMD
            mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
            total_loss += mmd

        return total_loss / len(gammas)


class CoralLoss(nn.Module):
    """
    CORAL (Correlation Alignment) loss
    Aligns second-order statistics between source and target domains
    """
    def __init__(self):
        super().__init__()

    def _compute_covariance(self, features):
        """Compute covariance matrix"""
        n = features.shape[0]
        features_centered = features - features.mean(dim=0, keepdim=True)
        cov = (features_centered.t() @ features_centered) / (n - 1)
        return cov

    def forward(self, source_features, target_features):
        """
        Compute CORAL loss

        Args:
            source_features: Source domain features [B1, C, H, W] or [B1, C]
            target_features: Target domain features [B2, C, H, W] or [B2, C]

        Returns:
            CORAL loss
        """
        # Flatten features if needed
        if source_features.dim() > 2:
            source_features = F.adaptive_avg_pool2d(source_features, 1).flatten(1)
        if target_features.dim() > 2:
            target_features = F.adaptive_avg_pool2d(target_features, 1).flatten(1)

        # Compute covariance matrices
        source_cov = self._compute_covariance(source_features)
        target_cov = self._compute_covariance(target_features)

        # Compute Frobenius norm of difference
        d = source_features.shape[1]
        loss = torch.sum((source_cov - target_cov) ** 2) / (4 * d * d)

        return loss


class DomainAdaptationManager:
    """
    Manages domain adaptation strategies for cross-dataset training
    """
    def __init__(
        self,
        method='mmd',
        feature_dim=64,
        lambda_domain=0.1,
        num_domains=2
    ):
        """
        Args:
            method: Domain adaptation method ('mmd', 'coral', 'dann', or 'none')
            feature_dim: Dimension of features for adaptation
            lambda_domain: Weight for domain adaptation loss
            num_domains: Number of domains
        """
        self.method = method
        self.lambda_domain = lambda_domain

        if method == 'mmd':
            self.criterion = MaximumMeanDiscrepancy()
        elif method == 'coral':
            self.criterion = CoralLoss()
        elif method == 'dann':
            self.criterion = DomainAdversarialLoss(
                feature_dim=feature_dim,
                num_domains=num_domains
            )
        elif method == 'none':
            self.criterion = None
        else:
            raise ValueError(f"Unknown domain adaptation method: {method}")

    def compute_adaptation_loss(
        self,
        source_features,
        target_features=None,
        domain_labels=None,
        alpha=1.0
    ):
        """
        Compute domain adaptation loss

        Args:
            source_features: Features from source domain
            target_features: Features from target domain (required for MMD/CORAL)
            domain_labels: Domain labels (required for DANN)
            alpha: Gradient reversal strength (for DANN)

        Returns:
            Domain adaptation loss
        """
        if self.criterion is None or self.method == 'none':
            return torch.tensor(0.0, device=source_features.device)

        if self.method in ['mmd', 'coral']:
            if target_features is None:
                return torch.tensor(0.0, device=source_features.device)
            loss = self.criterion(source_features, target_features)
        elif self.method == 'dann':
            if domain_labels is None:
                return torch.tensor(0.0, device=source_features.device)
            # Combine source and target features if available
            if target_features is not None:
                features = torch.cat([source_features, target_features], dim=0)
            else:
                features = source_features
            loss = self.criterion(features, domain_labels, alpha)
        else:
            loss = torch.tensor(0.0, device=source_features.device)

        return self.lambda_domain * loss


class StatisticsAlignment:
    """
    Simple statistics-based domain alignment
    Normalizes features to have similar statistics across domains
    """
    source_mean: Optional[torch.Tensor]
    source_std: Optional[torch.Tensor]
    target_mean: Optional[torch.Tensor]
    target_std: Optional[torch.Tensor]

    def __init__(self, momentum: float = 0.1, eps: float = 1e-5):
        self.momentum = momentum
        self.eps = eps
        self.source_mean = None
        self.source_std = None
        self.target_mean = None
        self.target_std = None

    def update_source_stats(self, features: torch.Tensor) -> None:
        """Update running statistics for source domain"""
        batch_mean = features.mean(dim=[0, 2, 3], keepdim=True)
        batch_std = features.std(dim=[0, 2, 3], keepdim=True)

        if self.source_mean is None or self.source_std is None:
            self.source_mean = batch_mean.detach()
            self.source_std = batch_std.detach()
        else:
            self.source_mean = (
                (1 - self.momentum) * self.source_mean +
                self.momentum * batch_mean.detach()
            )
            self.source_std = (
                (1 - self.momentum) * self.source_std +
                self.momentum * batch_std.detach()
            )

    def update_target_stats(self, features: torch.Tensor) -> None:
        """Update running statistics for target domain"""
        batch_mean = features.mean(dim=[0, 2, 3], keepdim=True)
        batch_std = features.std(dim=[0, 2, 3], keepdim=True)

        if self.target_mean is None or self.target_std is None:
            self.target_mean = batch_mean.detach()
            self.target_std = batch_std.detach()
        else:
            self.target_mean = (
                (1 - self.momentum) * self.target_mean +
                self.momentum * batch_mean.detach()
            )
            self.target_std = (
                (1 - self.momentum) * self.target_std +
                self.momentum * batch_std.detach()
            )

    def align_to_target(self, source_features: torch.Tensor) -> torch.Tensor:
        """
        Align source features to target statistics

        Args:
            source_features: Source domain features [B, C, H, W]

        Returns:
            Aligned features
        """
        if (self.source_mean is None or self.source_std is None or
                self.target_mean is None or self.target_std is None):
            return source_features

        # Normalize to source statistics
        normalized = (source_features - self.source_mean) / (self.source_std + self.eps)

        # Denormalize to target statistics
        aligned = normalized * (self.target_std + self.eps) + self.target_mean

        return aligned


class PseudoLabelingStrategy:
    """
    Pseudo-labeling for semi-supervised domain adaptation
    Generates confident predictions on target domain as pseudo-labels
    """
    def __init__(
        self,
        confidence_threshold=0.95,
        use_entropy=True,
        entropy_threshold=0.5
    ):
        self.confidence_threshold = confidence_threshold
        self.use_entropy = use_entropy
        self.entropy_threshold = entropy_threshold

    def generate_pseudo_labels(self, model, target_rgb, target_hsi_pred):
        """
        Generate pseudo-labels for target domain

        Args:
            model: The HSI reconstruction model
            target_rgb: Target domain RGB images
            target_hsi_pred: Predicted HSI for target

        Returns:
            mask: Boolean mask indicating which pixels to use [B, 1, H, W]
            pseudo_labels: Pseudo HSI labels [B, C, H, W]
        """
        # Use prediction as pseudo-label
        pseudo_labels = target_hsi_pred.detach()

        # Compute confidence metric (using prediction stability or entropy)
        if self.use_entropy:
            # Compute entropy across spectral dimension (normalized)
            prob = F.softmax(target_hsi_pred, dim=1)
            entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1, keepdim=True)
            max_entropy = np.log(target_hsi_pred.shape[1])
            normalized_entropy = entropy / max_entropy

            # Keep low-entropy (high-confidence) predictions
            mask = (normalized_entropy < self.entropy_threshold).float()
        else:
            # Use max value as confidence (simple approach)
            max_vals, _ = target_hsi_pred.max(dim=1, keepdim=True)
            min_vals, _ = target_hsi_pred.min(dim=1, keepdim=True)
            confidence = (max_vals - min_vals) / (max_vals.abs() + 1e-8)

            mask = (confidence > self.confidence_threshold).float()

        return mask, pseudo_labels

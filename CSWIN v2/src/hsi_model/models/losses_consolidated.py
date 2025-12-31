# src/hsi_model/models/losses_consolidated.py
"""
Unified Loss Functions for HSI Reconstruction

Combines all loss components with clean interfaces:
- Reconstruction losses (Charbonnier, L1)
- Spectral losses (SAM)
- Perceptual losses
- Adversarial losses (Sinkhorn divergence, GAN)
- Combined loss with adaptive weighting

Version: 3.0 - Consolidated and cleaned
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Mapping, Iterable, Protocol
import logging
from einops import rearrange

from ..constants import (
    CHARBONNIER_EPSILON,
    DEFAULT_SINKHORN_EPSILON,
    DEFAULT_SINKHORN_ITERATIONS,
    SINKHORN_EPS_STABILITY,
    EPSILON_LARGE,
    SAM_COSINE_CLAMP,
    DEFAULT_LAMBDA_REC,
    DEFAULT_LAMBDA_PERCEPTUAL,
    DEFAULT_LAMBDA_ADVERSARIAL,
    DEFAULT_LAMBDA_SAM,
    MAX_LOSS_VALUE
)

logger = logging.getLogger(__name__)

ConfigDict = Mapping[str, object]


class FeatureExtractor(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def parameters(self, recurse: bool = True) -> Iterable[nn.Parameter]:
        ...


# ============================================
# Reconstruction Losses
# ============================================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1) for better gradient flow.
    
    More robust to outliers than MSE while providing smoother gradients than L1.
    """
    def __init__(self, epsilon: float = CHARBONNIER_EPSILON):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Charbonnier loss value
        """
        diff = pred - target
        loss = torch.sqrt(diff.pow(2) + self.epsilon**2)
        return loss.mean()


# ============================================
# Spectral Losses
# ============================================

class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss.
    
    Measures spectral similarity between predicted and target HSI by computing
    the angle between spectra. Invariant to scaling, making it robust to
    illumination differences.
    """
    def __init__(self, epsilon: float = EPSILON_LARGE):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted HSI tensor of shape (B, C, H, W)
            target: Target HSI tensor of shape (B, C, H, W)
            
        Returns:
            Mean spectral angle in radians
        """
        # Reshape to (B, C, H*W)
        pred_flat = rearrange(pred, 'b c h w -> b c (h w)')
        target_flat = rearrange(target, 'b c h w -> b c (h w)')
        
        # L2 normalize each spectrum
        pred_norm = F.normalize(pred_flat, dim=1, eps=self.epsilon)
        target_norm = F.normalize(target_flat, dim=1, eps=self.epsilon)
        
        # Compute cosine similarity
        cosine_sim = torch.sum(pred_norm * target_norm, dim=1)
        
        # Clamp to prevent numerical issues in arccos
        cosine_sim = torch.clamp(cosine_sim, -SAM_COSINE_CLAMP, SAM_COSINE_CLAMP)
        
        # Compute angle in radians
        angles = torch.acos(cosine_sim)
        
        # Check for NaN
        if torch.isnan(angles).any():
            logger.warning("NaN detected in SAM loss, replacing with zeros")
            angles = torch.where(torch.isnan(angles), torch.zeros_like(angles), angles)
        
        return angles.mean()


# ============================================
# Adversarial Losses
# ============================================

class SinkhornDivergence(nn.Module):
    """
    Entropic Optimal Transport (Sinkhorn) divergence.
    
    Computes S_eps(P, Q) = OT_eps(P,Q) - 0.5*(OT_eps(P,P) + OT_eps(Q,Q))
    using stabilized Sinkhorn algorithm. Assumes uniform weights.
    
    More stable than vanilla GAN losses and provides better gradients.
    """
    def __init__(
        self,
        epsilon: float = DEFAULT_SINKHORN_EPSILON,
        n_iters: int = DEFAULT_SINKHORN_ITERATIONS,
        eps_stab: float = SINKHORN_EPS_STABILITY
    ):
        super().__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.eps_stab = eps_stab

    @staticmethod
    def _cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise squared Euclidean distance matrix.
        
        Args:
            x: Tensor of shape (n, d)
            y: Tensor of shape (m, d)
            
        Returns:
            Cost matrix of shape (n, m)
        """
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError("Cost matrix expects 2D tensors of shape (n, d) and (m, d)")
        return torch.cdist(x, y, p=2) ** 2

    def _sinkhorn_cost(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute entropic OT cost between two point clouds.
        
        Args:
            X: Point cloud of shape (N, d) with uniform weights
            Y: Point cloud of shape (M, d) with uniform weights
            
        Returns:
            OT cost (scalar)
        """
        n, m = X.size(0), Y.size(0)
        device = X.device

        C = self._cost_matrix(X, Y)  # (n, m)
        if C.dtype in (torch.float16, torch.bfloat16):
            C = C.float()
        K = torch.exp(-C / self.epsilon)  # (n, m)

        # Validate point cloud sizes
        if n <= 0 or m <= 0:
            raise ValueError(f"Empty point clouds: X has {n} points, Y has {m} points")

        # Create uniform distributions
        a = torch.full((n,), 1.0 / n, device=device, dtype=C.dtype)
        b = torch.full((m,), 1.0 / m, device=device, dtype=C.dtype)

        # Validate probability distributions sum to 1
        assert abs(a.sum().item() - 1.0) < 1e-6, f"Invalid distribution a: sum={a.sum().item()}"
        assert abs(b.sum().item() - 1.0) < 1e-6, f"Invalid distribution b: sum={b.sum().item()}"

        u = torch.ones_like(a)
        v = torch.ones_like(b)

        # Sinkhorn iterations with stabilization to prevent division by zero
        eps_stab = max(self.eps_stab, 1e-6)
        for _ in range(self.n_iters):
            Kv = K @ v + eps_stab
            # Use torch.where and clamp to prevent division by zero/near-zero values
            u = torch.where(Kv.abs() < eps_stab, a, a / torch.clamp(Kv, min=eps_stab))
            Ktu = K.transpose(0, 1) @ u + eps_stab
            v = torch.where(Ktu.abs() < eps_stab, b, b / torch.clamp(Ktu, min=eps_stab))

        # Transport plan π = diag(u) K diag(v)
        pi = (u[:, None] * K) * v[None, :]
        cost = torch.sum(pi * C)
        return cost

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute Sinkhorn divergence.
        
        Args:
            X: Source point cloud
            Y: Target point cloud
            
        Returns:
            Sinkhorn divergence S_eps(X, Y)
        """
        ot_xy = self._sinkhorn_cost(X, Y)
        ot_xx = self._sinkhorn_cost(X, X.detach())
        ot_yy = self._sinkhorn_cost(Y, Y.detach())
        divergence = ot_xy - 0.5 * (ot_xx + ot_yy)

        if torch.isnan(divergence) or torch.isinf(divergence):
            logger.warning("Invalid Sinkhorn divergence detected; returning zero")
            return torch.zeros_like(divergence)
        if (divergence < -1e-6).any():
            logger.warning("Sinkhorn divergence negative (min=%.6f); clamping to zero", divergence.min().item())
        return torch.clamp(divergence, min=0.0)


class SinkhornLoss(nn.Module):
    """
    Legacy Sinkhorn loss using log-space stabilization.
    
    Kept for backward compatibility. For new code, use SinkhornDivergence directly.
    """
    def __init__(
        self,
        epsilon: float = DEFAULT_SINKHORN_EPSILON,
        num_iterations: int = DEFAULT_SINKHORN_ITERATIONS,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.reduction = reduction
        
    def forward(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Sinkhorn divergence with automatic spatial pooling.
        
        Args:
            real_features: Discriminator outputs for real samples
            fake_features: Discriminator outputs for fake samples
            
        Returns:
            Sinkhorn divergence (scalar)
        """
        # Handle spatial dimensions via global pooling
        if real_features.dim() > 2:
            real_features = real_features.mean(dim=[2, 3])
        if fake_features.dim() > 2:
            fake_features = fake_features.mean(dim=[2, 3])
            
        batch_size = real_features.size(0)
        
        # Cost matrix: squared Euclidean distance
        real_expanded = real_features.unsqueeze(1)  # B x 1 x C
        fake_expanded = fake_features.unsqueeze(0)  # 1 x B x C
        cost_matrix = ((real_expanded - fake_expanded) ** 2).sum(dim=2)  # B x B
        
        # Initialize dual variables in log-space
        log_a = torch.zeros_like(cost_matrix[:, 0])
        log_b = torch.zeros_like(cost_matrix[0, :])
        
        # Sinkhorn iterations in log-space (no gradients needed)
        with torch.no_grad():
            for _ in range(self.num_iterations):
                # Update log_a
                log_kernel_a = -cost_matrix / self.epsilon + log_b.unsqueeze(0)
                log_kernel_a_max = torch.max(log_kernel_a, dim=1, keepdim=True)[0]
                log_kernel_a_stable = log_kernel_a - log_kernel_a_max
                log_sum_a = log_kernel_a_max.squeeze(1) + torch.log(
                    torch.sum(torch.exp(log_kernel_a_stable), dim=1) + 1e-10
                )
                log_a = -self.epsilon * log_sum_a
                
                # Update log_b
                log_kernel_b = -cost_matrix / self.epsilon + log_a.unsqueeze(1)
                log_kernel_b_max = torch.max(log_kernel_b, dim=0, keepdim=True)[0]
                log_kernel_b_stable = log_kernel_b - log_kernel_b_max
                log_sum_b = log_kernel_b_max.squeeze(0) + torch.log(
                    torch.sum(torch.exp(log_kernel_b_stable), dim=0) + 1e-10
                )
                log_b = -self.epsilon * log_sum_b
        
        # Compute transport plan (this needs gradients)
        log_transport = -cost_matrix / self.epsilon + log_a.unsqueeze(1) + log_b.unsqueeze(0)
        transport_plan = torch.exp(log_transport)
        
        # Compute Sinkhorn divergence
        sinkhorn_divergence = torch.sum(transport_plan * cost_matrix) / batch_size
        
        return sinkhorn_divergence


# ============================================
# Perceptual Loss
# ============================================

class ImprovedPerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained feature extractor.
    
    Computes loss on intermediate features from a pre-trained network,
    capturing semantic similarity beyond pixel-wise metrics.
    """
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None):
        super().__init__()
        self.feature_extractor = feature_extractor
        if feature_extractor:
            # Freeze feature extractor
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        self.charbonnier = CharbonnierLoss()
                
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Perceptual loss value
        """
        if self.feature_extractor is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Extract RGB channels (assuming first 3 channels)
        pred_rgb = pred[:, :3, :, :].contiguous()
        target_rgb = target[:, :3, :, :].contiguous()
        
        # Ensure valid range [0, 1]
        pred_rgb = torch.clamp(pred_rgb, 0, 1)
        target_rgb = torch.clamp(target_rgb, 0, 1)
        
        # Extract features (gradients flow through!)
        pred_features = self.feature_extractor(pred_rgb)
        target_features = self.feature_extractor(target_rgb)
        
        # Use Charbonnier loss on features
        loss = self.charbonnier(pred_features, target_features)
        
        # Clamp to prevent extreme values
        loss = torch.clamp(loss, 0, 10.0)
        
        return loss


# ============================================
# Combined Loss
# ============================================

class NoiseRobustLoss(nn.Module):
    """
    Combined loss function for HSI reconstruction with Sinkhorn adversarial training.
    
    Integrates:
    - Reconstruction loss (Charbonnier)
    - Spectral loss (SAM)
    - Perceptual loss
    - Adversarial loss (Sinkhorn divergence)
    
    Features:
    - Adaptive loss weighting based on training progress
    - Supports both Sinkhorn and traditional GAN losses
    - Comprehensive NaN handling
    """
    def __init__(self, config: ConfigDict):
        super().__init__()
        
        # Loss weights
        self.lambda_rec = config.get("lambda_rec", DEFAULT_LAMBDA_REC)
        self.lambda_perc = config.get("lambda_perceptual", DEFAULT_LAMBDA_PERCEPTUAL)
        self.lambda_adv = config.get("lambda_adversarial", DEFAULT_LAMBDA_ADVERSARIAL)
        self.lambda_sam = config.get("lambda_sam", DEFAULT_LAMBDA_SAM)
        
        # Loss components
        self.charbonnier = CharbonnierLoss()
        self.perceptual_loss = ImprovedPerceptualLoss()
        self.sam_loss = SAMLoss()
        
        # Adversarial loss configuration
        self.use_sinkhorn_adv = config.get("use_sinkhorn_adversarial", True)
        self.sinkhorn = SinkhornDivergence(
            epsilon=config.get("sinkhorn_epsilon", DEFAULT_SINKHORN_EPSILON),
            n_iters=config.get("sinkhorn_iters", DEFAULT_SINKHORN_ITERATIONS)
        )
        
        # Adaptive weighting
        self.use_adaptive_weights = config.get("use_adaptive_weights", True)
        self.warmup_iterations = config.get("loss_warmup_iterations", 5000)
        self._last_iteration: Optional[int] = None
        
        # Safety parameters
        self.max_loss_value = config.get("max_loss_value", MAX_LOSS_VALUE)
        
    def compute_adversarial_loss(
        self,
        disc_real: Optional[torch.Tensor],
        disc_fake: Optional[torch.Tensor],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute adversarial loss via Sinkhorn divergence.
        
        Treats discriminator spatial logits as point clouds and computes
        optimal transport distance between real and fake distributions.
        
        Args:
            disc_real: Discriminator output for real data (B, 1, H', W')
            disc_fake: Discriminator output for fake data (B, 1, H', W')
            device: Device to place tensors on
            
        Returns:
            Adversarial loss (scalar)
        """
        if disc_fake is None or disc_real is None:
            if device is None:
                device = 'cpu'
            return torch.tensor(0.0, device=device)

        # Check for NaN/Inf in inputs
        if torch.isnan(disc_fake).any() or torch.isnan(disc_real).any():
            logger.warning("NaN detected in discriminator outputs for Sinkhorn loss")
            return torch.tensor(0.0, device=device)

        B = disc_fake.shape[0]
        loss_batch = []
        
        try:
            for b in range(B):
                # Flatten spatial dimensions to point clouds
                f = disc_fake[b].reshape(1, -1).squeeze(0)  # (S,)
                r = disc_real[b].reshape(1, -1).squeeze(0)  # (S,)

                # Build point clouds in R^1: (S, 1)
                # Keep gradients on fake, detach real
                X = f.view(-1, 1).detach() * 1.0 + (f.view(-1, 1) - f.view(-1, 1).detach())
                Y = r.view(-1, 1).detach()

                # Normalize for stability
                X = torch.nan_to_num(F.normalize(X, dim=0), nan=0.0, posinf=0.0, neginf=0.0)
                Y = torch.nan_to_num(F.normalize(Y, dim=0), nan=0.0, posinf=0.0, neginf=0.0)

                # Skip if all zeros after normalization
                if X.abs().sum() < 1e-12 or Y.abs().sum() < 1e-12:
                    loss_b = torch.tensor(0.0, device=device)
                else:
                    loss_b = self.sinkhorn(X, Y)
                    
                loss_batch.append(loss_b)

            adv_loss = torch.stack(loss_batch).mean()
            adv_loss = torch.clamp(adv_loss, 0, 5.0)
            return adv_loss
            
        except Exception as e:
            logger.warning(f"Error in Sinkhorn computation: {e}, returning zero loss")
            return torch.tensor(0.0, device=device)
    
    def get_adaptive_weights(self, iteration: int) -> Dict[str, float]:
        """
        Get adaptive loss weights based on training progress.
        
        During warmup: Gradually introduce perceptual and adversarial losses
        After warmup: Adjust based on training progress
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Dictionary of loss weights
        """
        if not self.use_adaptive_weights:
            return {
                'rec': self.lambda_rec,
                'perc': self.lambda_perc,
                'adv': self.lambda_adv,
                'sam': self.lambda_sam
            }

        if self._last_iteration is not None and iteration < self._last_iteration:
            logger.warning(
                "Iteration decreased from %s to %s; clamping adaptive progress.",
                self._last_iteration,
                iteration,
            )
        iteration = max(iteration, 0)
        self._last_iteration = iteration
        
        # Warmup phase
        if iteration < self.warmup_iterations:
            warmup_factor = iteration / self.warmup_iterations
            return {
                'rec': self.lambda_rec,
                'perc': self.lambda_perc * warmup_factor,
                'adv': self.lambda_adv * warmup_factor * 0.5,
                'sam': self.lambda_sam * warmup_factor
            }
        
        # After warmup: gradually adjust
        progress = min(1.0, (iteration - self.warmup_iterations) / 50000)
        return {
            'rec': self.lambda_rec * (1.0 - 0.3 * progress),
            'perc': self.lambda_perc * (1.0 + progress),
            'adv': self.lambda_adv * (1.0 + 0.5 * progress),
            'sam': self.lambda_sam
        }
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        disc_real: Optional[torch.Tensor] = None,
        disc_fake: Optional[torch.Tensor] = None,
        current_iteration: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            pred: Predicted HSI (B, C, H, W)
            target: Ground truth HSI (B, C, H, W)
            disc_real: Discriminator output for real data (optional)
            disc_fake: Discriminator output for fake data (optional)
            current_iteration: Current training iteration for adaptive weighting

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Validate dtype and device consistency
        if pred.dtype != target.dtype:
            raise TypeError(f"Dtype mismatch: pred={pred.dtype}, target={target.dtype}")
        if pred.device != target.device:
            raise RuntimeError(f"Device mismatch: pred={pred.device}, target={target.device}")

        loss_components = {}

        # Check inputs for NaN
        if torch.isnan(pred).any() or torch.isnan(target).any():
            logger.error("NaN detected in loss inputs!")
            return torch.tensor(1.0, device=pred.device, requires_grad=True), {
                'reconstruction': torch.tensor(1.0),
                'perceptual': torch.tensor(0.0),
                'adversarial': torch.tensor(0.0),
                'sam': torch.tensor(0.0)
            }
        
        # Get adaptive weights
        iteration = current_iteration if current_iteration is not None else 0
        weights = self.get_adaptive_weights(iteration)
        
        # Compute individual losses
        rec_loss = self.charbonnier(pred, target)
        loss_components['reconstruction'] = rec_loss
        
        perc_loss = self.perceptual_loss(pred, target)
        loss_components['perceptual'] = perc_loss
        
        adv_loss = self.compute_adversarial_loss(disc_real, disc_fake, device=pred.device)
        loss_components['adversarial'] = adv_loss
        
        sam_loss = self.sam_loss(pred, target)
        loss_components['sam'] = sam_loss
        
        # Combine with adaptive weights
        total_loss = (
            weights['rec'] * rec_loss +
            weights['perc'] * perc_loss +
            weights['adv'] * adv_loss +
            weights['sam'] * sam_loss
        )
        
        # Log weighted components
        loss_components['weighted_rec'] = weights['rec'] * rec_loss
        loss_components['weighted_perc'] = weights['perc'] * perc_loss
        loss_components['weighted_adv'] = weights['adv'] * adv_loss
        loss_components['weighted_sam'] = weights['sam'] * sam_loss
        
        # Final safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error("NaN/Inf in total loss! Using fallback.")
            total_loss = torch.tensor(1.0, device=pred.device, requires_grad=True)
        else:
            total_loss = torch.clamp(total_loss, 0, self.max_loss_value)
        
        return total_loss, loss_components


# ============================================
# Discriminator Loss
# ============================================

class ComputeSinkhornDiscriminatorLoss(nn.Module):
    """
    Discriminator loss using Sinkhorn divergence.
    
    The discriminator maximizes divergence between real and fake distributions,
    which we implement by minimizing the negative divergence.
    """
    def __init__(self, criterion: NoiseRobustLoss):
        super().__init__()
        self.criterion = criterion

    def forward(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            real_pred: Discriminator output for real samples (B, 1, H', W')
            fake_pred: Discriminator output for fake samples (B, 1, H', W')
            
        Returns:
            Discriminator loss (scalar)
        """
        try:
            sinkhorn_terms = []
            for b in range(real_pred.shape[0]):
                # Use reshape() instead of view() to handle non-contiguous tensors safely
                R = real_pred[b].reshape(-1, 1).detach()
                Fk = fake_pred[b].reshape(-1, 1).detach()

                # Normalize for stability
                R = torch.nan_to_num(F.normalize(R, dim=0), nan=0.0, posinf=0.0, neginf=0.0)
                Fk = torch.nan_to_num(F.normalize(Fk, dim=0), nan=0.0, posinf=0.0, neginf=0.0)

                # Skip if all zeros after normalization
                if R.abs().sum() < 1e-12 or Fk.abs().sum() < 1e-12:
                    sinkhorn_terms.append(torch.tensor(0.0, device=real_pred.device))
                else:
                    sinkhorn_terms.append(self.criterion.sinkhorn(R, Fk))

            sinkhorn_val = torch.stack(sinkhorn_terms).mean()
            # Discriminator maximizes divergence -> minimize negative
            disc_loss = -sinkhorn_val

            return disc_loss

        except Exception as e:
            logger.warning(f"Error in discriminator Sinkhorn computation: {e}, returning fallback")
            return torch.tensor(0.1, device=real_pred.device, requires_grad=True)


# ============================================
# Exports
# ============================================

__all__ = [
    'CharbonnierLoss',
    'SAMLoss',
    'SinkhornDivergence',
    'SinkhornLoss',
    'ImprovedPerceptualLoss',
    'NoiseRobustLoss',
    'ComputeSinkhornDiscriminatorLoss',
]

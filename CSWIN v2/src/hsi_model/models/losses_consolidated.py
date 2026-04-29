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
    DEFAULT_SINKHORN_MAX_POINTS,
    DEFAULT_SINKHORN_KERNEL_CLAMP,
    DEFAULT_SINKHORN_LOSS_CLIP,
    SINKHORN_EPS_STABILITY,
    EPSILON_LARGE,
    EPSILON_SMALL,
    SAM_COSINE_CLAMP,
    DEFAULT_LAMBDA_REC,
    DEFAULT_LAMBDA_PERCEPTUAL,
    DEFAULT_LAMBDA_ADVERSARIAL,
    DEFAULT_LAMBDA_SAM,
    MAX_LOSS_VALUE
)

logger = logging.getLogger(__name__)

ConfigDict = Mapping[str, object]


def _cap_points(points: torch.Tensor, max_points: int) -> torch.Tensor:
    """Deterministically subsample points to cap Sinkhorn memory usage."""
    if max_points <= 0 or points.shape[0] <= max_points:
        return points

    idx = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=max_points,
        device=points.device,
        dtype=torch.float32,
    ).round().long()
    return points.index_select(0, idx)


def _safe_normalize(points: torch.Tensor, eps: float = EPSILON_SMALL) -> torch.Tensor:
    """Sanitize a point cloud tensor.

    NOTE (audit): the prior implementation called ``F.normalize(points, dim=0)``
    which divided every point in the cloud by the cloud's overall L2 norm.
    For 1D discriminator features this collapsed two clouds with vastly
    different magnitudes onto the unit sphere and erased the scale signal that
    the discriminator is supposed to convey to the Sinkhorn loss. Replaced by
    a NaN/Inf sanitize that keeps relative scale; joint scaling between
    real/fake clouds is now done by ``_jointly_rescale``.
    """
    return torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)


def _jointly_rescale(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Divide both point clouds by the same scalar so the OT cost stays finite.

    The shared scale is the maximum absolute value across both clouds. Using a
    *shared* scale (instead of per-cloud normalization) preserves the relative
    magnitude difference that distinguishes real from fake disc outputs, while
    still bounding values to [-1, 1] so Sinkhorn's kernel exponent ``-C/eps``
    does not under/overflow.
    """
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    scale = torch.maximum(a.abs().amax(), b.abs().amax())
    scale = torch.clamp(scale, min=eps)
    return a / scale, b / scale


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
        eps_stab: float = SINKHORN_EPS_STABILITY,
        max_points: int = DEFAULT_SINKHORN_MAX_POINTS,
        kernel_clamp: float = DEFAULT_SINKHORN_KERNEL_CLAMP,
        force_fp32: bool = True,
    ):
        super().__init__()
        self.epsilon = max(float(epsilon), 1e-4)
        self.n_iters = max(int(n_iters), 1)
        self.eps_stab = max(float(eps_stab), 1e-8)
        self.max_points = max(int(max_points), 0)
        self.kernel_clamp = max(float(kernel_clamp), 1.0)
        self.force_fp32 = bool(force_fp32)

        if epsilon <= 0:
            logger.warning(
                "Non-positive Sinkhorn epsilon=%s detected; clamped to %.6f",
                epsilon,
                self.epsilon,
            )

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
        cost = torch.cdist(x, y, p=2) ** 2
        return torch.nan_to_num(cost, nan=0.0, posinf=1e6, neginf=0.0)

    def _prepare_points(self, points: torch.Tensor) -> torch.Tensor:
        """Sanitize and optionally subsample point cloud for stable Sinkhorn."""
        if points.dim() != 2:
            raise ValueError(f"Point cloud must be 2D, got shape {tuple(points.shape)}")

        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        points = _cap_points(points, self.max_points)

        if self.force_fp32 and points.dtype in (torch.float16, torch.bfloat16):
            points = points.float()

        return points

    def _sinkhorn_cost(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute entropic OT cost between two point clouds.
        
        Args:
            X: Point cloud of shape (N, d) with uniform weights
            Y: Point cloud of shape (M, d) with uniform weights
            
        Returns:
            OT cost (scalar)
        """
        X = self._prepare_points(X)
        Y = self._prepare_points(Y)
        n, m = X.size(0), Y.size(0)
        device = X.device
        differentiable_zero = (X.sum() + Y.sum()) * 0.0

        # Validate point cloud sizes
        if n <= 0 or m <= 0:
            return differentiable_zero

        C = self._cost_matrix(X, Y)  # (n, m)
        if self.force_fp32 and C.dtype in (torch.float16, torch.bfloat16):
            C = C.float()
        scaled_cost = torch.clamp(-C / self.epsilon, min=-self.kernel_clamp, max=0.0)
        K = torch.exp(scaled_cost)
        K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(self.eps_stab)

        # Create uniform distributions
        a = torch.full((n,), 1.0 / n, device=device, dtype=C.dtype)
        b = torch.full((m,), 1.0 / m, device=device, dtype=C.dtype)

        u = torch.ones_like(a)
        v = torch.ones_like(b)

        # Sinkhorn iterations with stabilization to prevent division by zero
        for _ in range(self.n_iters):
            Kv = (K @ v).clamp_min(self.eps_stab)
            u = a / Kv
            Ktu = (K.transpose(0, 1) @ u).clamp_min(self.eps_stab)
            v = b / Ktu

            if not torch.isfinite(u).all() or not torch.isfinite(v).all():
                logger.warning("Non-finite Sinkhorn iterates detected; returning zero OT cost")
                return differentiable_zero.to(dtype=C.dtype, device=device)

        # Transport plan pi = diag(u) K diag(v)
        pi = (u[:, None] * K) * v[None, :]
        cost = torch.sum(pi * C)
        if not torch.isfinite(cost):
            logger.warning("Non-finite Sinkhorn cost detected; returning zero OT cost")
            return differentiable_zero.to(dtype=C.dtype, device=device)
        return torch.clamp(cost, min=0.0)

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

        if not torch.isfinite(divergence):
            logger.warning("Invalid Sinkhorn divergence detected; returning zero")
            return divergence * 0.0
        if (divergence < -1e-6).any():
            logger.warning("Sinkhorn divergence negative (min=%.6f); clamping to zero", divergence.min().item())
        return torch.clamp(divergence, min=0.0, max=1e4)


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

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# Local CMF cache. Kept here (instead of importing from utils.metrics) so the
# losses module has no transitive dependency on utils.__init__, which would
# otherwise pull in optional deps (tensorboard, scipy, h5py, cv2) used by
# the dataloaders.
_PERCEPTUAL_CMF_CACHE: Dict[Tuple[int, str, Tuple[float, float]], torch.Tensor] = {}


def _gaussian_cmf(
    num_bands: int,
    device: torch.device,
    wavelength_range: Tuple[float, float] = (400.0, 700.0),
) -> torch.Tensor:
    """Return a Gaussian approximation of CIE-1931 color-matching functions.

    Shape is (num_bands, 3). Matches utils.metrics.create_cmf_tensor so
    training-time perceptual conversion is consistent with eval metrics.
    """
    device_key = "cpu" if device is None else str(device.type)
    if getattr(device, "index", None) is not None:
        device_key += f":{device.index}"
    key = (num_bands, device_key, wavelength_range)
    cached = _PERCEPTUAL_CMF_CACHE.get(key)
    if cached is not None:
        return cached

    import numpy as _np  # local to avoid a global numpy import at module load

    wavelengths = _np.linspace(wavelength_range[0], wavelength_range[1], num_bands)
    # CIE-1931-ish Gaussian peaks (nm, width, amplitude).
    r_peak, r_width, r_amp = 599.8, 33.0, 0.264
    g_peak, g_width, g_amp = 549.1, 57.0, 0.323
    b_peak, b_width, b_amp = 445.8, 33.0, 0.272
    r = r_amp * _np.exp(-((wavelengths - r_peak) / r_width) ** 2)
    g = g_amp * _np.exp(-((wavelengths - g_peak) / g_width) ** 2)
    b = b_amp * _np.exp(-((wavelengths - b_peak) / b_width) ** 2)
    cmf = _np.stack([r, g, b], axis=1).astype(_np.float32)
    tensor = torch.from_numpy(cmf).to(device)
    _PERCEPTUAL_CMF_CACHE[key] = tensor
    return tensor


class ImprovedPerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained feature extractor.

    Computes loss on intermediate features from a pre-trained (RGB) network.
    For hyperspectral inputs, HSI is converted to RGB via a color-matching
    function (CMF) BEFORE feeding the feature extractor. Using the first 3
    HSI bands as pseudo-RGB — the previous behaviour — is not meaningful
    because those bands are typically ~400-430 nm (near-blue).

    Args:
        feature_extractor: Pretrained RGB feature extractor (e.g. VGG).
        num_bands: Number of spectral bands in the input HSI. If 3 the CMF
            step is skipped (already RGB).
        use_imagenet_norm: Apply ImageNet normalization (mean/std) before
            feeding the extractor. Required for standard torchvision VGG.
        wavelength_range: Band wavelength range in nm.
    """

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        num_bands: int = 31,
        use_imagenet_norm: bool = True,
        wavelength_range: Tuple[float, float] = (400.0, 700.0),
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_bands = int(num_bands)
        self.use_imagenet_norm = bool(use_imagenet_norm)
        self.wavelength_range = wavelength_range
        if feature_extractor is not None:
            # Freeze feature extractor
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.charbonnier = CharbonnierLoss()
        # Register constants as buffers so they follow .to(device) / .half()
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self._warned_disabled = False

    def _hsi_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Convert HSI (B,C,H,W) to RGB (B,3,H,W) via cached CMF; passthrough if C==3."""
        if x.shape[1] == 3:
            return torch.clamp(x, 0.0, 1.0)
        cmf = _gaussian_cmf(
            num_bands=x.shape[1],
            device=x.device,
            wavelength_range=self.wavelength_range,
        ).to(dtype=x.dtype)
        rgb = torch.einsum("bchw,cd->bdhw", x, cmf)
        return torch.clamp(rgb, 0.0, 1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted HSI tensor (B, C, H, W)
            target: Target HSI tensor (B, C, H, W)

        Returns:
            Perceptual loss value (scalar, differentiable zero if no extractor).
        """
        if self.feature_extractor is None:
            if not self._warned_disabled:
                logger.warning(
                    "ImprovedPerceptualLoss: no feature_extractor provided; "
                    "perceptual loss will always be zero. Pass a pretrained "
                    "RGB network (e.g. VGG16 features) to actually use "
                    "lambda_perceptual."
                )
                self._warned_disabled = True
            # Keep it differentiable so the combined loss graph is stable.
            return pred.sum() * 0.0

        # Convert HSI -> RGB in a principled way (CMF), then normalize.
        pred_rgb = self._hsi_to_rgb(pred)
        target_rgb = self._hsi_to_rgb(target)

        if self.use_imagenet_norm:
            mean = self._imagenet_mean.to(dtype=pred_rgb.dtype, device=pred_rgb.device)
            std = self._imagenet_std.to(dtype=pred_rgb.dtype, device=pred_rgb.device)
            pred_rgb = (pred_rgb - mean) / std
            target_rgb = (target_rgb - mean) / std

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
        perceptual_num_bands = int(config.get("perceptual_num_bands", 31))
        perceptual_feature_extractor = config.get("perceptual_feature_extractor", None)
        if perceptual_feature_extractor is None and self.lambda_perc > 0:
            logger.warning(
                "lambda_perceptual=%s > 0 but no perceptual_feature_extractor "
                "was provided; setting effective lambda_perceptual=0. "
                "Pass a pretrained RGB model (e.g. torchvision.models.vgg16(...).features) "
                "via config['perceptual_feature_extractor'] to enable it.",
                self.lambda_perc,
            )
            self.lambda_perc = 0.0
        self.perceptual_loss = ImprovedPerceptualLoss(
            feature_extractor=perceptual_feature_extractor,
            num_bands=perceptual_num_bands,
        )
        self.sam_loss = SAMLoss()
        
        # Adversarial loss configuration
        self.use_sinkhorn_adv = config.get("use_sinkhorn_adversarial", True)
        self.sinkhorn = SinkhornDivergence(
            epsilon=config.get("sinkhorn_epsilon", DEFAULT_SINKHORN_EPSILON),
            n_iters=config.get("sinkhorn_iters", DEFAULT_SINKHORN_ITERATIONS),
            max_points=config.get("sinkhorn_max_points", DEFAULT_SINKHORN_MAX_POINTS),
            kernel_clamp=config.get("sinkhorn_kernel_clamp", DEFAULT_SINKHORN_KERNEL_CLAMP),
            force_fp32=config.get("sinkhorn_force_fp32", True),
        )
        self.sinkhorn_loss_clip = config.get("sinkhorn_loss_clip", DEFAULT_SINKHORN_LOSS_CLIP)
        
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
        differentiable_zero: torch.Tensor
        if disc_fake is not None:
            differentiable_zero = torch.nan_to_num(
                disc_fake, nan=0.0, posinf=0.0, neginf=0.0
            ).sum() * 0.0
        elif disc_real is not None:
            differentiable_zero = torch.nan_to_num(
                disc_real, nan=0.0, posinf=0.0, neginf=0.0
            ).sum() * 0.0
        else:
            if device is None:
                device = torch.device("cpu")
            differentiable_zero = torch.zeros((), device=device)

        if disc_fake is None or disc_real is None:
            return differentiable_zero

        # Check for NaN/Inf in inputs
        if not torch.isfinite(disc_fake).all() or not torch.isfinite(disc_real).all():
            logger.warning("Non-finite discriminator outputs for Sinkhorn loss; using zero fallback")
            return differentiable_zero

        B = disc_fake.shape[0]
        loss_batch = []
        
        try:
            max_points = getattr(self.sinkhorn, "max_points", 0)
            for b in range(B):
                # Flatten spatial dimensions to point clouds
                f = disc_fake[b].reshape(-1, 1)  # (S, 1)
                r = disc_real[b].reshape(-1, 1)  # (S, 1)
                f = _cap_points(f, max_points)
                r = _cap_points(r, max_points)

                # Generator path: gradients flow only through `f` (fake);
                # `r` (real) is detached so it does not couple back into D.
                X = f
                Y = r.detach()

                # Joint rescale by max-abs across BOTH clouds preserves the
                # relative magnitude difference that the discriminator uses to
                # signal real vs fake. The previous _safe_normalize divided
                # each cloud by its own L2 norm and erased that signal.
                X, Y = _jointly_rescale(X, Y)

                if X.abs().sum() < 1e-12 and Y.abs().sum() < 1e-12:
                    loss_b = differentiable_zero
                else:
                    loss_b = self.sinkhorn(X, Y)

                loss_batch.append(loss_b)

            adv_loss = torch.stack(loss_batch).mean()
            adv_loss = torch.clamp(adv_loss, 0, self.sinkhorn_loss_clip)
            if not torch.isfinite(adv_loss):
                logger.warning("Non-finite adversarial loss computed; using zero fallback")
                return differentiable_zero
            return adv_loss
            
        except Exception as e:
            logger.warning(f"Error in Sinkhorn computation: {e}, returning zero loss")
            return differentiable_zero
    
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

        # Check inputs for NaN/Inf
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            logger.error("Non-finite values detected in loss inputs!")
            return torch.tensor(1.0, device=pred.device, requires_grad=True), {
                'reconstruction': torch.tensor(1.0, device=pred.device),
                'perceptual': torch.tensor(0.0, device=pred.device),
                'adversarial': torch.tensor(0.0, device=pred.device),
                'sam': torch.tensor(0.0, device=pred.device)
            }

        # Compute losses in fp32 for mixed-precision stability.
        pred_loss = pred.float()
        target_loss = target.float()
        
        # Get adaptive weights
        iteration = current_iteration if current_iteration is not None else 0
        weights = self.get_adaptive_weights(iteration)
        
        # Compute individual losses
        rec_loss = self.charbonnier(pred_loss, target_loss)
        loss_components['reconstruction'] = rec_loss
        
        perc_loss = self.perceptual_loss(pred_loss, target_loss)
        loss_components['perceptual'] = perc_loss
        
        adv_loss = self.compute_adversarial_loss(disc_real, disc_fake, device=pred.device)
        loss_components['adversarial'] = adv_loss
        
        sam_loss = self.sam_loss(pred_loss, target_loss)
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
        if not torch.isfinite(total_loss):
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
        # Keep a differentiable zero fallback anchored to discriminator outputs.
        differentiable_zero = (
            torch.nan_to_num(real_pred, nan=0.0, posinf=0.0, neginf=0.0).sum()
            + torch.nan_to_num(fake_pred, nan=0.0, posinf=0.0, neginf=0.0).sum()
        ) * 0.0
        if not torch.isfinite(real_pred).all() or not torch.isfinite(fake_pred).all():
            logger.warning("Non-finite discriminator logits detected; using zero fallback")
            return differentiable_zero

        try:
            sinkhorn_terms = []
            max_points = getattr(self.criterion.sinkhorn, "max_points", 0)
            for b in range(real_pred.shape[0]):
                # Use reshape() instead of view() to handle non-contiguous tensors safely.
                # IMPORTANT: do NOT detach here; discriminator loss must backpropagate to
                # discriminator parameters through both real and fake logits.
                R = real_pred[b].reshape(-1, 1)
                Fk = fake_pred[b].reshape(-1, 1)
                R = _cap_points(R, max_points)
                Fk = _cap_points(Fk, max_points)

                # Joint rescale (shared scale across real+fake) — preserves
                # the magnitude gap the discriminator should be exploiting,
                # while keeping inputs O(1) so Sinkhorn's kernel stays stable.
                R, Fk = _jointly_rescale(R, Fk)

                if R.abs().sum() < 1e-12 and Fk.abs().sum() < 1e-12:
                    sinkhorn_terms.append(differentiable_zero)
                else:
                    sinkhorn_terms.append(self.criterion.sinkhorn(R, Fk))

            if not sinkhorn_terms:
                return differentiable_zero

            sinkhorn_val = torch.stack(sinkhorn_terms).mean()
            # Discriminator maximizes divergence -> minimize negative
            disc_loss = -sinkhorn_val
            if not torch.isfinite(disc_loss):
                logger.warning("Non-finite discriminator Sinkhorn loss detected; using zero fallback")
                return differentiable_zero

            return disc_loss

        except Exception as e:
            logger.warning(f"Error in discriminator Sinkhorn computation: {e}, returning fallback")
            return differentiable_zero


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


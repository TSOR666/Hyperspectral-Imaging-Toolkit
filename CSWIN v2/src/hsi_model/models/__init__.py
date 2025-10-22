# src/hsi_model/models/__init__.py
"""
HSI reconstruction models package.

Exports:
- NoiseRobustCSWinModel: Combined GAN model with generator and discriminator
- Loss functions: core reconstruction, perceptual, and adversarial losses
- Discriminator utilities: Sinkhorn-compatible discriminator loss
"""

from .model import NoiseRobustCSWinModel
from .generator import NoiseRobustCSWinGenerator
from .discriminator import SpatialSpectralDiscriminator
from .losses import (
    CharbonnierLoss,
    SAMLoss,
    SinkhornDivergence,
    SinkhornLoss,
    ImprovedPerceptualLoss,
    NoiseRobustLoss,
    ComputeSinkhornDiscriminatorLoss,
)

__all__ = [
    # Models
    "NoiseRobustCSWinModel",
    "NoiseRobustCSWinGenerator",
    "SpatialSpectralDiscriminator",
    # Reconstruction losses
    "CharbonnierLoss",
    "SAMLoss",
    # Adversarial losses
    "SinkhornDivergence",
    "SinkhornLoss",
    # Perceptual loss
    "ImprovedPerceptualLoss",
    # Combined losses
    "NoiseRobustLoss",
    "ComputeSinkhornDiscriminatorLoss",
]

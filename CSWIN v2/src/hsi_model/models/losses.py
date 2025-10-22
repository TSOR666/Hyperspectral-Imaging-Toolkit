# src/hsi_model/models/losses.py
"""
Loss facade module that re-exports the consolidated loss implementations.
"""

from .losses_consolidated import *  # noqa: F401,F403

__all__ = [
    "CharbonnierLoss",
    "SAMLoss",
    "SinkhornDivergence",
    "SinkhornLoss",
    "ImprovedPerceptualLoss",
    "NoiseRobustLoss",
    "ComputeSinkhornDiscriminatorLoss",
]

# src/hsi_model/models/discriminator.py
"""
Discriminator wrapper exposing the production spectral-spatial discriminator.
"""

from .discriminator_v2 import SpatialSpectralDiscriminator

__all__ = ["SpatialSpectralDiscriminator"]

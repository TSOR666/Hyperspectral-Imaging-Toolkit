"""High level model wrapper that bundles the generator and discriminator."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .discriminator import SpatialSpectralDiscriminator
from .generator import NoiseRobustCSWinGenerator

class NoiseRobustCSWinModel(nn.Module):
    """
    Complete noise-robust model for HSI reconstruction with CSWin architecture.
    
    Combines the generator and discriminator into a single model for training.
    
    Args:
        config: Model configuration from hydra
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        # The training code expects direct access to both sub-modules for
        # optimizer/scheduler construction.  We therefore keep them as public
        # attributes, but still expose a couple of lightweight helpers for
        # readability.
        self.generator = NoiseRobustCSWinGenerator(config)
        self.discriminator = SpatialSpectralDiscriminator(config)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Run the generator to predict a hyperspectral cube from RGB input."""

        return self.generator(rgb)

    def discriminate(self, rgb: torch.Tensor, hsi: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that forwards to the discriminator."""

        return self.discriminator(rgb, hsi)


__all__ = ["NoiseRobustCSWinModel"]

"""Backward-compatible model import path.

Historically callers imported ``NoiseRobustCSWinModel`` from
``hsi_model.model``. The canonical definition now lives in
``hsi_model.models.model``; this shim keeps the old import stable.
"""

from .models.model import NoiseRobustCSWinModel

__all__ = ["NoiseRobustCSWinModel"]

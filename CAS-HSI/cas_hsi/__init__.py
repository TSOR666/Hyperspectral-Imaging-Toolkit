"""CAS-HSI -- Convolutional Attention Stack for hyperspectral reconstruction.

RGB (B x 3 x H x W) -> HSI (B x 31 x H x W), any H and W, no resizing.

    from cas_hsi import build_cas_hsi
    model = build_cas_hsi("tiny")          # research backend
    edge  = build_cas_hsi("tiny", backend="edge")   # convolutional, exportable

See docs/CAS_HSI_Implementation_Steps_3_to_9.md for the normative specification.
"""

from .config import BASE, TINY, VARIANTS, CASHSIConfig, Depths, validate_config, variant_config
from .inference import tiled_inference
from .model import (
    CASHSI,
    build_cas_hsi,
    build_edge_model,
    create_cas_hsi_base,
    create_cas_hsi_tiny,
)

__version__ = "1.0.0"

__all__ = [
    "BASE",
    "CASHSI",
    "CASHSIConfig",
    "Depths",
    "TINY",
    "VARIANTS",
    "__version__",
    "build_cas_hsi",
    "build_edge_model",
    "create_cas_hsi_base",
    "create_cas_hsi_tiny",
    "tiled_inference",
    "validate_config",
    "variant_config",
]

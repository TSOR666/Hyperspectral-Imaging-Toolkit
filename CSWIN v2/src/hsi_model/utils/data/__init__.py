# src/hsi_model/utils/data/__init__.py
"""
Data loading and processing package.

Organized modules:
- mst_dataset: MST++ training and validation datasets
- arad_dataset: ARAD-1K validation dataset with caching
- loaders: DataLoader creation utilities
- transforms: Data transformation and conversion helpers
- diagnostics: Memory and performance diagnostics
"""

from .mst_dataset import MST_TrainDataset, MST_ValidDataset
from .arad_dataset import ARAD1KDataset, create_arad1k_dataloader, DatasetCache
from .loaders import (
    create_mst_dataloaders,
    create_dataloaders,
    worker_init_fn_mst,
)
from .transforms import (
    mst_to_gan_batch,
    compute_mst_center_crop_metrics,
)
from .diagnostics import show_dataloader_diagnostics

__all__ = [
    # Datasets
    "MST_TrainDataset",
    "MST_ValidDataset",
    "ARAD1KDataset",
    "DatasetCache",
    # Loaders
    "create_mst_dataloaders",
    "create_dataloaders",
    "create_arad1k_dataloader",
    "worker_init_fn_mst",
    # Transforms
    "mst_to_gan_batch",
    "compute_mst_center_crop_metrics",
    # Diagnostics
    "show_dataloader_diagnostics",
]

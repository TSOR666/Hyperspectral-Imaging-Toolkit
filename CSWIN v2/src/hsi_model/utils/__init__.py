# src/hsi_model/utils/__init__.py
"""
Utilities package for HSI reconstruction.

Exposes checkpointing, logging, metrics, patch inference, and data utilities.
"""

# Core utilities
from .checkpoint import save_checkpoint, load_checkpoint
from .logging import setup_logging, MetricsLogger
from .metrics import (
    hsi_to_rgb,
    create_cmf_tensor,
    profile_model,
    export_model,
    compute_metrics_arad1k,
    crop_center_arad1k,
    compute_psnr,
    compute_metrics,
    validate_model_architecture,
    save_metrics,
    get_cached_cmf,
    create_error_report,
    compute_ssim,
    compute_sam_value,
    compute_mrae,
    compute_mae,
    compute_rmse,
)
from .patch_inference import PatchInference

# Data utilities (re-export for backwards compatibility)
from .data import (
    MST_TrainDataset,
    MST_ValidDataset,
    ARAD1KDataset,
    DatasetCache,
    create_mst_dataloaders,
    create_dataloaders,
    create_arad1k_dataloader,
    worker_init_fn_mst,
    mst_to_gan_batch,
    compute_mst_center_crop_metrics,
    show_dataloader_diagnostics,
)

__all__ = [
    # Checkpoint utilities
    "save_checkpoint",
    "load_checkpoint",
    "crop_center_arad1k",
    # Logging utilities
    "setup_logging",
    "MetricsLogger",
    # Metrics utilities
    "compute_metrics_arad1k",
    "hsi_to_rgb",
    "create_cmf_tensor",
    "profile_model",
    "save_metrics",
    "export_model",
    "compute_metrics",
    "validate_model_architecture",
    "PatchInference",
    "compute_psnr",
    "get_cached_cmf",
    "create_error_report",
    "compute_ssim",
    "compute_sam_value",
    "compute_mrae",
    "compute_mae",
    "compute_rmse",
    # Data utilities
    "MST_TrainDataset",
    "MST_ValidDataset",
    "ARAD1KDataset",
    "DatasetCache",
    "create_mst_dataloaders",
    "create_dataloaders",
    "create_arad1k_dataloader",
    "worker_init_fn_mst",
    "mst_to_gan_batch",
    "compute_mst_center_crop_metrics",
    "show_dataloader_diagnostics",
]

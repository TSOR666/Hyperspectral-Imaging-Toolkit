# src/hsi_model/utils/data/loaders.py
"""
DataLoader creation utilities with MST++-style memory optimisations.
"""

import os
import logging
import random
from typing import Dict, Tuple, Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .mst_dataset import MST_TrainDataset, MST_ValidDataset
from ...constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PATCH_SIZE,
    DEFAULT_STRIDE,
    DEFAULT_NUM_WORKERS,
    H5PY_CACHE_SIZE_MB,
    OMP_NUM_THREADS,
    DEFAULT_DATA_DIR,
)

logger = logging.getLogger(__name__)


def worker_init_fn_mst(worker_id: int, base_seed: int = 42, rank: int = 0) -> None:
    """
    MST++ style worker initialization with memory optimisation.
    """
    worker_seed = base_seed + rank * 100 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # Reduce h5py cache from 64MB to 4MB.
    try:
        h5py.get_config().chunk_cache_mem_size = H5PY_CACHE_SIZE_MB * 1024 * 1024
    except Exception:
        try:
            import h5py._hl.base

            h5py._hl.base.phil.acquire()
            h5py._hl.base.default_file_cache_size = (
                H5PY_CACHE_SIZE_MB * 1024 * 1024
            )
            h5py._hl.base.phil.release()
        except Exception:
            pass

    # Limit torch threads in workers to reduce memory pressure.
    num_threads = int(os.environ.get("OMP_NUM_THREADS", str(OMP_NUM_THREADS)))
    torch.set_num_threads(num_threads)

    logger.debug(
        "Worker %s initialised: seed=%s, h5py_cache=%sMB, threads=%s",
        worker_id,
        worker_seed,
        H5PY_CACHE_SIZE_MB,
        num_threads,
    )


def create_mst_dataloaders(
    config: Dict[str, Any],
    distributed: bool = False,
    seed: int = 42,
    rank: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create MST++ exact DataLoaders for GAN training.
    """
    data_root = config.get("data_dir", DEFAULT_DATA_DIR)
    batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)
    val_batch_size = config.get("val_batch_size", 1)
    num_workers = config.get("num_workers", DEFAULT_NUM_WORKERS)

    crop_size = config.get("patch_size", DEFAULT_PATCH_SIZE)
    stride = config.get("stride", DEFAULT_STRIDE)

    logger.info("=" * 60)
    logger.info("Creating MST++ DataLoaders")
    logger.info("=" * 60)
    logger.info("  - Data root: %s", data_root)
    logger.info("  - Crop size: %s", crop_size)
    logger.info("  - Stride: %s", stride)
    logger.info("  - Batch size: %s", batch_size)
    logger.info("  - Val batch size: %s", val_batch_size)
    logger.info("  - Workers: %s", num_workers)
    logger.info("  - h5py cache: %sMB per worker (fixed from 64MB)", H5PY_CACHE_SIZE_MB)
    logger.info("  - Distributed: %s", distributed)

    train_dataset = MST_TrainDataset(
        data_root=data_root,
        crop_size=crop_size,
        arg=True,
        bgr2rgb=True,
        stride=stride,
    )

    val_dataset = MST_ValidDataset(
        data_root=data_root,
        bgr2rgb=True,
    )

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=seed)
        logger.info("  - Using DistributedSampler")

    def worker_init_wrapper(worker_id: int) -> None:
        worker_init_fn_mst(worker_id, seed, rank)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_wrapper,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_wrapper,
        persistent_workers=False,
    )

    logger.info("=" * 60)
    logger.info("DataLoaders created:")
    logger.info("  - Training samples: %s", len(train_dataset))
    logger.info("  - Validation samples: %s", len(val_dataset))
    logger.info("  - Training batches per epoch: %s", len(train_loader))
    logger.info(
        "  - Expected memory per worker: ~%sMB",
        H5PY_CACHE_SIZE_MB + 100,
    )
    logger.info("=" * 60)

    return train_loader, val_loader


def create_dataloaders(
    config: Dict[str, Any],
    distributed: bool = False,
    seed: int = 42,
    rank: int = 0,
) -> Dict[int, Dict[str, DataLoader]]:
    """
    Backwards-compatible wrapper returning nested dictionaries.
    """
    train_loader, val_loader = create_mst_dataloaders(
        config, distributed, seed, rank
    )

    patch_size = config.get("patch_size", DEFAULT_PATCH_SIZE)
    return {
        patch_size: {
            "train": train_loader,
            "val": val_loader,
        }
    }

# src/hsi_model/utils/training_setup.py
"""
Shared training-setup helpers.

Factored out of `train_optimized.py` so both production trainers
(`training_script_fixed.py`, `train_optimized.py`) can use them without
cross-importing one script from the other.

Public API:
    setup_paths(config)                 -> dict
    setup_distributed_training(config)  -> (device, rank, world_size, is_distributed)
    setup_seed(seed, rank=0)            -> None
    cleanup()                           -> None   # destroys the NCCL process group
"""

from __future__ import annotations

import atexit
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist

from hsi_model.constants import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_LOG_DIR,
)

logger = logging.getLogger(__name__)


def setup_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Set up and validate all required paths.

    Validates the MST++/ARAD-1K dataset layout, creates checkpoint/log
    directories, and rewrites the config with absolute path strings.
    """
    setup_logger = logging.getLogger("hsi_model.setup")

    data_dir = Path(config.get("data_dir", DEFAULT_DATA_DIR))
    checkpoint_dir = Path(config.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))
    log_dir = Path(config.get("log_dir", DEFAULT_LOG_DIR))

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir.resolve()} "
            "(set `data_dir` in the Hydra config or environment)."
        )

    required_dirs = ["Train_RGB", "Train_Spec", "Test_RGB", "split_txt"]
    for req_dir in required_dirs:
        full_path = data_dir / req_dir
        if not full_path.exists():
            raise FileNotFoundError(f"Required MST++ directory not found: {full_path}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config["data_dir"] = str(data_dir)
    config["checkpoint_dir"] = str(checkpoint_dir)
    config["log_dir"] = str(log_dir)

    setup_logger.info("Data directory: %s", data_dir)
    setup_logger.info("Checkpoint directory: %s", checkpoint_dir)
    setup_logger.info("Log directory: %s", log_dir)

    return config


def cleanup() -> None:
    """Tear down the distributed process group if one is active."""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_distributed_training(
    config: Dict[str, Any],
) -> Tuple[torch.device, int, int, bool]:
    """Initialize distributed training when requested.

    Returns (device, rank, world_size, is_distributed). When
    `config['distributed']` is falsy, returns a single-process default
    without touching the process group.
    """
    if not config.get("distributed", False):
        return (
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            0,
            1,
            False,
        )

    local_rank = int(os.environ.get("LOCAL_RANK", config.get("local_rank", 0)))
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", config.get("world_size", 1)))

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    atexit.register(cleanup)

    return torch.device(f"cuda:{local_rank}"), rank, world_size, True


def setup_seed(seed: int, rank: int = 0) -> None:
    """Seed all RNGs deterministically for a given rank.

    cuDNN benchmark is disabled and deterministic mode enabled. This may
    reduce throughput 5-15%; set `cudnn.benchmark = True` after calling
    this helper if reproducibility is not required.
    """
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Deterministic mode enabled: cudnn.deterministic=True, benchmark=False")
    logger.warning(
        "Performance may be reduced. To disable: set cudnn.benchmark=True (loses determinism)"
    )


__all__ = [
    "setup_paths",
    "setup_distributed_training",
    "setup_seed",
    "cleanup",
]

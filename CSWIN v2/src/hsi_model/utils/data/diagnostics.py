# src/hsi_model/utils/data/diagnostics.py
"""
DataLoader diagnostics and memory monitoring helpers.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available. Install with: pip install psutil")

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _bytes_to_gb(value: float) -> float:
    return value / 1024**3


def show_dataloader_diagnostics(detailed: bool = False) -> Dict[str, Any]:
    """
    Inspect live DataLoader worker processes and memory usage.
    """
    if not HAS_PSUTIL:
        logger.error(
            "psutil is required for diagnostics. Install with: pip install psutil"
        )
        return {}

    diagnostics: Dict[str, Any] = {}

    process = psutil.Process()
    children = process.children(recursive=True)

    main_rss_gb = _bytes_to_gb(process.memory_info().rss)
    diagnostics["main_process_rss_gb"] = main_rss_gb
    diagnostics["worker_count"] = len(children)

    print("\n" + "=" * 60)
    print("DataLoader Memory Diagnostics")
    print("=" * 60)
    print(f"Main process RSS: {main_rss_gb:.2f} GB")
    print(f"Live worker processes: {len(children)}")

    worker_info = []
    total_worker_memory = 0.0

    for idx, child in enumerate(children):
        try:
            mem_gb = _bytes_to_gb(child.memory_info().rss)
            total_worker_memory += mem_gb
            entry = {"pid": child.pid, "memory_gb": mem_gb}

            if detailed:
                entry["cpu_percent"] = child.cpu_percent(interval=0.1)
                entry["num_threads"] = child.num_threads()

            worker_info.append(entry)

            if detailed:
                print(
                    f"  Worker {idx} (PID {child.pid}): "
                    f"{mem_gb:.2f} GB, CPU {entry['cpu_percent']:.1f}%, "
                    f"Threads {entry['num_threads']}"
                )
            else:
                print(f"  Worker {idx} (PID {child.pid}): {mem_gb:.2f} GB")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print(f"  Worker {idx}: process terminated or inaccessible")

    diagnostics["worker_info"] = worker_info
    diagnostics["total_worker_memory_gb"] = total_worker_memory
    diagnostics["total_memory_gb"] = main_rss_gb + total_worker_memory

    print(f"\nTotal worker memory: {total_worker_memory:.2f} GB")
    print(f"Total system usage: {main_rss_gb + total_worker_memory:.2f} GB")

    if HAS_TORCH:
        gc.collect()
        loaders = [
            obj for obj in gc.get_objects() if "DataLoader" in str(type(obj))
        ]
        diagnostics["dangling_loaders"] = len(loaders)
        print(f"Dangling DataLoader objects: {len(loaders)}")

    print("=" * 60 + "\n")
    return diagnostics


def check_dataloader_health(dataloader: Any) -> Dict[str, Any]:
    """
    Perform a set of heuristic checks on a DataLoader configuration.
    """
    health = {
        "issues": [],
        "warnings": [],
        "recommendations": [],
        "config": {},
    }

    config = {
        "batch_size": dataloader.batch_size,
        "num_workers": dataloader.num_workers,
        "pin_memory": dataloader.pin_memory,
        "drop_last": dataloader.drop_last,
        "persistent_workers": getattr(dataloader, "persistent_workers", False),
    }
    health["config"] = config

    if config["num_workers"] == 0:
        health["warnings"].append(
            "Using 0 workers (single-threaded). Consider num_workers > 0 for faster loading."
        )
    elif config["num_workers"] > 8:
        health["warnings"].append(
            f"High num_workers ({config['num_workers']}). May cause memory issues."
        )
        health["recommendations"].append(
            "Consider reducing num_workers to the 4-8 range for better memory efficiency."
        )

    if HAS_TORCH and torch.cuda.is_available() and not config["pin_memory"]:
        health["recommendations"].append(
            "CUDA available but pin_memory=False. Set pin_memory=True for faster GPU transfer."
        )

    if config["num_workers"] > 0 and not config["persistent_workers"]:
        health["recommendations"].append(
            "Consider persistent_workers=True to avoid recreating workers each epoch."
        )

    if config["batch_size"] == 1:
        health["warnings"].append(
            "Batch size is 1. This may be very slow for training."
        )

    if HAS_PSUTIL:
        vm = psutil.virtual_memory()
        if vm.percent > 90:
            health["issues"].append(
                f"System memory usage is high ({vm.percent:.1f}%). Risk of OOM."
            )

    return health


def get_worker_info_summary() -> Dict[str, Any]:
    """
    Return a summary of active DataLoader workers.
    """
    if not HAS_PSUTIL:
        return {"error": "psutil not available"}

    try:
        process = psutil.Process()
        children = process.children(recursive=True)
        return {
            "main_pid": process.pid,
            "active_workers": len(children),
            "worker_pids": [child.pid for child in children],
            "main_memory_mb": process.memory_info().rss / 1024**2,
            "worker_memory_mb": [
                child.memory_info().rss / 1024**2 for child in children
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


__all__ = [
    "show_dataloader_diagnostics",
    "check_dataloader_health",
    "get_worker_info_summary",
]

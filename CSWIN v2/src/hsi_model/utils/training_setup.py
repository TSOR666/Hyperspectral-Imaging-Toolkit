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
import contextlib
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    dataset_source = str(
        config.get("dataset_source", config.get("data_source", "mst"))
    ).strip().lower()

    uses_huggingface = dataset_source in {"huggingface", "hf", "hf_arad", "arad_hsdb"}

    if not uses_huggingface:
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {data_dir.resolve()} "
                "(set `data_dir` in the Hydra config or environment)."
            )

        required_dirs = ["Train_RGB", "Train_Spec", "split_txt"]
        for req_dir in required_dirs:
            full_path = data_dir / req_dir
            if not full_path.exists():
                raise FileNotFoundError(f"Required MST++ directory not found: {full_path}")

        valid_rgb_dirs = ("Valid_RGB", "Validation_RGB", "Val_RGB", "Test_RGB", "Train_RGB")
        valid_spec_dirs = ("Valid_Spec", "Validation_Spec", "Val_Spec", "Test_Spec", "Train_Spec")
        if not any((data_dir / name).exists() for name in valid_rgb_dirs):
            raise FileNotFoundError(
                f"Required validation RGB directory not found under {data_dir}; "
                f"expected one of {valid_rgb_dirs}"
            )
        if not any((data_dir / name).exists() for name in valid_spec_dirs):
            raise FileNotFoundError(
                f"Required validation spectral directory not found under {data_dir}; "
                f"expected one of {valid_spec_dirs}"
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config["data_dir"] = str(data_dir)
    config["checkpoint_dir"] = str(checkpoint_dir)
    config["log_dir"] = str(log_dir)

    if uses_huggingface:
        setup_logger.info("Dataset source: %s (local data_dir validation skipped)", dataset_source)
    else:
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


_DTYPE_ALIASES: Dict[str, Optional[torch.dtype]] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": None,
    "float32": None,
    "off": None,
    "none": None,
    "disabled": None,
}


def pick_amp_dtype(config: Dict[str, Any]) -> Optional[torch.dtype]:
    """Choose an autocast dtype based on config + GPU capability.

    Decision rule (when ``mixed_precision_dtype == 'auto'``, the default):

    * compute capability >= 8.0 (A100, A40, H100, RTX 30/40)  -> bf16
      bf16 has fp32's 8-bit exponent, so backward gradients cannot overflow
      the way fp16 does; this removes the GradScaler-cycling-down failure
      mode entirely on Ampere+ GPUs.
    * compute capability >= 7.0 (V100, T4)  -> fp16
      Tensor Cores accelerate fp16 here but bf16 isn't supported; fall back.
    * older GPUs / CPU                       -> None (run fp32, skip autocast)

    The explicit overrides ``"bf16"``, ``"fp16"``, ``"fp32"`` (and aliases)
    bypass the detection and pick the named dtype, returning ``None`` for
    fp32 to mean "do not enable autocast".

    Legacy ``mixed_precision: bool`` is honoured: ``False`` forces fp32,
    ``True`` falls through to the auto rule.
    """
    override = config.get("mixed_precision_dtype")
    if override is None and "mixed_precision" in config:
        # Map the legacy bool. True -> let auto-detect choose; False -> off.
        if not config.get("mixed_precision", True):
            return None
        override = "auto"

    if override is None:
        override = "auto"

    if isinstance(override, str):
        key = override.strip().lower()
        if key == "auto":
            pass  # fall through to detection
        elif key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
        else:
            raise ValueError(
                f"Unknown mixed_precision_dtype={override!r}; expected one of "
                "'auto', 'bf16', 'fp16', 'fp32' (or aliases)."
            )
    elif isinstance(override, torch.dtype):
        # Explicit dtype object — trust it.
        return override if override in (torch.bfloat16, torch.float16) else None

    # Auto path.
    if not torch.cuda.is_available():
        return None
    try:
        major, _minor = torch.cuda.get_device_capability()
    except (RuntimeError, AssertionError):
        return None

    if major >= 8:
        return torch.bfloat16
    if major >= 7:
        return torch.float16
    return None


def _normalize_cuda_rng_state_all(cuda_rng_state_all: Any) -> list[torch.Tensor]:
    """Return CUDA RNG states as CPU uint8 tensors for torch.cuda restore."""
    if isinstance(cuda_rng_state_all, torch.Tensor):
        states = [cuda_rng_state_all]
    else:
        states = list(cuda_rng_state_all)

    normalized = []
    for state in states:
        if isinstance(state, torch.Tensor):
            normalized.append(state.detach().to(device="cpu", dtype=torch.uint8))
        else:
            normalized.append(torch.as_tensor(state, dtype=torch.uint8, device="cpu"))
    return normalized


class GeneratorEMA:
    """Exponential moving average (EMA) of a module's float parameters.

    Maintains a shadow copy updated after each optimizer step as::

        shadow = decay * shadow + (1 - decay) * param

    Only trainable floating-point parameters are tracked; buffers are left
    untouched on purpose (the generator's ``iteration_count`` is a counter, and
    its GroupNorm layers keep no running statistics, so there is nothing to
    average). Use :meth:`average_parameters` to temporarily swap the EMA
    weights into the live module for validation / checkpoint snapshotting, then
    automatically restore the training weights.

    The EMA produces a smoother set of weights than the raw SGD/Adam iterate,
    which damps the late-epoch metric noise typical of GAN training and makes
    the *final* weights as good as the best-epoch ones.
    """

    def __init__(self, module: "torch.nn.Module", decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in module.named_parameters()
            if param.requires_grad and param.dtype.is_floating_point
        }
        self._backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, module: "torch.nn.Module") -> None:
        """Pull the shadow weights toward the module's current weights."""
        decay = self.decay
        for name, param in module.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            if shadow.device != param.device:
                shadow = shadow.to(param.device)
                self.shadow[name] = shadow
            shadow.mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def copy_to(self, module: "torch.nn.Module") -> None:
        """Overwrite the module's parameters with the EMA (shadow) weights."""
        for name, param in module.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is not None:
                param.data.copy_(shadow.to(param.device))

    @torch.no_grad()
    def store(self, module: "torch.nn.Module") -> None:
        """Snapshot the module's current (training) weights for later restore."""
        self._backup = {
            name: param.detach().clone()
            for name, param in module.named_parameters()
            if name in self.shadow
        }

    @torch.no_grad()
    def restore(self, module: "torch.nn.Module") -> None:
        """Put the snapshotted training weights back into the module."""
        if self._backup is None:
            return
        for name, param in module.named_parameters():
            backup = self._backup.get(name)
            if backup is not None:
                param.data.copy_(backup)
        self._backup = None

    @contextlib.contextmanager
    def average_parameters(self, module: "torch.nn.Module"):
        """Temporarily swap EMA weights into ``module`` for the block body."""
        self.store(module)
        self.copy_to(module)
        try:
            yield
        finally:
            self.restore(module)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(
        self, state: Dict[str, Any], device: Optional[torch.device] = None
    ) -> None:
        """Restore shadow weights from a checkpoint, tolerating missing keys."""
        if not state:
            return
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {}) or {}
        loaded = skipped = 0
        for name, tensor in shadow.items():
            target = self.shadow.get(name)
            if target is not None and isinstance(tensor, torch.Tensor):
                src = tensor if device is None else tensor.to(device)
                target.copy_(src.to(target.device))
                loaded += 1
            else:
                skipped += 1
        logger.info("Restored EMA shadow: %d params loaded, %d skipped", loaded, skipped)


def resume_training_state(
    checkpoint_path: str,
    model: "torch.nn.Module",
    optimizers: Dict[str, "torch.optim.Optimizer"],
    schedulers: Dict[str, Any],
    scalers: Dict[str, Any],
    device: torch.device,
    ema: Optional["GeneratorEMA"] = None,
) -> Dict[str, Any]:
    """Restore full training state from a checkpoint saved by either trainer.

    Loads (in this order, all keys optional except ``state_dict``):
      * ``state_dict`` (or ``model_state_dict``) -> model weights
      * ``optimizer_g`` / ``optimizer_d``         -> optimizer states
      * ``scheduler_g`` / ``scheduler_d``         -> LR-scheduler states
      * ``scaler_g`` / ``scaler_d``               -> GradScaler states
      * ``torch_rng_state`` / ``cuda_rng_state_all`` / ``numpy_rng_state``
        -> RNG states (deterministic resume)

    Returns ``resume_info`` shaped for the trainer loop:
        {"iteration": int, "best_mrae": float, "epoch": int}

    Uses ``weights_only=False`` because the checkpoint contains non-tensor
    state (config dict, numpy RNG, etc.) that PyTorch's safe-pickle does
    not handle. Only resume from checkpoints YOU produced.
    """
    if not checkpoint_path:
        return {}
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Model state. Tolerate both 'state_dict' (trainer schema) and
    # 'model_state_dict' (utils.checkpoint schema).
    state_dict = ck.get("state_dict") or ck.get("model_state_dict")
    if not state_dict:
        raise KeyError(
            f"Checkpoint {checkpoint_path} contains no model state "
            f"(looked for 'state_dict' and 'model_state_dict')."
        )
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state_dict, strict=True)

    # Optimizers / schedulers / scalers — each block tolerates missing keys
    # because pre-fix checkpoints may not have every component.
    for name, opt in optimizers.items():
        if opt is not None and name in ck:
            opt.load_state_dict(ck[name])
    for name, sched in schedulers.items():
        if sched is not None and name in ck:
            sched.load_state_dict(ck[name])
    for name, sc in scalers.items():
        if sc is not None and name in ck:
            sc.load_state_dict(ck[name])

    # EMA shadow weights — optional. Missing on pre-EMA checkpoints, in which
    # case the freshly-initialized shadow (a clone of the loaded model weights)
    # is the correct starting point, so silently skip.
    if ema is not None and ck.get("ema"):
        try:
            ema.load_state_dict(ck["ema"], device=device)
        except (RuntimeError, TypeError, ValueError) as e:
            logger.warning("Could not restore EMA state: %s", e)

    # RNG state — best-effort. Mismatched CUDA device counts at resume can
    # raise; log and continue rather than die because RNG drift across
    # resumes is acceptable for GAN training.
    if "torch_rng_state" in ck:
        try:
            torch.set_rng_state(ck["torch_rng_state"].cpu())
        except (RuntimeError, AttributeError) as e:
            logger.warning("Could not restore torch RNG state: %s", e)
    if ck.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        try:
            cuda_rng_state_all = _normalize_cuda_rng_state_all(ck["cuda_rng_state_all"])
            torch.cuda.set_rng_state_all(cuda_rng_state_all)
        except (RuntimeError, TypeError, ValueError) as e:
            logger.warning("Could not restore CUDA RNG state: %s", e)
    if "numpy_rng_state" in ck:
        try:
            np.random.set_state(ck["numpy_rng_state"])
        except (TypeError, ValueError) as e:
            logger.warning("Could not restore numpy RNG state: %s", e)

    info = {
        "iteration": int(ck.get("iter", ck.get("iteration", 0))),
        "epoch": int(ck.get("epoch", 0)),
        "best_mrae": float(ck.get("best_mrae", float("inf"))),
    }
    logger.info(
        "Resumed from %s | iter=%d, epoch=%d, best_mrae=%.6f",
        checkpoint_path,
        info["iteration"],
        info["epoch"],
        info["best_mrae"],
    )
    return info


__all__ = [
    "setup_paths",
    "setup_distributed_training",
    "setup_seed",
    "cleanup",
    "pick_amp_dtype",
    "resume_training_state",
    "GeneratorEMA",
]

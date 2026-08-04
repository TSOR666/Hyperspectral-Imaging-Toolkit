# src/hsi_model/train_generator.py
"""
Generator-only (no-GAN) training for CSWin RGB->HSI reconstruction.

This is the post-GAN trainer: the discriminator, Sinkhorn-OT adversarial loss,
and R1 regularization are gone. The objective is a configurable reconstruction
loss (see ``objective`` in config.yaml; the active recipe uses annealed
MRAE-only training while validation still reports exact MRAE).

Dropping the GAN removes 4 discriminator forwards + a discriminator backward +
the per-batch Sinkhorn loop per iteration, which frees most of the compute
budget for a larger/better generator.

Reuses the shared infrastructure (dataset, metrics, EMA, warmup-cosine
scheduler, atomic checkpointing) from training_setup / train_optimized.

Checkpoint format: ``state_dict`` holds the BARE generator weights. To load for
inference:

    gen = NoiseRobustCSWinGenerator(ckpt['config'])
    gen.load_state_dict(ckpt['state_dict'])   # use best_model.pth for EMA weights
"""

import hashlib
import logging
import math
import os
import statistics
import sys
import time
import traceback
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
try:  # torch>=2.3 exposes the non-deprecated GradScaler here (typing only).
    from torch.amp import GradScaler
except ImportError:  # pragma: no cover - very old torch
    from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
from hsi_model.models.losses_consolidated import (
    L1PlusMRAELoss,
    MRAELoss,
    MRAEPlusL1Loss,
    RelativeMRAELoss,
)
from hsi_model.constants import (
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_WARMUP_STEPS,
    CHECKPOINT_BEST_NAME,
    CHECKPOINT_LATEST_NAME,
    CHECKPOINT_KEEP_COUNT,
)
from hsi_model.utils import setup_logging, MetricsLogger
from hsi_model.utils.patch_inference import PatchInference
from hsi_model.utils.training_setup import (
    autocast_context,
    cleanup,
    GeneratorEMA,
    load_finetune_weights,
    make_grad_scaler,
    pick_amp_dtype,
    resume_training_state,
    resolve_resume_stage_position,
    setup_distributed_training,
    setup_paths,
    setup_seed,
)
from hsi_model.utils.data import (
    DistributedEvalSampler,
    create_training_datasets,
    mst_to_gan_batch,
    compute_mst_center_crop_metrics,
    make_worker_init_fn,
)
# Reuse the warmup->cosine scheduler and checkpoint helpers (acyclic imports).
from hsi_model.training_script_fixed import WarmupCosineScheduler
from hsi_model.train_optimized import (
    _atomic_torch_save,
    _prune_epoch_checkpoints,
    first_nonfinite_parameter_name,
    memory_cleanup,
    report_memory,
)

logger = logging.getLogger(__name__)

_ANNEALED_MRAE_OBJECTIVES = ("mrae_annealed", "annealed_mrae")


def _update_early_stopping(
    current_mrae: float,
    best_mrae: float,
    bad_epochs: int,
    patience: int,
    min_delta: float,
    epoch: int,
    warmup_epochs: int,
) -> tuple[float, int, bool]:
    """Update validation-MRAE early-stopping state."""
    improved = math.isfinite(current_mrae) and current_mrae < best_mrae - min_delta
    if improved:
        best_mrae = current_mrae
        bad_epochs = 0
    elif epoch >= warmup_epochs:
        bad_epochs += 1

    should_stop = (
        patience > 0
        and epoch >= warmup_epochs
        and bad_epochs >= patience
    )
    return best_mrae, bad_epochs, should_stop


def _sync_early_stopping_state(
    record_mrae_loss: float,
    early_stopping_best_mrae: float,
    early_stopping_bad_epochs: int,
    should_stop: bool,
    device: torch.device,
    distributed: bool,
) -> tuple[float, float, int, bool]:
    """Broadcast rank-zero stopping state so every DDP worker exits together."""
    if not distributed:
        return (
            record_mrae_loss,
            early_stopping_best_mrae,
            early_stopping_bad_epochs,
            should_stop,
        )

    state = torch.tensor(
        [
            record_mrae_loss,
            early_stopping_best_mrae,
            float(early_stopping_bad_epochs),
            float(should_stop),
        ],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.broadcast(state, src=0)
    return (
        float(state[0].item()),
        float(state[1].item()),
        int(state[2].item()),
        bool(state[3].item()),
    )


def build_criterion(config: Dict[str, Any]) -> nn.Module:
    """Build the configured reconstruction criterion (MRAE by default)."""
    objective = str(config.get("objective", "mrae")).lower()
    if objective in ("l1", "mae"):
        # HSIFormer/SS-Transformer (paper Sec 4.2) trains with L1 on [0,1]
        # targets; this is the validated objective for this architecture.
        return nn.L1Loss()
    if objective in _ANNEALED_MRAE_OBJECTIVES:
        return MRAELoss(
            epsilon=float(
                config.get("mrae_epsilon_start", config.get("mrae_epsilon", 1e-3))
            )
        )
    if objective in ("mrae", "mst", "mst++", "mrae_stable", "stable_mrae"):
        return MRAELoss(epsilon=float(config.get("mrae_epsilon", 1e-8)))
    if objective in ("mrae_l1", "mrae+l1", "l1_mrae"):
        return MRAEPlusL1Loss(
            mrae_epsilon=float(config.get("mrae_epsilon", 1e-2)),
            l1_weight=float(config.get("l1_weight", 0.3)),
        )
    if objective in ("l1_with_mrae", "l1+mrae_small", "balanced_l1_mrae"):
        return L1PlusMRAELoss(
            mrae_epsilon=float(config.get("mrae_epsilon", 1e-2)),
            mrae_weight=float(config.get("mrae_weight", 0.1)),
            l1_weight=float(config.get("l1_weight", 1.0)),
        )
    if objective in ("relative_mrae", "smooth_mrae"):
        return RelativeMRAELoss(
            denominator_epsilon=float(config.get("relative_mrae_epsilon", 1e-2))
        )
    raise ValueError(
        f"Unknown objective={objective!r}. Expected one of: l1, mrae, mrae_stable, "
        "mrae_annealed, mrae_l1, "
        "l1_with_mrae, relative_mrae (a typo here silently changed the training "
        "objective in earlier versions; failing loudly instead)."
    )


def _resolve_annealed_mrae_epsilon(config: Dict[str, Any], iteration: int) -> float:
    """Log-linearly decay the MRAE denominator floor toward the exact metric."""
    start = max(
        float(config.get("mrae_epsilon_start", config.get("mrae_epsilon", 1e-3))),
        1e-12,
    )
    end = max(float(config.get("mrae_epsilon_end", 1e-8)), 1e-12)
    anneal_iters = max(int(config.get("mrae_epsilon_anneal_iters", 50_000)), 1)
    progress = min(max(float(iteration), 0.0), float(anneal_iters)) / float(anneal_iters)
    log_epsilon = math.log(start) + progress * (math.log(end) - math.log(start))
    return float(math.exp(log_epsilon))


def update_mrae_epsilon_schedule(
    criterion: nn.Module,
    config: Dict[str, Any],
    iteration: int,
) -> Optional[float]:
    """Update scheduled MRAE epsilon in-place and return the active value."""
    objective = str(config.get("objective", "mrae")).lower()
    if objective not in _ANNEALED_MRAE_OBJECTIVES:
        return None

    epsilon = _resolve_annealed_mrae_epsilon(config, iteration)
    updated = False
    for module in criterion.modules():
        if isinstance(module, MRAELoss):
            module.epsilon = epsilon
            updated = True
    if not updated:
        raise TypeError("objective=mrae_annealed requires a criterion containing MRAELoss.")
    return epsilon


def _restore_from_latest_checkpoint(
    generator: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    ema: Optional["GeneratorEMA"],
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_dir: str,
    device: torch.device,
    lr_decay: float,
    train_logger: logging.Logger,
) -> Optional[int]:
    """Roll the model/optimizer back to the last saved checkpoint after a
    divergence, and permanently decay the LR.

    Restores generator/optimizer/scaler/EMA (the state that blows up), then
    scales ``scheduler.base_lrs`` so every subsequent cosine LR is smaller. The
    scheduler's ``last_epoch`` is deliberately NOT rewound: the training loop's
    iteration counter keeps advancing, so we accept losing the (already
    diverging) iterations since the checkpoint rather than desync the schedule.

    Returns the checkpoint's iteration on success, or None if no checkpoint
    exists yet (caller should then abort — nothing safe to fall back to).
    """
    path = os.path.join(checkpoint_dir, CHECKPOINT_LATEST_NAME)
    if not os.path.exists(path):
        return None

    # weights_only=False: the checkpoint holds config/RNG/non-tensor state, same
    # as utils.training_setup.resume_training_state. Only ever loads our own file.
    ck = torch.load(path, map_location=device, weights_only=False)
    target = generator.module if hasattr(generator, "module") else generator
    state_dict = ck.get("state_dict") or ck.get("model_state_dict")
    if not state_dict:
        raise KeyError(f"Rollback checkpoint {path} has no model state.")
    target.load_state_dict(state_dict, strict=True)

    if ck.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(ck["optimizer"])
        except (RuntimeError, ValueError, KeyError) as e:
            train_logger.warning("Rollback: could not restore optimizer state: %s", e)
            optimizer.zero_grad(set_to_none=True)
    if scaler is not None and ck.get("scaler") is not None:
        try:
            scaler.load_state_dict(ck["scaler"])
        except (RuntimeError, ValueError, KeyError) as e:
            train_logger.warning("Rollback: could not restore GradScaler state: %s", e)
    if ema is not None:
        ema.reinit_from(target)
        if ck.get("ema"):
            try:
                ema.load_state_dict(ck["ema"], device=device)
            except (RuntimeError, TypeError, ValueError) as e:
                train_logger.warning("Rollback: could not restore EMA state: %s", e)

    if lr_decay != 1.0 and hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [float(lr) * lr_decay for lr in scheduler.base_lrs]
        for group in optimizer.param_groups:
            group["lr"] = float(group.get("lr", 0.0)) * lr_decay

    return int(ck.get("iter", -1))


def _resolve_stages(config: Dict[str, Any]) -> list:
    """Resolve the progressive-training stage list (HSIFormer Sec 4.2).

    Each stage = {patch_size, iterations, init_lr, batch_size, warmup_steps}.
    If ``progressive_stages`` is absent, fall back to a single stage built from
    the top-level ``patch_size`` / ``epochs`` * ``iterations_per_epoch`` /
    ``generator_lr`` / ``batch_size`` (the standard single-resolution run used
    for ablations).
    """
    default_lr = float(config.get("generator_lr", config.get("learning_rate", 4e-4)))
    default_warmup = int(config.get("warmup_steps", DEFAULT_WARMUP_STEPS))
    default_iters = int(config.get("iterations_per_epoch", 1000)) * int(config.get("epochs", 100))
    raw = config.get("progressive_stages")
    if raw:
        stages = []
        for i, s in enumerate(raw):
            s = dict(s)
            stages.append({
                "patch_size": int(s.get("patch_size", config.get("patch_size", 128))),
                "iterations": int(s.get("iterations", default_iters)),
                "init_lr": float(s.get("init_lr", default_lr)),
                "batch_size": int(s.get("batch_size", config.get("batch_size", 20))),
                # Warmup only the first stage by default; later stages continue
                # from trained weights at a low LR and need no warmup.
                "warmup_steps": int(s.get("warmup_steps", default_warmup if i == 0 else 0)),
            })
        return stages
    return [{
        "patch_size": int(config.get("patch_size", 128)),
        "iterations": default_iters,
        "init_lr": default_lr,
        "batch_size": int(config.get("batch_size", 20)),
        "warmup_steps": default_warmup,
    }]


def _contract_value_matches(actual: Any, expected: Any) -> bool:
    """Compare resolved config values without making YAML numeric types brittle."""
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(
                float(actual),
                float(expected),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def _validate_training_contract(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Fail early when a named recipe silently resolves to another experiment.

    Hydra overrides are intentionally flexible, but that flexibility can make
    a named production run resolve to a shortened diagnostic schedule or an
    incompatible objective. A recipe can opt into this guard with
    ``training_contract``. Intentional ablations may set that key to ``null``
    (or update the expected values explicitly).
    """
    stages = _resolve_stages(config)
    contract = config.get("training_contract")
    if not contract:
        return stages

    contract = dict(contract)
    name = str(contract.get("name", "unnamed"))
    errors: list[str] = []

    expected_values = dict(contract.get("expected", {}))
    for key, expected in expected_values.items():
        actual = config.get(key)
        if not _contract_value_matches(actual, expected):
            errors.append(f"{key}={actual!r} (expected {expected!r})")

    expected_stages = contract.get("stages")
    if expected_stages is not None:
        expected_stages = [dict(stage) for stage in expected_stages]
        if len(stages) != len(expected_stages):
            errors.append(
                f"stage_count={len(stages)} (expected {len(expected_stages)})"
            )
        else:
            for index, (actual_stage, expected_stage) in enumerate(
                zip(stages, expected_stages)
            ):
                for key, expected in expected_stage.items():
                    actual = actual_stage.get(key)
                    if not _contract_value_matches(actual, expected):
                        errors.append(
                            f"stage[{index}].{key}={actual!r} "
                            f"(expected {expected!r})"
                        )

    if errors:
        details = "; ".join(errors)
        raise ValueError(
            f"Resolved training configuration violates contract {name!r}: "
            f"{details}. Refusing to launch a mislabeled run. For an "
            "intentional ablation, set training_contract=null or update the "
            "contract together with the overrides."
        )
    return stages


def _persist_resolved_config(config: Dict[str, Any]) -> tuple[str, Path]:
    """Atomically persist the exact resolved run config and return its hash."""
    fingerprint_config = dict(config)
    fingerprint_config.pop("config_fingerprint", None)
    config_text = OmegaConf.to_yaml(
        OmegaConf.create(fingerprint_config), resolve=True
    )
    fingerprint = hashlib.sha256(config_text.encode("utf-8")).hexdigest()[:12]
    config["config_fingerprint"] = fingerprint
    config_text = OmegaConf.to_yaml(OmegaConf.create(config), resolve=True)

    output_path = Path(str(config["log_dir"])) / "resolved_config.yaml"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(config_text, encoding="utf-8")
    os.replace(temporary_path, output_path)
    return fingerprint, output_path


_WARNED_RESIDENT_SPAWN = False


def _warn_resident_workers_under_spawn(config, num_workers):
    """Warn when resident scenes get copied into each spawn worker.

    In ``standard``/``float16`` memory modes the dataset holds every scene in
    RAM. Under the *fork* start method (Linux default) workers inherit those
    arrays copy-on-write and read-only access never duplicates them. Under
    *spawn* (Windows/macOS, or an explicit context) the whole resident set is
    pickled into every worker -> host RAM scales with ``num_workers`` and
    startup is slow. ``memory_mode=lazy`` keeps fp32 targets and reads on
    demand, avoiding both.
    """
    global _WARNED_RESIDENT_SPAWN
    if _WARNED_RESIDENT_SPAWN or num_workers <= 0:
        return
    memory_mode = str(config.get("memory_mode", "standard")).strip().lower()
    if memory_mode not in ("standard", "float16"):
        return
    try:
        import multiprocessing as _mp

        start_method = _mp.get_start_method(allow_none=True) or _mp.get_start_method()
    except (ValueError, RuntimeError):
        start_method = None
    if start_method == "spawn":
        _WARNED_RESIDENT_SPAWN = True
        logging.getLogger("hsi_model.training").warning(
            "memory_mode=%r with num_workers=%d under the 'spawn' start method "
            "pickles the full resident scene set into EVERY worker (host RAM "
            "scales with num_workers, slow startup). Use memory_mode='lazy' to "
            "read scenes on demand, or num_workers=0.",
            memory_mode, num_workers,
        )


def _build_train_loader(dataset, batch_size, config, distributed, seed, rank):
    """Build a training DataLoader for a given dataset/batch_size."""
    seed_base = seed + rank * 1000
    # DistributedSampler requires the SAME seed on every rank: each rank takes
    # its rank-th stride of one shared permutation. Per-rank seeds would make
    # the rank shards overlap (~35% duplicated samples per epoch at world=4).
    # Rank-dependent augmentation randomness comes from worker_init_fn_mst.
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed) if distributed else None
    num_workers = int(config.get("num_workers", 8))
    _warn_resident_workers_under_spawn(config, num_workers)
    return DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=make_worker_init_fn(seed_base, rank),
        persistent_workers=(num_workers > 0),
    )


def validate_generator(
    net: nn.Module,
    val_dataset: Any,
    criterion: nn.Module,
    device: torch.device,
    iteration: int,
    config: Dict[str, Any],
    distributed: bool,
    seed: int,
    rank: int,
) -> Dict[str, float]:
    """MST++-protocol validation for the bare generator (temporary DataLoader)."""
    update_mrae_epsilon_schedule(criterion, config, iteration)
    # DDP forward may broadcast buffers. Non-padding validation shards can have
    # different lengths (or be empty), so forwarding through the wrapper would
    # issue a different number of collectives per rank and can deadlock. The
    # synchronized underlying module is sufficient for inference-only eval.
    eval_net = net.module if hasattr(net, "module") else net
    eval_net.eval()
    val_logger = logging.getLogger("hsi_model.validation")

    val_sampler = None
    if distributed:
        val_sampler = DistributedEvalSampler(
            val_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
        )
    # This loader is rebuilt and torn down at EVERY validation (per
    # ``iterations_per_epoch``), and ``persistent_workers=False`` means each
    # rebuild spawns ``num_workers`` fresh processes. Under spawn (Windows) /
    # standard|float16 memory modes that also re-pickles the whole resident
    # scene set into every worker — dozens of GB copied per validation. The
    # val set is tiny (val_batch_size=1, full-frame forward dominates), so cap
    # workers low. ``val_num_workers`` overrides; default min(num_workers, 2).
    train_num_workers = int(config.get("num_workers", 8))
    val_num_workers = config.get("val_num_workers", None)
    if val_num_workers is None:
        val_num_workers = min(train_num_workers, 2)
    val_num_workers = max(0, int(val_num_workers))
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.get("val_batch_size", 1),
        shuffle=False,
        sampler=val_sampler,
        num_workers=val_num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=make_worker_init_fn(seed, rank),
        persistent_workers=False,
    )

    amp_dtype = pick_amp_dtype(config) if device.type == "cuda" else None
    use_amp = amp_dtype is not None
    autocast_dtype = amp_dtype if amp_dtype is not None else torch.float16
    clamp_prediction = bool(config.get("validation_clamp_output", False))
    report_raw_mrae = bool(config.get("validation_report_raw_mrae", False))
    metric_keys = ["psnr", "mrae", "rmse", "ssim", "sam", "mae"]
    if clamp_prediction:
        metric_keys.append("out_of_range_fraction")
        if report_raw_mrae:
            metric_keys.append("raw_mrae")

    total_loss = torch.tensor(0.0, device=device)
    # Keep the reduction schema identical on every rank, including ranks whose
    # non-padding shard contains zero samples.
    total_metrics: Dict[str, torch.Tensor] = {
        key: torch.tensor(0.0, device=device) for key in metric_keys
    }
    num_samples = 0

    patch_infer = None
    if bool(config.get("validation_tiled_inference", False)):
        patch_infer = PatchInference(
            model=eval_net,
            patch_size=int(
                config.get("validation_patch_size", config.get("patch_size", 128))
            ),
            overlap=int(config.get("validation_patch_overlap", 16)),
            batch_size=int(config.get("validation_patch_batch_size", 4)),
            device=device,
            amp_dtype=amp_dtype,
            apply_sigmoid=False,
        )
    validation_max_batches = config.get("validation_max_batches", None)
    if validation_max_batches is not None:
        validation_max_batches = int(validation_max_batches)

    validation_error: Optional[str] = None
    with torch.inference_mode():
        for batch_idx, (bgr_batch, hyper_batch) in enumerate(val_loader):
            if validation_max_batches is not None and validation_max_batches > 0:
                if batch_idx >= validation_max_batches:
                    break
            elif validation_max_batches == 0:
                break
            try:
                rgb_tensor, hsi_tensor = mst_to_gan_batch(bgr_batch, hyper_batch)
                rgb_tensor = rgb_tensor.to(device, non_blocking=True)
                hsi_tensor = hsi_tensor.to(device, non_blocking=True)
                if patch_infer is not None:
                    pred_hsi = torch.cat(
                        [
                            patch_infer.predict(
                                rgb_tensor[index : index + 1],
                                show_progress=False,
                            )
                            for index in range(int(rgb_tensor.shape[0]))
                        ],
                        dim=0,
                    )
                else:
                    with autocast_context(device.type, use_amp, autocast_dtype):
                        pred_hsi = eval_net(rgb_tensor)
                batch_size = int(hsi_tensor.shape[0])
                metrics = compute_mst_center_crop_metrics(
                    pred_hsi,
                    hsi_tensor,
                    criterion=criterion,
                    clamp_prediction=clamp_prediction,
                    report_raw_mrae=report_raw_mrae,
                )
                if "loss" not in metrics:
                    raise RuntimeError("Validation criterion did not produce a loss.")
                loss = torch.tensor(float(metrics["loss"]), device=device)
                total_loss += loss * batch_size
                missing_metrics = set(metric_keys) - set(metrics)
                if missing_metrics:
                    raise RuntimeError(
                        "Validation metric schema changed unexpectedly; missing "
                        f"{sorted(missing_metrics)}"
                    )
                for key in metric_keys:
                    value = metrics[key]
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(value, device=device)
                    total_metrics[key] += value * batch_size
                num_samples += batch_size
            except Exception as e:  # noqa: BLE001 - validation must not kill training
                val_logger.warning("Validation error: %s", str(e))
                validation_error = f"batch {batch_idx}: {e}"
                break

    del val_loader

    validation_failed = torch.tensor(
        1 if validation_error is not None else 0,
        device=device,
        dtype=torch.int32,
    )
    if distributed and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            validation_failed, op=torch.distributed.ReduceOp.MAX
        )
    if validation_failed.item():
        detail = validation_error or "validation failed on another distributed rank"
        raise RuntimeError(
            f"Validation aborted because at least one batch failed ({detail}). "
            "Refusing to select checkpoints from partial or empty metrics."
        )

    # Each rank evaluates a disjoint, non-padding shard. Aggregate sample sums
    # before averaging so uneven shards and partial final batches are weighted
    # correctly.
    if distributed and torch.distributed.is_initialized():
        samples_t = torch.tensor(float(num_samples), device=device)
        torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(samples_t, op=torch.distributed.ReduceOp.SUM)
        for value in total_metrics.values():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        num_samples = int(samples_t.item())

    if num_samples <= 0:
        raise RuntimeError(
            "Validation produced zero samples. Check the validation split, "
            "validation_max_batches, and distributed world size."
        )

    denom = num_samples
    avg = {"gen_loss": total_loss.item() / denom}
    avg.update({k: v.item() / denom for k, v in total_metrics.items()})
    val_logger.info(
        "Validation Iter: %d | MRAE: %.4f | PSNR: %.2fdB",
        iteration, avg.get("mrae", 0.0), avg.get("psnr", 0.0),
    )
    return avg


def _forward_backward_microbatched(
    net: nn.Module,
    criterion: nn.Module,
    scaler: GradScaler,
    rgb_batch: torch.Tensor,
    hsi_batch: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    autocast_dtype: torch.dtype,
    check_finite: bool,
    microbatch_size: int,
) -> tuple[torch.Tensor, bool]:
    """Run one optimizer-step batch, optionally split into gradient microbatches.

    The returned loss is the sample-weighted mean of the unscaled chunk losses.
    Scaling each chunk by its share of the original batch keeps the accumulated
    gradient equivalent to a single forward/backward over that batch (apart
    from harmless floating-point reduction order differences).  DDP all-reduce
    is deferred until the final chunk.
    """
    batch_size = int(rgb_batch.shape[0])
    if batch_size < 1:
        raise ValueError("Training batch must contain at least one sample.")
    microbatch_size = max(1, min(int(microbatch_size), batch_size))
    step_loss = torch.zeros((), device=device)
    pred_is_finite = True

    for start in range(0, batch_size, microbatch_size):
        end = min(start + microbatch_size, batch_size)
        chunk_size = end - start
        is_final_chunk = end == batch_size
        no_sync = getattr(net, "no_sync", None)
        sync_context = (
            no_sync()
            if callable(no_sync) and not is_final_chunk
            else nullcontext()
        )
        with sync_context:
            rgb_tensor = rgb_batch[start:end].to(device, non_blocking=True)
            hsi_tensor = hsi_batch[start:end].to(device, non_blocking=True)
            with autocast_context(device.type, use_amp, autocast_dtype):
                pred = net(rgb_tensor)

            if check_finite and not torch.isfinite(pred).all():
                pred_is_finite = False
                break

            with autocast_context(device.type, use_amp, autocast_dtype):
                chunk_loss = criterion(pred.float(), hsi_tensor.float())
            chunk_weight = chunk_size / batch_size
            scaler.scale(chunk_loss * chunk_weight).backward()

        # Keep logging loss detached so this does not retain chunk graphs.
        step_loss = step_loss + chunk_loss.detach() * chunk_weight

    return step_loss, pred_is_finite


def _run_stage(
    net, generator, optimizer, scheduler, scaler, criterion, ema,
    train_loader, val_dataset, config, device, metrics_logger,
    distributed, seed, rank, stage_idx, stage_iterations,
    global_iter, record_mrae_loss, early_stopping_best_mrae,
    early_stopping_bad_epochs, early_stopping_enabled,
    start_stage_iter=0,
):
    """Run one progressive-training stage and return training/stopping state.

    ``global_iter`` is the cumulative iteration across all stages (used for epoch
    numbering, logging and checkpoints); ``stage_iter`` is local to this stage
    and bounds the loop / drives the per-stage cosine scheduler.
    """
    train_logger = logging.getLogger("hsi_model.training")
    per_epoch_iteration = config.get("iterations_per_epoch", 1000)
    amp_dtype = pick_amp_dtype(config) if device.type == "cuda" else None
    use_amp = amp_dtype is not None
    autocast_dtype = amp_dtype if amp_dtype is not None else torch.float16
    max_consecutive_nonfinite = int(config.get("max_consecutive_nonfinite_generator_outputs", 3))
    max_consecutive_ooms = max(1, int(config.get("max_consecutive_ooms", 3)))
    finite_check_interval = max(1, int(config.get("finite_check_interval", 100)))
    ema_update_every = max(1, int(config.get("ema_update_every", 1)))
    early_stopping_patience = max(0, int(config.get("early_stopping_patience", 0)))
    early_stopping_min_delta = max(0.0, float(config.get("early_stopping_min_delta", 0.0)))
    early_stopping_warmup_epochs = max(0, int(config.get("early_stopping_warmup_epochs", 0)))
    gradient_clip_norm = float(
        config.get("gradient_clip_norm", DEFAULT_GRADIENT_CLIP_NORM)
    )
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm < 0.0:
        raise ValueError("gradient_clip_norm must be finite and non-negative")
    # clip_grad_norm_(..., inf) still computes the total norm (and catches
    # NaN/Inf) without changing gradients. This preserves the existing finite
    # guard while allowing the benchmark recipe to disable clipping explicitly.
    effective_clip_norm = (
        gradient_clip_norm if gradient_clip_norm > 0.0 else float("inf")
    )

    # Divergence tripwire (see config.yaml). Auto-disabled under DDP, where a
    # per-rank loss spike cannot be acted on without a cross-rank collective.
    divergence_guard_enabled = bool(config.get("divergence_guard_enabled", True))
    divergence_guard_active = divergence_guard_enabled and not distributed
    if divergence_guard_enabled and distributed:
        train_logger.warning(
            "divergence_guard_enabled=true but this run is distributed; the "
            "tripwire is disabled under DDP."
        )
    divergence_spike_factor = float(config.get("divergence_spike_factor", 5.0))
    divergence_spike_patience = max(1, int(config.get("divergence_spike_patience", 3)))
    divergence_history_windows = max(5, int(config.get("divergence_history_windows", 50)))
    divergence_val_factor = float(config.get("divergence_val_factor", 3.0))
    divergence_lr_decay = float(config.get("divergence_lr_decay", 0.5))
    max_divergence_rollbacks = max(0, int(config.get("max_divergence_rollbacks", 2)))
    divergence_warmup = max(0, int(config.get("warmup_steps", DEFAULT_WARMUP_STEPS)))

    data_iter = iter(train_loader)
    is_distributed = isinstance(train_loader.sampler, DistributedSampler)
    epoch_losses = []
    loss_history: "deque[float]" = deque(maxlen=divergence_history_windows)
    divergence_spike_count = 0
    divergence_rollbacks = 0
    loader_epoch = 0
    consecutive_nonfinite = 0
    consecutive_ooms = 0
    stage_iter = start_stage_iter
    iteration = global_iter
    log_window_start = time.perf_counter()
    log_window_iteration = iteration
    # Start with the configured loader batch. If it does not fit, retain the
    # same effective optimizer batch by accumulating progressively smaller
    # chunks instead of repeating the impossible forward three times.
    adaptive_microbatch_size: Optional[int] = None
    retry_batch: Optional[tuple[Any, Any]] = None

    def _do_rollback(reason: str) -> bool:
        """Restore the last-good checkpoint and decay LR after a divergence.

        Returns True if recovered (caller should ``continue``), False if it
        could not recover (caller must abort loudly).
        """
        nonlocal divergence_rollbacks, divergence_spike_count
        divergence_rollbacks += 1
        train_logger.error(
            "DIVERGENCE at iter %d (%s); rollback %d/%d.",
            iteration, reason, divergence_rollbacks, max_divergence_rollbacks,
        )
        if divergence_rollbacks > max_divergence_rollbacks:
            return False
        restored_iter = _restore_from_latest_checkpoint(
            generator, optimizer, scaler, ema, scheduler,
            config["checkpoint_dir"], device, divergence_lr_decay, train_logger,
        )
        if restored_iter is None:
            train_logger.error(
                "No checkpoint to roll back to (divergence before the first "
                "validation); cannot recover."
            )
            return False
        base_lr = (
            scheduler.base_lrs[0]
            if getattr(scheduler, "base_lrs", None)
            else optimizer.param_groups[0]["lr"]
        )
        train_logger.error(
            "Rolled back to checkpoint iter %d; base LR decayed x%.3g -> %.3g. "
            "Resuming.",
            restored_iter, divergence_lr_decay, base_lr,
        )
        divergence_spike_count = 0
        loss_history.clear()
        epoch_losses.clear()
        optimizer.zero_grad(set_to_none=True)
        return True

    while stage_iter < stage_iterations:
        net.train()
        if retry_batch is not None:
            bgr_batch, hyper_batch = retry_batch
            retry_batch = None
        else:
            try:
                bgr_batch, hyper_batch = next(data_iter)
            except StopIteration:
                if is_distributed:
                    loader_epoch += 1
                    train_loader.sampler.set_epoch(loader_epoch)
                data_iter = iter(train_loader)
                bgr_batch, hyper_batch = next(data_iter)

        try:
            rgb_batch, hsi_batch = mst_to_gan_batch(bgr_batch, hyper_batch)
            step_batch_size = int(rgb_batch.shape[0])
            check_finite = iteration % finite_check_interval == 0
            if check_finite and (
                not torch.isfinite(rgb_batch).all()
                or not torch.isfinite(hsi_batch).all()
            ):
                train_logger.warning("Skipping non-finite batch at iteration %s", iteration)
                continue

            lr = optimizer.param_groups[0]["lr"]
            if hasattr(generator, "set_iteration"):
                generator.set_iteration(iteration)

            optimizer.zero_grad(set_to_none=True)
            update_mrae_epsilon_schedule(criterion, config, iteration)
            microbatch_size = min(
                adaptive_microbatch_size or step_batch_size,
                step_batch_size,
            )
            loss, pred_is_finite = _forward_backward_microbatched(
                net=net,
                criterion=criterion,
                scaler=scaler,
                rgb_batch=rgb_batch,
                hsi_batch=hsi_batch,
                device=device,
                use_amp=use_amp,
                autocast_dtype=autocast_dtype,
                check_finite=check_finite,
                microbatch_size=microbatch_size,
            )

            max_initial_train_loss = config.get("max_initial_train_loss", None)
            if stage_iter == 0 and max_initial_train_loss is not None:
                initial_loss = float(loss.detach().item())
                max_initial_train_loss = float(max_initial_train_loss)
                if (
                    math.isfinite(initial_loss)
                    and max_initial_train_loss > 0.0
                    and initial_loss > max_initial_train_loss
                ):
                    optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError(
                        "Initial training loss is pathologically high: "
                        f"{initial_loss:.6g} > max_initial_train_loss="
                        f"{max_initial_train_loss:.6g}. Refusing to optimize "
                        "toward a collapsed low-output solution. Use the shipped "
                        "residual-balanced fresh-run config and do not resume "
                        "an architecture-incompatible checkpoint."
                    )

            if check_finite and not pred_is_finite:
                consecutive_nonfinite += 1
                train_logger.warning(
                    "Non-finite generator output at iter %s; retry %s/%s",
                    iteration, consecutive_nonfinite, max_consecutive_nonfinite,
                )
                optimizer.zero_grad(set_to_none=True)
                bad = first_nonfinite_parameter_name(generator)
                if bad is not None:
                    raise FloatingPointError(
                        f"Generator parameter contains NaN/Inf: {bad}. Resume from "
                        "the last finite checkpoint with a lower LR."
                    )
                if max_consecutive_nonfinite > 0 and consecutive_nonfinite >= max_consecutive_nonfinite:
                    raise FloatingPointError(
                        f"Generator produced non-finite outputs on {consecutive_nonfinite} "
                        f"consecutive batches at iteration {iteration}."
                    )
                continue
            # (consecutive_nonfinite is reset only after a successful optimizer
            # step below, so repeated NaN losses/grads cannot loop forever.)

            # A non-finite loss is caught below via the grad-norm check (NaN
            # loss -> NaN grads -> NaN grad norm), which avoids a dedicated
            # per-step host-device sync on the loss here.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=effective_clip_norm
            )
            if not torch.isfinite(grad_norm):
                consecutive_nonfinite += 1
                train_logger.warning(
                    "Non-finite loss/grad norm at iter %s; skipping step (%s/%s)",
                    iteration, consecutive_nonfinite, max_consecutive_nonfinite,
                )
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    # unscale_() was already called for this optimizer; update()
                    # consumes that state (otherwise the next unscale_() raises)
                    # and shrinks the loss scale since non-finite grads were
                    # found.
                    scaler.update()
                bad = first_nonfinite_parameter_name(generator)
                if bad is not None:
                    raise FloatingPointError(
                        f"Generator parameter contains NaN/Inf: {bad}. Resume from "
                        "the last finite checkpoint with a lower LR."
                    )
                if max_consecutive_nonfinite > 0 and consecutive_nonfinite >= max_consecutive_nonfinite:
                    raise FloatingPointError(
                        f"Non-finite loss/gradients on {consecutive_nonfinite} "
                        f"consecutive batches at iteration {iteration}."
                    )
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            consecutive_nonfinite = 0
            consecutive_ooms = 0
            if ema is not None and (iteration + 1) % ema_update_every == 0:
                ema.update(generator)

            # Keep the running loss on-device; it is materialised to Python
            # floats only at logging/validation cadence (saves one host-device
            # sync per step).
            epoch_losses.append(loss.detach())
            iteration += 1
            stage_iter += 1

            if iteration % 20 == 0:
                avg_loss = (
                    torch.stack(epoch_losses[-20:]).mean().item() if epoch_losses else 0.0
                )
                elapsed = max(time.perf_counter() - log_window_start, 1e-9)
                completed = max(iteration - log_window_iteration, 1)
                train_logger.info(
                    "[iter:%d | stage %d %d/%d] lr=%.9f train_loss=%.6f "
                    "time=%.3fs/iter throughput=%.2f samples/s",
                    iteration,
                    stage_idx,
                    stage_iter,
                    stage_iterations,
                    lr,
                    avg_loss,
                    elapsed / completed,
                    completed * step_batch_size / elapsed,
                )
                log_window_start = time.perf_counter()
                log_window_iteration = iteration

                # Fast divergence path: a sudden blow-up of the smoothed train
                # loss (finite but >> its recent rolling median) is caught here,
                # within ~divergence_spike_patience*20 iters -- long before the
                # per-epoch validation would notice. The rolling median tracks
                # the slow annealed-epsilon drift so only genuine spikes fire.
                if (
                    divergence_guard_active
                    and stage_iter > divergence_warmup
                    and math.isfinite(avg_loss)
                    and avg_loss > 0.0
                ):
                    if len(loss_history) >= max(5, divergence_history_windows // 5):
                        median = statistics.median(loss_history)
                        if median > 0.0 and avg_loss > divergence_spike_factor * median:
                            divergence_spike_count += 1
                            train_logger.warning(
                                "Train-loss spike %d/%d at iter %d: %.4f > %.1fx "
                                "median(%.4f)",
                                divergence_spike_count, divergence_spike_patience,
                                iteration, avg_loss, divergence_spike_factor, median,
                            )
                            if divergence_spike_count >= divergence_spike_patience:
                                if _do_rollback(
                                    f"train-loss {avg_loss:.4g} > "
                                    f"{divergence_spike_factor:g}x median {median:.4g}"
                                ):
                                    continue
                                raise FloatingPointError(
                                    f"Training diverged at iter {iteration} and could "
                                    "not be recovered by checkpoint rollback."
                                )
                        else:
                            divergence_spike_count = 0
                            loss_history.append(avg_loss)
                    else:
                        loss_history.append(avg_loss)

            if iteration % per_epoch_iteration == 0:
                if ema is not None:
                    with ema.average_parameters(generator):
                        val_metrics = validate_generator(
                            net, val_dataset, criterion, device, iteration,
                            config, distributed, seed, rank,
                        )
                else:
                    val_metrics = validate_generator(
                        net, val_dataset, criterion, device, iteration,
                        config, distributed, seed, rank,
                    )

                current_mrae = val_metrics.get("mrae", float("inf"))
                epoch_num = iteration // per_epoch_iteration
                stage_epoch_num = stage_iter // per_epoch_iteration
                avg_train_loss = (
                    torch.stack(epoch_losses).mean().item() if epoch_losses else 0.0
                )
                objective_name = str(config.get("objective", "mrae")).upper()
                scheduled_epsilon = update_mrae_epsilon_schedule(
                    criterion, config, iteration
                )
                train_logger.info(
                    "Iter[%06d] Epoch[%06d] TrainLoss[%s]: %.6f ValMRAE: %.6f",
                    iteration, epoch_num, objective_name, avg_train_loss, current_mrae,
                )
                train_scalars = {
                    "train_loss": avg_train_loss,
                    "lr": lr,
                    "grad_norm": float(grad_norm.detach().item()),
                    "gradient_clip_norm": gradient_clip_norm,
                }
                if scheduled_epsilon is not None:
                    train_scalars["mrae_epsilon"] = scheduled_epsilon
                metrics_logger.log_scalars(
                    train_scalars, epoch_num, "train"
                )
                metrics_logger.log_scalars(val_metrics, epoch_num, "val")

                # Backstop divergence path: catches a gradual collapse that the
                # train-loss spike detector misses (e.g. the frozen-dead-network
                # signature where val MRAE balloons past the best-so-far and
                # stays there). Checked BEFORE the checkpoint save below, so the
                # on-disk "latest" is still the pre-divergence epoch we roll back
                # to; on success we skip this epoch's save entirely.
                if (
                    divergence_guard_active
                    and stage_iter > divergence_warmup
                    and math.isfinite(record_mrae_loss)
                    and record_mrae_loss > 0.0
                    and current_mrae > divergence_val_factor * record_mrae_loss
                ):
                    if _do_rollback(
                        f"val MRAE {current_mrae:.4g} > {divergence_val_factor:g}x "
                        f"best {record_mrae_loss:.4g}"
                    ):
                        continue
                    raise FloatingPointError(
                        f"Validation MRAE diverged at iter {iteration} "
                        f"({current_mrae:.4g} vs best {record_mrae_loss:.4g}) and "
                        "could not be recovered by checkpoint rollback."
                    )

                should_stop = False
                if rank == 0:
                    is_best = current_mrae < record_mrae_loss
                    previous_best = record_mrae_loss
                    if is_best:
                        record_mrae_loss = current_mrae

                    if early_stopping_enabled:
                        (
                            early_stopping_best_mrae,
                            early_stopping_bad_epochs,
                            should_stop,
                        ) = _update_early_stopping(
                            current_mrae=current_mrae,
                            best_mrae=early_stopping_best_mrae,
                            bad_epochs=early_stopping_bad_epochs,
                            patience=early_stopping_patience,
                            min_delta=early_stopping_min_delta,
                            epoch=stage_epoch_num,
                            warmup_epochs=early_stopping_warmup_epochs,
                        )

                (
                    record_mrae_loss,
                    early_stopping_best_mrae,
                    early_stopping_bad_epochs,
                    should_stop,
                ) = _sync_early_stopping_state(
                    record_mrae_loss=record_mrae_loss,
                    early_stopping_best_mrae=early_stopping_best_mrae,
                    early_stopping_bad_epochs=early_stopping_bad_epochs,
                    should_stop=should_stop,
                    device=device,
                    distributed=distributed,
                )

                if rank == 0:
                    checkpoint_dir = config["checkpoint_dir"]
                    checkpoint_dict = {
                        "epoch": epoch_num,
                        "iter": iteration,
                        "stage_idx": stage_idx,
                        "stage_iter": stage_iter,
                        "state_dict": generator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_mrae": record_mrae_loss,
                        "early_stopping_best_mrae": early_stopping_best_mrae,
                        "early_stopping_bad_epochs": early_stopping_bad_epochs,
                        "val_metrics": val_metrics,
                        "config": config,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": (
                            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                        ),
                        "numpy_rng_state": np.random.get_state(),
                        "ema": ema.state_dict() if ema is not None else None,
                    }
                    _atomic_torch_save(
                        checkpoint_dict, os.path.join(checkpoint_dir, CHECKPOINT_LATEST_NAME)
                    )
                    if is_best:
                        if ema is not None:
                            with ema.average_parameters(generator):
                                best_dict = dict(checkpoint_dict)
                                best_dict["state_dict"] = generator.state_dict()
                                best_dict["ema_applied"] = True
                                _atomic_torch_save(
                                    best_dict, os.path.join(checkpoint_dir, CHECKPOINT_BEST_NAME)
                                )
                        else:
                            _atomic_torch_save(
                                checkpoint_dict, os.path.join(checkpoint_dir, CHECKPOINT_BEST_NAME)
                            )
                        train_logger.info(
                            "NEW BEST: MRAE %.6f -> %.6f at iter %d (epoch %d)",
                            previous_best, current_mrae, iteration, epoch_num,
                        )
                    if early_stopping_enabled:
                        train_logger.info(
                            "Early stopping: best MRAE %.6f, no improvement %d/%d",
                            early_stopping_best_mrae,
                            early_stopping_bad_epochs,
                            early_stopping_patience,
                        )
                    if iteration % 5000 == 0:
                        _atomic_torch_save(
                            checkpoint_dict,
                            os.path.join(checkpoint_dir, f"net_{epoch_num}epoch.pth"),
                        )
                        _prune_epoch_checkpoints(
                            checkpoint_dir,
                            int(config.get("checkpoint_keep", CHECKPOINT_KEEP_COUNT)),
                        )

                epoch_losses = []
                memory_cleanup()
                log_window_start = time.perf_counter()
                log_window_iteration = iteration
                if should_stop:
                    if rank == 0:
                        train_logger.info(
                            "EARLY STOP at iter %d (stage epoch %d): validation MRAE "
                            "did not improve by %.6f for %d checks.",
                            iteration,
                            stage_epoch_num,
                            early_stopping_min_delta,
                            early_stopping_patience,
                        )
                    return (
                        iteration,
                        record_mrae_loss,
                        early_stopping_best_mrae,
                        early_stopping_bad_epochs,
                        True,
                    )

            if stage_iter >= stage_iterations:
                break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                consecutive_ooms += 1
                batch_size = int(bgr_batch.shape[0])
                current_microbatch = min(
                    adaptive_microbatch_size or batch_size,
                    batch_size,
                )
                next_microbatch = max(1, (current_microbatch + 1) // 2)
                can_reduce_microbatch = next_microbatch < current_microbatch
                if can_reduce_microbatch:
                    adaptive_microbatch_size = next_microbatch
                    # Retry the exact samples that overflowed. Dropping them
                    # would change the epoch's sample distribution and hides
                    # the true effective batch from the scheduler.
                    retry_batch = (bgr_batch, hyper_batch)
                    train_logger.error(
                        "OOM at iter %s; clearing cache and retrying with "
                        "microbatch=%s (effective batch=%s; retry %s/%s)",
                        iteration,
                        next_microbatch,
                        batch_size,
                        consecutive_ooms,
                        max_consecutive_ooms,
                    )
                else:
                    train_logger.error(
                        "OOM at iter %s with microbatch=1; clearing cache "
                        "(retry %s/%s)",
                        iteration,
                        consecutive_ooms,
                        max_consecutive_ooms,
                    )
                # The caught exception retains traceback frames from the
                # failed forward. Clear their locals before emptying CUDA's
                # cache; otherwise those frames can keep the very activations
                # that caused the OOM alive through the retry.
                traceback.clear_frames(e.__traceback__)
                optimizer.zero_grad(set_to_none=True)
                memory_cleanup()
                if consecutive_ooms >= max_consecutive_ooms:
                    raise RuntimeError(
                        f"Training hit {consecutive_ooms} consecutive OOMs at "
                        f"iteration {iteration}; automatic microbatch recovery "
                        "reached microbatch=1. Reduce the stage batch size or "
                        "patch size before retrying."
                    ) from e
                continue
            raise

    return (
        iteration,
        record_mrae_loss,
        early_stopping_best_mrae,
        early_stopping_bad_epochs,
        False,
    )


def train_generator_only(
    net: nn.Module,
    train_dataset: Any,
    val_dataset: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    config: Dict[str, Any],
    device: torch.device,
    metrics_logger: MetricsLogger,
    distributed: bool,
    seed: int,
    rank: int,
    resume_info: Optional[Dict[str, Any]] = None,
    ema: Optional[GeneratorEMA] = None,
) -> None:
    """Progressive (multi-stage) generator-only training.

    Runs each stage in ``progressive_stages`` (patch_size, iterations, init_lr,
    batch_size); the model, optimizer and EMA persist across stages while the LR
    schedule and DataLoader are rebuilt per stage (HSIFormer Sec 4.2 trains
    128 -> 256 -> 512). Falls back to a single stage when ``progressive_stages``
    is unset.
    """
    train_logger = logging.getLogger("hsi_model.training")
    generator = net.module if hasattr(net, "module") else net

    stages = _resolve_stages(config)
    global_iter = int(resume_info.get("iteration", 0)) if resume_info else 0
    record_mrae_loss = float(resume_info.get("best_mrae", float("inf"))) if resume_info else float("inf")
    early_stopping_best_mrae = (
        float(resume_info.get("early_stopping_best_mrae", float("inf")))
        if resume_info else float("inf")
    )
    early_stopping_bad_epochs = (
        int(resume_info.get("early_stopping_bad_epochs", 0)) if resume_info else 0
    )
    resume_stage_idx, resume_stage_iter = resolve_resume_stage_position(stages, resume_info)
    early_stopping_patience = max(0, int(config.get("early_stopping_patience", 0)))
    early_stopping_final_stage_only = bool(
        config.get("early_stopping_final_stage_only", True)
    )

    if resume_info and not ({"stage_idx", "stage_iter"} <= set(resume_info)):
        train_logger.info(
            "Resume checkpoint has no explicit stage position; derived stage %d "
            "iteration %d from global_iter %d",
            resume_stage_idx, resume_stage_iter, global_iter,
        )

    train_logger.info(
        "Progressive training: %d stage(s); resuming at stage %d (global_iter %d)",
        len(stages), resume_stage_idx, global_iter,
    )

    for stage_idx, stage in enumerate(stages):
        if stage_idx < resume_stage_idx:
            continue  # already completed before the resume checkpoint
        start_stage_iter = resume_stage_iter if stage_idx == resume_stage_idx else 0

        # Reset the LR schedule for this stage (fresh cosine from init_lr).
        for group in optimizer.param_groups:
            group["lr"] = stage["init_lr"]
            group["initial_lr"] = stage["init_lr"]
        last_epoch = (start_stage_iter - 1) if start_stage_iter > 0 else -1
        scheduler = WarmupCosineScheduler(
            optimizer, stage["warmup_steps"], stage["iterations"], eta_min=1e-6, last_epoch=last_epoch
        )

        # Build the train loader for this stage's patch size. Reuse the dataset
        # passed in when it already matches (stage 0 at the config patch size);
        # otherwise rebuild at the new patch size.
        if stage_idx == 0 and train_dataset is not None and stage["patch_size"] == int(config.get("patch_size", 128)):
            stage_dataset = train_dataset
        elif train_dataset is not None and hasattr(train_dataset, "set_patch_geometry"):
            # The in-RAM dataset stores full-resolution scenes; only the patch
            # indexing depends on crop size. Mutating it avoids re-loading the
            # ~30 GB scene set (and a discarded val-set copy) at every stage
            # boundary, which previously doubled host RAM from stage 1 onward.
            train_dataset.set_patch_geometry(stage["patch_size"])
            stage_dataset = train_dataset
        else:
            # Fallback for dataset classes without set_patch_geometry (e.g. the
            # HF wrapper). Note: the caller's reference keeps the previous copy
            # alive for the duration of the run; only MST datasets avoid the
            # double-residency via the mutation path above.
            stage_cfg = dict(config)
            stage_cfg["patch_size"] = stage["patch_size"]
            stage_dataset, stage_val = create_training_datasets(stage_cfg, seed=int(config.get("seed", 42)))
            del stage_val  # only the train split is needed; free the val copy
            train_dataset = stage_dataset
        train_loader = _build_train_loader(
            stage_dataset, stage["batch_size"], config, distributed, seed, rank
        )

        train_logger.info(
            "=== Stage %d/%d: patch=%d, iters=%d, init_lr=%.2e, batch=%d (start_iter=%d) ===",
            stage_idx + 1, len(stages), stage["patch_size"], stage["iterations"],
            stage["init_lr"], stage["batch_size"], start_stage_iter,
        )
        report_memory(f"Before stage {stage_idx}")
        early_stopping_enabled = (
            early_stopping_patience > 0
            and (
                not early_stopping_final_stage_only
                or stage_idx == len(stages) - 1
            )
        )
        (
            global_iter,
            record_mrae_loss,
            early_stopping_best_mrae,
            early_stopping_bad_epochs,
            should_stop,
        ) = _run_stage(
            net, generator, optimizer, scheduler, scaler, criterion, ema,
            train_loader, val_dataset, config, device, metrics_logger,
            distributed, seed, rank, stage_idx, stage["iterations"],
            global_iter, record_mrae_loss, early_stopping_best_mrae,
            early_stopping_bad_epochs, early_stopping_enabled,
            start_stage_iter=start_stage_iter,
        )
        del train_loader
        memory_cleanup()
        if should_stop:
            break

    train_logger.info("Progressive training complete. Best MRAE: %.6f", record_mrae_loss)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    cfg = OmegaConf.to_container(config, resolve=True)
    main_logger = logging.getLogger("hsi_model.main")
    try:
        cfg = setup_paths(cfg)
        device, rank, world_size, is_distributed = setup_distributed_training(cfg)
        cfg["local_rank"] = rank
        cfg["world_size"] = world_size
        cfg["distributed"] = is_distributed

        cfg.setdefault("patch_size", 128)
        cfg.setdefault("batch_size", 20)
        cfg.setdefault("iterations_per_epoch", 1000)
        cfg.setdefault("epochs", 100)
        cfg.setdefault("num_workers", 8)
        cfg.setdefault("mixed_precision_dtype", "auto")
        cfg.setdefault("resume_checkpoint", None)
        cfg.setdefault("finetune_checkpoint", None)

        log_level = getattr(logging, cfg.get("log_level", "INFO"))
        main_logger = setup_logging(cfg["log_dir"], log_level, rank)
        resolved_stages = _validate_training_contract(cfg)
        if rank == 0:
            config_fingerprint, resolved_config_path = _persist_resolved_config(cfg)
            stage_summary = ", ".join(
                (
                    f"{stage['patch_size']}px/{stage['iterations']} steps/"
                    f"lr={stage['init_lr']:.3g}/batch={stage['batch_size']}/"
                    f"warmup={stage['warmup_steps']}"
                )
                for stage in resolved_stages
            )
            contract = cfg.get("training_contract") or {}
            main_logger.info(
                "Resolved config: fingerprint=%s, recipe=%s, stages=[%s], "
                "mrae_epsilon=%s->%s over %s steps, gradient_clip_norm=%s; "
                "saved to %s",
                config_fingerprint,
                contract.get("name", "unconstrained"),
                stage_summary,
                cfg.get("mrae_epsilon_start", cfg.get("mrae_epsilon")),
                cfg.get("mrae_epsilon_end", cfg.get("mrae_epsilon")),
                cfg.get("mrae_epsilon_anneal_iters", 0),
                cfg.get("gradient_clip_norm", DEFAULT_GRADIENT_CLIP_NORM),
                resolved_config_path,
            )
            main_logger.info("=" * 60)
            main_logger.info("GENERATOR-ONLY (NO-GAN) RECONSTRUCTION TRAINING")
            main_logger.info(
                "objective=%s, sampling=%s, spectral=%s, cswin=%s, "
                "base_channels=%s, stage_depths=%s, cascade=%s, "
                "input_skip=%s, feature_norm=%s, input_denoising=%s, "
                "smsa_output_norm=%s, sstb_residual_scale=%s, "
                "output_head_init_scale=%s, refinement_blocks=%s",
                cfg.get("objective", "mrae"),
                cfg.get("sampling"),
                cfg.get("spectral_attention_type"),
                cfg.get("cswin_attention_mode"),
                cfg.get("base_channels"),
                cfg.get("stage_depths", cfg.get("blocks_per_stage")),
                cfg.get("cascade_stages", 1),
                cfg.get("use_spectral_input_skip", False),
                cfg.get("use_feature_norm", True),
                cfg.get("use_input_denoising", True),
                cfg.get("smsa_output_norm", True),
                cfg.get("sstb_outer_residual_scale", 1.0),
                cfg.get("output_head_init_scale"),
                cfg.get("refinement_blocks", 0),
            )
            main_logger.info("=" * 60)

        setup_seed(
            cfg.get("seed", 42),
            rank,
            deterministic=cfg.get("deterministic", True),
            allow_tf32=cfg.get("allow_tf32", False),
        )
        memory_cleanup()

        train_dataset, val_dataset = create_training_datasets(cfg, seed=int(cfg.get("seed", 42)))
        report_memory("After loading datasets")

        _amp_dtype = pick_amp_dtype(cfg) if device.type == "cuda" else None
        if rank == 0:
            main_logger.info(
                "AMP dtype=%s, spectral_attention_force_fp32=%s",
                str(_amp_dtype).replace("torch.", "") if _amp_dtype else "fp32",
                cfg.get("spectral_attention_force_fp32", True),
            )

        net = NoiseRobustCSWinGenerator(cfg).to(device)
        report_memory("After creating generator")

        lr = float(cfg.get("generator_lr", cfg.get("learning_rate", 4e-4)))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        opt_name = str(cfg.get("optimizer", "adam")).lower()
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

        # Schedulers are built PER progressive stage inside train_generator_only.
        scaler = make_grad_scaler(device.type, enabled=(_amp_dtype == torch.float16))

        criterion = build_criterion(cfg)

        if is_distributed:
            # device_ids must be the node-LOCAL device index (the global rank
            # would index nonexistent GPUs on multi-node runs). The generator
            # objective touches every parameter, so the per-iteration
            # unused-parameter graph walk is pure overhead by default.
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            net = DDP(
                net,
                device_ids=[local_rank],
                find_unused_parameters=bool(cfg.get("ddp_find_unused_parameters", False)),
            )

        ema: Optional[GeneratorEMA] = None
        if cfg.get("use_ema", True):
            gen_for_ema = net.module if hasattr(net, "module") else net
            ema_update_every = max(1, int(cfg.get("ema_update_every", 1)))
            per_step_decay = float(cfg.get("ema_decay", 0.999))
            ema = GeneratorEMA(
                gen_for_ema,
                decay=per_step_decay ** ema_update_every,
            )
            main_logger.info(
                "Generator EMA enabled (per-step decay=%.4f, update every %d steps)",
                per_step_decay,
                ema_update_every,
            )

        resume_info: Optional[Dict[str, Any]] = None
        resume_path = cfg.get("resume_checkpoint")
        finetune_path = cfg.get("finetune_checkpoint")
        if resume_path and finetune_path:
            raise ValueError(
                "resume_checkpoint and finetune_checkpoint are mutually "
                "exclusive. Resume continues the same optimizer/stage state; "
                "fine-tune starts a new optimization phase from model weights."
            )
        if resume_path:
            resume_info = resume_training_state(
                checkpoint_path=str(resume_path),
                model=net,
                optimizers={"optimizer": optimizer},
                schedulers={},   # schedulers are rebuilt per progressive stage
                scalers={"scaler": scaler},
                device=device,
                ema=ema,
                expected_objective=str(cfg.get("objective", "mrae")),
                allow_objective_mismatch=bool(
                    cfg.get("allow_objective_mismatch_resume", False)
                ),
            )
        elif finetune_path:
            load_finetune_weights(
                checkpoint_path=str(finetune_path),
                model=net,
                device=device,
                ema=ema,
            )

        metrics_logger = MetricsLogger(cfg["log_dir"], rank)
        train_generator_only(
            net, train_dataset, val_dataset, optimizer, criterion, scaler,
            cfg, device, metrics_logger, is_distributed, cfg.get("seed", 42), rank,
            resume_info=resume_info, ema=ema,
        )
        if rank == 0:
            main_logger.info("TRAINING COMPLETED!")
    except Exception as e:
        main_logger.error("Training failed: %s", str(e), exc_info=True)
        raise
    finally:
        memory_cleanup()
        if "metrics_logger" in locals():
            metrics_logger.close()
        cleanup()


if __name__ == "__main__":
    main()

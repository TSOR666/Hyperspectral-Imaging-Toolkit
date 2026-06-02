# src/hsi_model/train_generator.py
"""
Generator-only (no-GAN) training for CSWin RGB->HSI reconstruction.

This is the post-GAN trainer: the discriminator, Sinkhorn-OT adversarial loss,
and R1 regularization are gone. The objective is pure reconstruction (MRAE by
default — the ARAD-1K leaderboard metric and MST++'s training loss), so the
reported metric is optimized directly and results are comparable to MST++.

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

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
from hsi_model.models.losses_consolidated import MRAELoss, RelativeMRAELoss
from hsi_model.constants import (
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_WARMUP_STEPS,
    CHECKPOINT_BEST_NAME,
    CHECKPOINT_LATEST_NAME,
    CHECKPOINT_KEEP_COUNT,
)
from hsi_model.utils import setup_logging, MetricsLogger
from hsi_model.utils.training_setup import (
    cleanup,
    GeneratorEMA,
    pick_amp_dtype,
    resume_training_state,
    setup_distributed_training,
    setup_paths,
    setup_seed,
)
from hsi_model.utils.data import (
    create_training_datasets,
    mst_to_gan_batch,
    compute_mst_center_crop_metrics,
    worker_init_fn_mst,
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


def build_criterion(config: Dict[str, Any]) -> nn.Module:
    """Reconstruction criterion. Default 'mrae' = the MST++ / leaderboard MRAE."""
    objective = str(config.get("objective", "mrae")).lower()
    if objective in ("mrae", "mst", "mst++"):
        return MRAELoss(epsilon=float(config.get("mrae_epsilon", 1e-8)))
    if objective in ("relative_mrae", "smooth_mrae"):
        return RelativeMRAELoss(
            denominator_epsilon=float(config.get("relative_mrae_epsilon", 1e-2))
        )
    logger.warning("Unknown objective=%r; defaulting to MRAE.", objective)
    return MRAELoss(epsilon=float(config.get("mrae_epsilon", 1e-8)))


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
    net.eval()
    val_logger = logging.getLogger("hsi_model.validation")

    val_sampler = None
    if distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=seed)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.get("val_batch_size", 1),
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.get("num_workers", 8),
        pin_memory=True,
        worker_init_fn=lambda w: worker_init_fn_mst(w, seed, rank),
        persistent_workers=False,
    )

    total_loss = torch.tensor(0.0, device=device)
    total_metrics: Dict[str, torch.Tensor] = {}
    num_batches = 0
    amp_dtype = pick_amp_dtype(config) if device.type == "cuda" else None
    use_amp = amp_dtype is not None
    autocast_dtype = amp_dtype if amp_dtype is not None else torch.float16
    validation_max_batches = config.get("validation_max_batches", None)
    if validation_max_batches is not None:
        validation_max_batches = int(validation_max_batches)

    with torch.no_grad():
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
                with autocast(enabled=use_amp, dtype=autocast_dtype):
                    pred_hsi = net(rgb_tensor)
                    loss = criterion(pred_hsi.float(), hsi_tensor.float())
                total_loss += loss
                metrics = compute_mst_center_crop_metrics(pred_hsi, hsi_tensor)
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = torch.tensor(0.0, device=device)
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(value, device=device)
                    total_metrics[key] += value
                num_batches += 1
            except Exception as e:  # noqa: BLE001 - validation must not kill training
                val_logger.warning("Validation error: %s", str(e))
                continue

    del val_loader
    denom = max(num_batches, 1)
    avg = {"gen_loss": total_loss.item() / denom}
    avg.update({k: v.item() / denom for k, v in total_metrics.items()})
    val_logger.info(
        "Validation Iter: %d | MRAE: %.4f | PSNR: %.2fdB",
        iteration, avg.get("mrae", 0.0), avg.get("psnr", 0.0),
    )
    return avg


def train_generator_only(
    net: nn.Module,
    train_dataset: Any,
    val_dataset: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
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
    train_logger = logging.getLogger("hsi_model.training")

    per_epoch_iteration = config.get("iterations_per_epoch", 1000)
    total_epochs = config.get("epochs", 100)
    total_iteration = per_epoch_iteration * total_epochs
    batch_size = config.get("batch_size", 20)
    num_workers = config.get("num_workers", 8)

    start_iteration = resume_info.get("iteration", 0) if resume_info else 0
    best_mrae = resume_info.get("best_mrae", float("inf")) if resume_info else float("inf")

    # The underlying generator (unwrap DDP) is what EMA + set_iteration act on.
    generator = net.module if hasattr(net, "module") else net

    train_logger.info("Generator-only training: %d total iters, batch %d", total_iteration, batch_size)
    report_memory("Before creating DataLoader")

    seed_base = seed + rank * 1000
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed_base)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda w: worker_init_fn_mst(w, seed_base, rank),
        persistent_workers=(num_workers > 0),
    )
    data_iter = iter(train_loader)
    is_distributed = isinstance(train_loader.sampler, DistributedSampler)

    iteration = start_iteration
    record_mrae_loss = best_mrae
    amp_dtype = pick_amp_dtype(config) if device.type == "cuda" else None
    use_amp = amp_dtype is not None
    autocast_dtype = amp_dtype if amp_dtype is not None else torch.float16
    epoch_losses = []
    current_epoch = 0
    consecutive_nonfinite = 0
    max_consecutive_nonfinite = int(config.get("max_consecutive_nonfinite_generator_outputs", 3))

    while iteration < total_iteration:
        net.train()
        try:
            bgr_batch, hyper_batch = next(data_iter)
        except StopIteration:
            if is_distributed:
                current_epoch += 1
                train_loader.sampler.set_epoch(current_epoch)
            data_iter = iter(train_loader)
            bgr_batch, hyper_batch = next(data_iter)

        try:
            rgb_tensor, hsi_tensor = mst_to_gan_batch(bgr_batch, hyper_batch)
            rgb_tensor = rgb_tensor.to(device, non_blocking=True)
            hsi_tensor = hsi_tensor.to(device, non_blocking=True)
            if not torch.isfinite(rgb_tensor).all() or not torch.isfinite(hsi_tensor).all():
                train_logger.warning("Skipping non-finite batch at iteration %s", iteration)
                continue

            lr = optimizer.param_groups[0]["lr"]
            if hasattr(generator, "set_iteration"):
                generator.set_iteration(iteration)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp, dtype=autocast_dtype):
                pred = net(rgb_tensor)

            if not torch.isfinite(pred).all():
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
            consecutive_nonfinite = 0

            with autocast(enabled=use_amp, dtype=autocast_dtype):
                loss = criterion(pred.float(), hsi_tensor.float())

            if not torch.isfinite(loss):
                train_logger.warning("Non-finite loss at iter %s; skipping batch", iteration)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=DEFAULT_GRADIENT_CLIP_NORM
            )
            if not torch.isfinite(grad_norm):
                train_logger.warning("Non-finite grad norm at iter %s; skipping step", iteration)
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(generator)

            epoch_losses.append(loss.item())
            iteration += 1

            if iteration % 20 == 0:
                avg_loss = np.mean(epoch_losses[-20:]) if epoch_losses else 0.0
                train_logger.info(
                    "[iter:%d/%d] lr=%.9f train_mrae=%.6f",
                    iteration, total_iteration, lr, avg_loss,
                )

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
                avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                train_logger.info(
                    "Iter[%06d] Epoch[%06d] TrainMRAE: %.6f TestMRAE: %.6f",
                    iteration, epoch_num, avg_train_loss, current_mrae,
                )
                metrics_logger.log_scalars(
                    {"train_loss": avg_train_loss, "lr": lr}, epoch_num, "train"
                )
                metrics_logger.log_scalars(val_metrics, epoch_num, "val")

                if config.get("local_rank", 0) == 0:
                    is_best = current_mrae < record_mrae_loss
                    checkpoint_dir = config["checkpoint_dir"]
                    checkpoint_dict = {
                        "epoch": epoch_num,
                        "iter": iteration,
                        "state_dict": generator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_mrae": min(current_mrae, record_mrae_loss),
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
                        previous_best = record_mrae_loss
                        record_mrae_loss = current_mrae
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
                    if iteration % 5000 == 0:
                        _atomic_torch_save(
                            checkpoint_dict,
                            os.path.join(checkpoint_dir, f"net_{epoch_num}epoch.pth"),
                        )
                        _prune_epoch_checkpoints(checkpoint_dir, CHECKPOINT_KEEP_COUNT)

                epoch_losses = []
                memory_cleanup()

            if iteration >= total_iteration:
                break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                train_logger.error("OOM at iter %s; clearing cache", iteration)
                optimizer.zero_grad(set_to_none=True)
                memory_cleanup()
                continue
            raise

    train_logger.info("Generator-only training complete. Best MRAE: %.6f", record_mrae_loss)


@hydra.main(config_path="../configs", config_name="config")
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

        log_level = getattr(logging, cfg.get("log_level", "INFO"))
        main_logger = setup_logging(cfg["log_dir"], log_level, rank)
        if rank == 0:
            main_logger.info("=" * 60)
            main_logger.info("GENERATOR-ONLY (NO-GAN) MRAE TRAINING")
            main_logger.info("objective=%s, sampling=%s, spectral=%s, base_channels=%s, blocks/stage=%s",
                             cfg.get("objective", "mrae"), cfg.get("sampling"),
                             cfg.get("spectral_attention_type"), cfg.get("base_channels"),
                             cfg.get("blocks_per_stage"))
            main_logger.info("=" * 60)

        setup_seed(cfg.get("seed", 42), rank)
        memory_cleanup()

        train_dataset, val_dataset = create_training_datasets(cfg, seed=int(cfg.get("seed", 42)))
        report_memory("After loading datasets")

        net = NoiseRobustCSWinGenerator(cfg).to(device)
        report_memory("After creating generator")

        lr = float(cfg.get("generator_lr", cfg.get("learning_rate", 4e-4)))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        opt_name = str(cfg.get("optimizer", "adam")).lower()
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

        total_iterations = cfg["iterations_per_epoch"] * cfg["epochs"]
        warmup_steps = cfg.get("warmup_steps", DEFAULT_WARMUP_STEPS)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_iterations, eta_min=1e-6)

        _amp_dtype = pick_amp_dtype(cfg) if device.type == "cuda" else None
        scaler = GradScaler(enabled=(_amp_dtype == torch.float16))

        criterion = build_criterion(cfg)

        if is_distributed:
            net = DDP(net, device_ids=[rank], find_unused_parameters=True)

        ema: Optional[GeneratorEMA] = None
        if cfg.get("use_ema", True):
            gen_for_ema = net.module if hasattr(net, "module") else net
            ema = GeneratorEMA(gen_for_ema, decay=float(cfg.get("ema_decay", 0.999)))
            main_logger.info("Generator EMA enabled (decay=%.4f)", ema.decay)

        resume_info: Optional[Dict[str, Any]] = None
        resume_path = cfg.get("resume_checkpoint")
        if resume_path:
            resume_info = resume_training_state(
                checkpoint_path=str(resume_path),
                model=net,
                optimizers={"optimizer": optimizer},
                schedulers={"scheduler": scheduler},
                scalers={"scaler": scaler},
                device=device,
                ema=ema,
            )

        metrics_logger = MetricsLogger(cfg["log_dir"], rank)
        train_generator_only(
            net, train_dataset, val_dataset, optimizer, scheduler, criterion, scaler,
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

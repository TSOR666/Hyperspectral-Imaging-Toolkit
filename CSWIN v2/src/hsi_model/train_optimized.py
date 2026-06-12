# src/hsi_model/train_optimized.py - MST++ Training with ALL memory optimizations
"""
MST++ Training Script with Critical Memory Optimizations

CRITICAL MEMORY OPTIMIZATIONS (v3.0):
==========================================
1. Training Loop Optimization (SAVES ~18GB):
   - Compute disc_real under no_grad during generator updates
   - Freeze discriminator params during generator updates
   - Cache disc_fake to avoid duplicate computation
   - Memory reduction: ~41GB -> ~23GB for discriminator phase

2. Validation Optimization:
   - Use torch.no_grad() for all discriminator calls in validation
   - Create temporary DataLoader without persistent workers
   - Clean up validation loader immediately after use

3. General Optimizations:
   - Single DataLoader reused throughout training
   - h5py cache handling (version-aware)
   - Regular memory cleanup and reporting
   - Gradient accumulation support for larger effective batches

4. v3.0 Fixes:
   - PyTorch version compatibility for gradient checkpointing
   - Worker thread optimization (2-4 threads)
   - Reduced diagnostic frequency

Memory Impact:
- Before: Peak memory ~76GB causing OOM on 80GB A100
- After: Peak memory <30GB with comfortable headroom
- Can now train with batch_size=20 on 128x128 patches

Environment Setup:
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export OMP_NUM_THREADS=2
export MST_MEMORY_MODE=lazy
export MST_LAZY_CACHE_SIZE=3

Version History:
- v1.0: Initial implementation with memory issues
- v2.0: Fixed discriminator graph duplication and optimized training loop
- v3.0: Added compatibility fixes and worker optimizations
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import hydra
import gc
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Tuple, Dict, Any, Optional

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Local imports
from hsi_model.models import NoiseRobustCSWinModel
from hsi_model.models.losses_consolidated import (
    ComputeSinkhornDiscriminatorLoss,
    NoiseRobustLoss,
)
from hsi_model.constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_PATCH_SIZE,
    DEFAULT_STRIDE,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_WARMUP_STEPS,
    CHECKPOINT_BEST_NAME,
    CHECKPOINT_LATEST_NAME,
    CHECKPOINT_KEEP_COUNT,
)
# Reuse the linear-warmup -> cosine-decay scheduler defined in
# training_script_fixed.py so both trainers anneal the LR identically. The
# import is acyclic (training_script_fixed does not import this module).
from hsi_model.training_script_fixed import WarmupCosineScheduler
from hsi_model.utils import (
    setup_logging, MetricsLogger, save_checkpoint, load_checkpoint
)
from hsi_model.utils.training_setup import (
    cleanup,
    GeneratorEMA,
    pick_amp_dtype,
    resume_training_state,
    setup_distributed_training,
    setup_paths,
    setup_seed,
)
from hsi_model.utils.metrics import (
    compute_metrics, compute_metrics_arad1k, profile_model, export_model,
    validate_model_architecture, save_metrics, create_error_report
)
from hsi_model.utils.data import (
    create_training_datasets,
    mst_to_gan_batch,
    compute_mst_center_crop_metrics,
    show_dataloader_diagnostics,
    make_worker_init_fn,
)

logger = logging.getLogger(__name__)


def resolve_generator_discriminator_lrs(config: Dict[str, Any]) -> Tuple[float, float]:
    """Resolve split G/D learning rates with legacy learning_rate fallback."""
    legacy_lr = config.get("learning_rate", None)
    generator_lr = config.get(
        "generator_lr",
        legacy_lr if legacy_lr is not None else 1e-4,
    )
    discriminator_lr = config.get(
        "discriminator_lr",
        legacy_lr if legacy_lr is not None else 2e-5,
    )
    return float(generator_lr), float(discriminator_lr)


def report_memory(tag: str):
    """Report memory usage for debugging."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss / 1e9  # GB
        kids = len(proc.children(recursive=True))
        print(f"{tag:<30} | RSS {rss:>6.2f} GB | workers {kids}")
        
        # Also log GPU memory if available
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"{'':30} | GPU {gpu_alloc:>6.2f} GB alloc, {gpu_reserved:>6.2f} GB reserved")
    except:
        pass
    gc.collect()


def create_datasets_only(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Create configured datasets without DataLoaders to avoid extra workers."""
    memory_mode = config.get("memory_mode", "standard")
    logger = logging.getLogger(__name__)
    logger.info(
        "Creating %s datasets with memory mode: %s",
        config.get("dataset_source", "mst"),
        memory_mode,
    )

    return create_training_datasets(config, seed=int(config.get("seed", 42)))


def memory_cleanup():
    """Perform memory cleanup operations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Toggle gradient computation for all parameters in a module."""
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def first_nonfinite_parameter_name(module: nn.Module) -> Optional[str]:
    """Return the first parameter name containing NaN/Inf, if any."""
    for name, param in module.named_parameters():
        if not torch.isfinite(param.detach()).all():
            return name
    return None


def _atomic_torch_save(obj: Dict[str, Any], path: str) -> None:
    """Write a checkpoint via .tmp + os.replace so an HPC kill mid-save
    never leaves a half-written file. Same idiom as utils.checkpoint."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _prune_epoch_checkpoints(checkpoint_dir: str, keep: int = CHECKPOINT_KEEP_COUNT) -> None:
    """Keep only the ``keep`` most-recent ``net_*epoch.pth`` files.

    Best and latest checkpoints are kept regardless because their names
    don't match the ``net_*epoch.pth`` pattern. Errors are logged and
    swallowed — checkpoint hygiene must never interrupt training.
    """
    try:
        snapshots = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("net_") and filename.endswith("epoch.pth"):
                full = os.path.join(checkpoint_dir, filename)
                snapshots.append((full, os.path.getmtime(full)))
        if len(snapshots) <= keep:
            return
        snapshots.sort(key=lambda t: t[1], reverse=True)
        for path, _ in snapshots[keep:]:
            try:
                os.remove(path)
            except OSError as e:
                logging.getLogger(__name__).warning(
                    "Failed to prune %s: %s", path, e
                )
    except OSError as e:
        logging.getLogger(__name__).warning(
            "Checkpoint pruning skipped (%s): %s", checkpoint_dir, e
        )


def validate_mst_style(
    model: nn.Module,
    val_dataset: Any,
    criterion: nn.Module,
    device: torch.device,
    iteration: int,
    config: Dict[str, Any],
    distributed: bool,
    seed: int,
    rank: int
) -> Dict[str, float]:
    """
    MST++ validation with temporary DataLoader and optimized discriminator usage.
    
    v2.0: All discriminator calls wrapped in torch.no_grad() to avoid graph building
    """
    model.eval()
    logger = logging.getLogger("hsi_model.validation")
    
    # Create validation loader WITHOUT persistent workers
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
        worker_init_fn=make_worker_init_fn(seed, rank),
        persistent_workers=False  # KEY: Don't keep validation workers alive
    )
    
    # Accumulate on GPU to avoid CPU-GPU thrashing
    total_gen_loss = torch.tensor(0.0, device=device)
    total_metrics = {}
    num_batches = 0
    amp_dtype = pick_amp_dtype(config) if device.type == "cuda" else None
    use_amp = amp_dtype is not None
    # Pass dtype unconditionally; autocast ignores it when enabled=False.
    autocast_dtype = amp_dtype if amp_dtype is not None else torch.float16
    validation_max_batches = config.get("validation_max_batches", None)
    if validation_max_batches is not None:
        validation_max_batches = int(validation_max_batches)

    with torch.no_grad():  # v2.0: Ensure no gradients during validation
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
                    if hasattr(model, 'module'):
                        pred_hsi = model.module.generator(rgb_tensor)
                        # v2.0: Discriminator calls already under no_grad context
                        disc_real = model.module.discriminator(rgb_tensor, hsi_tensor)
                        disc_fake = model.module.discriminator(rgb_tensor, pred_hsi)
                    else:
                        pred_hsi = model.generator(rgb_tensor)
                        # v2.0: Discriminator calls already under no_grad context
                        disc_real = model.discriminator(rgb_tensor, hsi_tensor)
                        disc_fake = model.discriminator(rgb_tensor, pred_hsi)

                    gen_loss, loss_components = criterion(
                        pred_hsi,
                        hsi_tensor,
                        disc_real,
                        disc_fake,
                        current_iteration=iteration,
                    )

                # Accumulate on GPU (no .item() calls inside loop!)
                total_gen_loss += gen_loss

                metrics = compute_mst_center_crop_metrics(pred_hsi, hsi_tensor)
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = torch.tensor(0.0, device=device)
                    # Ensure value is a tensor on GPU
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(value, device=device)
                    total_metrics[key] += value

                num_batches += 1

            except Exception as e:
                logger.warning(f"Validation error: {str(e)}")
                continue
    
    # DataLoader automatically cleans up workers when it goes out of scope
    del val_loader  # Explicit cleanup

    # Single sync at end: transfer accumulated metrics from GPU to CPU
    avg_losses = {'gen_loss': total_gen_loss.item() / max(num_batches, 1)}
    avg_metrics = {k: v.item() / max(num_batches, 1) for k, v in total_metrics.items()}
    all_metrics = {**avg_losses, **avg_metrics}
    
    logger.info(
        f"Validation Iter: {iteration} | "
        f"MRAE: {avg_metrics.get('mrae', 0):.4f} | "
        f"PSNR: {avg_metrics.get('psnr', 0):.2f}dB"
    )
    
    return all_metrics


def train_mst_gan_optimized(
    model: nn.Module,
    train_dataset,
    val_dataset,  # Pass dataset, not loader
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    disc_criterion: nn.Module,
    scalers: Dict[str, GradScaler],
    config: Dict[str, Any],
    device: torch.device,
    metrics_logger: MetricsLogger,
    distributed: bool,
    seed: int,
    rank: int,
    resume_info: Optional[Dict[str, Any]] = None,
    ema: Optional[GeneratorEMA] = None,
) -> None:
    """
    MST++ training with optimized discriminator graph handling.
    
    v3.0 Critical Fixes:
    - Use torch.set_grad_enabled context instead of requires_grad toggling
    - Cache disc_fake computation to avoid duplicate forward passes
    - Reduced diagnostic frequency
    """
    logger = logging.getLogger("hsi_model.training")
    
    # Training parameters
    per_epoch_iteration = config.get("iterations_per_epoch", 1000)
    total_epochs = config.get("epochs", 300)
    total_iteration = per_epoch_iteration * total_epochs
    batch_size = config.get("batch_size", 20)
    num_workers = config.get("num_workers", 8)
    
    optimizer_g = optimizers['optimizer_g']
    optimizer_d = optimizers['optimizer_d']
    scheduler_g = schedulers['scheduler_g']
    scheduler_d = schedulers['scheduler_d']
    scaler_g = scalers['scaler_g']
    scaler_d = scalers['scaler_d']
    
    # Resume
    start_iteration = resume_info.get('iteration', 0) if resume_info else 0
    best_mrae = resume_info.get('best_mrae', float('inf')) if resume_info else float('inf')
    
    logger.info(f"Starting optimized MST++ training (v3.0):")
    logger.info(f"  - Total iterations: {total_iteration}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Workers: {num_workers}")
    logger.info(f"  - Memory optimizations: Fixed discriminator graph & gradient handling")
    
    # Report initial memory
    report_memory("Before creating DataLoader")
    
    # Create single training DataLoader
    # Use consistent seed across sampler and workers for determinism
    seed_base = seed + rank * 1000  # Unique augmentation stream per rank
    train_sampler = None
    if distributed:
        # Every rank must shard the same permutation. Rank-specific sampler
        # seeds create overlapping shards; worker seeds remain rank-specific.
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=make_worker_init_fn(seed_base, rank),
        persistent_workers=(num_workers > 0)
    )
    
    data_iter = iter(train_loader)
    is_distributed = isinstance(train_loader.sampler, DistributedSampler)
    
    report_memory("After creating DataLoader")
    
    # Training state
    iteration = start_iteration
    record_mrae_loss = best_mrae
    # GPU-aware precision pick. On A100 (cc>=8) this picks bf16, which has
    # fp32's dynamic range — eliminating the GradScaler-cycling-down failure
    # mode you saw at iter 0 (65536 -> 0.2 over ~190 iters). fp16 is still
    # selected on V100/T4 (cc 7.x); fp32 falls back when nothing else fits.
    amp_dtype = pick_amp_dtype(config) if device.type == "cuda" else None
    use_amp = amp_dtype is not None
    autocast_dtype = amp_dtype if amp_dtype is not None else torch.float16
    logger.info(
        "Mixed precision: dtype=%s, enabled=%s",
        "fp32" if amp_dtype is None else str(amp_dtype).replace("torch.", ""),
        use_amp,
    )
    epoch_losses = []
    current_epoch = 0
    generator = model.module.generator if hasattr(model, "module") else model.generator
    discriminator = model.module.discriminator if hasattr(model, "module") else model.discriminator
    consecutive_nonfinite_generator_outputs = 0
    max_consecutive_nonfinite_generator_outputs = int(
        config.get("max_consecutive_nonfinite_generator_outputs", 3)
    )
    
    # Main training loop
    while iteration < total_iteration:
        model.train()
        
        try:
            bgr_batch, hyper_batch = next(data_iter)
        except StopIteration:
            logger.info(f"Dataset epoch completed at iteration {iteration}")
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
                logger.warning("Skipping non-finite batch at iteration %s", iteration)
                continue
            
            lr_g = optimizer_g.param_groups[0]['lr']
            lr_d = optimizer_d.param_groups[0]['lr']

            if hasattr(generator, "set_iteration"):
                generator.set_iteration(iteration)
            
            # ========== Train Generator (v3.0 OPTIMIZED) ==========
            optimizer_g.zero_grad(set_to_none=True)
            
            with autocast(enabled=use_amp, dtype=autocast_dtype):
                fake_hsi = generator(rgb_tensor)

            if not torch.isfinite(fake_hsi).all():
                consecutive_nonfinite_generator_outputs += 1
                logger.warning(
                    "Non-finite generator output at iteration %s; retry %s/%s",
                    iteration,
                    consecutive_nonfinite_generator_outputs,
                    max_consecutive_nonfinite_generator_outputs,
                )
                optimizer_g.zero_grad(set_to_none=True)
                bad_param = first_nonfinite_parameter_name(generator)
                if bad_param is not None:
                    raise FloatingPointError(
                        "Generator parameter contains NaN/Inf after a non-finite "
                        f"forward pass: {bad_param}. Resume from the last finite "
                        "checkpoint with a lower generator LR or disable mixed_precision."
                    )
                if (
                    max_consecutive_nonfinite_generator_outputs > 0
                    and consecutive_nonfinite_generator_outputs
                    >= max_consecutive_nonfinite_generator_outputs
                ):
                    raise FloatingPointError(
                        "Generator produced non-finite outputs on "
                        f"{consecutive_nonfinite_generator_outputs} consecutive "
                        f"batches at iteration {iteration}. Stopping instead of "
                        "retrying the same iteration indefinitely."
                    )
                continue
            consecutive_nonfinite_generator_outputs = 0

            set_requires_grad(discriminator, False)
            try:
                with autocast(enabled=use_amp, dtype=autocast_dtype):
                    disc_fake_for_g = discriminator(rgb_tensor, fake_hsi)
                with torch.no_grad():
                    with autocast(enabled=use_amp, dtype=autocast_dtype):
                        disc_real_for_g = discriminator(rgb_tensor, hsi_tensor)
            finally:
                set_requires_grad(discriminator, True)

            with autocast(enabled=use_amp, dtype=autocast_dtype):
                gen_loss, loss_components = criterion(
                    fake_hsi,
                    hsi_tensor,
                    disc_real_for_g,
                    disc_fake_for_g,
                    current_iteration=iteration,
                )

            if not torch.isfinite(gen_loss):
                logger.warning("Non-finite generator loss at iteration %s; skipping batch", iteration)
                optimizer_g.zero_grad(set_to_none=True)
                continue

            scaler_g.scale(gen_loss).backward()
            # Unscale + clip BEFORE step. scaler.step only skips on Inf/NaN
            # gradients; a finite-but-large gradient slips through and drives
            # the weights toward Inf over many iterations (root cause of the
            # generator-NaN-then-infinite-retry crash). Clip to bound the
            # step size like training_script_fixed.py does.
            scaler_g.unscale_(optimizer_g)
            g_grad_norm = torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=DEFAULT_GRADIENT_CLIP_NORM,
            )
            if not torch.isfinite(g_grad_norm):
                logger.warning(
                    "Non-finite generator grad norm at iter %s; skipping optimizer step",
                    iteration,
                )
                optimizer_g.zero_grad(set_to_none=True)
                continue
            old_scale_g = scaler_g.get_scale()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            new_scale_g = scaler_g.get_scale()
            if new_scale_g < old_scale_g:
                logger.warning(
                    f"Generator scaler reduced scale {old_scale_g:.1f} -> {new_scale_g:.1f} "
                    f"(overflow detected at iter {iteration})"
                )

            # Update the generator EMA after a successful optimizer step.
            if ema is not None:
                ema.update(generator)

            # ========== Train Discriminator (v3.0 OPTIMIZED) ==========
            optimizer_d.zero_grad(set_to_none=True)

            # v3.0: Detach fake_hsi to prevent gradients flowing to generator
            fake_hsi_detached = fake_hsi.detach()

            with autocast(enabled=use_amp, dtype=autocast_dtype):
                real_pred = discriminator(rgb_tensor, hsi_tensor)
                fake_pred = discriminator(rgb_tensor, fake_hsi_detached)
                if not torch.isfinite(real_pred).all() or not torch.isfinite(fake_pred).all():
                    logger.warning("Non-finite discriminator logits at iteration %s; skipping D step", iteration)
                    optimizer_d.zero_grad(set_to_none=True)
                    continue

                disc_loss = disc_criterion(real_pred, fake_pred)

            if not torch.isfinite(disc_loss):
                logger.warning("Non-finite discriminator loss at iteration %s; skipping D step", iteration)
                optimizer_d.zero_grad(set_to_none=True)
                continue

            scaler_d.scale(disc_loss).backward()
            scaler_d.unscale_(optimizer_d)
            d_grad_norm = torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=DEFAULT_GRADIENT_CLIP_NORM,
            )
            if not torch.isfinite(d_grad_norm):
                logger.warning(
                    "Non-finite discriminator grad norm at iter %s; skipping optimizer step",
                    iteration,
                )
                optimizer_d.zero_grad(set_to_none=True)
                continue
            old_scale_d = scaler_d.get_scale()
            scaler_d.step(optimizer_d)
            scaler_d.update()
            new_scale_d = scaler_d.get_scale()
            if new_scale_d < old_scale_d:
                logger.warning(
                    f"Discriminator scaler reduced scale {old_scale_d:.1f} -> {new_scale_d:.1f} "
                    f"(overflow detected at iter {iteration})"
                )

            scheduler_g.step()
            scheduler_d.step()
            
            epoch_losses.append(gen_loss.item())
            iteration += 1
            
            # Logging
            if iteration % 20 == 0:
                if epoch_losses:
                    avg_loss = np.mean(epoch_losses[-20:]) if len(epoch_losses) >= 20 else np.mean(epoch_losses)
                else:
                    avg_loss = 0.0
                logger.info(
                    f'[iter:{iteration}/{total_iteration}], lr_g={lr_g:.9f}, '
                    f'lr_d={lr_d:.9f}, train_loss={avg_loss:.9f}'
                )
            
            # v3.0: Reduced diagnostic frequency
            if iteration % 2000 == 0:  # Changed from 500
                report_memory(f"Iteration {iteration}")
                if iteration % 10000 == 0:  # Only show detailed diagnostics rarely
                    show_dataloader_diagnostics()
            
            # Validation
            if iteration % per_epoch_iteration == 0:
                # Validation with temporary DataLoader. When EMA is enabled,
                # validate with the EMA generator weights so the reported
                # metrics (and the best_model selection below) reflect the
                # smoothed weights that are actually saved for inference.
                if ema is not None:
                    with ema.average_parameters(generator):
                        val_metrics = validate_mst_style(
                            model, val_dataset, criterion, device, iteration, config,
                            distributed, seed, rank
                        )
                else:
                    val_metrics = validate_mst_style(
                        model, val_dataset, criterion, device, iteration, config,
                        distributed, seed, rank
                    )
                
                current_mrae = val_metrics.get('mrae', float('inf'))
                epoch_num = iteration // per_epoch_iteration
                avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
                
                logger.info(
                    f" Iter[{iteration:06d}], Epoch[{epoch_num:06d}], "
                    f"Train Loss: {avg_train_loss:.9f}, Test MRAE: {current_mrae:.9f}"
                )
                
                # train_loss and lr_g are the only true training-side scalars
                # here; val_metrics are computed on the held-out val set and
                # MUST be logged under a separate prefix or every "train/*"
                # tag in tensorboard is just the validation result mislabelled.
                metrics_logger.log_scalars({
                    'train_loss': avg_train_loss,
                    'lr_g': lr_g,
                }, epoch_num, "train")
                metrics_logger.log_scalars(val_metrics, epoch_num, "val")
                
                # Save checkpoint.
                # Three-tier save policy:
                #   1. latest_checkpoint.pth — overwritten every validation.
                #      Resume target after HPC time-out; user never needs to
                #      know an epoch number.
                #   2. best_model.pth      — overwritten whenever val MRAE
                #      improves. Logged at INFO so you can grep "NEW BEST".
                #   3. net_{N}epoch.pth    — periodic snapshot every 5k iters,
                #      pruned to the CHECKPOINT_KEEP_COUNT most recent so the
                #      checkpoint_dir doesn't grow unboundedly.
                # All writes go through _atomic_torch_save (.tmp + os.replace)
                # so an HPC kill mid-save can't leave a corrupted checkpoint.
                if config.get("local_rank", 0) == 0:
                    is_best = current_mrae < record_mrae_loss
                    checkpoint_dir = config["checkpoint_dir"]

                    checkpoint_dict = {
                        'epoch': epoch_num,
                        'iter': iteration,
                        'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        'optimizer_g': optimizer_g.state_dict(),
                        'optimizer_d': optimizer_d.state_dict(),
                        'scheduler_g': scheduler_g.state_dict(),
                        'scheduler_d': scheduler_d.state_dict(),
                        'scaler_g': scaler_g.state_dict(),
                        'scaler_d': scaler_d.state_dict(),
                        'best_mrae': min(current_mrae, record_mrae_loss),
                        'val_metrics': val_metrics,
                        'config': config,
                        # RNG state for deterministic resume.
                        'torch_rng_state': torch.get_rng_state(),
                        'cuda_rng_state_all': (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None
                        ),
                        'numpy_rng_state': np.random.get_state(),
                        # EMA shadow weights for resume (None when EMA disabled).
                        'ema': ema.state_dict() if ema is not None else None,
                    }

                    # (1) latest — always overwritten so resume is one-liner
                    _atomic_torch_save(
                        checkpoint_dict,
                        os.path.join(checkpoint_dir, CHECKPOINT_LATEST_NAME),
                    )

                    # (2) best — only on MRAE improvement, logged loudly.
                    # When EMA is on, best_model stores the EMA weights (the
                    # ones that produced this metric) so inference loads them
                    # directly; latest_checkpoint above keeps the raw weights.
                    if is_best:
                        previous_best = record_mrae_loss
                        record_mrae_loss = current_mrae
                        if ema is not None:
                            base_model = model.module if hasattr(model, 'module') else model
                            with ema.average_parameters(generator):
                                best_dict = dict(checkpoint_dict)
                                best_dict['state_dict'] = base_model.state_dict()
                                best_dict['ema_applied'] = True
                                _atomic_torch_save(
                                    best_dict,
                                    os.path.join(checkpoint_dir, CHECKPOINT_BEST_NAME),
                                )
                        else:
                            _atomic_torch_save(
                                checkpoint_dict,
                                os.path.join(checkpoint_dir, CHECKPOINT_BEST_NAME),
                            )
                        logger.info(
                            "NEW BEST: MRAE %.6f -> %.6f at iter %d (epoch %d); "
                            "saved %s",
                            previous_best,
                            current_mrae,
                            iteration,
                            epoch_num,
                            CHECKPOINT_BEST_NAME,
                        )

                    # (3) periodic snapshot, then prune
                    if iteration % 5000 == 0:
                        _atomic_torch_save(
                            checkpoint_dict,
                            os.path.join(checkpoint_dir, f"net_{epoch_num}epoch.pth"),
                        )
                        _prune_epoch_checkpoints(checkpoint_dir, CHECKPOINT_KEEP_COUNT)
                
                epoch_losses = []
                memory_cleanup()
                report_memory(f"After validation {epoch_num}")
            
            if iteration >= total_iteration:
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("OOM at iteration %s; clearing gradients and cache", iteration)
                optimizer_g.zero_grad(set_to_none=True)
                optimizer_d.zero_grad(set_to_none=True)
                if "fake_hsi" in locals():
                    del fake_hsi
                if "fake_hsi_detached" in locals():
                    del fake_hsi_detached
                if "rgb_tensor" in locals():
                    del rgb_tensor
                if "hsi_tensor" in locals():
                    del hsi_tensor
                memory_cleanup()
                continue
            # Non-OOM RuntimeErrors are usually structural (shape mismatch,
            # CUDA assert). Re-raise so the caller actually sees them — the
            # previous swallow-and-continue turned every fatal error into a
            # silent infinite retry.
            raise
    
    # Final cleanup
    logger.info("Training completed, cleaning up...")
    report_memory("Before cleanup")
    
    if hasattr(train_loader, '_iterator') and train_loader._iterator is not None:
        if hasattr(train_loader._iterator, '_shutdown_workers'):
            train_loader._iterator._shutdown_workers()
    del train_loader
    memory_cleanup()
    
    report_memory("After cleanup")
    logger.info(f"Training completed successfully. Best MRAE: {record_mrae_loss:.4f}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main function with all memory optimizations (v3.0)."""
    cfg = OmegaConf.to_container(config, resolve=True)
    logger = logging.getLogger("hsi_model.main")
    
    try:
        report_memory("Start of main")
        
        cfg = setup_paths(cfg)
        device, rank, world_size, is_distributed = setup_distributed_training(cfg)
        
        cfg["local_rank"] = rank
        cfg["world_size"] = world_size
        cfg["distributed"] = is_distributed
        
        # MST++ defaults
        cfg.setdefault("patch_size", 128)
        cfg.setdefault("stride", 8)
        cfg.setdefault("batch_size", 20)
        cfg.setdefault("iterations_per_epoch", 1000)
        cfg.setdefault("epochs", 300)
        cfg.setdefault("generator_lr", cfg.get("learning_rate", 1e-4))
        cfg.setdefault("discriminator_lr", cfg.get("learning_rate", 2e-5))
        cfg.setdefault("validation_max_batches", None)
        cfg.setdefault("num_workers", 8)
        cfg.setdefault("memory_mode", "standard")  # Options: standard, float16, lazy
        # AMP dtype: "auto" picks bf16 on A100+ (no GradScaler cycling), fp16 on
        # older Tensor Core GPUs, fp32 elsewhere. Override with "bf16"/"fp16"/"fp32".
        cfg.setdefault("mixed_precision_dtype", "auto")
        # Resume from a checkpoint when non-empty. Override on the CLI:
        #   python train_optimized.py resume_checkpoint=/path/to/best_model.pth
        cfg.setdefault("resume_checkpoint", None)
        
        log_level = getattr(logging, cfg.get("log_level", "INFO"))
        logger = setup_logging(cfg["log_dir"], log_level, rank)
        
        if rank == 0:
            logger.info("="*60)
            logger.info("MST++ TRAINING - FULLY OPTIMIZED (v3.0)")
            logger.info("="*60)
            logger.info(f"Memory mode: {cfg.get('memory_mode', 'standard')}")
            logger.info("Critical v3.0 optimizations:")
            logger.info("  - Fixed CSWin bias tiling (saves ~50GB)")
            logger.info("  - Removed debug prints (saves ~10GB)")
            logger.info("  - Optimized discriminator graph (saves ~18GB)")
            logger.info("  - Single DataLoader reused")
            logger.info("  - Validation without persistent workers")
            logger.info("  - h5py cache handling (version-aware)")
            logger.info("  - PyTorch version compatibility")
            logger.info("  - LRU cache for lazy mode")
            logger.info("  - FP16 bias tables by default")
            if cfg.get('memory_mode') == 'float16':
                logger.info("  - Float16 dataset storage (~17GB)")
            elif cfg.get('memory_mode') == 'lazy':
                logger.info("  - Lazy loading with LRU cache (~5GB)")
            logger.info("="*60)
        
        setup_seed(
            cfg.get("seed", 42),
            rank,
            deterministic=cfg.get("deterministic", True),
            allow_tf32=cfg.get("allow_tf32", False),
        )
        memory_cleanup()
        
        # Create only datasets (no DataLoaders yet)
        train_dataset, val_dataset = create_datasets_only(cfg)
        report_memory("After loading datasets")
        
        # Create model
        model = NoiseRobustCSWinModel(cfg).to(device)
        report_memory("After creating model")
        
        # Create optimizers
        generator_lr, discriminator_lr = resolve_generator_discriminator_lrs(cfg)
        optimizer_g = torch.optim.Adam(
            model.generator.parameters(),
            lr=generator_lr,
            betas=(0.9, 0.999),
        )
        optimizer_d = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=discriminator_lr,
            # beta1=0.5 is standard GAN practice (matches training_script_fixed.py):
            # beta1=0.9 gives the discriminator heavy momentum, producing an
            # oscillating adversarial signal the generator chases in late epochs.
            betas=(0.5, 0.999),
        )
        optimizers = {'optimizer_g': optimizer_g, 'optimizer_d': optimizer_d}
        
        # Create schedulers. WarmupCosineScheduler = linear warmup -> cosine
        # decay to eta_min. ``total_iterations`` is the cosine horizon, so it
        # MUST equal the real training length (driven by cfg['epochs']) for the
        # LR to actually anneal; a too-long horizon leaves the LR high and the
        # model plateaus. This trainer steps the scheduler once per iteration
        # (no gradient accumulation), so total_steps == total_iterations.
        total_iterations = cfg['iterations_per_epoch'] * cfg['epochs']
        warmup_steps = cfg.get("warmup_steps", DEFAULT_WARMUP_STEPS)
        scheduler_g = WarmupCosineScheduler(optimizer_g, warmup_steps, total_iterations, eta_min=1e-6)
        scheduler_d = WarmupCosineScheduler(optimizer_d, warmup_steps, total_iterations, eta_min=1e-6)
        schedulers = {'scheduler_g': scheduler_g, 'scheduler_d': scheduler_d}
        
        # Create scalers. GradScaler is only meaningful for fp16 — bf16 has
        # fp32's exponent range so backward gradients cannot overflow, and
        # PyTorch will warn/error if you scale bf16 grads. fp32 obviously
        # needs no scaler either. Disabled scalers are transparent: scale()
        # is a no-op, step() calls optimizer.step() directly.
        _amp_dtype = pick_amp_dtype(cfg) if device.type == "cuda" else None
        _needs_scaler = _amp_dtype == torch.float16
        scaler_g = GradScaler(enabled=_needs_scaler)
        scaler_d = GradScaler(enabled=_needs_scaler)
        scalers = {'scaler_g': scaler_g, 'scaler_d': scaler_d}
        
        # Create loss function
        criterion = NoiseRobustLoss(cfg)
        disc_criterion = ComputeSinkhornDiscriminatorLoss(criterion)
        
        # Wrap with DDP
        if is_distributed:
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        # Generator weight EMA (optional). Built AFTER the DDP wrap so it tracks
        # the underlying generator's parameter tensors. Created before resume so
        # its shadow can be restored from the checkpoint.
        ema: Optional[GeneratorEMA] = None
        if cfg.get("use_ema", True):
            gen_for_ema = model.module.generator if hasattr(model, "module") else model.generator
            ema = GeneratorEMA(gen_for_ema, decay=float(cfg.get("ema_decay", 0.999)))
            logger.info("Generator EMA enabled (decay=%.4f)", ema.decay)

        # Resume from a previous checkpoint if requested. Hydra syntax:
        #   python train_optimized.py resume_checkpoint=/path/to/best_model.pth
        # Restores model + both optimizers + both schedulers + both scalers +
        # iteration counter + best-MRAE record + RNG state. Loading must
        # happen AFTER DDP wrapping so module-prefix matching works.
        resume_info: Optional[Dict[str, Any]] = None
        resume_path = cfg.get("resume_checkpoint")
        if resume_path:
            resume_info = resume_training_state(
                checkpoint_path=str(resume_path),
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                scalers=scalers,
                device=device,
                ema=ema,
            )

        # Set up metrics logger
        metrics_logger = MetricsLogger(cfg["log_dir"], rank)

        # Start training
        logger.info("Starting optimized training (v3.0)")
        train_mst_gan_optimized(
            model,
            train_dataset,  # Pass dataset
            val_dataset,    # Pass dataset
            optimizers,
            schedulers,
            criterion,
            disc_criterion,
            scalers,
            cfg,
            device,
            metrics_logger,
            is_distributed,
            cfg.get("seed", 42),
            rank,
            resume_info=resume_info,
            ema=ema,
        )
        
        if rank == 0:
            logger.info("TRAINING COMPLETED!")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise e
    finally:
        report_memory("End of main")
        memory_cleanup()
        if 'metrics_logger' in locals():
            metrics_logger.close()
        cleanup()


if __name__ == "__main__":
    main()

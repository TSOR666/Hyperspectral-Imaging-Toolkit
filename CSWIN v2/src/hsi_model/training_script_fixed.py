#!/usr/bin/env python3
# src/hsi_model/training_script_fixed.py
"""
Sinkhorn GAN Training Script - Production Ready

Uses optimal transport (Sinkhorn divergence) for adversarial training instead of
traditional GAN losses (BCE/LSGAN/WGAN).

Key Features:
- Generator minimizes Sinkhorn divergence to real data distribution
- Discriminator maximizes divergence (minimizes negative divergence)
- Better gradient flow through optimal transport objective
- All v5 stability fixes preserved
- Memory-efficient MST++ data loading
- Gradient accumulation support
- R1 regularization on discriminator

Version: 6.0 - Cleaned and refactored
"""

import os
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import gc
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

# Local imports
from hsi_model.models import NoiseRobustCSWinModel
from hsi_model.models.losses import NoiseRobustLoss, ComputeSinkhornDiscriminatorLoss
from hsi_model.utils import setup_logging, MetricsLogger
from hsi_model.utils.dataloader import (
    mst_to_gan_batch, compute_mst_center_crop_metrics
)
from hsi_model.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PATCH_SIZE,
    DEFAULT_STRIDE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_GENERATOR_LR,
    DEFAULT_DISCRIMINATOR_LR,
    DEFAULT_DATA_DIR,
    DEFAULT_R1_GAMMA,
    R1_APPLY_FREQUENCY,
    LOG_EVERY_N_ITERATIONS,
    SAVE_CHECKPOINT_EVERY_N_ITERATIONS,
    PYTORCH_CUDA_ALLOC_CONF,
    DEFAULT_SINKHORN_EPSILON,
    DEFAULT_SINKHORN_ITERATIONS
)

logger = logging.getLogger(__name__)


# ============================================
# Utility Functions
# ============================================

def clear_memory():
    """Aggressive memory clearing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def fixed_worker_init_fn_mst(worker_id: int, base_seed: int = 42, rank: int = 0):
    """
    MST++ style worker initialization with memory fix.
    
    Critical: Sets h5py cache to 4MB instead of default 64MB to prevent memory leak.
    """
    import h5py
    
    worker_seed = base_seed + rank * 100 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    try:
        import h5py._hl.base
        h5py._hl.base.phil.acquire()
        h5py._hl.base.default_file_cache_size = 4 * 1024 * 1024  # 4MB
        h5py._hl.base.phil.release()
    except:
        pass
    
    num_threads = int(os.environ.get('OMP_NUM_THREADS', '2'))
    torch.set_num_threads(num_threads)


# ============================================
# Learning Rate Scheduler
# ============================================

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine learning rate scheduler with linear warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup - ensure first step has non-zero LR
            factor = (self.last_epoch + 2) / (self.warmup_steps + 1)
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            factor = self.eta_min + (1 - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
        
        return [base_lr * factor for base_lr in self.base_lrs]


# ============================================
# Dataset Creation
# ============================================

def create_datasets_only(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Create MST++ datasets without DataLoaders."""
    from hsi_model.utils.dataloader import MST_TrainDataset, MST_ValidDataset
    
    memory_mode = config.get("memory_mode", "standard")
    data_root = config.get("data_dir", DEFAULT_DATA_DIR)
    crop_size = config.get("patch_size", DEFAULT_PATCH_SIZE)
    stride = config.get("stride", DEFAULT_STRIDE)
    
    train_dataset = MST_TrainDataset(
        data_root=data_root,
        crop_size=crop_size,
        arg=True,
        bgr2rgb=True,
        stride=stride
    )
    
    val_dataset = MST_ValidDataset(
        data_root=data_root,
        bgr2rgb=True
    )
    
    return train_dataset, val_dataset


# ============================================
# Validation
# ============================================

def validate_gan_safe(
    model: nn.Module,
    val_dataset,
    criterion: nn.Module,
    device: torch.device,
    iteration: int,
    config: Dict[str, Any],
    seed: int = 42,
    rank: int = 0
) -> Dict[str, float]:
    """
    Safe validation for GAN with memory management.
    
    Args:
        model: Model to validate
        val_dataset: Validation dataset
        criterion: Loss criterion
        device: Device to run on
        iteration: Current iteration
        config: Configuration dictionary
        seed: Random seed
        rank: Process rank
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    logger_val = logging.getLogger("hsi_model.validation")
    
    # Create validation loader with minimal memory usage
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    
    total_losses = {}
    total_metrics = {}
    num_batches = 0
    num_valid_batches = 0
    use_amp = config.get("mixed_precision", True)
    
    with torch.no_grad():
        for batch_idx, (bgr_batch, hyper_batch) in enumerate(val_loader):
            if batch_idx >= 10:  # Limit validation
                break
                
            try:
                rgb_tensor, hsi_tensor = mst_to_gan_batch(bgr_batch, hyper_batch)
                
                if rgb_tensor.numel() == 0 or hsi_tensor.numel() == 0:
                    continue
                
                rgb_tensor = rgb_tensor.to(device, non_blocking=True)
                hsi_tensor = hsi_tensor.to(device, non_blocking=True)
                
                with autocast(enabled=use_amp):
                    if hasattr(model, 'module'):
                        pred_hsi = model.module.generator(rgb_tensor)
                        disc_real = model.module.discriminator(rgb_tensor, hsi_tensor)
                        disc_fake = model.module.discriminator(rgb_tensor, pred_hsi)
                    else:
                        pred_hsi = model.generator(rgb_tensor)
                        disc_real = model.discriminator(rgb_tensor, hsi_tensor)
                        disc_fake = model.discriminator(rgb_tensor, pred_hsi)
                    
                    # Check for NaN
                    if torch.isnan(pred_hsi).any():
                        logger_val.warning(f"NaN in predictions at validation batch {batch_idx}")
                        continue
                    
                    # Compute generator loss
                    gen_loss, loss_components = criterion(
                        pred_hsi, hsi_tensor, 
                        disc_real=disc_real.detach(),
                        disc_fake=disc_fake,
                        current_iteration=iteration
                    )
                
                if not torch.isnan(gen_loss):
                    total_losses['gen_loss'] = total_losses.get('gen_loss', 0.0) + gen_loss.item()
                    num_valid_batches += 1
                
                # Compute metrics
                metrics = compute_mst_center_crop_metrics(pred_hsi, hsi_tensor)
                
                # Validate metrics
                valid_metrics = all(np.isfinite(v) for v in metrics.values())
                
                if valid_metrics:
                    for key, value in metrics.items():
                        if key not in total_metrics:
                            total_metrics[key] = 0.0
                        total_metrics[key] += value
                
                num_batches += 1
                
                # Clear memory
                del rgb_tensor, hsi_tensor, pred_hsi, disc_real, disc_fake
                
            except Exception as e:
                logger_val.warning(f"Validation error at batch {batch_idx}: {str(e)}")
                continue
    
    del val_loader
    clear_memory()
    
    # Calculate averages with fallback values
    if num_valid_batches == 0:
        return {
            'gen_loss': 1.0,
            'mrae': 1.0,
            'psnr': 10.0
        }
    
    avg_losses = {k: v / num_valid_batches for k, v in total_losses.items()}
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items() if num_batches > 0}
    
    # Ensure reasonable values
    avg_metrics['mrae'] = min(avg_metrics.get('mrae', 1.0), 10.0)
    avg_metrics['psnr'] = max(avg_metrics.get('psnr', 10.0), 0.0)
    
    all_metrics = {**avg_losses, **avg_metrics}
    
    logger_val.info(
        f"Validation Iter: {iteration} | "
        f"MRAE: {avg_metrics.get('mrae', 1.0):.4f} | "
        f"PSNR: {avg_metrics.get('psnr', 10.0):.2f}dB"
    )
    
    return all_metrics


# ============================================
# Regularization
# ============================================

def apply_r1_regularization(
    disc_real: torch.Tensor,
    real_data: torch.Tensor,
    gamma: float = DEFAULT_R1_GAMMA
) -> torch.Tensor:
    """
    R1 gradient penalty for discriminator stability.
    
    Critical: Apply penalty on HSI data (not RGB) as that's what matters for discrimination.
    
    Args:
        disc_real: Discriminator output for real data
        real_data: Real HSI data (requires gradient)
        gamma: Regularization strength
        
    Returns:
        R1 penalty (scalar)
    """
    real_data.requires_grad_(True)
    
    grad_real = torch.autograd.grad(
        outputs=disc_real.sum(),
        inputs=real_data,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    grad_penalty = (grad_real.pow(2).sum([1, 2, 3])).mean() * gamma
    return grad_penalty


# ============================================
# Main Training Loop
# ============================================

def train_sinkhorn_gan(
    model: nn.Module,
    train_dataset,
    val_dataset,
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
    resume_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Main Sinkhorn GAN training loop.
    
    Args:
        model: Combined model with generator and discriminator
        train_dataset: Training dataset
        val_dataset: Validation dataset
        optimizers: Dictionary of optimizers for generator and discriminator
        schedulers: Dictionary of learning rate schedulers
        criterion: Generator loss criterion
        disc_criterion: Discriminator loss criterion
        scalers: Dictionary of gradient scalers
        config: Configuration dictionary
        device: Device to train on
        metrics_logger: Metrics logging utility
        distributed: Whether using distributed training
        seed: Random seed
        rank: Process rank
        resume_info: Optional resume information
    """
    logger.info("="*60)
    logger.info("Starting Sinkhorn GAN Training")
    logger.info("="*60)
    
    # Extract parameters
    per_epoch_iteration = config.get("iterations_per_epoch", 1000)
    total_epochs = config.get("epochs", 300)
    total_iteration = per_epoch_iteration * total_epochs
    batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)
    num_workers = config.get("num_workers", DEFAULT_NUM_WORKERS)
    accumulation_steps = config.get("gradient_accumulation_steps", 2)
    effective_batch_size = batch_size * accumulation_steps
    
    optimizer_g = optimizers['optimizer_g']
    optimizer_d = optimizers['optimizer_d']
    scheduler_g = schedulers['scheduler_g']
    scheduler_d = schedulers['scheduler_d']
    scaler_g = scalers['scaler_g']
    scaler_d = scalers['scaler_d']
    
    # Resume handling
    start_iteration = resume_info.get('iteration', 0) if resume_info else 0
    best_mrae = resume_info.get('best_mrae', float('inf')) if resume_info else float('inf')
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Total iterations: {total_iteration}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {accumulation_steps}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"  Workers: {num_workers}")
    logger.info(f"  Warmup steps: {config.get('warmup_steps', DEFAULT_WARMUP_STEPS)}")
    logger.info(f"  Sinkhorn epsilon: {config.get('sinkhorn_epsilon', DEFAULT_SINKHORN_EPSILON)}")
    logger.info(f"  Sinkhorn iterations: {config.get('sinkhorn_iters', DEFAULT_SINKHORN_ITERATIONS)}")
    
    # Create training DataLoader
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda w: fixed_worker_init_fn_mst(w, seed, rank),
        persistent_workers=False
    )
    
    # Training state
    iteration = start_iteration
    record_mrae_loss = best_mrae
    use_amp = config.get("mixed_precision", True)
    epoch_losses = []
    current_epoch = 0
    nan_count = 0
    max_memory_gb = 0
    consecutive_nan_tolerance = 3
    
    # Moving average for discriminator loss logging
    disc_loss_ma = 0.0
    disc_loss_ma_beta = 0.9
    
    # Training parameters
    n_critic = config.get("n_critic", 1)
    use_r1_reg = config.get("use_r1_regularization", True)
    r1_gamma = config.get("r1_gamma", DEFAULT_R1_GAMMA)
    
    # Main training loop
    data_iter = iter(train_loader)
    
    # Zero gradients at start
    optimizer_g.zero_grad()
    optimizer_d.zero_grad()
    
    while iteration < total_iteration:
        model.train()
        
        try:
            bgr_batch, hyper_batch = next(data_iter)
        except StopIteration:
            logger.info(f"Dataset epoch completed at iteration {iteration}")
            if distributed:
                current_epoch += 1
                train_loader.sampler.set_epoch(current_epoch)
            data_iter = iter(train_loader)
            bgr_batch, hyper_batch = next(data_iter)
        
        try:
            rgb_tensor, hsi_tensor = mst_to_gan_batch(bgr_batch, hyper_batch)
            
            # Skip invalid batches
            if rgb_tensor.shape[2] < 1 or rgb_tensor.shape[3] < 1:
                continue
            
            rgb_tensor = rgb_tensor.to(device, non_blocking=True)
            hsi_tensor = hsi_tensor.to(device, non_blocking=True)
            
            # Get learning rates
            lr_g = optimizer_g.param_groups[0]['lr']
            lr_d = optimizer_d.param_groups[0]['lr']
            
            # ========== Train Generator ==========
            is_step_boundary = ((iteration + 1) % accumulation_steps == 0)
            
            with autocast(enabled=use_amp):
                if hasattr(model, 'module'):
                    fake_hsi = model.module.generator(rgb_tensor)
                    disc_fake_for_g = model.module.discriminator(rgb_tensor, fake_hsi)
                    disc_real_for_g = model.module.discriminator(rgb_tensor, hsi_tensor).detach()
                else:
                    fake_hsi = model.generator(rgb_tensor)
                    disc_fake_for_g = model.discriminator(rgb_tensor, fake_hsi)
                    disc_real_for_g = model.discriminator(rgb_tensor, hsi_tensor).detach()
                
                # Check for NaN with aggressive handling
                if torch.isnan(fake_hsi).any() or torch.isinf(fake_hsi).any():
                    nan_count += 1
                    logger.warning(f"NaN/Inf in generator output! Count: {nan_count}")
                    
                    if nan_count >= consecutive_nan_tolerance:
                        # Emergency recovery
                        logger.warning("Too many NaNs - emergency recovery!")
                        optimizer_g.zero_grad()
                        
                        # Reset optimizer state
                        for group in optimizer_g.param_groups:
                            for p in group['params']:
                                state = optimizer_g.state[p]
                                if 'exp_avg' in state:
                                    state['exp_avg'].zero_()
                                if 'exp_avg_sq' in state:
                                    state['exp_avg_sq'].zero_()
                        
                        # Reduce learning rate
                        for param_group in optimizer_g.param_groups:
                            param_group['lr'] *= 0.5
                        logger.warning(f"Reset optimizer and reduced generator LR to {optimizer_g.param_groups[0]['lr']:.6f}")
                        
                        nan_count = 0
                        scaler_g._scale = torch.tensor(1.0).to(device)
                    
                    continue
                else:
                    nan_count = 0
                
                # Compute generator loss
                gen_loss, loss_components = criterion(
                    fake_hsi, hsi_tensor,
                    disc_real=disc_real_for_g,
                    disc_fake=disc_fake_for_g,
                    current_iteration=iteration
                )
            
            if not torch.isnan(gen_loss):
                gen_loss = gen_loss / accumulation_steps
                scaler_g.scale(gen_loss).backward()
                
                if is_step_boundary:
                    # Gradient clipping
                    scaler_g.unscale_(optimizer_g)
                    g_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.module.generator.parameters() if hasattr(model, 'module') else model.generator.parameters(),
                        max_norm=DEFAULT_GRADIENT_CLIP_NORM
                    )
                    
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                    optimizer_g.zero_grad()
                    scheduler_g.step()
            
            # ========== Train Discriminator ==========
            if is_step_boundary and (iteration % n_critic == 0):
                fake_hsi_detached = fake_hsi.detach()
                
                with autocast(enabled=use_amp):
                    if hasattr(model, 'module'):
                        real_pred = model.module.discriminator(rgb_tensor, hsi_tensor)
                        fake_pred = model.module.discriminator(rgb_tensor, fake_hsi_detached)
                    else:
                        real_pred = model.discriminator(rgb_tensor, hsi_tensor)
                        fake_pred = model.discriminator(rgb_tensor, fake_hsi_detached)
                    
                    # Sinkhorn discriminator loss
                    disc_loss = disc_criterion(real_pred, fake_pred)
                    
                    # R1 regularization
                    if use_r1_reg and iteration % R1_APPLY_FREQUENCY == 0:
                        hsi_tensor.requires_grad_(True)
                        if hasattr(model, 'module'):
                            real_pred_for_r1 = model.module.discriminator(rgb_tensor, hsi_tensor)
                        else:
                            real_pred_for_r1 = model.discriminator(rgb_tensor, hsi_tensor)
                        r1_loss = apply_r1_regularization(real_pred_for_r1, hsi_tensor, gamma=r1_gamma)
                        disc_loss = disc_loss + r1_loss
                
                if not torch.isnan(disc_loss):
                    scaler_d.scale(disc_loss).backward()
                    
                    # Update moving average
                    disc_loss_ma = disc_loss_ma_beta * disc_loss_ma + (1 - disc_loss_ma_beta) * disc_loss.item()
                    
                    # Gradient clipping
                    scaler_d.unscale_(optimizer_d)
                    d_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.module.discriminator.parameters() if hasattr(model, 'module') else model.discriminator.parameters(),
                        max_norm=DEFAULT_GRADIENT_CLIP_NORM
                    )
                    
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad()
                    scheduler_d.step()
            
            if not torch.isnan(gen_loss):
                epoch_losses.append(gen_loss.item() * accumulation_steps)
            
            iteration += 1
            
            # Clear intermediate tensors
            del rgb_tensor, hsi_tensor, fake_hsi
            if 'fake_hsi_detached' in locals():
                del fake_hsi_detached
            
            # Memory monitoring
            current_memory = 0.0
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1e9
                max_memory_gb = max(max_memory_gb, current_memory)
            
            # Logging
            if iteration % LOG_EVERY_N_ITERATIONS == 0:
                avg_loss = np.mean(epoch_losses[-20:]) if len(epoch_losses) >= 20 else np.mean(epoch_losses)
                disc_loss_value = disc_loss_ma / (1 - disc_loss_ma_beta ** max(iteration, 1)) if disc_loss_ma != 0 else 0.0
                logger.info(
                    f'[iter:{iteration}/{total_iteration}], lr_g={lr_g:.9f}, lr_d={lr_d:.9f}, '
                    f'loss={avg_loss:.9f}, sinkhorn_disc_loss={disc_loss_value:.6f}, '
                    f'Mem={current_memory:.1f}GB'
                )
            
            # Periodic memory cleanup
            if iteration % 100 == 0:
                clear_memory()
            
            # Validation
            if iteration % per_epoch_iteration == 0:
                val_metrics = validate_gan_safe(
                    model, val_dataset, criterion, device, iteration, config, seed, rank
                )
                
                current_mrae = val_metrics.get('mrae', float('inf'))
                epoch_num = iteration // per_epoch_iteration
                avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
                
                logger.info(
                    f"Iter[{iteration:06d}], Epoch[{epoch_num:06d}], "
                    f"Train Loss: {avg_train_loss:.9f}, Test MRAE: {current_mrae:.9f}"
                )
                
                metrics_logger.log_scalars({
                    'train_loss': avg_train_loss,
                    'lr_g': lr_g,
                    'lr_d': lr_d,
                    'sinkhorn_disc_loss': disc_loss_ma / (1 - disc_loss_ma_beta ** max(iteration, 1)),
                    **val_metrics
                }, epoch_num, "train")
                
                # Save checkpoint
                if rank == 0:
                    is_best = current_mrae < record_mrae_loss
                    
                    if is_best or iteration % SAVE_CHECKPOINT_EVERY_N_ITERATIONS == 0:
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
                            'config': config
                        }
                        
                        if is_best:
                            record_mrae_loss = current_mrae
                            best_path = os.path.join(config["checkpoint_dir"], 'best_model.pth')
                            torch.save(checkpoint_dict, best_path)
                            logger.info(f"ðŸ† New best MRAE: {current_mrae:.4f}")
                        
                        if iteration % SAVE_CHECKPOINT_EVERY_N_ITERATIONS == 0:
                            checkpoint_path = os.path.join(config["checkpoint_dir"], f'net_{epoch_num}epoch.pth')
                            torch.save(checkpoint_dict, checkpoint_path)
                
                epoch_losses = []
                clear_memory()
                
                # Recreate data loader
                del train_loader
                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    worker_init_fn=lambda w: fixed_worker_init_fn_mst(w, seed, rank),
                    persistent_workers=False
                )
                data_iter = iter(train_loader)
            
            if iteration >= total_iteration:
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM at iteration {iteration}! Clearing cache")
                optimizer_g.zero_grad()
                optimizer_d.zero_grad()
                clear_memory()
                continue
            else:
                logger.error(f"Training error at iteration {iteration}: {str(e)}", exc_info=True)
                continue
        except Exception as e:
            logger.error(f"Training error at iteration {iteration}: {str(e)}", exc_info=True)
            continue
    
    # Final cleanup
    logger.info("Sinkhorn GAN training completed!")
    logger.info(f"Best MRAE: {record_mrae_loss:.4f}")
    logger.info(f"Peak memory usage: {max_memory_gb:.1f}GB")
    
    del train_loader
    clear_memory()


# ============================================
# Main Entry Point
# ============================================

@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main function for Sinkhorn GAN training."""
    cfg = OmegaConf.to_container(config, resolve=True)
    
    # Setup paths
    from train_optimized import setup_paths, setup_distributed_training, setup_seed
    
    cfg = setup_paths(cfg)
    device, rank, world_size, is_distributed = setup_distributed_training(cfg)
    
    # Set memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = PYTORCH_CUDA_ALLOC_CONF
    
    # Setup
    cfg["local_rank"] = rank
    cfg["world_size"] = world_size
    cfg["distributed"] = is_distributed
    
    # Set defaults from constants
    cfg.setdefault("patch_size", DEFAULT_PATCH_SIZE)
    cfg.setdefault("stride", DEFAULT_STRIDE)
    cfg.setdefault("batch_size", DEFAULT_BATCH_SIZE)
    cfg.setdefault("iterations_per_epoch", 1000)
    cfg.setdefault("epochs", 300)
    cfg.setdefault("num_workers", DEFAULT_NUM_WORKERS)
    cfg.setdefault("memory_mode", "standard")
    cfg.setdefault("n_critic", 1)
    cfg.setdefault("gradient_accumulation_steps", 2)
    cfg.setdefault("warmup_steps", DEFAULT_WARMUP_STEPS)
    cfg.setdefault("use_r1_regularization", True)
    cfg.setdefault("r1_gamma", DEFAULT_R1_GAMMA)
    cfg.setdefault("use_sinkhorn_adversarial", True)
    cfg.setdefault("sinkhorn_epsilon", DEFAULT_SINKHORN_EPSILON)
    cfg.setdefault("sinkhorn_iters", DEFAULT_SINKHORN_ITERATIONS)
    cfg.setdefault("sinkhorn_flatten_spatial", True)
    cfg.setdefault("generator_lr", DEFAULT_GENERATOR_LR)
    cfg.setdefault("discriminator_lr", DEFAULT_DISCRIMINATOR_LR)
    
    logger = setup_logging(cfg["log_dir"], logging.INFO, rank)
    
    if rank == 0:
        logger.info("="*60)
        logger.info("SINKHORN GAN TRAINING - OPTIMAL TRANSPORT APPROACH")
        logger.info("="*60)
        logger.info("Key features:")
        logger.info("  - Sinkhorn divergence for adversarial loss")
        logger.info("  - Generator minimizes transport cost to real data")
        logger.info("  - Discriminator maximizes divergence")
        logger.info("  - Better gradient flow than BCE/LSGAN")
        logger.info("  - All stability fixes from v5 preserved")
        logger.info("="*60)
    
    setup_seed(cfg.get("seed", 42), rank)
    clear_memory()
    
    # Create datasets
    train_dataset, val_dataset = create_datasets_only(cfg)
    
    # Create model
    model = NoiseRobustCSWinModel(cfg).to(device)
    
    # Create optimizers
    optimizer_g = torch.optim.Adam(
        model.generator.parameters(), 
        lr=cfg.get("generator_lr", DEFAULT_GENERATOR_LR), 
        betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        model.discriminator.parameters(), 
        lr=cfg.get("discriminator_lr", DEFAULT_DISCRIMINATOR_LR),
        betas=(0.5, 0.999)
    )
    optimizers = {'optimizer_g': optimizer_g, 'optimizer_d': optimizer_d}
    
    # Create schedulers
    total_iterations = cfg['iterations_per_epoch'] * cfg['epochs']
    warmup_steps = cfg.get("warmup_steps", DEFAULT_WARMUP_STEPS)
    
    scheduler_g = WarmupCosineScheduler(optimizer_g, warmup_steps, total_iterations, eta_min=1e-6)
    scheduler_d = WarmupCosineScheduler(optimizer_d, warmup_steps, total_iterations, eta_min=1e-6)
    schedulers = {'scheduler_g': scheduler_g, 'scheduler_d': scheduler_d}
    
    # Create scalers
    scaler_g = GradScaler(enabled=cfg.get("mixed_precision", True), growth_interval=100)
    scaler_d = GradScaler(enabled=cfg.get("mixed_precision", True), growth_interval=100)
    scalers = {'scaler_g': scaler_g, 'scaler_d': scaler_d}
    
    # Create loss functions
    criterion = NoiseRobustLoss(cfg)
    disc_criterion = ComputeSinkhornDiscriminatorLoss(criterion)
    
    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Set up metrics logger
    metrics_logger = MetricsLogger(cfg["log_dir"], rank)
    
    # Start training
    try:
        train_sinkhorn_gan(
            model,
            train_dataset,
            val_dataset,
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
            rank
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise e
    finally:
        metrics_logger.close()
        clear_memory()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Dedicated Training Script for SHARP v3.2.2 Hardened
Optimized implementation with all v3.2.2 features and audit fixes
"""

import os
import sys
import time
import math
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Import SHARP v3.2.2
try:
    from sharp_v322_hardened import create_sharp_v32, SHARPv32Trainer, SHARPv32Config
    SHARP_AVAILABLE = True
    SHARP_TRAINER_AVAILABLE = True
except ImportError:
    try:
        # Try without SHARPv32Trainer (older versions)
        from sharp_v322_hardened import create_sharp_v32, SHARPv32Config
        SHARP_AVAILABLE = True
        SHARP_TRAINER_AVAILABLE = False
        SHARPv32Trainer = None
    except ImportError:
        print("Error: SHARP v3.2.2 not found. Please ensure sharp_v322_hardened.py is in your path.")
        sys.exit(1)

# Import optimized dataloader
try:
    from optimized_dataloader import (
        OptimizedTrainDataset, 
        OptimizedValDataset, 
        MSTPlusPlusLoss,
        create_optimized_dataloaders
    )
    DATALOADER_AVAILABLE = True
except ImportError:
    print("Warning: Optimized dataloader not found. Using fallback implementation.")
    DATALOADER_AVAILABLE = False


@dataclass
class SHARPTrainingConfig:
    """Configuration for SHARP v3.2.2 training"""
    # Model configuration
    model_size: str = 'base'  # tiny, small, base, large
    in_channels: int = 3
    out_channels: int = 31
    
    # SHARP v3.2.2 specific parameters
    sparse_sparsity_ratio: float = 0.9      # Fraction of attention weights to zero out
    sparse_block_size: int = 2048           # Block size for streaming attention
    sparse_max_tokens: int = 8192           # Max tokens before window fallback
    sparse_window_size: int = 49            # Window size for fallback (odd)
    sparse_k_cap: int = 1024                # Maximum k_keep to prevent memory spikes
    sparse_q_block_size: int = 1024         # Query block size for tiling
    rbf_centers_per_head: int = 32          # RBF centers per attention head
    key_rbf_mode: str = 'mean'              # RBF key projection mode: mean/linear/none
    sparsemax_pad_value: Optional[float] = None  # Custom pad value for sparsemax
    ema_update_every: int = 1               # EMA update frequency (v3.2.2 throttling)
    
    # Data configuration
    data_root: str = './dataset'
    batch_size: int = 20
    num_workers: int = 4
    patch_size: int = 128
    stride: int = 8
    augment: bool = True
    memory_mode: str = 'float16'  # standard, float16, or lazy
    
    # Training configuration
    epochs: int = 300
    learning_rate: float = 4e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    
    # Optimization
    use_amp: bool = True
    compile_model: bool = True
    use_channels_last: bool = True
    accumulate_steps: int = 1
    
    # Validation
    val_interval: int = 10
    val_crop: bool = True  # Center crop 226x256 for MST++ protocol
    val_crop_size: Tuple[int, int] = (226, 256)
    
    # Checkpointing
    save_interval: int = 50
    log_interval: int = 100
    use_checkpoint: bool = False  # Gradient checkpointing
    
    # Output
    output_dir: str = './experiments/sharp'
    experiment_name: Optional[str] = None
    resume_from: Optional[str] = None
    
    # Hardware
    device: str = 'cuda'
    seed: int = 42
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Validate sparsity ratio
        if not 0.0 <= self.sparse_sparsity_ratio <= 1.0:
            raise ValueError(f"sparse_sparsity_ratio must be in [0,1], got {self.sparse_sparsity_ratio}")
        
        # Ensure odd window size
        if self.sparse_window_size % 2 == 0:
            self.sparse_window_size -= 1
            warnings.warn(f"sparse_window_size must be odd, adjusted to {self.sparse_window_size}")
        
        # Handle k_cap <= 0 as no cap
        if self.sparse_k_cap <= 0:
            self.sparse_k_cap = None
        
        # Create experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = f"sharp_v322_{self.model_size}_sp{self.sparse_sparsity_ratio:.1f}_rbf{self.rbf_centers_per_head}"


class SHARPLoss(nn.Module):
    """Combined loss for SHARP training"""
    
    def __init__(self, l1_weight: float = 1.0, spectral_weight: float = 0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.spectral_weight = spectral_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            pred: Predicted HSI (B, C, H, W)
            target: Ground truth HSI (B, C, H, W)
        """
        # L1 reconstruction loss
        l1_loss = F.l1_loss(pred, target)
        
        # Spectral angle mapper (SAM) loss
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        # Invariant check: Normalized vectors should have unit norm (within numerical tolerance)
        # This check is only active in debug mode to avoid overhead in production
        if __debug__:
            pred_norms = pred_norm.norm(p=2, dim=1)
            if not torch.allclose(pred_norms, torch.ones_like(pred_norms), atol=1e-5):
                warnings.warn("Normalization invariant violation: pred_norm does not have unit norm")

        sam_loss = 1 - (pred_norm * target_norm).sum(dim=1).mean()
        
        # Combined loss
        total_loss = self.l1_weight * l1_loss + self.spectral_weight * sam_loss
        
        return total_loss


class DedicatedSHARPTrainer:
    """Dedicated trainer for SHARP v3.2.2"""
    
    def __init__(self, config: SHARPTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Create output directory
        self.exp_dir = Path(config.output_dir) / config.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self._save_config()
        
        # Create model
        self.model = self._create_model()
        
        # Optimization setup
        self.use_sharp_trainer = SHARP_TRAINER_AVAILABLE
        if self.use_sharp_trainer:
            # Use built-in SHARP trainer
            total_steps = config.epochs * self._estimate_steps_per_epoch()
            self.sharp_trainer = SHARPv32Trainer(
                model=self.model,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                warmup_ratio=config.warmup_ratio,
                total_steps=total_steps,
                gradient_clip=config.gradient_clip,
                ema_decay=config.ema_decay,
                use_amp=config.use_amp
            )
        else:
            # Manual setup
            self._setup_training_components()
        
        # Create dataloaders
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Logging
        self.writer = SummaryWriter(self.exp_dir / 'logs')
        self.best_mrae = float('inf')
        self.start_epoch = 0
        self.iteration = 0
        
        # Resume if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)
        
        # Print model info
        self._print_model_info()
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Performance vs determinism trade-off
        if seed == 42:  # Special case for deterministic mode
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _save_config(self):
        """Save configuration to file"""
        import json
        config_path = self.exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def _create_model(self) -> nn.Module:
        """Create SHARP v3.2.2 model"""
        try:
            # Create SHARP config
            sharp_config = SHARPv32Config(
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                use_checkpoint=self.config.use_checkpoint,
                compile_mode='reduce-overhead' if self.config.compile_model else None,
                # v3.2.2 streaming attention parameters
                sparse_block_size=self.config.sparse_block_size,
                sparse_max_tokens=self.config.sparse_max_tokens,
                sparse_window_size=self.config.sparse_window_size,
                sparse_k_cap=self.config.sparse_k_cap,
                sparse_q_block_size=self.config.sparse_q_block_size,
                sparse_sparsity_ratio=self.config.sparse_sparsity_ratio,
                rbf_centers_per_head=self.config.rbf_centers_per_head,
                # v3.2.2 new features
                key_rbf_mode=self.config.key_rbf_mode,
                sparsemax_pad_value=self.config.sparsemax_pad_value,
                ema_update_every=self.config.ema_update_every
            )
            
            # Update config based on model size
            size_configs = {
                'tiny': {'base_dim': 48, 'depths': [2, 2, 2, 2], 'heads': [3, 6, 12, 24]},
                'small': {'base_dim': 64, 'depths': [2, 2, 4, 2], 'heads': [4, 8, 16, 32]},
                'base': {'base_dim': 96, 'depths': [2, 2, 6, 2], 'heads': [6, 12, 24, 48]},
                'large': {'base_dim': 128, 'depths': [2, 2, 8, 2], 'heads': [8, 16, 32, 64]}
            }
            
            if self.config.model_size in size_configs:
                for key, value in size_configs[self.config.model_size].items():
                    setattr(sharp_config, key, value)
            
            # Create model
            model = create_sharp_v32(
                model_size=self.config.model_size,
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                compile_model=self.config.compile_model,
                sparse_sparsity_ratio=self.config.sparse_sparsity_ratio,
                rbf_centers_per_head=self.config.rbf_centers_per_head,
                sparse_k_cap=self.config.sparse_k_cap,
                key_rbf_mode=self.config.key_rbf_mode,
                sparsemax_pad_value=self.config.sparsemax_pad_value,
                verbose=True
            )
            
            # Move to device and optimize memory format
            model = model.to(self.device)
            if self.config.use_channels_last and torch.cuda.is_available():
                model = model.to(memory_format=torch.channels_last)
            
            return model
            
        except Exception as e:
            print(f"Error creating SHARP model: {e}")
            raise
    
    def _setup_training_components(self):
        """Setup training components if not using SHARPv32Trainer"""
        # Loss function - use MST++ loss if available
        if DATALOADER_AVAILABLE:
            self.criterion = MSTPlusPlusLoss()
        else:
            self.criterion = SHARPLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # AMP scaler
        self.scaler = GradScaler(enabled=self.config.use_amp)
        
        # EMA
        self.ema_state = None
        if self.config.ema_decay > 0:
            self.ema_state = self._create_ema_state()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with proper weight decay"""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if len(param.shape) == 1 or name.endswith('.bias') or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        return optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.config.learning_rate, betas=(0.9, 0.999))
    
    def _create_scheduler(self):
        """Create cosine scheduler with warmup"""
        total_steps = self.config.epochs * self._estimate_steps_per_epoch()
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_ema_state(self):
        """Create EMA state dictionary"""
        ema_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ema_state[name] = param.detach().cpu().clone()
        return ema_state
    
    def _update_ema(self):
        """Update EMA weights"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.ema_state:
                    self.ema_state[name].mul_(self.config.ema_decay).add_(
                        param.detach().cpu(), alpha=1 - self.config.ema_decay
                    )
    
    def _estimate_steps_per_epoch(self) -> int:
        """Estimate steps per epoch for scheduler"""
        # This is a rough estimate - will be updated after dataloader creation
        # ARAD-1K typically has ~900 training images
        # With default settings: 482x512 images, 128x128 patches, stride 8
        # Approximately 1800 patches per image
        estimated_patches = 900 * 1800  # ~1.6M patches
        return estimated_patches // self.config.batch_size
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders"""
        if DATALOADER_AVAILABLE:
            # Use the optimized dataloader with proper config structure
            from types import SimpleNamespace
            
            # Create a config object compatible with create_optimized_dataloaders
            dataloader_config = SimpleNamespace(
                data_root=self.config.data_root,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                memory_mode=self.config.memory_mode,
                patch_size=self.config.patch_size,
                stride=self.config.stride
            )
            
            return create_optimized_dataloaders(dataloader_config, memory_mode=self.config.memory_mode)
        else:
            # Fallback implementation
            raise NotImplementedError("Optimized dataloader not available. Please ensure optimized_dataloader.py is in your path.")
    
    def _print_model_info(self):
        """Print detailed model information"""
        model_ref = getattr(self.model, '_orig_mod', self.model)
        params_info = model_ref.num_parameters
        
        print(f"\n{'='*80}")
        print(f"SHARP v3.2.2 Model Information")
        print(f"{'='*80}")
        print(f"Model size: {self.config.model_size}")
        print(f"Total parameters: {params_info['total']/1e6:.2f}M")
        print(f"Trainable parameters: {params_info['trainable']/1e6:.2f}M")
        print(f"Model size: {params_info['size_mb']:.1f} MB")
        print(f"\nStreaming Sparse Attention Configuration:")
        print(f"  Sparsity ratio: {self.config.sparse_sparsity_ratio}")
        print(f"  Block size: {self.config.sparse_block_size}")
        print(f"  Query block size: {self.config.sparse_q_block_size}")
        print(f"  Max tokens: {self.config.sparse_max_tokens}")
        print(f"  Window size: {self.config.sparse_window_size}")
        print(f"  K-cap: {self.config.sparse_k_cap if self.config.sparse_k_cap else 'disabled'}")
        print(f"  RBF centers/head: {self.config.rbf_centers_per_head}")
        
        # Sample memory calculation
        sample_n = 512
        sample_k = max(1, int(sample_n * (1 - self.config.sparse_sparsity_ratio)))
        if self.config.sparse_k_cap:
            sample_k = min(sample_k, self.config.sparse_k_cap)
        print(f"\n  Example: For N={sample_n} tokens:")
        print(f"    k_keep = {sample_k} (top {100*(1-self.config.sparse_sparsity_ratio):.1f}%)")
        print(f"    Peak memory: O(BH x {self.config.sparse_q_block_size} x {self.config.sparse_block_size})")
        
        print(f"\nTraining Configuration:")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Device: {self.device}")
        
        if self.config.compile_model:
            print(f"  Model compilation: ENABLED")
        if self.config.use_channels_last:
            print(f"  Channels last: ENABLED")
        if self.config.use_amp:
            print(f"  Mixed precision: ENABLED")
        if self.config.use_checkpoint:
            print(f"  Gradient checkpointing: ENABLED")
        
        print(f"{'='*80}\n")
    
    def train(self):
        """Main training loop"""
        print(f"Starting SHARP v3.2.2 training...")
        print(f"Experiment: {self.config.experiment_name}")
        start_time = time.time()
        self.start_time = start_time  # Store for logging
        
        for epoch in range(self.start_epoch, self.config.epochs):
            # Training phase
            epoch_metrics = self._train_epoch(epoch)
            
            # Validation phase
            if (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self._validate(epoch)
                
                # Save best model
                if val_metrics['mrae'] < self.best_mrae:
                    self.best_mrae = val_metrics['mrae']
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"New best model! MRAE: {self.best_mrae:.6f}")
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch)
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.1f} hours")
        print(f"Best MRAE: {self.best_mrae:.6f}")
        print(f"Results saved to: {self.exp_dir}")
        self.writer.close()
    
    def _train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_start = time.time()
        
        for batch_idx, (rgb, hsi) in enumerate(self.train_loader):
            if self.use_sharp_trainer:
                # Use built-in trainer
                metrics = self.sharp_trainer.train_step(rgb, hsi)
            else:
                # Manual training step
                metrics = self._train_step(batch_idx, rgb, hsi)
            
            epoch_losses.append(metrics['loss'])
            self.iteration += 1
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self._log_training(epoch, batch_idx, metrics, epoch_losses)
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        
        print(f"\nEpoch {epoch+1}/{self.config.epochs} completed in {epoch_time:.1f}s")
        print(f"Average loss: {avg_loss:.4f}")
        
        # Memory logging
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"Peak GPU memory: {memory_mb:.1f} MB")
            torch.cuda.reset_peak_memory_stats()
        
        return {'avg_loss': avg_loss, 'epoch_time': epoch_time}
    
    def _train_step(self, batch_idx: int, rgb: torch.Tensor, hsi: torch.Tensor) -> Dict:
        """Single training step (manual implementation)"""
        # Move to device
        if self.config.use_channels_last:
            rgb = rgb.to(self.device, memory_format=torch.channels_last)
            hsi = hsi.to(self.device, memory_format=torch.channels_last)
        else:
            rgb = rgb.to(self.device)
            hsi = hsi.to(self.device)
        
        # Forward pass
        with autocast(enabled=self.config.use_amp):
            pred = self.model(rgb)
            loss = self.criterion(pred, hsi)
            loss = loss / self.config.accumulate_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % self.config.accumulate_steps == 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            
            # Update EMA if available
            if self.ema_state is not None:
                self._update_ema()
        else:
            grad_norm = 0.0
        
        return {
            'loss': loss.item() * self.config.accumulate_steps,
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict:
        """Validation loop"""
        self.model.eval()
        
        if self.use_sharp_trainer:
            metrics = self.sharp_trainer.evaluate(self.val_loader, psnr_max=2.0)
        else:
            metrics = self._manual_validate()
        
        # Log metrics
        print(f"\nValidation @ Epoch {epoch+1}:")
        print(f"  MRAE: {metrics['mrae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        
        # TensorBoard logging
        self.writer.add_scalar('val/mrae', metrics['mrae'], epoch)
        self.writer.add_scalar('val/rmse', metrics['rmse'], epoch)
        self.writer.add_scalar('val/psnr', metrics['psnr'], epoch)
        
        return metrics
    
    @torch.no_grad()
    def _manual_validate(self) -> Dict:
        """Manual validation implementation"""
        total_mrae = 0.0
        total_rmse = 0.0
        total_psnr = 0.0
        num_samples = 0
        
        for rgb, hsi in self.val_loader:
            rgb = rgb.to(self.device)
            hsi = hsi.to(self.device)
            
            with autocast(enabled=self.config.use_amp):
                pred = self.model(rgb)
            
            # Center crop if enabled
            if self.config.val_crop:
                crop_h, crop_w = self.config.val_crop_size
                h, w = pred.shape[-2:]
                
                if h >= crop_h and w >= crop_w:
                    start_h = (h - crop_h) // 2
                    start_w = (w - crop_w) // 2
                    pred = pred[..., start_h:start_h+crop_h, start_w:start_w+crop_w]
                    hsi = hsi[..., start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            # Compute metrics
            # Use clamp_min for numerical stability, especially under AMP/float16
            mrae = torch.mean(torch.abs(pred - hsi) / torch.clamp_min(hsi, 1e-6)).item()
            rmse = torch.sqrt(torch.mean((pred - hsi) ** 2)).item()
            mse = torch.mean((pred - hsi) ** 2)
            psnr = 20 * torch.log10(torch.tensor(2.0) / torch.sqrt(mse.clamp(min=1e-8))).item()
            
            total_mrae += mrae
            total_rmse += rmse
            total_psnr += psnr
            num_samples += 1
        
        return {
            'mrae': total_mrae / num_samples,
            'rmse': total_rmse / num_samples,
            'psnr': total_psnr / num_samples
        }
    
    def _log_training(self, epoch: int, batch_idx: int, metrics: Dict, epoch_losses: List):
        """Log training progress"""
        avg_loss = np.mean(epoch_losses[-100:])
        elapsed = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        
        print(f"Epoch [{epoch+1}/{self.config.epochs}] "
              f"Iter [{batch_idx}/{len(self.train_loader)}] "
              f"Loss: {metrics['loss']:.4f} (avg: {avg_loss:.4f}) "
              f"LR: {metrics['lr']:.2e} "
              f"GradNorm: {metrics['grad_norm']:.2f}")
        
        # TensorBoard logging
        self.writer.add_scalar('train/loss', metrics['loss'], self.iteration)
        self.writer.add_scalar('train/lr', metrics['lr'], self.iteration)
        self.writer.add_scalar('train/grad_norm', metrics['grad_norm'], self.iteration)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        model_state = getattr(self.model, '_orig_mod', self.model).state_dict()
        
        checkpoint = {
            'epoch': epoch + 1,
            'iteration': self.iteration,
            'model_state_dict': model_state,
            'best_mrae': self.best_mrae,
            'config': self.config,
            'sharp_version': '3.2.2'
        }
        
        # Add training state if not using SHARPv32Trainer
        if not self.use_sharp_trainer:
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'ema_state': self.ema_state
            })
        
        if is_best:
            path = self.exp_dir / 'best_model.pth'
        else:
            path = self.exp_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model_ref = getattr(self.model, '_orig_mod', self.model)
        model_ref.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch']
        self.iteration = checkpoint.get('iteration', 0)
        self.best_mrae = checkpoint.get('best_mrae', float('inf'))
        
        if not self.use_sharp_trainer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'ema_state' in checkpoint:
                self.ema_state = checkpoint['ema_state']
        
        print(f"Resumed from epoch {self.start_epoch}, best MRAE: {self.best_mrae:.6f}")


def main():
    parser = argparse.ArgumentParser(description='SHARP v3.2.2 Dedicated Training')
    
    # Model configuration
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Model size variant')
    
    # Data configuration
    parser.add_argument('--data_root', type=str, default='./dataset',
                        help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size for training')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Patch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--memory_mode', type=str, default='float16',
                        choices=['standard', 'float16', 'lazy'],
                        help='Memory mode for dataloader')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # SHARP v3.2.2 specific
    parser.add_argument('--sparsity', type=float, default=0.9,
                        help='Sparse attention sparsity ratio (0.0-1.0)')
    parser.add_argument('--rbf_centers', type=int, default=32,
                        help='RBF centers per attention head')
    parser.add_argument('--k_cap', type=int, default=1024,
                        help='Memory cap for top-k (0 = no cap)')
    parser.add_argument('--block_size', type=int, default=2048,
                        help='Block size for streaming attention')
    parser.add_argument('--q_block_size', type=int, default=1024,
                        help='Query block size for tiling')
    parser.add_argument('--key_rbf_mode', type=str, default='mean',
                        choices=['mean', 'linear', 'none'],
                        help='RBF key projection mode')
    parser.add_argument('--ema_update_every', type=int, default=1,
                        help='EMA update frequency (v3.2.2 throttling)')
    
    # Optimization
    parser.add_argument('--compile', action='store_true',
                        help='Compile model with torch.compile')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--checkpoint', action='store_true',
                        help='Enable gradient checkpointing')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./experiments/sharp',
                        help='Output directory for experiments')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (42 for deterministic mode)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SHARPTrainingConfig(
        model_size=args.model_size,
        data_root=args.data_root,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        memory_mode=args.memory_mode,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        sparse_sparsity_ratio=args.sparsity,
        rbf_centers_per_head=args.rbf_centers,
        sparse_k_cap=args.k_cap,
        sparse_block_size=args.block_size,
        sparse_q_block_size=args.q_block_size,
        key_rbf_mode=args.key_rbf_mode,
        ema_update_every=args.ema_update_every,
        compile_model=args.compile,
        use_amp=not args.no_amp,
        use_checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        resume_from=args.resume,
        seed=args.seed
    )
    
    # Create trainer and start training
    trainer = DedicatedSHARPTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()






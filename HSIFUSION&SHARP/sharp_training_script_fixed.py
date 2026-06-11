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
import contextlib
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[override]
        """No-op fallback so training/imports still work without tensorboard installed."""

        def __init__(self, *args, **kwargs) -> None:
            warnings.warn(
                "tensorboard is not installed; SummaryWriter logging is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )

        def add_scalar(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None

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


def _torch_load_compat(path: str, map_location: torch.device):
    """Load checkpoints across PyTorch versions (2.6+ defaults to weights_only=True)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying SHARP module through DDP and torch.compile wrappers."""
    if isinstance(model, DDP):
        model = model.module
    return getattr(model, "_orig_mod", model)


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class _NoOpWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def should_optimizer_step(batch_idx: int, total_batches: int, accumulate_steps: int) -> bool:
    """Return whether an optimizer update should be executed for this batch index."""
    if accumulate_steps <= 0:
        raise ValueError("accumulate_steps must be > 0")
    return ((batch_idx + 1) % accumulate_steps == 0) or ((batch_idx + 1) == total_batches)


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
    sparse_max_tokens: int = 8192           # Legacy hard cap for exact all-key scans
    sparse_window_size: int = 49            # Number of 2D local candidates
    sparse_k_cap: int = 1024                # Maximum k_keep to prevent memory spikes
    sparse_q_block_size: int = 1024         # Query block size for tiling
    sparse_exact_topk_max_tokens: int = 1024 # Exact all-key scan threshold
    sparse_landmark_tokens: int = 256        # Pooled global candidates above threshold
    max_global_tokens: Optional[int] = 1024 # Bound dense global/cross-attention context
    rbf_centers_per_head: int = 32          # RBF centers per attention head
    key_rbf_mode: str = 'linear'            # Preserve query-dependent key rankings
    sparsemax_pad_value: Optional[float] = None  # Custom pad value for sparsemax
    output_activation: str = 'sigmoid'      # Output param for [0,1] reflectance (sigmoid|tanh|relu|softplus|none)
    ema_update_every: int = 1               # EMA update frequency (v3.2.2 throttling)
    # Training objective. 'mrae' matches the MST++/ARAD-1K selection+eval metric and the
    # sibling HSIFusion baseline (fair benchmark comparability). 'l1_curvature' uses the
    # model's compute_loss (L1 + 0.1*spectral-curvature) and is kept for ablation only.
    loss_type: str = 'mrae'                  # mrae | l1_curvature
    
    # Data configuration
    data_root: str = './dataset'
    batch_size: int = 20
    num_workers: int = 4
    patch_size: int = 128
    stride: int = 8
    augment: bool = True
    memory_mode: str = 'float16'  # standard, float16, or lazy
    cache_size: int = 4
    
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
    psnr_data_range: float = 1.0
    early_stopping_patience: int = 35
    early_stopping_min_delta: float = 1e-5
    
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
    distributed: bool = False
    ddp_find_unused_parameters: bool = False
    local_rank: int = 0
    rank: int = field(default=0, init=False)
    world_size: int = field(default=1, init=False)
    min_mrae_denom: float = 1e-6
    max_consecutive_nonfinite: int = 8
    skip_oom_batches: bool = True
    
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

        if self.accumulate_steps <= 0:
            raise ValueError("accumulate_steps must be > 0")
        if self.psnr_data_range <= 0:
            raise ValueError("psnr_data_range must be > 0")
        if self.max_global_tokens is not None and self.max_global_tokens <= 0:
            self.max_global_tokens = None
        if self.sparse_exact_topk_max_tokens <= 0:
            raise ValueError("sparse_exact_topk_max_tokens must be > 0")
        if self.sparse_landmark_tokens <= 0:
            raise ValueError("sparse_landmark_tokens must be > 0")
        if self.cache_size < 0:
            raise ValueError("cache_size must be >= 0")
        if self.ema_update_every <= 0:
            raise ValueError("ema_update_every must be > 0")
        if self.key_rbf_mode not in {'mean', 'linear', 'none'}:
            raise ValueError("key_rbf_mode must be one of mean/linear/none")
        if self.loss_type not in {'mrae', 'l1_curvature'}:
            raise ValueError("loss_type must be one of mrae/l1_curvature")
        if self.output_activation not in {'sigmoid', 'tanh', 'relu', 'softplus', 'none'}:
            raise ValueError("output_activation must be one of sigmoid/tanh/relu/softplus/none")


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
        self._setup_distributed()
        self.device = self._resolve_device()
        self.is_main_process = self.rank == 0
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Create output directory
        self.exp_dir = Path(config.output_dir) / config.experiment_name
        if self.is_main_process:
            self.exp_dir.mkdir(parents=True, exist_ok=True)
        if self.distributed:
            dist.barrier()
        
        # Save configuration
        if self.is_main_process:
            self._save_config()
        
        # Create model
        self.model = self._create_model()

        # Build the optimization objective once so both trainer paths agree with the
        # MRAE selection/eval metric (None -> built-in SHARPv32Trainer uses model.compute_loss).
        self._criterion = self._build_criterion()

        # Create dataloaders before trainers/schedulers so step counts reflect the real dataset.
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Optimization setup
        self.use_sharp_trainer = SHARP_TRAINER_AVAILABLE and config.accumulate_steps == 1
        if self.use_sharp_trainer:
            # Use built-in SHARP trainer
            total_steps = config.epochs * max(1, math.ceil(len(self.train_loader) / config.accumulate_steps))
            self.sharp_trainer = SHARPv32Trainer(
                model=self.model,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                warmup_ratio=config.warmup_ratio,
                total_steps=total_steps,
                gradient_clip=config.gradient_clip,
                ema_decay=config.ema_decay,
                use_amp=config.use_amp,
                ema_update_every=config.ema_update_every,
                criterion=self._criterion,
            )
        else:
            if SHARP_TRAINER_AVAILABLE and config.accumulate_steps > 1:
                warnings.warn(
                    "accumulate_steps > 1 is not supported by SHARPv32Trainer; "
                    "falling back to the manual training loop to preserve the requested behavior."
                )
            # Manual setup
            self._setup_training_components()
        
        # Logging
        self.writer = (
            SummaryWriter(self.exp_dir / 'logs')
            if self.is_main_process
            else _NoOpWriter()
        )
        self.best_mrae = float('inf')
        self.start_epoch = 0
        self.iteration = 0
        self.bad_val_epochs = 0
        self.consecutive_nonfinite = 0
        self.skipped_oom_batches = 0
        self.skipped_nonfinite_batches = 0
        
        # Resume if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)
        
        # Print model info
        self._print_model_info()

    def _setup_distributed(self) -> None:
        """Initialize torchrun-provided distributed state."""
        env_world_size = _env_int("WORLD_SIZE", 1)
        self.distributed = bool(self.config.distributed or env_world_size > 1)
        self.local_rank = _env_int("LOCAL_RANK", self.config.local_rank)
        self.rank = _env_int("RANK", 0)
        self.world_size = env_world_size if self.distributed else 1

        if self.distributed:
            if not dist.is_available():
                raise RuntimeError(
                    "Distributed training requested but torch.distributed is unavailable."
                )
            requested_type = torch.device(self.config.device).type
            use_cuda = requested_type == "cuda" and torch.cuda.is_available()
            backend = "nccl" if use_cuda else "gloo"
            if use_cuda:
                torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend=backend, init_method="env://")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.config.distributed = self.distributed
        self.config.local_rank = self.local_rank
        self.config.rank = self.rank
        self.config.world_size = self.world_size

    def _resolve_device(self) -> torch.device:
        requested = torch.device(self.config.device)
        if requested.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            if self.distributed:
                return torch.device("cuda", self.local_rank)
            return requested
        return requested

    def _print(self, *args, **kwargs) -> None:
        if getattr(self, "is_main_process", True):
            print(*args, **kwargs)

    def _all_ranks_true(self, value: bool) -> bool:
        if not getattr(self, "distributed", False):
            return value
        flag = torch.tensor(
            int(value), device=self.device, dtype=torch.int32
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())
    
    def _is_oom_error(self, exc: BaseException) -> bool:
        return "out of memory" in str(exc).lower()

    def _has_finite_gradients(self) -> bool:
        for param in self.model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return False
        return True

    def _safe_mrae(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp_min(torch.abs(target), self.config.min_mrae_denom)
        return torch.mean(torch.abs(pred - target) / denom)
    
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
            # Create model
            model = create_sharp_v32(
                model_size=self.config.model_size,
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                compile_model=self.config.compile_model,
                use_checkpoint=self.config.use_checkpoint,
                sparse_sparsity_ratio=self.config.sparse_sparsity_ratio,
                rbf_centers_per_head=self.config.rbf_centers_per_head,
                sparse_k_cap=self.config.sparse_k_cap,
                sparse_block_size=self.config.sparse_block_size,
                sparse_q_block_size=self.config.sparse_q_block_size,
                sparse_exact_topk_max_tokens=self.config.sparse_exact_topk_max_tokens,
                sparse_landmark_tokens=self.config.sparse_landmark_tokens,
                max_global_tokens=self.config.max_global_tokens,
                sparse_window_size=self.config.sparse_window_size,
                sparse_max_tokens=self.config.sparse_max_tokens,
                key_rbf_mode=self.config.key_rbf_mode,
                sparsemax_pad_value=self.config.sparsemax_pad_value,
                output_activation=self.config.output_activation,
                ema_update_every=self.config.ema_update_every,
                verbose=getattr(self, "is_main_process", True),
            )
            
            # Move to device and optimize memory format
            model = model.to(self.device)
            if self.config.use_channels_last and torch.cuda.is_available():
                model = model.to(memory_format=torch.channels_last)

            if getattr(self, "distributed", False):
                ddp_kwargs = {
                    "find_unused_parameters": self.config.ddp_find_unused_parameters,
                }
                if self.device.type == "cuda":
                    ddp_kwargs.update({
                        "device_ids": [self.local_rank],
                        "output_device": self.local_rank,
                    })
                model = DDP(model, **ddp_kwargs)
            
            return model
            
        except Exception as e:
            self._print(f"Error creating SHARP model: {e}")
            raise
    
    def _build_criterion(self):
        """Build the training objective shared by both trainer paths.

        Returns MSTPlusPlusLoss (MRAE) for loss_type='mrae' so the optimization target
        matches the MRAE selection/eval metric and the sibling HSIFusion baseline. Returns
        None for loss_type='l1_curvature' so callers fall back to the model's compute_loss
        (L1 + 0.1*spectral-curvature), retained for ablation.
        """
        if self.config.loss_type == 'mrae':
            if not DATALOADER_AVAILABLE:
                raise RuntimeError("loss_type='mrae' requires optimized_dataloader.MSTPlusPlusLoss")
            return MSTPlusPlusLoss(eps=self.config.min_mrae_denom)
        return None  # 'l1_curvature' -> use model.compute_loss

    def _setup_training_components(self):
        """Setup training components if not using SHARPv32Trainer"""
        # Keep the optimization objective aligned with the built-in trainer path.
        if self._criterion is not None:
            self.criterion = self._criterion
        elif hasattr(unwrap_model(self.model), "compute_loss"):
            self.criterion = unwrap_model(self.model).compute_loss
        elif DATALOADER_AVAILABLE:
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
        total_steps = self.config.epochs * max(1, math.ceil(len(self.train_loader) / self.config.accumulate_steps))
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_ema_state(self):
        """Create EMA state dictionary"""
        ema_state = {}
        for name, param in unwrap_model(self.model).named_parameters():
            if param.requires_grad:
                ema_state[name] = param.detach().cpu().clone()
        return ema_state
    
    def _update_ema(self):
        """Update EMA weights"""
        with torch.no_grad():
            for name, param in unwrap_model(self.model).named_parameters():
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
                augment=self.config.augment,
                memory_mode=self.config.memory_mode,
                patch_size=self.config.patch_size,
                stride=self.config.stride,
                cache_size=self.config.cache_size,
                seed=self.config.seed,
                val_num_workers=min(2, self.config.num_workers),
                distributed=self.distributed,
                rank=self.rank,
                world_size=self.world_size,
            )
            
            return create_optimized_dataloaders(dataloader_config, memory_mode=self.config.memory_mode)
        else:
            # Fallback implementation
            raise NotImplementedError("Optimized dataloader not available. Please ensure optimized_dataloader.py is in your path.")
    
    def _print_model_info(self):
        """Print detailed model information"""
        if not self.is_main_process:
            return
        model_ref = unwrap_model(self.model)
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
        print(f"  Exact hard cap: {self.config.sparse_max_tokens}")
        print(f"  Exact top-k threshold: {self.config.sparse_exact_topk_max_tokens}")
        print(f"  Global landmarks: {self.config.sparse_landmark_tokens}")
        print(f"  Global context cap: {self.config.max_global_tokens or 'disabled'}")
        print(f"  Window size: {self.config.sparse_window_size}")
        print(f"  K-cap: {self.config.sparse_k_cap if self.config.sparse_k_cap else 'disabled'}")
        print(f"  RBF centers/head: {self.config.rbf_centers_per_head}")
        
        # Sample memory calculation
        sample_n = 512
        sample_k = max(1, int(sample_n * (1 - self.config.sparse_sparsity_ratio)))
        if self.config.sparse_k_cap:
            sample_k = min(sample_k, self.config.sparse_k_cap)
        print(f"\n  Exact-path example: For N={sample_n} tokens:")
        print(f"    k_keep = {sample_k} (top {100*(1-self.config.sparse_sparsity_ratio):.1f}%)")
        print(f"    Peak memory: O(BH x {self.config.sparse_q_block_size} x {self.config.sparse_block_size})")
        print(
            "  Hybrid-path compute: "
            f"O(N x ({self.config.sparse_window_size} + "
            f"{self.config.sparse_landmark_tokens}))"
        )
        
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
        self._print("Starting SHARP v3.2.2 training...")
        self._print(f"Experiment: {self.config.experiment_name}")
        if self.distributed:
            self._print(
                f"DDP: {self.world_size} processes "
                f"(backend={dist.get_backend()})"
            )
        start_time = time.time()
        self.start_time = start_time  # Store for logging
        
        for epoch in range(self.start_epoch, self.config.epochs):
            # Training phase
            epoch_metrics = self._train_epoch(epoch)
            
            # Validation phase
            if (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self._validate(epoch)
                
                # Save best model
                if val_metrics['mrae'] < (self.best_mrae - self.config.early_stopping_min_delta):
                    self.best_mrae = val_metrics['mrae']
                    self.bad_val_epochs = 0
                    self._save_checkpoint(epoch, is_best=True)
                    self._print(f"New best model! MRAE: {self.best_mrae:.6f}")
                else:
                    self.bad_val_epochs += 1

                if (
                    self.config.early_stopping_patience > 0
                    and self.bad_val_epochs >= self.config.early_stopping_patience
                ):
                    self._print(
                        f"Early stopping triggered after {self.bad_val_epochs} validation checks "
                        f"without improvement > {self.config.early_stopping_min_delta}."
                    )
                    break
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch)
        
        # Training completed
        total_time = time.time() - start_time
        self._print(f"\nTraining completed in {total_time/3600:.1f} hours")
        self._print(f"Best MRAE: {self.best_mrae:.6f}")
        self._print(
            f"Skipped batches - OOM: {self.skipped_oom_batches}, "
            f"non-finite: {self.skipped_nonfinite_batches}"
        )
        self._print(f"Results saved to: {self.exp_dir}")
        self.writer.close()
        if self.distributed:
            dist.barrier()
    
    def _train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_start = time.time()
        total_batches = len(self.train_loader)
        if isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, (rgb, hsi) in enumerate(self.train_loader):
            try:
                if self.use_sharp_trainer:
                    # Use built-in trainer
                    metrics = self.sharp_trainer.train_step(rgb, hsi)
                else:
                    # Manual training step
                    metrics = self._train_step(
                        batch_idx=batch_idx,
                        rgb=rgb,
                        hsi=hsi,
                        is_last_batch=(batch_idx + 1) == total_batches,
                    )
            except RuntimeError as exc:
                if (
                    self.config.skip_oom_batches
                    and not self.distributed
                    and self._is_oom_error(exc)
                ):
                    self.skipped_oom_batches += 1
                    if not self.use_sharp_trainer:
                        self.optimizer.zero_grad(set_to_none=True)
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    warnings.warn(
                        f"Skipped OOM batch at epoch={epoch+1}, batch={batch_idx+1}. "
                        f"Total skipped OOM batches: {self.skipped_oom_batches}"
                    )
                    continue
                raise

            if metrics.get('skipped', False):
                self.skipped_nonfinite_batches += 1
                continue
            
            epoch_losses.append(metrics['loss'])
            self.iteration += 1
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self._log_training(epoch, batch_idx, metrics, epoch_losses)
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start
        loss_sum = float(np.sum(epoch_losses)) if epoch_losses else 0.0
        loss_count = len(epoch_losses)
        if self.distributed:
            packed = torch.tensor(
                [loss_sum, float(loss_count)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            loss_sum, loss_count = packed.tolist()
        avg_loss = loss_sum / loss_count if loss_count else float('nan')
        
        self._print(
            f"\nEpoch {epoch+1}/{self.config.epochs} completed in {epoch_time:.1f}s"
        )
        self._print(f"Average loss: {avg_loss:.4f}")
        
        # Memory logging
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            self._print(f"Peak GPU memory: {memory_mb:.1f} MB")
            torch.cuda.reset_peak_memory_stats()
        
        return {'avg_loss': avg_loss, 'epoch_time': epoch_time}
    
    def _train_step(self, batch_idx: int, rgb: torch.Tensor, hsi: torch.Tensor, is_last_batch: bool = False) -> Dict:
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
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = self.criterion(pred, hsi)

        if not self._all_ranks_true(bool(torch.isfinite(loss).item())):
            self.consecutive_nonfinite += 1
            warnings.warn(
                f"Non-finite loss at batch={batch_idx+1}; skipping update "
                f"({self.consecutive_nonfinite} consecutive)."
            )
            self.optimizer.zero_grad(set_to_none=True)
            if self.consecutive_nonfinite >= self.config.max_consecutive_nonfinite:
                raise RuntimeError("Exceeded maximum consecutive non-finite losses; aborting to avoid divergence.")
            return {
                'loss': float('nan'),
                'grad_norm': 0.0,
                'lr': self.optimizer.param_groups[0]['lr'],
                'skipped': True,
            }

        self.consecutive_nonfinite = 0
        loss = loss / self.config.accumulate_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient accumulation
        if should_optimizer_step(batch_idx, len(self.train_loader), self.config.accumulate_steps) or is_last_batch:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
            if not self._all_ranks_true(self._has_finite_gradients()):
                self.consecutive_nonfinite += 1
                warnings.warn(f"Non-finite gradients at batch={batch_idx+1}; skipping optimizer step.")
                self.optimizer.zero_grad(set_to_none=True)
                if self.consecutive_nonfinite >= self.config.max_consecutive_nonfinite:
                    raise RuntimeError("Exceeded maximum consecutive non-finite gradients; aborting to avoid divergence.")
                return {
                    'loss': float('nan'),
                    'grad_norm': 0.0,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'skipped': True,
                }
            
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
            'lr': self.optimizer.param_groups[0]['lr'],
            'skipped': False,
        }
    
    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict:
        """Validation loop"""
        self.model.eval()
        
        if self.use_sharp_trainer:
            crop_size = self.config.val_crop_size if self.config.val_crop else None
            metrics = self.sharp_trainer.evaluate(
                self.val_loader,
                psnr_max=self.config.psnr_data_range,
                crop_size=crop_size,
                use_ema=self.config.ema_decay > 0,
            )
        else:
            metrics = self._manual_validate()
        
        # Log metrics
        self._print(f"\nValidation @ Epoch {epoch+1}:")
        self._print(f"  MRAE: {metrics['mrae']:.6f}")
        self._print(f"  RMSE: {metrics['rmse']:.6f}")
        self._print(f"  PSNR: {metrics['psnr']:.2f} dB")
        
        # TensorBoard logging
        self.writer.add_scalar('val/mrae', metrics['mrae'], epoch)
        self.writer.add_scalar('val/rmse', metrics['rmse'], epoch)
        self.writer.add_scalar('val/psnr', metrics['psnr'], epoch)
        
        return metrics
    
    @contextlib.contextmanager
    def _ema_weights_applied(self):
        """Temporarily load EMA params into the live model, restoring originals on exit.

        Only swaps requires_grad parameters (matching _create_ema_state); buffers are left
        as-is. Restoration is guaranteed via finally so a raised exception cannot leave the
        model in the EMA state.
        """
        model_ref = unwrap_model(self.model)
        if not getattr(self, 'ema_state', None):
            yield
            return
        backup = {}
        try:
            with torch.no_grad():
                for name, param in model_ref.named_parameters():
                    if name in self.ema_state:
                        backup[name] = param.detach().clone()
                        param.copy_(self.ema_state[name].to(device=param.device, dtype=param.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, param in model_ref.named_parameters():
                    if name in backup:
                        param.copy_(backup[name])

    @torch.no_grad()
    def _manual_validate(self) -> Dict:
        """Manual validation implementation (evaluates EMA weights when available)."""
        total_mrae = 0.0
        total_rmse = 0.0
        total_psnr = 0.0
        num_samples = 0

        with self._ema_weights_applied():
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
            mrae = self._safe_mrae(pred, hsi).item()
            rmse = torch.sqrt(torch.mean((pred - hsi) ** 2)).item()
            mse = torch.mean((pred - hsi) ** 2)
            psnr = 20 * torch.log10(
                torch.tensor(self.config.psnr_data_range, device=mse.device, dtype=mse.dtype)
                / torch.sqrt(mse.clamp(min=1e-8))
            ).item()
            
            total_mrae += mrae
            total_rmse += rmse
            total_psnr += psnr
            num_samples += 1
        
        if num_samples == 0:
            totals = torch.zeros(4, device=self.device, dtype=torch.float64)
        else:
            totals = torch.tensor(
                [total_mrae, total_rmse, total_psnr, float(num_samples)],
                device=self.device,
                dtype=torch.float64,
            )
        if getattr(self, "distributed", False):
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_mrae, total_rmse, total_psnr, num_samples = totals.tolist()
        if num_samples == 0:
            return {'mrae': 0.0, 'rmse': 0.0, 'psnr': 0.0}

        return {
            'mrae': total_mrae / num_samples,
            'rmse': total_rmse / num_samples,
            'psnr': total_psnr / num_samples
        }
    
    def _log_training(self, epoch: int, batch_idx: int, metrics: Dict, epoch_losses: List):
        """Log training progress"""
        avg_loss = np.mean(epoch_losses[-100:])
        elapsed = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        
        self._print(
            f"Epoch [{epoch+1}/{self.config.epochs}] "
            f"Iter [{batch_idx}/{len(self.train_loader)}] "
            f"Loss: {metrics['loss']:.4f} (avg: {avg_loss:.4f}) "
            f"LR: {metrics['lr']:.2e} "
            f"GradNorm: {metrics['grad_norm']:.2f}"
        )
        
        # TensorBoard logging
        self.writer.add_scalar('train/loss', metrics['loss'], self.iteration)
        self.writer.add_scalar('train/lr', metrics['lr'], self.iteration)
        self.writer.add_scalar('train/grad_norm', metrics['grad_norm'], self.iteration)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        if not self.is_main_process:
            if self.distributed:
                dist.barrier()
            return
        model_state = unwrap_model(self.model).state_dict()
        
        checkpoint = {
            'epoch': epoch + 1,
            'iteration': self.iteration,
            'model_state_dict': model_state,
            'best_mrae': self.best_mrae,
            'bad_val_epochs': self.bad_val_epochs,
            'skipped_oom_batches': self.skipped_oom_batches,
            'skipped_nonfinite_batches': self.skipped_nonfinite_batches,
            'config': self.config,
            'sharp_version': '3.2.2'
        }
        
        if self.use_sharp_trainer:
            checkpoint['trainer_state_dict'] = self.sharp_trainer.state_dict()
            if self.sharp_trainer.ema_state is not None:
                checkpoint['ema_model_state_dict'] = self.sharp_trainer.get_ema_model(
                    device='cpu', eval_mode=True
                ).state_dict()
        else:
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'ema_state': self.ema_state
            })
            # Persist an EMA-applied weight set so inference loads the weights that were
            # actually validated/selected (mirrors the built-in trainer path).
            if getattr(self, 'ema_state', None):
                ema_model_state = dict(model_state)
                for name, ema_val in self.ema_state.items():
                    if name in ema_model_state:
                        ema_model_state[name] = ema_val.to(
                            device='cpu', dtype=ema_model_state[name].dtype
                        )
                checkpoint['ema_model_state_dict'] = ema_model_state
        
        if is_best:
            path = self.exp_dir / 'best_model.pth'
        else:
            path = self.exp_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
        self._print(f"Saved checkpoint to {path}")
        if self.distributed:
            dist.barrier()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        self._print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = _torch_load_compat(checkpoint_path, self.device)
        
        # Load model state
        model_ref = unwrap_model(self.model)
        model_ref.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch']
        self.iteration = checkpoint.get('iteration', 0)
        self.best_mrae = checkpoint.get('best_mrae', float('inf'))
        self.bad_val_epochs = checkpoint.get('bad_val_epochs', 0)
        self.skipped_oom_batches = checkpoint.get('skipped_oom_batches', 0)
        self.skipped_nonfinite_batches = checkpoint.get('skipped_nonfinite_batches', 0)
        
        if self.use_sharp_trainer:
            trainer_state = checkpoint.get('trainer_state_dict')
            if trainer_state is not None:
                self.sharp_trainer.load_state_dict(trainer_state)
            else:
                warnings.warn(
                    "Legacy checkpoint has no built-in trainer state; optimizer, "
                    "scheduler, scaler, and EMA will restart."
                )
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'ema_state' in checkpoint:
                self.ema_state = checkpoint['ema_state']
        
        self._print(
            f"Resumed from epoch {self.start_epoch}, "
            f"best MRAE: {self.best_mrae:.6f}"
        )
        if self.distributed:
            dist.barrier()


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
    parser.add_argument('--stride', type=int, default=8,
                        help='Training patch stride')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--memory_mode', type=str, default='float16',
                        choices=['standard', 'float16', 'lazy'],
                        help='Memory mode for dataloader')
    parser.add_argument('--cache_size', type=int, default=4,
                        help='Per-worker full-image cache size in lazy mode')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable paired spatial augmentation')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Fraction of optimizer steps used for warmup')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient norm clipping threshold')
    parser.add_argument('--accumulate_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='Exponential moving average decay')
    
    # SHARP v3.2.2 specific
    parser.add_argument('--sparsity', '--sparse_sparsity_ratio', dest='sparsity',
                        type=float, default=0.9,
                        help='Sparse attention sparsity ratio (0.0-1.0)')
    parser.add_argument('--rbf_centers', type=int, default=32,
                        help='RBF centers per attention head')
    parser.add_argument('--k_cap', '--sparse_k_cap', dest='k_cap', type=int, default=1024,
                        help='Memory cap for top-k (0 = no cap)')
    parser.add_argument('--block_size', '--sparse_block_size', dest='block_size',
                        type=int, default=2048,
                        help='Block size for streaming attention')
    parser.add_argument('--q_block_size', '--sparse_q_block_size', dest='q_block_size',
                        type=int, default=1024,
                        help='Query block size for tiling')
    parser.add_argument('--exact_topk_max_tokens', '--sparse_exact_topk_max_tokens',
                        dest='exact_topk_max_tokens', type=int, default=1024,
                        help='Maximum token count for exact all-key streaming top-k')
    parser.add_argument('--landmark_tokens', '--sparse_landmark_tokens',
                        dest='landmark_tokens', type=int, default=256,
                        help='Maximum pooled global candidates above the exact top-k threshold')
    parser.add_argument('--max_tokens', '--sparse_max_tokens', dest='max_tokens',
                        type=int, default=8192,
                        help='Legacy hard cap for exact all-key top-k scans')
    parser.add_argument('--window_size', '--sparse_window_size', dest='window_size',
                        type=int, default=49,
                        help='Number of 2D local candidates in hybrid attention')
    parser.add_argument('--max_global_tokens', type=int, default=1024,
                        help='Maximum key/value tokens in dense global and cross attention (0 disables cap)')
    parser.add_argument('--key_rbf_mode', type=str, default='linear',
                        choices=['mean', 'linear', 'none'],
                        help='RBF key projection mode')
    parser.add_argument('--ema_update_every', type=int, default=1,
                        help='EMA update frequency (v3.2.2 throttling)')
    parser.add_argument('--output_activation', type=str, default='sigmoid',
                        choices=['sigmoid', 'tanh', 'relu', 'softplus', 'none'],
                        help='Output parameterization for [0,1] HSI reflectance (default sigmoid)')
    parser.add_argument('--loss_type', type=str, default='mrae',
                        choices=['mrae', 'l1_curvature'],
                        help='Training objective: mrae (matches eval metric/MST++) or l1_curvature (ablation)')

    # Optimization
    parser.add_argument('--compile', '--compile_model', dest='compile', action='store_true',
                        help='Compile model with torch.compile')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--checkpoint', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--early_stopping_patience', type=int, default=35,
                        help='Stop training after this many validations without improvement')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-5,
                        help='Minimum MRAE improvement to reset early stopping counter')
    parser.add_argument('--val_interval', type=int, default=10,
                        help='Validation interval in epochs')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Periodic checkpoint interval in epochs')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Training log interval in batches')
    parser.add_argument('--psnr_data_range', type=float, default=1.0,
                        help='Reference intensity range used when reporting PSNR')
    parser.add_argument('--min_mrae_denom', type=float, default=1e-6,
                        help='Clamp floor for MRAE denominator to avoid division by zero')
    parser.add_argument('--max_consecutive_nonfinite', type=int, default=8,
                        help='Abort after this many consecutive non-finite batches')
    parser.add_argument('--disable_oom_skip', action='store_true',
                        help='Disable OOM batch skipping and fail immediately')
    
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
    parser.add_argument('--device', type=str, default='cuda',
                        help='Training device (cuda or cpu)')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable DDP (automatically enabled when launched with torchrun)')
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank',
                        type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument('--ddp_find_unused_parameters', action='store_true',
                        help='Enable DDP unused-parameter graph traversal for debugging')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SHARPTrainingConfig(
        model_size=args.model_size,
        data_root=args.data_root,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        num_workers=args.num_workers,
        memory_mode=args.memory_mode,
        cache_size=args.cache_size,
        augment=not args.no_augment,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_clip=args.gradient_clip,
        accumulate_steps=args.accumulate_steps,
        ema_decay=args.ema_decay,
        sparse_sparsity_ratio=args.sparsity,
        rbf_centers_per_head=args.rbf_centers,
        sparse_k_cap=args.k_cap,
        sparse_block_size=args.block_size,
        sparse_q_block_size=args.q_block_size,
        sparse_exact_topk_max_tokens=args.exact_topk_max_tokens,
        sparse_landmark_tokens=args.landmark_tokens,
        sparse_max_tokens=args.max_tokens,
        sparse_window_size=args.window_size,
        max_global_tokens=args.max_global_tokens,
        key_rbf_mode=args.key_rbf_mode,
        output_activation=args.output_activation,
        loss_type=args.loss_type,
        ema_update_every=args.ema_update_every,
        compile_model=args.compile,
        use_amp=not args.no_amp,
        use_checkpoint=args.checkpoint,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        psnr_data_range=args.psnr_data_range,
        min_mrae_denom=args.min_mrae_denom,
        max_consecutive_nonfinite=args.max_consecutive_nonfinite,
        skip_oom_batches=not args.disable_oom_skip,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        resume_from=args.resume,
        seed=args.seed,
        device=args.device,
        distributed=args.distributed,
        local_rank=args.local_rank,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )
    
    # Create trainer and start training
    try:
        trainer = DedicatedSHARPTrainer(config)
        trainer.train()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()

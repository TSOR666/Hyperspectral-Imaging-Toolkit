
# --- BEGIN LOCAL PATCH: ensure local 'model' package is importable ---
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
_local_model_dir = os.path.join(_here, "model")
if _local_model_dir not in sys.path:
    sys.path.insert(0, _local_model_dir)
# --- END LOCAL PATCH ---
"""
Enhanced Training Script for MSWR-Net v2.1.2 (Production-Ready with SAM and Fixed Logging)
==========================================================================================

Production-ready training script with CNN-based wavelets and SAM loss integration.
Now with properly fixed logging that avoids duplicate output and ensures all messages are captured.

Key Updates:
1. Fixed create_logger with propagate=False to avoid duplicate console output
2. Proper handler cleanup to avoid file descriptor leaks when re-initializing
3. Fixed setup_environment to return paths for logging
4. Enhanced main() with proper error handling
5. All print statements replaced with logger calls
6. Proper log file and error file separation
7. Python warnings captured into logging system
8. Early logger also has propagate=False to prevent bubbling

Robustness Features:
- Closes existing handlers before clearing to prevent file descriptor leaks
- Captures Python warnings (e.g., NumPy deprecations) into log files
- Early bootstrap logger for pre-initialization errors
- Separate error log file for quick error scanning
- TQDM-compatible logging (progress bars and logs coexist cleanly)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from packaging import version

# Compatibility import for AMP (PyTorch < 2.4 uses torch.cuda.amp)
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

import os
import json
from datetime import datetime
import numpy as np
import random
from typing import Dict, Optional, Tuple, List, Any
import wandb
from tqdm import tqdm
import yaml
import logging
import sys
import time
import psutil
from collections import defaultdict
import warnings
import traceback
import copy

# Import the enhanced model - UPDATE THIS PATH TO YOUR FIXED MODEL FILE
try:
    from model.mswr_net_v212 import (
        MSWRDualConfig, 
        IntegratedMSWRNet,
        create_mswr_tiny,
        create_mswr_small,
        create_mswr_base,
        create_mswr_large,
        PerformanceMonitor
    )
except ImportError as e:
    # Use early logging for import errors
    early_logger = logging.getLogger('mswr_train_early')
    early_logger.setLevel(logging.ERROR)
    early_logger.propagate = False  # Prevent bubbling up to root logger
    if not early_logger.handlers:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.ERROR)
        early_logger.addHandler(ch)
    early_logger.error(f"Cannot import MSWR model: {e}")
    early_logger.error("Please ensure the model file is in the correct location and update the import path.")
    sys.exit(1)

# Import data and utilities
try:
    from dataloader import TrainDataset, ValidDataset
except ImportError as e:
    early_logger = logging.getLogger('mswr_train_early')
    early_logger.setLevel(logging.ERROR)
    early_logger.propagate = False  # Prevent bubbling up to root logger
    if not early_logger.handlers:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.ERROR)
        early_logger.addHandler(ch)
    early_logger.error(f"Cannot import dataloader: {e}")
    early_logger.error("Please ensure dataloader.py is available in your Python path.")
    sys.exit(1)

# UPDATED IMPORTS: Include Loss_SAM from utils
try:
    from utils import (
        AverageMeter, 
        initialize_logger, 
        save_checkpoint,
        time2file_name, 
        Loss_MRAE, 
        Loss_RMSE, 
        Loss_PSNR,
        Loss_SAM  # Added SAM import
    )
except ImportError as e:
    early_logger = logging.getLogger('mswr_train_early')
    early_logger.setLevel(logging.ERROR)
    early_logger.propagate = False  # Prevent bubbling up to root logger
    if not early_logger.handlers:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.ERROR)
        early_logger.addHandler(ch)
    early_logger.error(f"Cannot import utils: {e}")
    early_logger.error("Please ensure utils.py is available in your Python path with the fixed Loss_SAM.")
    sys.exit(1)

# Workspace defaults (can be overridden via environment variables or CLI)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.environ.get('MSWR_DATA_ROOT', os.path.join(REPO_ROOT, 'data', 'ARAD_1K')))
EXPERIMENTS_ROOT = os.path.abspath(os.environ.get('MSWR_EXPERIMENTS_ROOT', os.path.join(REPO_ROOT, 'experiments')))
LOG_DIR_BASE = os.path.join(EXPERIMENTS_ROOT, 'logs')
CHECKPOINT_DIR_BASE = os.path.join(EXPERIMENTS_ROOT, 'checkpoints')

TORCH_VERSION = version.parse(torch.__version__.split('+')[0])

# Model size mapping
MODEL_SIZES = {
    'tiny': create_mswr_tiny,
    'small': create_mswr_small,
    'base': create_mswr_base,
    'large': create_mswr_large
}

# UPDATED: Enhanced loss function using Loss_SAM from utils
class EnhancedMSWRLoss(nn.Module):
    """
    Enhanced loss function with multiple components including SAM from utils
    
    This version properly uses the fixed Loss_SAM that handles non-contiguous tensors
    """
    def __init__(self, 
                 l1_weight: float = 1.0,
                 mrae_weight: float = 0.0,
                 ssim_weight: float = 0.5,
                 sam_weight: float = 0.1,
                 gradient_weight: float = 0.1,
                 warmup_epochs: int = 10):
        super().__init__()
        self.l1_weight = l1_weight
        self.mrae_weight = mrae_weight
        self.ssim_weight = ssim_weight
        self.sam_weight = sam_weight
        self.gradient_weight = gradient_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Use the fixed losses from utils
        if self.mrae_weight > 0:
            self.mrae_loss = Loss_MRAE()
        
        # UPDATED: Use the fixed SAM loss from utils
        if self.sam_weight > 0:
            self.sam_loss = Loss_SAM()
        
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
    
    def _ssim_loss(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Simplified SSIM loss"""
        # Ensure minimum spatial size for SSIM
        if pred.shape[-2] < window_size or pred.shape[-1] < window_size:
            return torch.tensor(0.0, device=pred.device)
        
        mu1 = F.avg_pool2d(pred, window_size, 1, window_size//2)
        mu2 = F.avg_pool2d(target, window_size, 1, window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, window_size, 1, window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, 1, window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, 1, window_size//2) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()
    
    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gradient loss for edge preservation"""
        def gradient(x):
            # Avoid empty tensors
            if x.shape[-1] <= 1 or x.shape[-2] <= 1:
                return None, None
            grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
            grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
            return grad_x, grad_y
        
        pred_grad_x, pred_grad_y = gradient(pred)
        target_grad_x, target_grad_y = gradient(target)
        
        if pred_grad_x is None:
            return torch.tensor(0.0, device=pred.device)
        
        loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        loss_y = self.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass computing the combined loss
        
        Args:
            pred: Model predictions [B, C, H, W]
            target: Ground truth [B, C, H, W]
        
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary of individual loss components
        """
        total_loss = torch.tensor(0.0, device=pred.device)
        loss_dict = {}
        
        # L1 loss (base loss)
        if self.l1_weight > 0:
            l1_loss = self.l1_loss(pred, target)
            total_loss = total_loss + self.l1_weight * l1_loss
            loss_dict['l1'] = l1_loss.detach()
        
        # MRAE loss (using fixed version from utils)
        if self.mrae_weight > 0:
            mrae_weight = self.mrae_weight
            if self.current_epoch < self.warmup_epochs:
                mrae_weight *= (self.current_epoch / self.warmup_epochs)
            
            mrae_loss = self.mrae_loss(pred, target)
            total_loss = total_loss + mrae_weight * mrae_loss
            loss_dict['mrae'] = mrae_loss.detach()
        
        # SSIM loss (with warmup)
        if self.ssim_weight > 0:
            ssim_weight = self.ssim_weight
            if self.current_epoch < self.warmup_epochs:
                ssim_weight *= (self.current_epoch / self.warmup_epochs)
            
            ssim_loss = self._ssim_loss(pred, target)
            total_loss = total_loss + ssim_weight * ssim_loss
            loss_dict['ssim'] = ssim_loss.detach()
        
        # SAM loss (using fixed version from utils, for hyperspectral)
        if self.sam_weight > 0 and pred.size(1) > 3:  # Only for hyperspectral (>3 channels)
            sam_weight = self.sam_weight
            if self.current_epoch < self.warmup_epochs:
                sam_weight *= (self.current_epoch / self.warmup_epochs)
            
            sam_loss = self.sam_loss(pred, target)
            # Convert from radians to degrees for better interpretability in logging
            sam_loss_deg = sam_loss * 180.0 / np.pi
            total_loss = total_loss + sam_weight * sam_loss  # Use radians for optimization
            loss_dict['sam'] = sam_loss.detach()  # Store in radians
            loss_dict['sam_deg'] = sam_loss_deg.detach()  # Also store in degrees for display
        
        # Gradient loss (with warmup)
        if self.gradient_weight > 0:
            grad_weight = self.gradient_weight
            if self.current_epoch < self.warmup_epochs:
                grad_weight *= (self.current_epoch / self.warmup_epochs)
            
            grad_loss = self._gradient_loss(pred, target)
            total_loss = total_loss + grad_weight * grad_loss
            loss_dict['gradient'] = grad_loss.detach()
        
        # Ensure we have at least one loss component
        if len(loss_dict) == 0:
            l1_loss = self.l1_loss(pred, target)
            total_loss = l1_loss
            loss_dict['l1'] = l1_loss.detach()
        
        loss_dict['total'] = total_loss.detach()
        return total_loss, loss_dict

class TrainingConfig:
    """Enhanced configuration management for MSWR v2.1.2"""
    def __init__(self, args):
        # Basic training parameters
        self.batch_size = args.batch_size
        self.end_epoch = args.end_epoch
        self.init_lr = args.init_lr
        self.weight_decay = args.weight_decay
        self.patch_size = args.patch_size
        self.stride = args.stride
        self.gpu_id = args.gpu_id
        
        # Model parameters
        self.model_size = args.model_size
        self.attention_type = args.attention_type
        self.num_heads = args.num_heads
        self.base_channels = args.base_channels
        self.num_stages = args.num_stages
        self.use_checkpoint = args.use_checkpoint
        self.use_flash_attn = args.use_flash_attn
        self.window_size = args.window_size
        self.num_landmarks = args.num_landmarks
        self.landmark_pooling = args.landmark_pooling
        
        # CNN Wavelet parameters (no external dependencies)
        self.use_wavelet = args.use_wavelet
        self.wavelet_type = args.wavelet_type
        self.wavelet_levels = getattr(args, 'wavelet_levels', None)
        
        # Loss parameters (including MRAE weight)
        self.use_enhanced_loss = args.use_enhanced_loss
        self.l1_weight = args.l1_weight
        self.mrae_weight = getattr(args, 'mrae_weight', 0.0)  # Add MRAE weight
        self.ssim_weight = args.ssim_weight
        self.sam_weight = args.sam_weight
        self.gradient_weight = args.gradient_weight
        self.loss_warmup_epochs = args.loss_warmup_epochs
        
        # Advanced training parameters
        self.use_amp = args.use_amp
        self.gradient_clip = args.gradient_clip
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.save_frequency = args.save_frequency
        self.validate_frequency = args.validate_frequency
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.seed = args.seed
        
        # Optimization parameters
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.warmup_epochs = args.warmup_epochs
        self.min_lr = args.min_lr
        
        # Enhanced features
        self.early_stopping_patience = getattr(args, 'early_stopping_patience', 50)
        self.early_stopping_mode = getattr(args, 'early_stopping_mode', 'off')
        self.early_stopping_warmup = getattr(args, 'early_stopping_warmup', 0)
        self.memory_monitoring = getattr(args, 'memory_monitoring', True)
        self.profile_model = getattr(args, 'profile_model', False)
        
        # EMA settings
        self.use_ema = getattr(args, 'use_ema', True)
        self.ema_decay = getattr(args, 'ema_decay', 0.999)
        self.ema_start_epoch = getattr(args, 'ema_start_epoch', 5)
        self.ema_eval_mode = getattr(args, 'ema_eval_mode', 'ema')
        
        # Logging
        self.use_wandb = args.use_wandb
        self.experiment_name = args.experiment_name
        self.resume_path = args.pretrained_model_path
        
        # Paths
        self.data_root = args.data_root
        self.log_base = args.log_base
        self.checkpoint_base = args.checkpoint_base
        
        # Store run timestamp for consistent directory naming
        self.run_timestamp = time2file_name(str(datetime.now()))
        
    def to_dict(self):
        """Convert config to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

def parse_arguments():
    """Parse command line arguments with enhanced options"""
    parser = argparse.ArgumentParser(description="MSWR-Net v2.1.2 Training")
    
    # Configuration file
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    
    # Model loading
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    
    # Basic training parameters
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
    parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--patch_size", type=int, default=128, help="patch size")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument("--gpu_id", type=str, default='0', help='GPU ID(s)')
    
    # Model parameters
    parser.add_argument("--model_size", type=str, default='base', 
                       choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument("--attention_type", type=str, default='dual', 
                       choices=['window', 'dual', 'landmark', 'hybrid'])
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--num_landmarks", type=int, default=64)
    parser.add_argument("--landmark_pooling", type=str, default='learned',
                       choices=['learned', 'uniform', 'adaptive'])
    parser.add_argument("--use_checkpoint", action='store_true')
    parser.add_argument("--use_flash_attn", action='store_true', default=True)
    
    # CNN Wavelet parameters (always available, no dependencies)
    parser.add_argument("--use_wavelet", action='store_true', default=True)
    parser.add_argument("--wavelet_type", type=str, default='db2',
                       choices=['haar', 'db1', 'db2', 'db3', 'db4'])
    parser.add_argument("--wavelet_levels", type=int, nargs='+', default=None,
                       help='Wavelet levels for each stage')
    
    # Enhanced loss parameters (ADDED mrae_weight)
    parser.add_argument("--use_enhanced_loss", action='store_true', 
                       help='Use enhanced loss (L1+MRAE+SSIM+SAM+Gradient)')
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--mrae_weight", type=float, default=0.0,
                       help='Weight for MRAE loss')
    parser.add_argument("--ssim_weight", type=float, default=0.5)
    parser.add_argument("--sam_weight", type=float, default=0.1,
                       help='Weight for SAM loss (spectral angle mapper)')
    parser.add_argument("--gradient_weight", type=float, default=0.1)
    parser.add_argument("--loss_warmup_epochs", type=int, default=10)
    
    # Advanced training parameters
    parser.add_argument("--use_amp", action='store_true', default=True,
                       help='use automatic mixed precision')
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_frequency", type=int, default=5000)
    parser.add_argument("--validate_frequency", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=42)
    
    # Enhanced features
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--early_stopping_mode", type=str, default='off',
                       choices=['off', 'min', 'max'],
                       help="Disable early stopping ('off') or choose direction for monitoring")
    parser.add_argument("--early_stopping_warmup", type=int, default=5,
                       help="Number of epochs to skip before early stopping can trigger")
    parser.add_argument("--memory_monitoring", action='store_true', default=True)
    parser.add_argument("--profile_model", action='store_true')
    
    # Regularization / averaging
    parser.add_argument("--use_ema", dest="use_ema", action='store_true', default=True,
                       help="Enable exponential moving average of model weights (default: enabled)")
    parser.add_argument("--no_ema", dest="use_ema", action='store_false',
                       help="Disable exponential moving average of model weights")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                       help="Decay factor for EMA updates (closer to 1.0 means slower updates)")
    parser.add_argument("--ema_start_epoch", type=int, default=5,
                       help="Delay EMA updates until after this epoch to let training stabilize")
    parser.add_argument("--ema_eval_mode", type=str, default='ema',
                       choices=['ema', 'model', 'both'],
                       help="Which weights to use during validation ('ema' recommended)")
    
    # Optimization parameters
    parser.add_argument("--optimizer", type=str, default='adamw', 
                       choices=['adam', 'adamw', 'sgd'])
    parser.add_argument("--scheduler", type=str, default='cosine',
                       choices=['cosine', 'step', 'exponential'])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    
    # Logging
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--experiment_name", type=str, default='mswr_v212')
    
    # Paths
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--log_base", type=str, default=LOG_DIR_BASE)
    parser.add_argument("--checkpoint_base", type=str, default=CHECKPOINT_DIR_BASE)
    
    args = parser.parse_args()
    default_args = parser.parse_args([])
    args._parser_defaults = default_args
    args._explicit_cli_args = {
        key for key, value in vars(args).items()
        if getattr(default_args, key, None) != value
    }
    
    # Post-parse fix for wavelet_levels if single value provided
    if args.wavelet_levels is not None and len(args.wavelet_levels) == 1:
        # If only one value provided and we have multiple stages, repeat it
        if hasattr(args, 'num_stages') and args.num_stages > 1:
            args.wavelet_levels = args.wavelet_levels * args.num_stages
    
    return args

def load_config(args):
    """Load configuration from YAML file and merge with command line args"""
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        
        if not isinstance(yaml_cfg, dict):
            raise ValueError(f"Configuration file must define a mapping, got {type(yaml_cfg).__name__}")
        
        parser_defaults = vars(getattr(args, '_parser_defaults', argparse.Namespace()))
        explicit_cli = getattr(args, '_explicit_cli_args', set())
        
        # Update args with yaml config (preserving command line precedence)
        for key, value in yaml_cfg.items():
            if not hasattr(args, key):
                continue
            if key in explicit_cli:
                continue
            current_value = getattr(args, key)
            default_value = parser_defaults.get(key, current_value)
            if current_value == default_value:
                setattr(args, key, value)
    
    return args

def setup_environment(config: TrainingConfig):
    """Setup training environment with enhanced monitoring - FIXED VERSION"""
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    
    # Set random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    # Enhanced CUDA settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create directories
    log_dir = os.path.join(config.log_base, config.run_timestamp)
    checkpoint_dir = os.path.join(config.checkpoint_base, config.run_timestamp)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
    
    # System info logging
    system_info = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cpu_count': os.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3)
    }
    
    if torch.cuda.is_available():
        system_info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                   for i in range(torch.cuda.device_count())]
    
    system_info_path = os.path.join(log_dir, 'system_info.json')
    with open(system_info_path, 'w') as f:
        json.dump(system_info, f, indent=4)
    
    # FIX: Don't print here - return the paths and log them after logger is created
    return log_dir, checkpoint_dir, config_path, system_info_path

def create_logger(log_dir: str, name: str = 'mswr_train') -> logging.Logger:
    """Create enhanced logger with detailed formatting - FIXED VERSION"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Properly close and remove existing handlers to avoid file descriptor leaks
    for handler in logger.handlers[:]:  # Use slice to avoid modifying list during iteration
        handler.close()
        logger.removeHandler(handler)
    
    # CRITICAL FIX: Disable propagation to avoid duplicate console output
    logger.propagate = False
    
    # Enhanced formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = os.path.join(log_dir, 'train.log')
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Error handler (separate file for errors)
    error_file = os.path.join(log_dir, 'errors.log')
    eh = logging.FileHandler(error_file, mode='a')
    eh.setLevel(logging.ERROR)
    eh.setFormatter(formatter)
    logger.addHandler(eh)
    
    # FIX: Attach the log file path to the logger object
    logger.log_file_path = log_file
    logger.error_file_path = error_file
    
    # Capture Python warnings into the logging system
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.addHandler(fh)  # Send warnings to the main log file
    
    return logger

def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer with enhanced parameter grouping"""
    # Separate parameters by type for optimal training
    norm_params = []
    transformer_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'norm' in name or 'bias' in name:
            norm_params.append(param)
        elif any(x in name for x in ['attn', 'attention', 'transformer']):
            transformer_params.append(param)
        else:
            other_params.append(param)
    
    # Note: CNN wavelets use buffers, not parameters, so no wavelet param group
    param_groups = [
        {'params': transformer_params, 'weight_decay': config.weight_decay, 
         'lr': config.init_lr},
        {'params': other_params, 'weight_decay': config.weight_decay, 
         'lr': config.init_lr},
        {'params': norm_params, 'weight_decay': 0.0, 'lr': config.init_lr}
    ]
    
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optimizer

def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig, 
                    steps_per_epoch: int):
    """Create enhanced learning rate scheduler"""
    total_steps = steps_per_epoch * config.end_epoch
    warmup_steps = steps_per_epoch * config.warmup_epochs
    
    if config.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=config.min_lr
        )
    elif config.scheduler == 'step':
        milestones = [int(0.3 * config.end_epoch), int(0.6 * config.end_epoch), int(0.9 * config.end_epoch)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif config.scheduler == 'exponential':
        gamma = (config.min_lr / config.init_lr) ** (1.0 / config.end_epoch)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.min_lr
        )
    
    # Warmup wrapper
    if config.warmup_epochs > 0:
        scheduler = WarmupScheduler(scheduler, warmup_steps, config.init_lr)
    
    return scheduler

class WarmupScheduler:
    """Enhanced warmup scheduler with linear warmup"""
    def __init__(self, scheduler, warmup_steps: int, base_lr: float):
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_count = 0
        
    def step(self):
        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr_scale = (self.step_count + 1) / self.warmup_steps
            for param_group in self.scheduler.optimizer.param_groups:
                param_group['lr'] = param_group.get('initial_lr', self.base_lr) * lr_scale
        else:
            # Transition to main scheduler
            if self.step_count == self.warmup_steps:
                # Reset scheduler's last_epoch to avoid extra step
                self.scheduler.last_epoch = -1
            self.scheduler.step()
        self.step_count += 1
        
    def state_dict(self):
        return {
            'scheduler': self.scheduler.state_dict(),
            'warmup_steps': self.warmup_steps,
            'step_count': self.step_count,
            'base_lr': self.base_lr
        }
    
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.warmup_steps = state_dict['warmup_steps']
        self.step_count = state_dict['step_count']
        self.base_lr = state_dict.get('base_lr', self.base_lr)

class ModelEMA:
    """Exponential moving average (EMA) of model parameters for better generalization"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        if isinstance(model, nn.DataParallel):
            model = model.module
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def to(self, device: torch.device):
        self.ema_model.to(device)
        return self

    def update(self, model: nn.Module):
        source_model = model.module if isinstance(model, nn.DataParallel) else model
        ema_params = dict(self.ema_model.named_parameters())
        src_params = dict(source_model.named_parameters())
        src_buffers = dict(source_model.named_buffers())

        with torch.no_grad():
            for name, ema_param in ema_params.items():
                src_param = src_params[name].detach()
                ema_param.mul_(self.decay)
                ema_param.add_(src_param * (1.0 - self.decay))

            # Keep buffers (e.g., BatchNorm running stats) in sync
            for name, ema_buffer in self.ema_model.named_buffers():
                if name in src_buffers:
                    ema_buffer.copy_(src_buffers[name])

    def state_dict(self):
        return {
            'decay': self.decay,
            'ema_state': self.ema_model.state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.decay = state_dict.get('decay', self.decay)
        self.ema_model.load_state_dict(state_dict['ema_state'])


class EarlyStoppingMonitor:
    """Enhanced early stopping with best model tracking"""
    def __init__(self, patience: int = 50, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> Tuple[bool, bool]:
        improved = False
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
                improved = True
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
                improved = True
            else:
                self.counter += 1
        
        return improved, self.counter >= self.patience

class EnhancedTrainer:
    """Enhanced trainer class with advanced features and fixed logging"""
    def __init__(self, config: TrainingConfig, logger: logging.Logger, 
                 log_dir: str, checkpoint_dir: str):
        self.config = config
        self.logger = logger  # Now properly configured with no propagation
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state
        self.iteration = 0
        self.epoch = 0
        self.best_mrae = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.early_stopping = None
        self.early_stopping_warmup = max(0, config.early_stopping_warmup)
        self.early_stopping_metric_key = 'psnr' if config.early_stopping_mode == 'max' else 'mrae'
        self.ema = None
        
        # Setup components
        self._setup_model()
        self._setup_data()
        self._setup_optimization()
        self._setup_loss()

        # Enhanced monitoring setup
        if config.early_stopping_mode != 'off':
            self.early_stopping = EarlyStoppingMonitor(
                patience=config.early_stopping_patience,
                mode=config.early_stopping_mode
            )
            self.logger.info(
                f"Early stopping enabled (mode={config.early_stopping_mode}, "
                f"patience={config.early_stopping_patience}, warmup={self.early_stopping_warmup}, "
                f"metric={self.early_stopping_metric_key})"
            )
        else:
            self.logger.info("Early stopping is disabled (mode='off').")

        if self.config.use_ema:
            self.ema = ModelEMA(self.model, decay=self.config.ema_decay).to(self.device)
            self.logger.info(
                f"EMA enabled (decay={self.config.ema_decay}, "
                f"start_epoch={self.config.ema_start_epoch}, eval_mode={self.config.ema_eval_mode})"
            )
        else:
            self.logger.info("EMA is disabled.")
        
        # Mixed precision with compatibility
        if self.config.use_amp and torch.cuda.is_available():
            try:
                # PyTorch 2.0+ style
                self.scaler = GradScaler('cuda')
            except TypeError:
                # PyTorch < 2.0 style
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Wandb setup
        if config.use_wandb:
            wandb.init(
                project="mswr-v212-cnn-wavelets",
                name=config.experiment_name,
                config=config.to_dict(),
                tags=['cnn-wavelets', 'production', 'v2.1.2', 'fixed', 'sam', 'fixed-logging']
            )
            # Reduced frequency for transformer models to avoid overhead
            wandb.watch(self.model, log='gradients', log_freq=500)
    
    def _setup_model(self):
        """Initialize model with CNN wavelets"""
        if self.config.model_size in MODEL_SIZES:
            # Use predefined model size
            model_kwargs = {
                'attention_type': self.config.attention_type,
                'use_checkpoint': self.config.use_checkpoint,
                'use_flash_attn': self.config.use_flash_attn,
                'use_wavelet': self.config.use_wavelet,
                'wavelet_type': self.config.wavelet_type,
                'landmark_pooling': self.config.landmark_pooling,
                'performance_monitoring': self.config.memory_monitoring
            }
            
            if self.config.wavelet_levels:
                model_kwargs['wavelet_levels'] = self.config.wavelet_levels
                
            self.model = MODEL_SIZES[self.config.model_size](**model_kwargs)
        else:
            # Custom configuration
            model_config = MSWRDualConfig(
                input_channels=3,
                base_channels=self.config.base_channels,
                output_channels=31,
                attention_type=self.config.attention_type,
                num_heads=self.config.num_heads,
                num_stages=self.config.num_stages,
                window_size=self.config.window_size,
                num_landmarks=self.config.num_landmarks,
                landmark_pooling=self.config.landmark_pooling,
                use_checkpoint=self.config.use_checkpoint,
                use_flash_attn=self.config.use_flash_attn,
                use_wavelet=self.config.use_wavelet,
                wavelet_type=self.config.wavelet_type,
                wavelet_levels=self.config.wavelet_levels,
                performance_monitoring=self.config.memory_monitoring
            )
            self.model = IntegratedMSWRNet(model_config)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        
        # Model profiling
        if self.config.profile_model:
            self._profile_model()
        
        # Log model info
        model_info = self.model.module.get_model_info() if hasattr(self.model, 'module') else self.model.get_model_info()
        self.logger.info("="*60)
        self.logger.info("MODEL INFORMATION")
        self.logger.info("="*60)
        self.logger.info(f"Total parameters: {model_info['total_parameters']:,}")
        self.logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        self.logger.info(f"Model memory: {model_info['total_memory_mb']:.2f} MB")
        self.logger.info(f"Architecture: {model_info['architecture']}")
        self.logger.info("="*60)
    
    def _profile_model(self):
        """Profile model performance"""
        self.logger.info("Profiling model performance...")
        dummy_input = torch.randn(1, 3, self.config.patch_size, self.config.patch_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Profile
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(50):
                output = self.model(dummy_input)
        
        if has_cuda:
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 50 * 1000  # ms
        throughput = 1000 / avg_time  # FPS
        
        # Memory usage
        if has_cuda:
            max_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            self.logger.info(f"Average inference time: {avg_time:.2f}ms")
            self.logger.info(f"Throughput: {throughput:.2f} FPS @ {max_memory:.0f} MB")
        else:
            self.logger.info(f"Average inference time: {avg_time:.2f}ms")
            self.logger.info(f"Throughput: {throughput:.2f} FPS")
        
        # Get detailed performance summary
        if hasattr(self.model, 'module'):
            perf_summary = self.model.module.get_performance_summary()
        else:
            perf_summary = self.model.get_performance_summary()
        
        if perf_summary['stage_times_ms']:
            self.logger.info("Stage breakdown:")
            for stage, time_ms in perf_summary['stage_times_ms'].items():
                self.logger.info(f"  {stage}: {time_ms:.2f}ms")
    
    def _setup_data(self):
        """Initialize datasets with enhanced configuration"""
        self.train_dataset = TrainDataset(
            data_root=self.config.data_root,
            crop_size=self.config.patch_size,
            bgr2rgb=True,
            arg=True,
            stride=self.config.stride,
            logger=self.logger
        )
        
        self.val_dataset = ValidDataset(
            data_root=self.config.data_root,
            bgr2rgb=True,
            logger=self.logger
        )
        
        # Enhanced data loaders
        pin_memory_kwargs = {}
        if (
            torch.cuda.is_available()
            and TORCH_VERSION >= version.parse("2.0.0")
            and self.config.pin_memory
        ):
            pin_memory_kwargs["pin_memory_device"] = "cuda"

        def _prefetch_kwargs(worker_count: int):
            if worker_count > 0:
                return {"prefetch_factor": max(2, getattr(self.config, "prefetch_factor", 2))}
            return {}

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            **_prefetch_kwargs(self.config.num_workers),
            **pin_memory_kwargs,
        )
        
        # Validation loader with fixed persistent_workers handling
        val_num_workers = min(self.config.num_workers, 2)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if val_num_workers > 0 else False,
            **_prefetch_kwargs(val_num_workers),
            **pin_memory_kwargs,
        )
        
        self.logger.info(f"Training samples: {len(self.train_dataset):,}")
        self.logger.info(f"Validation samples: {len(self.val_dataset):,}")
        self.logger.info(f"Training batches per epoch: {len(self.train_loader):,}")
    
    def _setup_optimization(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = create_optimizer(self.model, self.config)
        
        # Store initial learning rates
        for group in self.optimizer.param_groups:
            group['initial_lr'] = group['lr']
        
        steps_per_epoch = len(self.train_loader)
        self.scheduler = create_scheduler(self.optimizer, self.config, steps_per_epoch)
        
        # Resume from checkpoint if specified
        if self.config.resume_path:
            self._load_checkpoint(self.config.resume_path)
        
        # Log optimization setup
        self.logger.info(f"Optimizer: {self.config.optimizer}")
        self.logger.info(f"Base learning rate: {self.config.init_lr}")
        self.logger.info(f"Scheduler: {self.config.scheduler}")
        self.logger.info(f"Weight decay: {self.config.weight_decay}")
    
    def _setup_loss(self):
        """Initialize enhanced loss functions"""
        if self.config.use_enhanced_loss:
            self.criterion = EnhancedMSWRLoss(
                l1_weight=self.config.l1_weight,
                mrae_weight=self.config.mrae_weight,
                ssim_weight=self.config.ssim_weight,
                sam_weight=self.config.sam_weight,
                gradient_weight=self.config.gradient_weight,
                warmup_epochs=self.config.loss_warmup_epochs
            ).to(self.device)
            
            # Log active loss components
            active_losses = []
            if self.config.l1_weight > 0:
                active_losses.append(f"L1({self.config.l1_weight})")
            if self.config.mrae_weight > 0:
                active_losses.append(f"MRAE({self.config.mrae_weight})")
            if self.config.ssim_weight > 0:
                active_losses.append(f"SSIM({self.config.ssim_weight})")
            if self.config.sam_weight > 0:
                active_losses.append(f"SAM({self.config.sam_weight})")
            if self.config.gradient_weight > 0:
                active_losses.append(f"Gradient({self.config.gradient_weight})")
            
            self.logger.info(f"Using enhanced loss: {' + '.join(active_losses)}")
        else:
            self.criterion = Loss_MRAE().to(self.device)
            self.logger.info("Using MRAE loss")
        
        # Validation metrics (always use these from utils)
        self.criterion_mrae = Loss_MRAE().to(self.device)
        self.criterion_rmse = Loss_RMSE().to(self.device)
        self.criterion_psnr = Loss_PSNR().to(self.device)
        self.criterion_sam = Loss_SAM().to(self.device)  # Add SAM for validation
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint with enhanced error handling"""
        if not os.path.isfile(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        try:
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            
            # Load optimizer state
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                self.logger.warning(f"Failed to load optimizer state: {e}")
            
            # Load scheduler state
            if 'scheduler' in checkpoint and checkpoint['scheduler']:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                except Exception as e:
                    self.logger.warning(f"Failed to load scheduler state: {e}")
            
            # Load training state
            self.epoch = checkpoint.get('epoch', 0)
            self.iteration = checkpoint.get('iter', 0)
            self.best_mrae = checkpoint.get('best_mrae', float('inf'))

            # Restore EMA state if available
            if self.ema is not None and checkpoint.get('ema'):
                try:
                    self.ema.load_state_dict(checkpoint['ema'])
                    self.ema.to(self.device)
                    self.logger.info("Loaded EMA state from checkpoint.")
                except Exception as e:
                    self.logger.warning(f"Failed to load EMA state: {e}")
            elif checkpoint.get('ema') and self.ema is None:
                self.logger.info("Checkpoint contains EMA weights but EMA is disabled in the current run.")

            # Restore early stopping state if applicable
            early_state = checkpoint.get('early_stopping_state')
            if self.early_stopping is not None and early_state:
                self.early_stopping.best_score = early_state.get('best_score', self.early_stopping.best_score)
                self.early_stopping.counter = early_state.get('counter', self.early_stopping.counter)
                self.early_stopping.best_epoch = early_state.get('best_epoch', self.early_stopping.best_epoch)

            self.logger.info(f"Resumed from epoch {self.epoch}, iteration {self.iteration}")
            self.logger.info(f"Best MRAE so far: {self.best_mrae:.6f}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def train_epoch(self) -> Dict[str, float]:
        """Enhanced training epoch with detailed monitoring including SAM and EMA updates"""
        self.model.train()
        losses = AverageMeter()
        loss_components = defaultdict(AverageMeter)
        grad_norms = AverageMeter()
        
        # Update loss epoch if using enhanced loss
        if hasattr(self.criterion, 'set_epoch'):
            self.criterion.set_epoch(self.epoch)
        
        # Memory monitoring
        if self.config.memory_monitoring and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for i, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Initialize per-step state
            loss_dict = {}
            grad_norm = 0.0
            
            # Forward pass
            if self.scaler is not None:
                with autocast('cuda'):
                    output = self.model(images)
                    if self.config.use_enhanced_loss:
                        loss, loss_dict = self.criterion(output, labels)
                        loss = loss / self.config.gradient_accumulation_steps
                    else:
                        loss = self.criterion(output, labels) / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                output = self.model(images)
                if self.config.use_enhanced_loss:
                    loss, loss_dict = self.criterion(output, labels)
                    loss = loss / self.config.gradient_accumulation_steps
                else:
                    loss = self.criterion(output, labels) / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation and optimization step
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    if self.config.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.config.gradient_clip > 0:
                        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                
                if self.ema is not None and self.epoch >= self.config.ema_start_epoch:
                    self.ema.update(self.model)
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                # Track gradient norm
                if isinstance(grad_norm, torch.Tensor):
                    grad_norms.update(grad_norm.item())
                else:
                    grad_norms.update(grad_norm)
            
            # Update metrics
            losses.update(loss.item() * self.config.gradient_accumulation_steps)
            
            # Update component losses
            if self.config.use_enhanced_loss and loss_dict:
                for key, value in loss_dict.items():
                    if key != 'total':
                        loss_components[key].update(value.item())
            
            self.iteration += 1
            
            # Progress bar update with SAM
            postfix = {
                'loss': f'{losses.avg:.4f}',
                'lr': f'{current_lr:.2e}',
                'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}G' if torch.cuda.is_available() else 'N/A'
            }
            
            # Add gradient norm to display
            if grad_norms.count > 0:
                postfix['grad'] = f'{grad_norms.avg:.3f}'
            
            # Add loss components including SAM
            if self.config.use_enhanced_loss and loss_components:
                if 'l1' in loss_components:
                    postfix['L1'] = f'{loss_components["l1"].avg:.4f}'
                if 'mrae' in loss_components:
                    postfix['MRAE'] = f'{loss_components["mrae"].avg:.4f}'
                if 'ssim' in loss_components:
                    postfix['SSIM'] = f'{loss_components["ssim"].avg:.4f}'
                if 'sam_deg' in loss_components:
                    postfix['SAM'] = f'{loss_components["sam_deg"].avg:.2f}deg'
                elif 'sam' in loss_components:
                    postfix['SAM'] = f'{loss_components["sam"].avg * 180.0 / np.pi:.2f}deg'
            
            pbar.set_postfix(postfix)
            
            # Logging
            if self.config.use_wandb and self.iteration % 20 == 0:
                log_dict = {
                    'train/loss': losses.val,
                    'train/lr': current_lr,
                    'train/epoch': self.epoch,
                    'iteration': self.iteration
                }
                
                if grad_norms.count > 0:
                    log_dict['train/grad_norm'] = grad_norms.val
                
                if self.config.use_enhanced_loss:
                    for key, meter in loss_components.items():
                        if meter.count > 0:
                            if key == 'sam_deg':
                                log_dict['train/sam_deg'] = meter.val
                            elif key == 'sam':
                                log_dict['train/sam_rad'] = meter.val
                                log_dict['train/sam_deg'] = meter.val * 180.0 / np.pi
                            else:
                                log_dict[f'train/{key}_loss'] = meter.val
                
                if torch.cuda.is_available():
                    log_dict['train/gpu_memory_gb'] = torch.cuda.memory_allocated() / 1024**3
                
                wandb.log(log_dict)
            
            # Validation and checkpointing
            if self.iteration % self.config.validate_frequency == 0:
                val_metrics = self.validate()
                evaluation_source = val_metrics.get('evaluation_model', 'model')
                
                # Track best checkpoint using validation MRAE
                new_best = val_metrics.get('mrae', float('inf')) < self.best_mrae
                if new_best:
                    self.best_mrae = val_metrics['mrae']
                    self.save_checkpoint('best.pth', is_best=True)
                    self.logger.info(
                        f"New best validation MRAE: {self.best_mrae:.6f} "
                        f"(weights={evaluation_source})"
                    )
                
                # Optional early stopping check
                should_stop = False
                if self.early_stopping is not None:
                    monitor_value = val_metrics.get(self.early_stopping_metric_key)
                    if monitor_value is None:
                        self.logger.warning(
                            f"Validation metrics missing '{self.early_stopping_metric_key}' "
                            "for early stopping; skipping check."
                        )
                    elif self.epoch < self.early_stopping_warmup:
                        if self.early_stopping.mode == 'min':
                            if monitor_value < self.early_stopping.best_score:
                                self.early_stopping.best_score = monitor_value
                                self.early_stopping.best_epoch = self.epoch
                        else:
                            if monitor_value > self.early_stopping.best_score:
                                self.early_stopping.best_score = monitor_value
                                self.early_stopping.best_epoch = self.epoch
                        self.early_stopping.counter = 0
                    else:
                        _, should_stop = self.early_stopping(monitor_value, self.epoch)
                
                if should_stop:
                    self.logger.info(f"Early stopping triggered at epoch {self.epoch}")
                    return {'loss': losses.avg, 'early_stop': True}
                
                # Log validation results with SAM
                log_msg = (
                    f"[Iter {self.iteration}] "
                    f"Train Loss: {losses.avg:.4f}, "
                    f"Val MRAE: {val_metrics['mrae']:.6f}, "
                    f"Val RMSE: {val_metrics['rmse']:.6f}, "
                    f"Val PSNR: {val_metrics['psnr']:.2f}dB"
                )
                if 'sam' in val_metrics:
                    log_msg += f", Val SAM: {val_metrics['sam']:.2f}deg"
                log_msg += f" [weights={evaluation_source}]"
                
                self.logger.info(log_msg)
                
                self.model.train()
            
            # Periodic checkpoint saving
            if self.iteration % self.config.save_frequency == 0:
                self.save_checkpoint(f'iter_{self.iteration}.pth')
        
        # End of epoch summary
        result = {'loss': losses.avg}
        for key, meter in loss_components.items():
            result[key] = meter.avg
        
        # Memory summary
        if self.config.memory_monitoring and torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            self.logger.info(f"Peak GPU memory usage: {peak_memory:.2f}GB")
        
        return result
    
    def _validation_pass(self, model: nn.Module, desc: str) -> Dict[str, float]:
        model.eval()
        metrics = {
            'mrae': AverageMeter(),
            'rmse': AverageMeter(),
            'psnr': AverageMeter(),
            'sam': AverageMeter()
        }
        
        pbar = tqdm(self.val_loader, desc=desc, leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.scaler is not None and torch.cuda.is_available():
                with autocast('cuda'):
                    output = model(images)
            else:
                output = model(images)
            
            if output.shape[-1] > 256:
                crop_size = min(128, output.shape[-1] // 4)
                output_crop = output[:, :, crop_size:-crop_size, crop_size:-crop_size]
                labels_crop = labels[:, :, crop_size:-crop_size, crop_size:-crop_size]
            else:
                output_crop, labels_crop = output, labels
            
            mrae = self.criterion_mrae(output_crop, labels_crop)
            rmse = self.criterion_rmse(output_crop, labels_crop)
            psnr = self.criterion_psnr(output_crop, labels_crop, data_range=1.0)
            
            metrics['mrae'].update(mrae.item())
            metrics['rmse'].update(rmse.item())
            metrics['psnr'].update(psnr.item())
            
            if output_crop.shape[1] > 3:
                sam = self.criterion_sam(output_crop, labels_crop)
                metrics['sam'].update(sam.item() * 180.0 / np.pi)
            
            postfix = {
                'MRAE': f'{metrics["mrae"].avg:.6f}',
                'RMSE': f'{metrics["rmse"].avg:.6f}',
                'PSNR': f'{metrics["psnr"].avg:.2f}dB'
            }
            if output_crop.shape[1] > 3:
                postfix['SAM'] = f'{metrics["sam"].avg:.2f}deg'
            
            pbar.set_postfix(postfix)
        
        return {key: meter.avg for key, meter in metrics.items() if meter.count > 0}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        eval_mode = self.config.ema_eval_mode
        ema_ready = self.ema is not None and self.epoch >= self.config.ema_start_epoch
        
        if eval_mode == 'both' and ema_ready:
            ema_results = self._validation_pass(self.ema.ema_model, desc='Validation (EMA)')
            model_results = self._validation_pass(self.model, desc='Validation (model)')
            
            results = dict(ema_results)
            results['evaluation_model'] = 'ema'
            for key, value in ema_results.items():
                results[f'ema_{key}'] = value
            for key, value in model_results.items():
                results[f'model_{key}'] = value
        else:
            use_ema = ema_ready and eval_mode != 'model'
            model_to_eval = self.ema.ema_model if use_ema else self.model
            desc = 'Validation (EMA)' if use_ema else 'Validation'
            results = self._validation_pass(model_to_eval, desc=desc)
            results['evaluation_model'] = 'ema' if use_ema else 'model'
        
        if self.config.use_wandb:
            log_dict = {
                'val/mrae': results['mrae'],
                'val/rmse': results['rmse'],
                'val/psnr': results['psnr'],
                'val/epoch': self.epoch,
                'val/evaluation_model': results['evaluation_model']
            }
            if 'sam' in results:
                log_dict['val/sam_deg'] = results['sam']
            for prefix in ('ema', 'model'):
                key_mrae = f'{prefix}_mrae'
                if key_mrae in results:
                    log_dict[f'val_{prefix}/mrae'] = results[key_mrae]
                    key_rmse = f'{prefix}_rmse'
                    key_psnr = f'{prefix}_psnr'
                    key_sam = f'{prefix}_sam'
                    if key_rmse in results:
                        log_dict[f'val_{prefix}/rmse'] = results[key_rmse]
                    if key_psnr in results:
                        log_dict[f'val_{prefix}/psnr'] = results[key_psnr]
                    if key_sam in results:
                        log_dict[f'val_{prefix}/sam_deg'] = results[key_sam]
            wandb.log(log_dict)
        
        self.model.train()
        return results

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Enhanced checkpoint saving with metadata"""
        # Prepare model state
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
            model_config = self.model.module.config.to_dict() if hasattr(self.model.module, 'config') else {}
        else:
            model_state = self.model.state_dict()
            model_config = self.model.config.to_dict() if hasattr(self.model, 'config') else {}
        
        # Track optional states
        early_state = None
        if self.early_stopping is not None:
            early_state = {
                'best_score': self.early_stopping.best_score,
                'counter': self.early_stopping.counter,
                'best_epoch': self.early_stopping.best_epoch,
                'mode': self.early_stopping.mode,
                'patience': self.early_stopping.patience
            }
        
        ema_state = self.ema.state_dict() if self.ema is not None else None
        
        # Create checkpoint with comprehensive metadata
        checkpoint = {
            'epoch': self.epoch,
            'iter': self.iteration,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'best_mrae': self.best_mrae,
            'config': self.config.to_dict(),
            'model_config': model_config,
            'torch_version': torch.__version__,
            'timestamp': datetime.now().isoformat(),
            'early_stopping_state': early_state,
            'ema': ema_state
        }
        
        # Add scaler state if using AMP
        if self.scaler is not None:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        try:
            torch.save(checkpoint, filepath)
            self.logger.info(f"Checkpoint saved: {filepath}")
            
            if is_best:
                # Save best model separately
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                
                # Save lightweight version (model only)
                lightweight_checkpoint = {
                    'state_dict': model_state,
                    'model_config': model_config,
                    'best_mrae': self.best_mrae,
                    'epoch': self.epoch,
                    'timestamp': datetime.now().isoformat()
                }
                if self.ema is not None:
                    lightweight_checkpoint['ema_state_dict'] = self.ema.ema_model.state_dict()
                    lightweight_checkpoint['ema_decay'] = self.config.ema_decay
                lightweight_path = os.path.join(self.checkpoint_dir, 'best_model_lightweight.pth')
                torch.save(lightweight_checkpoint, lightweight_path)
                
                self.logger.info(f"Best model saved: {best_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def train(self):
        """Enhanced training loop with comprehensive monitoring"""
        self.logger.info("="*80)
        self.logger.info("STARTING MSWR-NET v2.1.2 TRAINING WITH CNN WAVELETS AND SAM")
        self.logger.info("="*80)
        self.logger.info(f"Training for {self.config.end_epoch} epochs")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        self.logger.info(f"Using CNN-based wavelets: {self.config.use_wavelet} ({self.config.wavelet_type})")
        self.logger.info(f"Mixed precision: {self.config.use_amp}")
        self.logger.info(f"Flash attention: {self.config.use_flash_attn}")
        self.logger.info(f"SAM loss weight: {self.config.sam_weight}")
        self.logger.info("="*80)
        
        start_time = time.time()
        best_epoch = 0
        best_eval_source = 'model'
        
        try:
            for epoch in range(self.epoch, self.config.end_epoch):
                self.epoch = epoch
                epoch_start_time = time.time()
                
                # Training
                train_metrics = self.train_epoch()
                
                # Check for early stopping
                if train_metrics.get('early_stop', False):
                    break
                
                # End-of-epoch validation
                val_metrics = self.validate()
                
                # Track best performance
                if val_metrics['mrae'] < self.best_mrae:
                    self.best_mrae = val_metrics['mrae']
                    best_epoch = epoch
                    best_eval_source = val_metrics.get('evaluation_model', 'model')
                    self.save_checkpoint('best_epoch.pth', is_best=True)
                
                # Epoch timing
                epoch_time = time.time() - epoch_start_time
                
                # Comprehensive logging with SAM
                log_msg = (
                    f"[Epoch {epoch:3d}/{self.config.end_epoch}] "
                    f"Time: {epoch_time:.1f}s | "
                    f"Train Loss: {train_metrics['loss']:.6f}"
                )
                
                if self.config.use_enhanced_loss:
                    component_parts = []
                    if 'l1' in train_metrics:
                        component_parts.append(f"L1: {train_metrics['l1']:.6f}")
                    if 'mrae' in train_metrics:
                        component_parts.append(f"MRAE: {train_metrics['mrae']:.6f}")
                    if 'ssim' in train_metrics:
                        component_parts.append(f"SSIM: {train_metrics['ssim']:.6f}")
                    if 'sam_deg' in train_metrics:
                        component_parts.append(f"SAM: {train_metrics['sam_deg']:.2f}deg")
                    elif 'sam' in train_metrics:
                        component_parts.append(f"SAM: {train_metrics['sam'] * 180.0 / np.pi:.2f}deg")
                    if 'gradient' in train_metrics:
                        component_parts.append(f"Grad: {train_metrics['gradient']:.6f}")
                    
                    if component_parts:
                        log_msg += f" ({' | '.join(component_parts)})"
                
                log_msg += (
                    f" | Val MRAE: {val_metrics['mrae']:.6f} "
                    f"| Val RMSE: {val_metrics['rmse']:.6f} "
                    f"| Val PSNR: {val_metrics['psnr']:.2f}dB"
                )
                
                if 'sam' in val_metrics:
                    log_msg += f" | Val SAM: {val_metrics['sam']:.2f}deg"
                
                log_msg += f" | Eval Weights: {val_metrics.get('evaluation_model', 'model')}"
                
                # Add best performance info
                if val_metrics['mrae'] == self.best_mrae:
                    log_msg += " [best]"
                self.logger.info(log_msg)
                
                # Periodic checkpointing
                if epoch % 10 == 0:
                    self.save_checkpoint(f'epoch_{epoch:03d}.pth')
                
                # Track metrics for analysis
                self.train_losses.append(train_metrics['loss'])
                self.val_losses.append(val_metrics['mrae'])
                
                # Enhanced wandb logging
                if self.config.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'epoch_time': epoch_time,
                        'best_mrae': self.best_mrae,
                        'epochs_since_best': epoch - best_epoch,
                        'epoch_val/evaluation_model': val_metrics.get('evaluation_model', 'model'),
                        **{f'epoch_train/{k}': v for k, v in train_metrics.items() if isinstance(v, (int, float))},
                        **{f'epoch_val/{k}': v for k, v in val_metrics.items() if isinstance(v, (int, float))}
                    })
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.save_checkpoint('interrupted.pth')
        
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}", exc_info=True)
            self.save_checkpoint('error.pth')
            raise
        
        finally:
            # Training summary
            total_time = time.time() - start_time
            self.save_checkpoint('final.pth')
            
            self.logger.info("="*80)
            self.logger.info("TRAINING COMPLETED")
            self.logger.info("="*80)
            self.logger.info(f"Total training time: {total_time/3600:.2f} hours")
            self.logger.info(f"Best MRAE: {self.best_mrae:.6f} (epoch {best_epoch})")
            self.logger.info(f"Best checkpoint weights source: {best_eval_source}")
            self.logger.info(f"Final epoch: {self.epoch}")
            self.logger.info(f"Checkpoints saved to: {self.checkpoint_dir}")
            self.logger.info("="*80)
            
            if self.config.use_wandb:
                wandb.log({
                    'training_completed': True,
                    'total_training_hours': total_time / 3600,
                    'final_best_mrae': self.best_mrae,
                    'best_epoch': best_epoch
                })
                wandb.finish()

def main():
    """Main training entry point - FIXED VERSION with proper logging"""
    # Initialize basic logger for early errors
    early_logger = logging.getLogger('mswr_train_early')
    early_logger.setLevel(logging.ERROR)
    early_logger.propagate = False  # Prevent bubbling up to root logger
    
    # Add a basic console handler for early errors
    if not early_logger.handlers:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.ERROR)
        early_logger.addHandler(ch)
    
    try:
        # Parse arguments
        args = parse_arguments()
        args = load_config(args)
        config = TrainingConfig(args)
        
        # Setup environment (now returns paths for logging)
        log_dir, checkpoint_dir, config_path, system_info_path = setup_environment(config)
        
        # Initialize the main logger
        logger = create_logger(log_dir)
        
        # NOW log all the setup information
        logger.info("="*80)
        logger.info("MSWR-NET v2.1.2 TRAINING - CNN WAVELETS + SAM EDITION")
        logger.info("="*80)
        logger.info(f"Log directory: {log_dir}")
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        logger.info(f"Configuration saved to: {config_path}")
        logger.info(f"System info saved to: {system_info_path}")
        logger.info(f"Log file: {logger.log_file_path}")
        logger.info(f"Error log file: {logger.error_file_path}")
        logger.info("="*80)
        
        # Log the configuration
        logger.info("Training Configuration:")
        for key, value in config.to_dict().items():
            logger.info(f"  {key}: {value}")
        logger.info("="*80)
        
        # Create and run trainer
        trainer = EnhancedTrainer(config, logger, log_dir, checkpoint_dir)
        trainer.train()
        
    except KeyboardInterrupt:
        # Handle keyboard interrupt gracefully
        if 'logger' in locals():
            logger.info("Training interrupted by user (Ctrl+C)")
        else:
            early_logger.info("Training interrupted by user (Ctrl+C)")
        sys.exit(0)
        
    except Exception as e:
        # FIX: Use logger for exceptions instead of print
        if 'logger' in locals():
            logger.error("Fatal error in main()", exc_info=True)
            logger.error(f"Error details: {str(e)}")
        else:
            # Fall back to early logger if main logger isn't initialized
            early_logger.error("Fatal error in main() before logger initialization", exc_info=True)
            early_logger.error(f"Error details: {str(e)}")
            # Still print to console as backup
            traceback.print_exc()
        
        sys.exit(1)

if __name__ == '__main__':
    main()

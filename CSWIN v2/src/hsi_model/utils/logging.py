import os
import logging
import logging.handlers
from typing import Dict, Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from hsi_model.constants import DEFAULT_LOG_DIR

def setup_logging(log_dir: str = DEFAULT_LOG_DIR, log_level: int = logging.INFO, rank: int = 0) -> logging.Logger:
    """
    Set up rotating file logging, with separate files per rank in distributed mode.
    
    Args:
        log_dir: Directory for log files (default: DEFAULT_LOG_DIR)
        log_level: Logging level (default: INFO)
        rank: Process rank for distributed training
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("hsi_model")
    logger.setLevel(log_level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Console handler (only for rank 0 in distributed mode)
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Rotating file handler - 10MB max size, keep 5 backups
    # Create separate log file for each rank
    log_file = os.path.join(log_dir, f"training_rank{rank}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """
    Wrapper for logging metrics to both logger and tensorboard.
    
    Handles rank separation in distributed training.
    """
    def __init__(self, log_dir: str, local_rank: int = 0):
        self.local_rank = local_rank
        self.logger = logging.getLogger("hsi_model")
        
        # Only create tensorboard writer on rank 0
        self.writer = None
        if local_rank == 0:
            tensorboard_dir = os.path.join(log_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_dir)
    
    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """
        Log scalar metrics to both logger and tensorboard.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Current step (epoch or iteration)
            prefix: Optional prefix for metric names
        """
        for key, value in metrics.items():
            name = f"{prefix}/{key}" if prefix else key
            
            # Log to text logger
            if self.local_rank == 0:
                self.logger.info(f"{name}: {value:.4f}")
            
            # Log to tensorboard
            if self.writer is not None:
                self.writer.add_scalar(name, value, step)
                
    def log_images(self, name: str, images: torch.Tensor, step: int) -> None:
        """
        Log images to tensorboard.
        
        Args:
            name: Name for the image group
            images: Tensor of images (B, C, H, W) in range [0, 1]
            step: Current step (epoch or iteration)
        """
        if self.writer is not None:
            self.writer.add_images(name, images, step)
            
    def log_model_params(self, model: torch.nn.Module, step: int) -> None:
        """
        Log model parameter histograms to tensorboard.
        
        Args:
            model: Model to log parameters from
            step: Current step (epoch or iteration)
        """
        if self.writer is not None:
            for name, param in model.named_parameters():
                self.writer.add_histogram(f"parameters/{name}", param.data, step)
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad, step)
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int, name: str = "learning_rate") -> None:
        """
        Log learning rate to tensorboard.
        
        Args:
            optimizer: Optimizer to get learning rate from
            step: Current step (epoch or iteration)
            name: Name for the learning rate scalar
        """
        if self.writer is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                self.writer.add_scalar(f"{name}/group_{i}", param_group['lr'], step)
    
    def close(self) -> None:
        """Close tensorboard writer"""
        if self.writer is not None:
            self.writer.close()

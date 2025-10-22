import os
import logging
from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from hsi_model.constants import (
    DEFAULT_CHECKPOINT_DIR,
    CHECKPOINT_BEST_NAME,
    CHECKPOINT_LATEST_NAME,
    CHECKPOINT_KEEP_COUNT,
)

logger = logging.getLogger(__name__)

def save_checkpoint(
    model: nn.Module, 
    optimizers: Dict[str, torch.optim.Optimizer], 
    scalers: Dict[str, GradScaler],
    epoch: int, 
    resolution: int, 
    config: Dict[str, Any],
    is_best: bool = False,
    val_metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Save model and optimizer states with improved error handling.
    
    Args:
        model: Model to save
        optimizers: Dictionary of optimizers to save
        scalers: Dictionary of gradient scalers to save
        epoch: Current epoch
        resolution: Current resolution
        config: Model configuration
        is_best: Whether this is the best model so far
        val_metrics: Optional dictionary of validation metrics
    """
    # Only the main process saves checkpoints in distributed mode
    if config.get("distributed", False) and torch.distributed.get_rank() != 0:
        return
    
    try:
        # Create checkpoint directory
        checkpoint_dir = config.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Determine checkpoint filename
        if is_best:
            checkpoint_name = CHECKPOINT_BEST_NAME
        elif epoch == -1:  # Emergency save
            checkpoint_name = f"emergency_checkpoint_res{resolution}.pth"
        else:
            checkpoint_name = f"model_res{resolution}_epoch{epoch}.pth"
            
        save_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        # Get model state dict, handling DDP wrapper if present
        if isinstance(model, DDP):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        # Prepare checkpoint dictionary
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'resolution': resolution,
            'config': config,
        }
        
        # Save optimizer states
        if optimizers:
            for name, optimizer in optimizers.items():
                if optimizer is not None:
                    save_dict[f'{name}_state_dict'] = optimizer.state_dict()
        
        # Save scaler states
        if scalers:
            for name, scaler in scalers.items():
                if scaler is not None:
                    save_dict[f'{name}_scaler_state_dict'] = scaler.state_dict()
        
        # Save validation metrics if provided
        if val_metrics is not None:
            save_dict['val_metrics'] = val_metrics
        
        # Save checkpoint with atomic write
        temp_path = save_path + ".tmp"
        torch.save(save_dict, temp_path)
        
        # Atomic move to final location
        if os.path.exists(temp_path):
            os.replace(temp_path, save_path)
            logger.info(f"Checkpoint saved to {save_path}")
            
            # Also save latest checkpoint link
            latest_path = os.path.join(checkpoint_dir, CHECKPOINT_LATEST_NAME)
            if os.path.exists(save_path):
                # Create a copy for latest checkpoint
                torch.save(save_dict, latest_path)
                logger.info(f"Latest checkpoint updated: {latest_path}")
        else:
            logger.error(f"Failed to create temporary checkpoint file: {temp_path}")
            
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        # Try to save a minimal checkpoint as fallback
        try:
            fallback_path = os.path.join(checkpoint_dir, f"fallback_checkpoint_epoch{epoch}.pth")
            minimal_dict = {
                'epoch': epoch,
                'model_state_dict': model_state if 'model_state' in locals() else {},
                'resolution': resolution,
                'error': str(e)
            }
            torch.save(minimal_dict, fallback_path)
            logger.info(f"Saved fallback checkpoint to {fallback_path}")
        except Exception as fallback_e:
            logger.error(f"Even fallback checkpoint save failed: {str(fallback_e)}")


def load_checkpoint(
    model: nn.Module, 
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    scalers: Optional[Dict[str, GradScaler]] = None,
    checkpoint_path: str = "", 
    device: torch.device = torch.device("cpu")
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model weights from checkpoint with support for resuming training.
    
    Args:
        model: Model to load weights into
        optimizers: Optional dictionary of optimizers to load states into
        scalers: Optional dictionary of gradient scalers to load states into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (loaded_model, checkpoint_info_dict)
    """
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return model, {}
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle DDP module prefix mismatch
        is_model_ddp = isinstance(model, DDP)
        state_dict = checkpoint.get('model_state_dict', {})
        
        if not state_dict:
            logger.warning(f"No model state dict found in checkpoint: {checkpoint_path}")
            return model, {}
        
        # Check if keys exist in the state dict
        if len(state_dict) > 0:
            first_key = next(iter(state_dict.keys()))
            
            # Check if the state_dict has module prefix but model is not DDP
            if not is_model_ddp and first_key.startswith('module.'):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                logger.info("Removed 'module.' prefix from checkpoint keys")
            # Check if the state_dict doesn't have module prefix but model is DDP
            elif is_model_ddp and not first_key.startswith('module.'):
                state_dict = {f"module.{k}": v for k, v in state_dict.items()}
                logger.info("Added 'module.' prefix to checkpoint keys")
        
        # Load the state dict with strict=False to handle partial loading
        if is_model_ddp:
            missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        
        # If optimizers provided, load their states for resuming training
        if optimizers is not None:
            for name, optimizer in optimizers.items():
                if optimizer is not None and f'{name}_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint[f'{name}_state_dict'])
                        logger.info(f"Loaded optimizer state for {name}")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state for {name}: {str(e)}")
        
        # If scalers provided, load their states
        if scalers is not None:
            for name, scaler in scalers.items():
                if scaler is not None and f'{name}_scaler_state_dict' in checkpoint:
                    try:
                        scaler.load_state_dict(checkpoint[f'{name}_scaler_state_dict'])
                        logger.info(f"Loaded scaler state for {name}")
                    except Exception as e:
                        logger.warning(f"Failed to load scaler state for {name}: {str(e)}")
        
        # Extract training info for resuming
        training_info = {
            'epoch': checkpoint.get('epoch', 0),
            'resolution': checkpoint.get('resolution', 0),
            'config': checkpoint.get('config', None),
            'val_metrics': checkpoint.get('val_metrics', None)
        }
        
        logger.info(f"Successfully loaded checkpoint from {checkpoint_path} (epoch {training_info['epoch']})")
        return model, training_info
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {str(e)}")
        return model, {}


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        Path to the latest checkpoint file, or empty string if none found
    """
    if not os.path.exists(checkpoint_dir):
        return ""
    
    # First check for an explicit "latest" checkpoint
    latest_path = os.path.join(checkpoint_dir, CHECKPOINT_LATEST_NAME)
    if os.path.exists(latest_path):
        return latest_path
    
    # Otherwise find the checkpoint with highest epoch number
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth') and 'epoch' in filename:
            checkpoint_files.append(filename)
    
    if not checkpoint_files:
        return ""
    
    # Sort by epoch number (extract from filename)
    def extract_epoch(filename):
        try:
            # Extract epoch number from filename like "model_res256_epoch123.pth"
            parts = filename.split('_')
            for part in parts:
                if part.startswith('epoch'):
                    return int(part.replace('epoch', '').replace('.pth', ''))
            return 0
        except:
            return 0
    
    latest_file = max(checkpoint_files, key=extract_epoch)
    return os.path.join(checkpoint_dir, latest_file)


def cleanup_old_checkpoints(
    checkpoint_dir: str, keep_count: int = CHECKPOINT_KEEP_COUNT
) -> None:
    """
    Clean up old checkpoint files, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_count: Number of recent checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    try:
        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            if (
                filename.endswith(".pth")
                and "epoch" in filename
                and filename not in [CHECKPOINT_BEST_NAME, CHECKPOINT_LATEST_NAME]
            ):
                filepath = os.path.join(checkpoint_dir, filename)
                mtime = os.path.getmtime(filepath)
                checkpoint_files.append((filepath, mtime))
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old checkpoints
        for filepath, _ in checkpoint_files[keep_count:]:
            try:
                os.remove(filepath)
                logger.info(f"Removed old checkpoint: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {filepath}: {str(e)}")
                
    except Exception as e:
        logger.warning(f"Failed to cleanup old checkpoints: {str(e)}")

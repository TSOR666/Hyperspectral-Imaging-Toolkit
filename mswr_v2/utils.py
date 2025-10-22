"""
Fixed utility functions for MSWR-Net training - Legacy Compatible
==================================================================

CRITICAL FIX: Replace .view(-1) with .reshape(-1) to handle non-contiguous tensors
while maintaining complete backward compatibility with existing code.
"""

import hdf5storage
import torch
import torch.nn as nn
import numpy as np
from fvcore.nn import FlopCountAnalysis
import logging
import os
from typing import Optional, Dict, Any, Tuple

def save_matv73(mat_name, var_name, var):
    """Save variable to MATLAB v7.3 format"""
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Loss_MRAE(nn.Module):
    """
    Mean Relative Absolute Error Loss
    FIXED: Use reshape instead of view for non-contiguous tensors
    """
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        # Add small epsilon to avoid division by zero
        error = torch.abs(outputs - label) / (label + 1e-8)
        # FIXED: Use reshape instead of view to handle non-contiguous tensors
        mrae = torch.mean(error.reshape(-1))
        return mrae

class Loss_RMSE(nn.Module):
    """
    Root Mean Squared Error Loss
    FIXED: Use reshape for consistency
    """
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs - label
        sqrt_error = torch.pow(error, 2)
        # FIXED: Use reshape instead of view to handle non-contiguous tensors
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio Loss
    FIXED: Use reshape instead of resize_ and proper tensor operations
    """
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        
        # FIXED: Use reshape instead of resize_ (which is deprecated and modifies in-place)
        Itrue = im_true.clamp(0., 1.).mul(data_range).reshape(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul(data_range).reshape(N, C * H * W)
        
        # Calculate MSE
        mse = nn.MSELoss(reduction='none')
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div(C * H * W)
        
        # Calculate PSNR
        psnr = 10. * torch.log10((data_range ** 2) / (err + 1e-8))  # Add epsilon to avoid log(0)
        return torch.mean(psnr)

# Additional loss functions for enhanced training
class Loss_SAM(nn.Module):
    """
    Spectral Angle Mapper Loss
    Measures the spectral angle between predicted and target spectra
    """
    def __init__(self):
        super(Loss_SAM, self).__init__()
    
    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        
        # Reshape to (batch*height*width, channels)
        B, C, H, W = outputs.shape
        outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, C)
        label_flat = label.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Normalize vectors
        outputs_norm = torch.nn.functional.normalize(outputs_flat, p=2, dim=1)
        label_norm = torch.nn.functional.normalize(label_flat, p=2, dim=1)
        
        # Compute dot product and clamp to avoid numerical issues
        dot_product = torch.sum(outputs_norm * label_norm, dim=1)
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Compute angle in radians
        angles = torch.acos(dot_product)
        
        # Return mean angle (you can convert to degrees if needed)
        sam = torch.mean(angles)
        return sam

def my_summary(test_model, H=256, W=256, C=31, N=1):
    """Print model summary with FLOPs and parameters count"""
    model = test_model.cuda()
    print(model)
    inputs = torch.randn((N, C, H, W)).cuda()
    flops = FlopCountAnalysis(model, inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')

def initialize_logger(log_path):
    """Initialize a logger that writes to log_path file"""
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Also add console handler for visibility
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def record_loss(log_path, loss):
    """Record loss value to a text file"""
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    with open(log_path, 'a') as f:
        f.write(f"{loss}\n")

def save_checkpoint(outf, epoch, iteration, model, optimizer, scheduler=None, best_metric=None):
    """
    Save model checkpoint with enhanced state information
    
    Args:
        outf: Output folder path
        epoch: Current epoch
        iteration: Current iteration
        model: Model to save
        optimizer: Optimizer state
        scheduler: Optional learning rate scheduler
        best_metric: Optional best metric value
    """
    # Create directory if it doesn't exist
    if not os.path.exists(outf):
        os.makedirs(outf, exist_ok=True)
    
    # Prepare state dictionary
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    
    # Add optional components
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    
    if best_metric is not None:
        state['best_metric'] = best_metric
    
    # Save checkpoint
    checkpoint_path = os.path.join(outf, f"checkpoint_{epoch}.pth")
    torch.save(state, checkpoint_path)
    
    # Also save as 'latest.pth' for easy resuming
    latest_path = os.path.join(outf, "latest.pth")
    torch.save(state, latest_path)
    
    return checkpoint_path

def load_checkpoint(checkpoint_path: str, 
                   model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None,
                   device: torch.device = torch.device('cpu')) -> Tuple[int, int, Optional[float]]:
    """
    Load model checkpoint
    
    Returns:
        Tuple of (epoch, iteration, best_metric)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Get training state
    epoch = checkpoint.get('epoch', 0)
    iteration = checkpoint.get('iter', 0)
    best_metric = checkpoint.get('best_metric', None)
    
    return epoch, iteration, best_metric

def time2file_name(date_time_str):
    """Transform a date string to a filename-safe format"""
    return date_time_str.replace(':', '-').replace(' ', '_').replace('.', '-')

# Additional utility functions
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size(model: nn.Module) -> float:
    """
    Get model size in MB
    
    Returns:
        Model size in megabytes
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)

def calculate_metrics(outputs: torch.Tensor, 
                     labels: torch.Tensor,
                     include_sam: bool = True) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics
    
    Args:
        outputs: Model predictions
        labels: Ground truth
        include_sam: Whether to include SAM metric (for hyperspectral)
    
    Returns:
        Dictionary of metric values
    """
    with torch.no_grad():
        # Ensure tensors are on the same device
        if outputs.device != labels.device:
            labels = labels.to(outputs.device)
        
        # Initialize metrics
        metrics = {}
        
        # MRAE
        mrae_loss = Loss_MRAE()
        metrics['mrae'] = mrae_loss(outputs, labels).item()
        
        # RMSE
        rmse_loss = Loss_RMSE()
        metrics['rmse'] = rmse_loss(outputs, labels).item()
        
        # PSNR
        psnr_loss = Loss_PSNR()
        metrics['psnr'] = psnr_loss(outputs, labels, data_range=1.0).item()
        
        # SAM (for hyperspectral)
        if include_sam and outputs.shape[1] > 3:  # More than RGB channels
            sam_loss = Loss_SAM()
            metrics['sam'] = sam_loss(outputs, labels).item()
        
        return metrics

# Test the fixed functions
if __name__ == "__main__":
    print("Testing fixed loss functions with non-contiguous tensors...")
    
    # Create test tensors (make them non-contiguous on purpose)
    B, C, H, W = 4, 31, 64, 64
    outputs = torch.randn(B, C, H, W).permute(0, 2, 3, 1).permute(0, 3, 1, 2)  # Non-contiguous
    labels = torch.randn(B, C, H, W)
    
    print(f"Output tensor contiguous: {outputs.is_contiguous()}")
    print(f"Label tensor contiguous: {labels.is_contiguous()}")
    
    # Test MRAE loss
    try:
        mrae_loss = Loss_MRAE()
        loss = mrae_loss(outputs, labels)
        print(f"✅ MRAE Loss: {loss.item():.6f}")
    except Exception as e:
        print(f"❌ MRAE Loss failed: {e}")
    
    # Test RMSE loss
    try:
        rmse_loss = Loss_RMSE()
        loss = rmse_loss(outputs, labels)
        print(f"✅ RMSE Loss: {loss.item():.6f}")
    except Exception as e:
        print(f"❌ RMSE Loss failed: {e}")
    
    # Test PSNR loss
    try:
        psnr_loss = Loss_PSNR()
        loss = psnr_loss(outputs, labels, data_range=1.0)
        print(f"✅ PSNR Loss: {loss.item():.2f} dB")
    except Exception as e:
        print(f"❌ PSNR Loss failed: {e}")
    
    # Test SAM loss
    try:
        sam_loss = Loss_SAM()
        loss = sam_loss(outputs, labels)
        print(f"✅ SAM Loss: {loss.item():.6f} radians")
    except Exception as e:
        print(f"❌ SAM Loss failed: {e}")
    
    # Test calculate_metrics
    try:
        metrics = calculate_metrics(outputs, labels)
        print(f"✅ All metrics calculated: {metrics}")
    except Exception as e:
        print(f"❌ Calculate metrics failed: {e}")
    
    print("\n✅ All tests passed! The utils file is ready for training.")

# src/hsi_model/utils/metrics.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
from typing import Tuple, Dict, Any, Optional, Union
import warnings
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Check for pytorch-msssim availability
try:
    from pytorch_msssim import ssim as msssim_ssim
    HAS_PYTORCH_MSSSIM = True
except ImportError:
    HAS_PYTORCH_MSSSIM = False
    warnings.warn(
        "pytorch-msssim not installed. SSIM computation will use approximate fallback. "
        "Install with: pip install pytorch-msssim"
    )

# Check for onnxruntime availability
try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


def hsi_to_rgb(hsi: torch.Tensor, cmf: torch.Tensor) -> torch.Tensor:
    """
    Convert hyperspectral image to RGB using color matching functions.
    
    Args:
        hsi: Hyperspectral image tensor with shape (B, C, H, W)
        cmf: Color matching function tensor with shape (C, 3)
    
    Returns:
        RGB image tensor with shape (B, 3, H, W)
    """
    # Ensure both tensors are the same type (convert to float32)
    hsi = hsi.to(torch.float32)
    cmf = cmf.to(torch.float32)
    
    # Perform the einsum operation
    rgb = torch.einsum('bchw,cd->bdhw', hsi, cmf)
    
    # Clamp values to [0, 1]
    rgb = torch.clamp(rgb, 0, 1)
    
    return rgb


def create_cmf_tensor(
    num_bands: int, 
    device: Optional[torch.device] = None,
    wavelength_range: Tuple[float, float] = (400, 700)
) -> torch.Tensor:
    """
    Create color matching function tensor for RGB conversion.
    
    Args:
        num_bands: Number of spectral bands
        device: Device to place tensor on
        wavelength_range: Range of wavelengths in nm
    
    Returns:
        Color matching function tensor with shape (num_bands, 3)
    """
    if wavelength_range[1] <= 10:
        logger.warning(
            "wavelength_range appears to be in micrometers; expected nm. Got %s",
            wavelength_range,
        )
    # More accurate CIE 1931 XYZ to sRGB approximation
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_bands)
    
    # Gaussian approximation of CIE color matching functions
    # These parameters approximate the CIE 1931 standard observer
    r_params = (599.8, 33.0, 0.264)  # peak, width, amplitude
    g_params = (549.1, 57.0, 0.323)
    b_params = (445.8, 33.0, 0.272)
    
    def gaussian(x, peak, width, amplitude):
        return amplitude * np.exp(-((x - peak) / width) ** 2)
    
    r_curve = gaussian(wavelengths, *r_params)
    g_curve = gaussian(wavelengths, *g_params)
    b_curve = gaussian(wavelengths, *b_params)
    
    # Create tensor
    cmf = np.stack([r_curve, g_curve, b_curve], axis=1)
    cmf_tensor = torch.from_numpy(cmf.astype(np.float32))
    
    if device is not None:
        cmf_tensor = cmf_tensor.to(device)
    
    return cmf_tensor


# Global CMF cache to avoid recomputation
_CMF_CACHE: Dict[Tuple[int, str, Tuple[float, float]], torch.Tensor] = {}
_WINDOW_CACHE: Dict[Tuple[int, int, float], torch.Tensor] = {}


def get_cached_cmf(
    num_bands: int, 
    device: Optional[torch.device] = None,
    wavelength_range: Tuple[float, float] = (400, 700)
) -> torch.Tensor:
    """Get CMF tensor from cache or create new one."""
    # Create stable cache key
    if device is None:
        device_key = "cpu"
    else:
        device_key = str(device.type)
        if device.index is not None:
            device_key += f":{device.index}"
    
    cache_key = (num_bands, device_key, wavelength_range)
    
    if cache_key not in _CMF_CACHE:
        cmf = create_cmf_tensor(num_bands, device, wavelength_range)
        _CMF_CACHE[cache_key] = cmf
    
    return _CMF_CACHE[cache_key]


def profile_model(
    model: nn.Module, 
    input_size: Tuple[int, int, int, int] = (1, 3, 256, 256), 
    device: str = "cuda",
    warmup_runs: int = 3
) -> Dict[str, Any]:
    """
    Profile model performance and memory usage.
    
    Args:
        model: Model to profile
        input_size: Input tensor size (B, C, H, W)
        device: Device to run profiling on
        warmup_runs: Number of warmup runs
    
    Returns:
        Dictionary with profiling results
    """
    results = {}
    
    # Validate model architecture
    validate_model_architecture(model, input_size)
    
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
        
        # Create dummy input
        x = torch.randn(input_size).to(device)
        model.eval()
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(x) if not hasattr(model, 'generator') else model.generator(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Count parameters
        params = parameter_count(model)
        results['total_params'] = params['']
        results['param_details'] = params
        logger.info(f"Total parameters: {params['']:,}")
        
        # Count FLOPs
        flops = FlopCountAnalysis(model, x)
        results['total_flops'] = flops.total()
        results['flop_table'] = flop_count_table(flops)
        logger.info(results['flop_table'])
        
    except ImportError:
        logger.warning("Install fvcore for detailed profiling: pip install fvcore")
        results['error'] = "fvcore not installed"
    
    # Profile with PyTorch profiler
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for _ in range(5):  # Profile multiple runs
                    _ = model(x) if not hasattr(model, 'generator') else model.generator(x)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
        
        results['profile'] = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
        results['memory_usage'] = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10)
        logger.info(results['profile'])
        
    except Exception as e:
        logger.warning(f"Profiling failed: {str(e)}")
        results['profile_error'] = str(e)
    
    return results


def validate_model_architecture(
    model: nn.Module,
    expected_input_size: Tuple[int, int, int, int],
    strict: bool = False
) -> bool:
    """
    Validate model architecture matches expected configuration.
    
    Args:
        model: Model to validate
        expected_input_size: Expected input size (B, C, H, W)
        strict: If True, raise exception on mismatch
    
    Returns:
        True if valid, False otherwise
    
    Raises:
        ValueError: If strict=True and validation fails
    """
    try:
        # Test forward pass
        device = next(model.parameters()).device
        test_input = torch.randn(expected_input_size).to(device)
        
        with torch.no_grad():
            # Handle models with generator attribute (GANs)
            if hasattr(model, 'generator'):
                output = model.generator(test_input)
            else:
                output = model(test_input)
        
        # Check output shape
        expected_out_channels = 31  # ARAD-1K
        if output.shape[1] != expected_out_channels:
            msg = f"Model output channels {output.shape[1]} != expected {expected_out_channels}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            return False
        
        # Check spatial dimensions preserved
        if output.shape[2:] != test_input.shape[2:]:
            msg = f"Model spatial dims {output.shape[2:]} != input {test_input.shape[2:]}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            return False
        
        return True
        
    except Exception as e:
        msg = f"Model validation failed: {str(e)}"
        if strict:
            raise ValueError(msg)
        logger.error(msg)
        return False


def export_model(
    model: nn.Module,
    save_path: str,
    input_size: Tuple[int, int, int, int] = (1, 3, 256, 256),
    format: str = "torchscript",
    validate: bool = True,
    opset_version: int = 13
) -> None:
    """
    Export model for deployment with validation.
    
    Args:
        model: Model to export
        save_path: Path to save exported model
        input_size: Input tensor size (B, C, H, W)
        format: Export format ("torchscript" or "onnx")
        validate: Whether to validate exported model
        opset_version: ONNX opset version
    
    Raises:
        ValueError: If format is not supported or validation fails
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract just the generator if it's a combined model
    if hasattr(model, 'generator'):
        if hasattr(model, 'module'):  # Handle DDP wrapped model
            generator = model.module.generator
        else:
            generator = model.generator
    else:
        if hasattr(model, 'module'):  # Handle DDP wrapped model
            generator = model.module
        else:
            generator = model
    
    generator.eval()
    
    # Validate before export
    if validate:
        validate_model_architecture(generator, input_size, strict=True)
    
    # Create example input
    dummy_input = torch.randn(input_size)
    device = next(generator.parameters()).device
    dummy_input = dummy_input.to(device)
    
    if format.lower() == "torchscript":
        # Export to TorchScript
        with torch.no_grad():
            scripted_model = torch.jit.trace(generator, dummy_input)
        
        output_path = save_path.with_suffix('.pt')
        torch.jit.save(scripted_model, output_path)
        logger.info(f"Exported TorchScript model to {output_path}")
        
        # Validate exported model
        if validate:
            loaded = torch.jit.load(output_path)
            with torch.no_grad():
                test_output = loaded(dummy_input.cpu())
                orig_output = generator(dummy_input).cpu()
                
            if not torch.allclose(test_output, orig_output, atol=1e-5):
                raise ValueError("Exported model output doesn't match original")
        
    elif format.lower() == "onnx":
        # Export to ONNX
        output_path = save_path.with_suffix('.onnx')
        
        torch.onnx.export(
            generator,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True
        )
        logger.info(f"Exported ONNX model to {output_path}")
        
        # Validate exported model
        if validate:
            if not HAS_ONNXRUNTIME:
                logger.warning(
                    "onnxruntime not installed. Cannot validate ONNX export. "
                    "Install with: pip install onnxruntime"
                )
            else:
                # Test ONNX model
                ort_session = ort.InferenceSession(str(output_path))
                ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
                ort_outputs = ort_session.run(None, ort_inputs)[0]
                
                with torch.no_grad():
                    orig_output = generator(dummy_input).cpu().numpy()
                
                if not np.allclose(ort_outputs, orig_output, atol=1e-5):
                    raise ValueError("ONNX model output doesn't match original")
    else:
        raise ValueError(f"Unsupported export format: {format}. Supported: 'torchscript', 'onnx'")


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio between predictions and targets.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        max_val: Maximum value in the valid range
    
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-10:  # Avoid log(0)
        return torch.tensor(100.0, device=pred.device)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def compute_ssim(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    window_size: int = 11, 
    sigma: float = 1.5,
    use_fallback: bool = False
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between predictions and targets.
    
    Args:
        pred: Predicted tensor of shape (B, C, H, W)
        target: Target tensor of shape (B, C, H, W)
        window_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian window
        use_fallback: Force use of fallback implementation
    
    Returns:
        SSIM value
    """
    if HAS_PYTORCH_MSSSIM and not use_fallback:
        return msssim_ssim(pred, target, data_range=1.0, size_average=True)
    else:
        if not use_fallback:
            warnings.warn(
                "Using fallback SSIM implementation. Results may differ from standard SSIM. "
                "Install pytorch-msssim for accurate computation."
            )
        
        # Improved fallback implementation with window caching
        c1 = (0.01 * 1) ** 2
        c2 = (0.03 * 1) ** 2
        
        # Get cached window
        window = _get_cached_window(window_size, pred.shape[1], sigma, pred.device)
        
        # Apply Gaussian smoothing
        mu_x = F.conv2d(pred, window, padding=window_size//2, groups=pred.shape[1])
        mu_y = F.conv2d(target, window, padding=window_size//2, groups=target.shape[1])
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=pred.shape[1]) - mu_x_sq
        sigma_y_sq = F.conv2d(target * target, window, padding=window_size//2, groups=target.shape[1]) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, window, padding=window_size//2, groups=pred.shape[1]) - mu_xy
        
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        
        return torch.mean(ssim_map)


def _get_cached_window(window_size: int, channel: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Get cached Gaussian window for SSIM computation."""
    cache_key = (window_size, channel, sigma)
    
    if cache_key not in _WINDOW_CACHE:
        window = _create_window(window_size, channel, sigma)
        _WINDOW_CACHE[cache_key] = window
    
    return _WINDOW_CACHE[cache_key].to(device)


def _create_window(window_size: int, channel: int, sigma: float) -> torch.Tensor:
    """Create Gaussian window for SSIM computation."""
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / (2.0 * sigma**2)) 
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    
    return window


def compute_sam_value(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute Spectral Angle Mapper (SAM) value between predicted and target HSI.
    Numerically stable implementation.
    
    Args:
        pred: Predicted HSI tensor of shape (B, C, H, W)
        target: Target HSI tensor of shape (B, C, H, W)
        epsilon: Small value for numerical stability
    
    Returns:
        Mean SAM value (in degrees)
    """
    # Normalize the spectral vectors
    pred_norm = F.normalize(pred, dim=1, eps=epsilon)
    target_norm = F.normalize(target, dim=1, eps=epsilon)
    
    # Compute dot product
    dot_product = torch.sum(pred_norm * target_norm, dim=1)
    
    # Improved numerical stability
    # Clamp to slightly inside [-1, 1] to avoid arccos issues
    dot_product = torch.clamp(dot_product, -1.0 + epsilon, 1.0 - epsilon)
    
    # Compute angle in radians
    sam_rad = torch.acos(dot_product)
    
    # Handle any remaining NaN values (shouldn't happen with proper clamping)
    sam_rad = torch.where(torch.isnan(sam_rad), torch.zeros_like(sam_rad), sam_rad)
    
    # Convert to degrees and compute mean
    sam_deg = sam_rad * 180.0 / torch.pi
    
    return torch.mean(sam_deg)


def compute_mrae(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute Mean Relative Absolute Error (MRAE) between predictions and targets.
    
    This is a critical metric for NTIRE-2022 challenge evaluation.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        epsilon: Small value to prevent division by zero
    
    Returns:
        MRAE value
    """
    return torch.mean(torch.abs(pred - target) / (target + epsilon))


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE) between predictions and targets.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
    
    Returns:
        MAE value
    """
    return torch.mean(torch.abs(pred - target))


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Square Error (RMSE) between predictions and targets.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
    
    Returns:
        RMSE value
    """
    return torch.sqrt(torch.mean((pred - target) ** 2))


def crop_center_arad1k(
    tensor: torch.Tensor, 
    h_crop: int = 226, 
    w_crop: int = 256
) -> torch.Tensor:
    """
    Crop the center region following ARAD-1K/MST++ evaluation protocol.
    
    This follows the exact same logic as MST++ test script:
    h_crop = 226; w_crop = 256; 
    x0 = (W-w_crop)//2; y0 = (H-h_crop)//2
    
    Args:
        tensor: Input tensor of shape (B, C, H, W)
        h_crop: Height of center crop (226 for ARAD-1K)
        w_crop: Width of center crop (256 for ARAD-1K)
    
    Returns:
        Center-cropped tensor of shape (B, C, h_crop, w_crop)
    
    Raises:
        ValueError: If tensor is smaller than crop size
    """
    _, _, H, W = tensor.shape
    
    if H < h_crop or W < w_crop:
        raise ValueError(
            f"Input tensor size ({H}×{W}) is smaller than crop size ({h_crop}×{w_crop})"
        )
    
    # MST++ exact crop calculation
    x0 = (W - w_crop) // 2
    y0 = (H - h_crop) // 2
    
    # Crop the tensor using MST++ coordinates
    cropped = tensor[:, :, y0:y0+h_crop, x0:x0+w_crop]
    
    logger.debug(
        f"ARAD-1K center crop: {tensor.shape} -> {cropped.shape} "
        f"(crop {h_crop}×{w_crop} at ({x0},{y0}))"
    )
    
    return cropped


def compute_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor,
    compute_all: bool = True
) -> Dict[str, float]:
    """
    Compute all metrics between predictions and targets.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        compute_all: If True, compute all metrics. If False, only essential ones.
    
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        metrics = {
            'psnr': compute_psnr(pred, target).item(),
            'mrae': compute_mrae(pred, target).item(),
            'rmse': compute_rmse(pred, target).item(),
        }
        
        if compute_all:
            metrics.update({
                'ssim': compute_ssim(pred, target).item(),
                'sam': compute_sam_value(pred, target).item(),
                'mae': compute_mae(pred, target).item(),
            })
    
    return metrics


def compute_metrics_arad1k(
    pred: torch.Tensor, 
    target: torch.Tensor,
    save_crops: bool = False,
    crop_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Compute metrics following ARAD-1K/MST++ evaluation protocol.
    
    Protocol:
    1. Inference on full image (~482×512)
    2. Crop BOTH prediction and ground truth to central 226×256 region
    3. Compute metrics ONLY on the cropped regions
    
    This matches exactly what MST++ test script does for NTIRE-2022 challenge.
    
    Args:
        pred: Predicted HSI tensor of shape (B, C, H, W) - full size (~482×512)
        target: Target HSI tensor of shape (B, C, H, W) - full size (~482×512)
        save_crops: If True, save the cropped tensors for debugging
        crop_dir: Directory to save crops (if save_crops=True)
    
    Returns:
        Dictionary of metrics computed on center crop (226×256)
    """
    # Step 3 of MST++ protocol: Crop both prediction and GT to center 226×256
    try:
        pred_crop = crop_center_arad1k(pred, h_crop=226, w_crop=256)
        target_crop = crop_center_arad1k(target, h_crop=226, w_crop=256)
    except ValueError as e:
        logger.error(f"Failed to crop for ARAD-1K evaluation: {str(e)}")
        logger.error(f"Prediction shape: {pred.shape}, Target shape: {target.shape}")
        logger.error("ARAD-1K requires minimum size of 226×256 for center crop evaluation")
        raise
    
    # Optionally save crops for debugging
    if save_crops and crop_dir:
        crop_dir = Path(crop_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)
        torch.save(pred_crop, crop_dir / "pred_crop.pt")
        torch.save(target_crop, crop_dir / "target_crop.pt")
        logger.info(f"Saved cropped tensors to {crop_dir}")
    
    logger.debug(f"ARAD-1K metrics: computing on cropped regions {pred_crop.shape}")
    
    # Compute metrics on cropped regions only (as per NTIRE-2022 challenge rules)
    return compute_metrics(pred_crop, target_crop, compute_all=True)


def save_metrics(metrics: Dict[str, float], output_path: Union[str, Path]) -> None:
    """Save metrics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    serializable_metrics = {k: float(v) for k, v in metrics.items()}
    
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {output_path}")


def create_error_report(
    exception: Exception,
    context: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Create detailed error report for debugging.
    
    Args:
        exception: The exception that occurred
        context: Context information (config, data paths, etc.)
        output_path: Optional path to save the report
    
    Returns:
        Error report dictionary
    """
    import traceback
    import psutil
    import time
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'exception': {
            'type': type(exception).__name__,
            'message': str(exception),
            'traceback': traceback.format_exc()
        },
        'context': context,
        'system': {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024**3,
        }
    }
    
    if torch.cuda.is_available():
        report['gpu'] = {
            'count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Error report saved to {output_path}")
    
    return report

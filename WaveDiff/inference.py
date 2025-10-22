import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Import model implementations
from models.base_model import HSILatentDiffusionModel
from models.wavelet_model import WaveletHSILatentDiffusionModel
from models.adaptive_model import AdaptiveWaveletHSILatentDiffusionModel

# Import evaluation metrics
from utils.spectral_utils import (
    spectral_angular_mapper,
    root_mean_square_error,
    peak_signal_to_noise_ratio,
    mean_relative_absolute_error,
    calculate_metrics
)

# Import visualization utilities
from utils.visualization import (
    visualize_rgb_hsi,
    visualize_reconstruction_comparison,
    visualize_spectral_signature,
    create_false_color_image
)

def load_model(checkpoint_path, device, model_type=None):
    """
    Load a model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        model_type: Optional model type to override checkpoint
        
    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', {})
    
    # Override model type if specified
    if model_type is not None:
        config['model_type'] = model_type
    elif 'model_type' not in config:
        # Try to infer model type from filename
        if 'wavelet' in checkpoint_path.lower() and 'adaptive' in checkpoint_path.lower():
            config['model_type'] = 'adaptive_wavelet'
        elif 'wavelet' in checkpoint_path.lower():
            config['model_type'] = 'wavelet'
        else:
            config['model_type'] = 'base'
    
    # Create model based on type
    latent_dim = config.get('latent_dim', 64)
    timesteps = config.get('timesteps', 1000)
    use_batchnorm = config.get('use_batchnorm', True)
    
    refinement_config = config.get('refinement_config')

    if config['model_type'] == 'base':
        print("Loading Base HSI Latent Diffusion Model")
        model = HSILatentDiffusionModel(
            latent_dim=latent_dim,
            out_channels=31,  # 31 HSI bands output for ARAD-1K
            timesteps=timesteps,
            use_batchnorm=use_batchnorm,
            refinement_config=refinement_config
        ).to(device)
    elif config['model_type'] == 'wavelet':
        print("Loading Wavelet-enhanced HSI Latent Diffusion Model")
        model = WaveletHSILatentDiffusionModel(
            latent_dim=latent_dim,
            out_channels=31,
            timesteps=timesteps,
            use_batchnorm=use_batchnorm,
            refinement_config=refinement_config
        ).to(device)
    elif config['model_type'] == 'adaptive_wavelet':
        print("Loading Adaptive Wavelet HSI Latent Diffusion Model")
        threshold_method = config.get('threshold_method', 'soft')
        init_threshold = config.get('init_threshold', 0.1)
        trainable_threshold = config.get('trainable_threshold', True)

        model = AdaptiveWaveletHSILatentDiffusionModel(
            latent_dim=latent_dim,
            out_channels=31,
            timesteps=timesteps,
            use_batchnorm=use_batchnorm,
            threshold_method=threshold_method,
            init_threshold=init_threshold,
            trainable_threshold=trainable_threshold,
            refinement_config=refinement_config
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model, config


def preprocess_image(image_path, image_size=256):
    """
    Preprocess an image for inference
    
    Args:
        image_path: Path to image file
        image_size: Size to resize image to
        
    Returns:
        Preprocessed image tensor
    """
    from torchvision import transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def load_hsi_ground_truth(path):
    """
    Load HSI ground truth data
    
    Args:
        path: Path to HSI data file (npy or mat)
        
    Returns:
        HSI tensor or None if file not found
    """
    import scipy.io as sio
    
    # Try different file extensions
    path = Path(path)
    npy_path = path.with_suffix('.npy')
    mat_path = path.with_suffix('.mat')
    
    if npy_path.exists():
        # Load from numpy file
        hsi_data = np.load(npy_path)
        hsi_tensor = torch.from_numpy(hsi_data).float()
        
        # Ensure channel first format (C, H, W)
        if hsi_tensor.shape[0] != 31:
            hsi_tensor = hsi_tensor.permute(2, 0, 1)
            
        return hsi_tensor.unsqueeze(0)  # Add batch dimension
    
    elif mat_path.exists():
        # Load from mat file
        mat_data = sio.loadmat(mat_path)
        
        # Extract HSI data (adjust key as needed)
        if 'cube' in mat_data:
            hsi_data = mat_data['cube']
        elif 'data' in mat_data:
            hsi_data = mat_data['data']
        else:
            # Try first array in mat file
            for k, v in mat_data.items():
                if isinstance(v, np.ndarray) and len(v.shape) == 3:
                    hsi_data = v
                    break
            else:
                print(f"Could not find HSI data in {mat_path}")
                return None
        
        # Convert to tensor
        hsi_tensor = torch.from_numpy(hsi_data).float()
        
        # Ensure channel first format (C, H, W)
        if hsi_tensor.shape[0] != 31:
            hsi_tensor = hsi_tensor.permute(2, 0, 1)
            
        return hsi_tensor.unsqueeze(0)  # Add batch dimension
    
    # No ground truth found
    return None


def run_inference(model, rgb_tensor, device, sampling_steps=20, apply_adaptive_threshold=True):
    """
    Run inference on an RGB image
    
    Args:
        model: Model to use for inference
        rgb_tensor: Preprocessed RGB image tensor
        device: Device to run inference on
        sampling_steps: Number of sampling steps for DPM Solver
        apply_adaptive_threshold: Whether to apply adaptive thresholding (for adaptive models)
        
    Returns:
        Tuple of (predicted HSI tensor, stage_outputs_dict)
    """
    # Move tensor to device
    rgb_tensor = rgb_tensor.to(device)
    
    # Run inference
    stage_outputs = {}
    with torch.no_grad():
        # Check if model has dedicated inference method
        if hasattr(model, 'rgb_to_hsi'):
            # For adaptive models, use the method that applies thresholding
            if isinstance(model, AdaptiveWaveletHSILatentDiffusionModel):
                result = model.rgb_to_hsi(
                    rgb_tensor,
                    sampling_steps=sampling_steps,
                    apply_adaptive_threshold=apply_adaptive_threshold,
                    return_stages=True
                )
            else:
                result = model.rgb_to_hsi(
                    rgb_tensor,
                    sampling_steps=sampling_steps,
                    return_stages=True
                )

            if isinstance(result, tuple) and len(result) == 2:
                hsi_output, stage_outputs = result
            else:
                hsi_output = result
        else:
            # Manual inference for other models
            latent = model.encode(rgb_tensor)

            # Sample latent using DPM Solver
            sampled_latent = model.dpm_ot.sample(
                latent.shape,
                latent.device,
                use_dpm_solver=True,
                steps=sampling_steps
            )

            # Decode to HSI
            hsi_output = model.decode(sampled_latent)
            stage_outputs = {'initial': hsi_output, 'final': hsi_output}

    if 'final' not in stage_outputs:
        stage_outputs['final'] = hsi_output

    return hsi_output, stage_outputs


def save_results(rgb_tensor, hsi_output, hsi_gt, output_dir, filename, metrics=None, stage_outputs=None):
    """
    Save inference results
    
    Args:
        rgb_tensor: Input RGB tensor
        hsi_output: Predicted HSI tensor
        hsi_gt: Ground truth HSI tensor (or None)
        output_dir: Directory to save results
        filename: Base filename for outputs
        metrics: Optional dictionary of evaluation metrics
        stage_outputs: Optional dictionary of intermediate stage tensors
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy
    rgb_np = rgb_tensor.squeeze(0).cpu().numpy()
    hsi_np = hsi_output.squeeze(0).cpu().numpy()
    
    # Save HSI as numpy array
    np.save(os.path.join(output_dir, f"{filename}_hsi.npy"), hsi_np)

    # Optionally save intermediate stage outputs for analysis
    if stage_outputs is not None:
        for stage_name, stage_tensor in stage_outputs.items():
            if stage_name == 'final':
                continue
            stage_np = stage_tensor.squeeze(0).cpu().numpy()
            np.save(os.path.join(output_dir, f"{filename}_{stage_name}_hsi.npy"), stage_np)
    
    # Create false color visualization using selected bands
    save_path = os.path.join(output_dir, f"{filename}_false_color.png")
    create_false_color_image(
        hsi_np, band_indices=[0, 15, 30],
        save_path=save_path, gamma=0.8
    )
    
    # Visualize RGB input and selected HSI bands
    save_path = os.path.join(output_dir, f"{filename}_rgb_hsi_bands.png")
    visualize_rgb_hsi(
        rgb_np, hsi_np,
        save_path=save_path,
        title=f"RGB Input and Selected HSI Bands for {filename}"
    )
    
    # If ground truth is available, create comparison
    if hsi_gt is not None:
        hsi_gt_np = hsi_gt.squeeze(0).cpu().numpy()
        
        # Save comparison visualization
        save_path = os.path.join(output_dir, f"{filename}_comparison.png")
        visualize_reconstruction_comparison(
            rgb_np, hsi_gt_np, hsi_np,
            save_path=save_path,
            error_metric=metrics
        )
        
        # Visualize spectral signatures at center point
        h, w = hsi_np.shape[1:]
        center_point = (h//2, w//2)
        
        save_path = os.path.join(output_dir, f"{filename}_spectral_signature.png")
        visualize_spectral_signature(
            [hsi_gt_np, hsi_np],  # List of HSI data to compare
            [center_point],  # Points to sample
            wavelengths=np.linspace(400, 700, hsi_np.shape[0]),  # Approximate wavelengths
            save_path=save_path,
            title=f"Spectral Signature Comparison at Center Point",
            labels=["Ground Truth", "Predicted"]
        )


def evaluate_metrics(hsi_output, hsi_gt):
    """
    Calculate evaluation metrics between predicted and ground truth HSI
    
    Args:
        hsi_output: Predicted HSI tensor
        hsi_gt: Ground truth HSI tensor
        
    Returns:
        Dictionary of metrics
    """
    # Ensure both tensors are on the same device
    device = hsi_output.device
    hsi_gt = hsi_gt.to(device)
    
    # Calculate metrics
    metrics = {}
    
    # RMSE
    metrics['rmse'] = root_mean_square_error(hsi_output, hsi_gt).item()
    
    # PSNR
    metrics['psnr'] = peak_signal_to_noise_ratio(hsi_output, hsi_gt).item()
    
    # SAM (Spectral Angle Mapper)
    metrics['sam'] = spectral_angular_mapper(hsi_output, hsi_gt).item()
    
    # MRAE (Mean Relative Absolute Error)
    metrics['mrae'] = mean_relative_absolute_error(hsi_output, hsi_gt).item()
    
    return metrics


def process_directory(input_dir, model, device, output_dir, sampling_steps=20, 
                      apply_adaptive_threshold=True, batch_size=1):
    """
    Process all RGB images in a directory
    
    Args:
        input_dir: Directory containing RGB images
        model: Model to use for inference
        device: Device to run inference on
        output_dir: Directory to save results
        sampling_steps: Number of sampling steps for DPM Solver
        apply_adaptive_threshold: Whether to apply adaptive thresholding (for adaptive models)
        batch_size: Batch size for processing (currently only supports batch_size=1)
    """
    from torchvision import transforms
    from PIL import Image
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f"*{ext}")))
    
    # Check if we have any images to process
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create results file for metrics
    results_file = os.path.join(output_dir, "metrics.txt")
    results_summary = {}
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Get base filename (without extension)
        filename = img_path.stem
        
        # Load and preprocess image
        try:
            rgb_tensor = preprocess_image(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
        
        # Try to load ground truth HSI data if available
        gt_dir = Path(input_dir).parent / "HSI"
        if gt_dir.exists():
            hsi_gt = load_hsi_ground_truth(gt_dir / filename)
        else:
            hsi_gt = None
        
        # Run inference
        hsi_output, stage_outputs = run_inference(
            model, rgb_tensor, device,
            sampling_steps=sampling_steps,
            apply_adaptive_threshold=apply_adaptive_threshold
        )

        # Calculate metrics if ground truth is available
        metrics = None
        stage_metrics = None
        if hsi_gt is not None:
            metrics = evaluate_metrics(hsi_output, hsi_gt)
            stage_metrics = {}
            for stage_name, stage_tensor in stage_outputs.items():
                if stage_name == 'final':
                    continue
                stage_metrics[stage_name] = evaluate_metrics(stage_tensor, hsi_gt)

            # Save metrics to results dictionary
            results_summary[filename] = {
                'final': metrics,
                'stages': stage_metrics
            }

            # Print metrics
            print(f"Metrics for {filename}:")
            for k, v in metrics.items():
                print(f"  final_{k}: {v:.6f}")
            for stage_name, stage_vals in stage_metrics.items():
                for k, v in stage_vals.items():
                    print(f"  {stage_name}_{k}: {v:.6f}")
            print()

        # Save results
        save_results(
            rgb_tensor, hsi_output, hsi_gt,
            output_dir, filename, metrics,
            stage_outputs=stage_outputs
        )
    
    # Calculate and save average metrics
    if results_summary:
        avg_metrics = {
            'rmse': np.mean([m['final']['rmse'] for m in results_summary.values()]),
            'psnr': np.mean([m['final']['psnr'] for m in results_summary.values()]),
            'sam': np.mean([m['final']['sam'] for m in results_summary.values()]),
            'mrae': np.mean([m['final']['mrae'] for m in results_summary.values()])
        }
        
        # Save detailed results
        with open(results_file, 'w') as f:
            f.write("# HSI Reconstruction Metrics\n\n")
            
            # Write average metrics
            f.write("## Average Metrics\n\n")
            for k, v in avg_metrics.items():
                f.write(f"{k}: {v:.6f}\n")
            f.write("\n")
            
            # Write per-image metrics
            f.write("## Per-image Metrics\n\n")
            for filename, metrics in results_summary.items():
                f.write(f"### {filename}\n")
                for k, v in metrics['final'].items():
                    f.write(f"final_{k}: {v:.6f}\n")
                for stage_name, stage_vals in metrics['stages'].items():
                    for k, v in stage_vals.items():
                        f.write(f"{stage_name}_{k}: {v:.6f}\n")
                f.write("\n")
        
        print("\nAverage Metrics:")
        for k, v in avg_metrics.items():
            print(f"  {k}: {v:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with HSI model")
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['base', 'wavelet', 'adaptive_wavelet'],
                        help='Model type (overrides checkpoint)')
    parser.add_argument('--sampling_steps', type=int, default=20,
                        help='Number of sampling steps for DPM Solver')
    parser.add_argument('--adaptive_threshold', dest='adaptive_threshold', action='store_true',
                        help='Apply adaptive thresholding (for adaptive models)')
    parser.add_argument('--no_adaptive_threshold', dest='adaptive_threshold', action='store_false',
                        help='Disable adaptive thresholding')
    
    # Input parameters
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, default=None,
                      help='Path to input RGB image')
    group.add_argument('--input_dir', type=str, default=None,
                      help='Directory containing RGB images')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='Path to ground truth HSI data (optional)')
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing (only for directory mode)')
    
    # Parse arguments
    parser.set_defaults(adaptive_threshold=True)
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device, args.model_type)
    print(f"Model loaded from {args.checkpoint}")
    
    # Process single image or directory
    if args.image:
        # Process single image
        print(f"Processing image: {args.image}")
        
        # Load and preprocess image
        rgb_tensor = preprocess_image(args.image)
        
        # Get base filename
        filename = Path(args.image).stem
        
        # Try to load ground truth HSI data if provided
        hsi_gt = None
        if args.gt_path:
            hsi_gt = load_hsi_ground_truth(args.gt_path)
        
        # Run inference
        hsi_output, stage_outputs = run_inference(
            model, rgb_tensor, device,
            sampling_steps=args.sampling_steps,
            apply_adaptive_threshold=args.adaptive_threshold
        )

        # Calculate metrics if ground truth is available
        metrics = None
        stage_metrics = None
        if hsi_gt is not None:
            metrics = evaluate_metrics(hsi_output, hsi_gt)
            stage_metrics = {}
            for stage_name, stage_tensor in stage_outputs.items():
                if stage_name == 'final':
                    continue
                stage_metrics[stage_name] = evaluate_metrics(stage_tensor, hsi_gt)

            # Print metrics
            print("Evaluation Metrics:")
            for k, v in metrics.items():
                print(f"  final_{k}: {v:.6f}")
            for stage_name, stage_vals in stage_metrics.items():
                for k, v in stage_vals.items():
                    print(f"  {stage_name}_{k}: {v:.6f}")

        # Save results
        save_results(
            rgb_tensor, hsi_output, hsi_gt,
            args.output_dir, filename, metrics,
            stage_outputs=stage_outputs
        )
        
        print(f"Results saved to {args.output_dir}")
        
    else:
        # Process directory
        print(f"Processing directory: {args.input_dir}")
        
        process_directory(
            args.input_dir, model, device, 
            args.output_dir,
            sampling_steps=args.sampling_steps,
            apply_adaptive_threshold=args.adaptive_threshold,
            batch_size=args.batch_size
        )
        
        print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
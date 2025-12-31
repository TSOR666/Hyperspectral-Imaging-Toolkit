"""
MSWR-Net v2.1.2 - NTIRE 2022 Challenge Testing Protocol
========================================================

Comprehensive evaluation suite following NTIRE 2022 Spectral Reconstruction Challenge:
- Standard metrics: MRAE, RMSE, PSNR, SAM, SSIM
- Per-band and per-pixel analysis
- Statistical significance testing
- Visualization and reporting
- Comparison with baselines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import yaml
from tqdm import tqdm
import h5py
import scipy.io as sio
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import model and utilities
from model.mswr_net_v212 import (
    IntegratedMSWRNet,
    MSWRDualConfig,
    create_mswr_tiny,
    create_mswr_small,
    create_mswr_base,
    create_mswr_large
)

from utils import (
    Loss_MRAE,
    Loss_RMSE,
    Loss_PSNR,
    Loss_SAM,
    AverageMeter
)

# Configure plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """NTIRE 2022 Testing Configuration"""
    # Model settings
    model_path: str
    model_size: str = 'base'
    device: str = 'cuda'
    
    # Data settings
    test_data_path: str = None
    gt_data_path: str = None
    data_format: str = 'mat'  # 'mat', 'h5', 'npy'
    
    # Testing settings
    batch_size: int = 1
    center_crop: bool = True
    crop_size: int = 256
    use_amp: bool = True
    
    # Metrics
    calculate_sam: bool = True
    calculate_ssim: bool = True
    per_band_metrics: bool = True
    per_pixel_metrics: bool = False
    
    # Analysis
    statistical_analysis: bool = True
    compare_baselines: bool = False
    baseline_results_path: str = None
    
    # Visualization
    save_visualizations: bool = True
    save_error_maps: bool = True
    save_spectral_plots: bool = True
    n_visualization_samples: int = 10
    
    # Output
    output_dir: str = './test_results'
    save_predictions: bool = False
    save_format: str = 'mat'
    
    # Advanced
    ensemble_mode: str = None  # None, 'flip', 'rotate', 'full'
    mc_dropout_samples: int = 0  # 0 = disabled, >0 = Monte Carlo dropout
    
    def to_dict(self) -> Dict:
        return asdict(self)

class MetricsCalculator:
    """Comprehensive metrics calculation following NTIRE 2022"""
    
    def __init__(self, calculate_sam: bool = True, calculate_ssim: bool = True):
        self.calculate_sam = calculate_sam
        self.calculate_ssim = calculate_ssim
        
        # Initialize metric functions from utils
        self.mrae_fn = Loss_MRAE()
        self.rmse_fn = Loss_RMSE()
        self.psnr_fn = Loss_PSNR()
        self.sam_fn = Loss_SAM() if calculate_sam else None
        
        # Metric storage
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            'mrae': AverageMeter(),
            'rmse': AverageMeter(),
            'psnr': AverageMeter(),
            'sam': AverageMeter() if self.calculate_sam else None,
            'ssim': AverageMeter() if self.calculate_ssim else None,
        }
        
        self.per_band_metrics = defaultdict(lambda: defaultdict(list))
        self.per_pixel_metrics = defaultdict(list)
        self.raw_errors = []
    
    def calculate_ssim_metric(self, pred: torch.Tensor, target: torch.Tensor, 
                              data_range: float = 1.0) -> torch.Tensor:
        """Calculate SSIM for hyperspectral images"""
        # Calculate SSIM per band and average
        ssim_values = []
        
        for i in range(pred.shape[1]):  # Iterate over channels
            pred_band = pred[:, i:i+1, :, :]
            target_band = target[:, i:i+1, :, :]
            
            # Simple SSIM implementation
            mu1 = F.avg_pool2d(pred_band, 11, 1, 5)
            mu2 = F.avg_pool2d(target_band, 11, 1, 5)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.avg_pool2d(pred_band * pred_band, 11, 1, 5) - mu1_sq
            sigma2_sq = F.avg_pool2d(target_band * target_band, 11, 1, 5) - mu2_sq
            sigma12 = F.avg_pool2d(pred_band * target_band, 11, 1, 5) - mu1_mu2

            # NUMERICAL STABILITY FIX: Use dtype-aware epsilon to handle fp16 underflow
            # In fp16 mode, eps=1e-8 is too small and can cause underflow
            eps = torch.finfo(pred.dtype).eps * 10  # Safety margin for the dtype
            sigma1_sq = torch.clamp(sigma1_sq, min=eps)
            sigma2_sq = torch.clamp(sigma2_sq, min=eps)

            C1 = (0.01 * data_range) ** 2
            C2 = (0.03 * data_range) ** 2

            # SSIM formula with clamped denominator for stability
            numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            ssim_map = numerator / torch.clamp(denominator, min=eps)

            ssim_values.append(ssim_map.mean())
        
        return torch.stack(ssim_values).mean()
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, 
              calculate_per_band: bool = False, calculate_per_pixel: bool = False):
        """Update metrics with new predictions"""
        # Ensure same device
        if pred.device != target.device:
            target = target.to(pred.device)
        
        # Basic metrics
        mrae = self.mrae_fn(pred, target)
        rmse = self.rmse_fn(pred, target)
        psnr = self.psnr_fn(pred, target, data_range=1.0)
        
        self.metrics['mrae'].update(mrae.item())
        self.metrics['rmse'].update(rmse.item())
        self.metrics['psnr'].update(psnr.item())
        
        # SAM (Spectral Angle Mapper)
        if self.calculate_sam and self.sam_fn is not None:
            sam = self.sam_fn(pred, target)
            sam_deg = sam.item() * 180.0 / np.pi  # Convert to degrees
            self.metrics['sam'].update(sam_deg)
        
        # SSIM
        if self.calculate_ssim:
            ssim = self.calculate_ssim_metric(pred, target)
            self.metrics['ssim'].update(ssim.item())
        
        # Per-band metrics
        if calculate_per_band:
            self._calculate_per_band_metrics(pred, target)
        
        # Per-pixel metrics
        if calculate_per_pixel:
            self._calculate_per_pixel_metrics(pred, target)
        
        # Store raw errors for statistical analysis
        error = torch.abs(pred - target).cpu().numpy()
        self.raw_errors.append(error)
    
    def _calculate_per_band_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """Calculate metrics for each spectral band"""
        n_bands = pred.shape[1]
        
        for band_idx in range(n_bands):
            pred_band = pred[:, band_idx:band_idx+1, :, :]
            target_band = target[:, band_idx:band_idx+1, :, :]
            
            # Calculate metrics for this band
            mrae_band = self.mrae_fn(pred_band, target_band).item()
            rmse_band = self.rmse_fn(pred_band, target_band).item()
            psnr_band = self.psnr_fn(pred_band, target_band, data_range=1.0).item()
            
            self.per_band_metrics['mrae'][band_idx].append(mrae_band)
            self.per_band_metrics['rmse'][band_idx].append(rmse_band)
            self.per_band_metrics['psnr'][band_idx].append(psnr_band)
    
    def _calculate_per_pixel_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """Calculate metrics for each pixel"""
        # Reshape to (n_pixels, n_bands)
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
        target_flat = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])

        # Calculate per-pixel relative error with proper handling of near-zero values
        # Use max(target, epsilon) to avoid division by very small values
        # A typical threshold is based on the data range
        eps = 1e-3  # Use a more reasonable epsilon for relative error
        abs_error = torch.abs(pred_flat - target_flat)
        denominator = torch.maximum(torch.abs(target_flat), torch.tensor(eps, device=target.device))
        pixel_mrae = (abs_error / denominator).mean(dim=1)

        pixel_rmse = torch.sqrt(((pred_flat - target_flat) ** 2).mean(dim=1))

        self.per_pixel_metrics['mrae'].extend(pixel_mrae.cpu().numpy().tolist())
        self.per_pixel_metrics['rmse'].extend(pixel_rmse.cpu().numpy().tolist())
    
    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive metrics results"""
        results = {}
        
        # Basic metrics
        for metric_name, meter in self.metrics.items():
            if meter is not None:
                results[metric_name] = {
                    'mean': meter.avg,
                    'count': meter.count,
                    'values': [] # Could store all values if needed
                }
        
        # Per-band statistics
        if self.per_band_metrics:
            results['per_band'] = {}
            for metric_name, band_data in self.per_band_metrics.items():
                band_means = {k: np.mean(v) for k, v in band_data.items()}
                band_stds = {k: np.std(v) for k, v in band_data.items()}
                results['per_band'][metric_name] = {
                    'mean': band_means,
                    'std': band_stds
                }
        
        # Per-pixel statistics
        if self.per_pixel_metrics:
            results['per_pixel'] = {}
            for metric_name, values in self.per_pixel_metrics.items():
                if values:
                    results['per_pixel'][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'percentiles': {
                            '25': np.percentile(values, 25),
                            '50': np.percentile(values, 50),
                            '75': np.percentile(values, 75),
                            '95': np.percentile(values, 95),
                            '99': np.percentile(values, 99)
                        }
                    }
        
        return results

class DataLoader:
    """Data loader for NTIRE 2022 test data"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_files = []
        self.gt_files = []
        
        self._load_file_lists()
    
    def _load_file_lists(self):
        """Load test and ground truth file lists"""
        test_path = Path(self.config.test_data_path)
        
        if self.config.data_format == 'mat':
            self.test_files = sorted(test_path.glob('*.mat'))
        elif self.config.data_format == 'h5':
            self.test_files = sorted(test_path.glob('*.h5'))
        elif self.config.data_format == 'npy':
            self.test_files = sorted(test_path.glob('*.npy'))
        
        # Match ground truth files if provided
        if self.config.gt_data_path:
            gt_path = Path(self.config.gt_data_path)
            for test_file in self.test_files:
                gt_file = gt_path / test_file.name
                if gt_file.exists():
                    self.gt_files.append(gt_file)
                else:
                    # Try alternative naming patterns
                    alternatives = [
                        gt_path / test_file.stem.replace('rgb', 'hsi') + test_file.suffix,
                        gt_path / test_file.stem.replace('RGB', 'HSI') + test_file.suffix,
                        gt_path / (test_file.stem + '_gt' + test_file.suffix)
                    ]
                    for alt in alternatives:
                        if alt.exists():
                            self.gt_files.append(alt)
                            break
                    else:
                        logger.warning(f"No ground truth found for {test_file.name}")
                        self.gt_files.append(None)
        
        logger.info(f"Found {len(self.test_files)} test files")
        logger.info(f"Found {len([f for f in self.gt_files if f is not None])} ground truth files")
    
    def load_data(self, file_path: Path) -> np.ndarray:
        """Load data from file"""
        if self.config.data_format == 'mat':
            mat_data = sio.loadmat(str(file_path))
            # Try common keys
            for key in ['data', 'img', 'image', 'rgb', 'RGB', 'hsi', 'HSI']:
                if key in mat_data:
                    return mat_data[key].astype(np.float32)
            # Use first non-metadata key
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    return value.astype(np.float32)
                    
        elif self.config.data_format == 'h5':
            with h5py.File(str(file_path), 'r') as f:
                # Try common keys
                for key in ['data', 'img', 'image', 'rgb', 'RGB', 'hsi', 'HSI']:
                    if key in f:
                        return f[key][:].astype(np.float32)
                # Use first dataset
                key = list(f.keys())[0]
                return f[key][:].astype(np.float32)
                
        elif self.config.data_format == 'npy':
            return np.load(str(file_path)).astype(np.float32)
        
        raise ValueError(f"Unable to load data from {file_path}")
    
    def preprocess(self, data: np.ndarray, is_rgb: bool = True) -> torch.Tensor:
        """Preprocess data for model input"""
        # Normalize to [0, 1] if needed
        if data.max() > 1.0:
            data = data / 255.0
        
        # Handle different input shapes
        if len(data.shape) == 2:
            # Single channel
            data = np.stack([data] * 3, axis=-1) if is_rgb else data[:, :, np.newaxis]
        elif len(data.shape) == 3:
            if data.shape[0] in [3, 31]:
                # Channel first
                data = np.transpose(data, (1, 2, 0))
        
        # Center crop if specified
        if self.config.center_crop and self.config.crop_size > 0:
            h, w = data.shape[:2]
            crop_h = min(self.config.crop_size, h)
            crop_w = min(self.config.crop_size, w)
            
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            
            data = data[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Convert to tensor
        tensor = torch.from_numpy(data.transpose(2, 0, 1)).float()
        
        # Add batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def __len__(self) -> int:
        return len(self.test_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
        """Get test sample and optionally ground truth"""
        test_file = self.test_files[idx]
        test_data = self.load_data(test_file)
        test_tensor = self.preprocess(test_data, is_rgb=True)
        
        gt_tensor = None
        if idx < len(self.gt_files) and self.gt_files[idx] is not None:
            gt_data = self.load_data(self.gt_files[idx])
            gt_tensor = self.preprocess(gt_data, is_rgb=False)
        
        return test_tensor, gt_tensor, test_file.stem

class Visualizer:
    """Comprehensive visualization for test results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.output_dir / 'plots'
        self.error_maps_dir = self.output_dir / 'error_maps'
        self.spectral_plots_dir = self.output_dir / 'spectral_plots'
        
        for dir_path in [self.plots_dir, self.error_maps_dir, self.spectral_plots_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def plot_metrics_summary(self, results: Dict[str, Any]):
        """Create comprehensive metrics summary plot"""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Main metrics bar plot
        ax1 = fig.add_subplot(gs[0, :2])
        metrics = ['MRAE', 'RMSE', 'PSNR', 'SAM', 'SSIM']
        values = []
        for metric in metrics:
            if metric.lower() in results:
                values.append(results[metric.lower()]['mean'])
            else:
                values.append(0)
        
        bars = ax1.bar(metrics, values, color=sns.color_palette("husl", len(metrics)))
        ax1.set_ylabel('Metric Value')
        ax1.set_title('Overall Metrics Summary')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}' if val < 10 else f'{val:.2f}',
                    ha='center', va='bottom')
        
        # Per-band MRAE plot
        if 'per_band' in results and 'mrae' in results['per_band']:
            ax2 = fig.add_subplot(gs[1, :])
            band_mrae = results['per_band']['mrae']['mean']
            bands = list(band_mrae.keys())
            band_values = [band_mrae[b] for b in bands]
            wavelengths = np.linspace(400, 700, len(bands))
            
            ax2.plot(wavelengths, band_values, 'b-', linewidth=2)
            ax2.fill_between(wavelengths, band_values, alpha=0.3)
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('MRAE')
            ax2.set_title('Per-Band MRAE')
            ax2.grid(True, alpha=0.3)
        
        # Error distribution
        ax3 = fig.add_subplot(gs[2, 0])
        if 'per_pixel' in results and 'mrae' in results['per_pixel']:
            pixel_errors = results['per_pixel']['mrae']
            ax3.hist(pixel_errors.get('values', []), bins=50, edgecolor='black', alpha=0.7)
            ax3.set_xlabel('MRAE')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Pixel Error Distribution')
            ax3.grid(axis='y', alpha=0.3)
        
        # Metrics comparison radar chart
        ax4 = fig.add_subplot(gs[0, 2], projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values_norm = []
        for metric, val in zip(metrics, values):
            if metric == 'PSNR':
                values_norm.append(val / 50)  # Normalize PSNR
            elif metric == 'SSIM':
                values_norm.append(val)
            else:
                values_norm.append(1 - val if val <= 1 else 0)  # Invert error metrics
        
        values_norm += values_norm[:1]  # Complete the circle
        angles += angles[:1]
        
        ax4.plot(angles, values_norm, 'o-', linewidth=2)
        ax4.fill(angles, values_norm, alpha=0.25)
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0, 1)
        ax4.set_title('Metrics Radar Chart')
        
        # Statistical summary table
        ax5 = fig.add_subplot(gs[2, 1:])
        ax5.axis('tight')
        ax5.axis('off')
        
        table_data = []
        for metric in metrics:
            if metric.lower() in results:
                val = results[metric.lower()]['mean']
                table_data.append([metric, f'{val:.6f}'])
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.suptitle('MSWR-Net Test Results - NTIRE 2022 Protocol', fontsize=16)
        plt.tight_layout()
        
        save_path = self.plots_dir / 'metrics_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics summary: {save_path}")
    
    def plot_error_maps(self, pred: np.ndarray, target: np.ndarray, 
                       name: str, n_bands: int = 6):
        """Plot error maps for visualization"""
        fig, axes = plt.subplots(3, n_bands, figsize=(n_bands*3, 9))
        
        # Select bands to visualize
        total_bands = pred.shape[2]
        band_indices = np.linspace(0, total_bands-1, n_bands, dtype=int)
        wavelengths = np.linspace(400, 700, total_bands)
        
        for idx, band_idx in enumerate(band_indices):
            # Prediction
            axes[0, idx].imshow(pred[:, :, band_idx], cmap='viridis', vmin=0, vmax=1)
            axes[0, idx].set_title(f'{wavelengths[band_idx]:.0f}nm')
            axes[0, idx].axis('off')
            
            # Ground truth
            axes[1, idx].imshow(target[:, :, band_idx], cmap='viridis', vmin=0, vmax=1)
            axes[1, idx].axis('off')
            
            # Error map
            error = np.abs(pred[:, :, band_idx] - target[:, :, band_idx])
            im = axes[2, idx].imshow(error, cmap='hot', vmin=0, vmax=0.1)
            axes[2, idx].axis('off')
        
        axes[0, 0].set_ylabel('Predicted', fontsize=12)
        axes[1, 0].set_ylabel('Ground Truth', fontsize=12)
        axes[2, 0].set_ylabel('Absolute Error', fontsize=12)
        
        # Add colorbar
        fig.colorbar(im, ax=axes[2, :], orientation='horizontal', 
                    fraction=0.046, pad=0.04, label='Absolute Error')
        
        plt.suptitle(f'Error Maps - {name}', fontsize=14)
        plt.tight_layout()
        
        save_path = self.error_maps_dir / f'{name}_error_maps.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved error maps: {save_path}")
    
    def plot_spectral_comparison(self, pred: np.ndarray, target: np.ndarray, 
                                name: str, n_pixels: int = 5):
        """Plot spectral signatures comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        wavelengths = np.linspace(400, 700, pred.shape[2])
        
        # Random pixels for comparison
        h, w = pred.shape[:2]
        pixels = [(np.random.randint(h), np.random.randint(w)) for _ in range(n_pixels)]
        
        # Plot 1: Random pixel comparisons
        ax = axes[0, 0]
        for i, (y, x) in enumerate(pixels):
            ax.plot(wavelengths, target[y, x, :], 'b-', alpha=0.5, label='GT' if i == 0 else '')
            ax.plot(wavelengths, pred[y, x, :], 'r--', alpha=0.5, label='Pred' if i == 0 else '')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.set_title('Random Pixel Spectral Signatures')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mean spectrum
        ax = axes[0, 1]
        mean_gt = target.mean(axis=(0, 1))
        mean_pred = pred.mean(axis=(0, 1))
        ax.plot(wavelengths, mean_gt, 'b-', linewidth=2, label='GT Mean')
        ax.plot(wavelengths, mean_pred, 'r--', linewidth=2, label='Pred Mean')
        ax.fill_between(wavelengths, mean_gt - target.std(axis=(0, 1)),
                        mean_gt + target.std(axis=(0, 1)), alpha=0.2, color='blue')
        ax.fill_between(wavelengths, mean_pred - pred.std(axis=(0, 1)),
                        mean_pred + pred.std(axis=(0, 1)), alpha=0.2, color='red')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.set_title('Mean Spectrum Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Spectral angle per pixel
        ax = axes[1, 0]
        sam_map = self._calculate_sam_map(pred, target)
        im = ax.imshow(sam_map, cmap='hot', vmin=0, vmax=10)
        ax.set_title('Spectral Angle Map (degrees)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot 4: Error spectrum
        ax = axes[1, 1]
        error_spectrum = np.abs(mean_pred - mean_gt)
        ax.bar(wavelengths, error_spectrum, width=300/len(wavelengths), color='red', alpha=0.7)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Mean Absolute Error per Band')
        ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Spectral Analysis - {name}', fontsize=14)
        plt.tight_layout()
        
        save_path = self.spectral_plots_dir / f'{name}_spectral.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved spectral plot: {save_path}")
    
    def _calculate_sam_map(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculate spectral angle mapper for each pixel"""
        # Normalize spectra
        pred_norm = pred / (np.linalg.norm(pred, axis=2, keepdims=True) + 1e-8)
        target_norm = target / (np.linalg.norm(target, axis=2, keepdims=True) + 1e-8)
        
        # Calculate dot product
        dot_product = np.sum(pred_norm * target_norm, axis=2)
        dot_product = np.clip(dot_product, -1, 1)
        
        # Calculate angle in degrees
        angles = np.arccos(dot_product) * 180.0 / np.pi
        
        return angles
    
    def create_interactive_plot(self, results: Dict[str, Any]):
        """Create interactive Plotly visualization"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Metrics Overview', 'Per-Band Performance',
                          'Error Distribution', 'Statistical Summary'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'histogram'}, {'type': 'table'}]]
        )
        
        # Metrics bar chart
        metrics = ['MRAE', 'RMSE', 'PSNR', 'SAM', 'SSIM']
        values = [results.get(m.lower(), {}).get('mean', 0) for m in metrics]
        
        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Metrics',
                  text=[f'{v:.4f}' for v in values],
                  textposition='auto'),
            row=1, col=1
        )
        
        # Per-band line plot
        if 'per_band' in results and 'mrae' in results['per_band']:
            band_mrae = results['per_band']['mrae']['mean']
            wavelengths = np.linspace(400, 700, len(band_mrae))
            band_values = list(band_mrae.values())
            
            fig.add_trace(
                go.Scatter(x=wavelengths, y=band_values,
                          mode='lines+markers', name='MRAE per Band'),
                row=1, col=2
            )
        
        # Error histogram
        if 'per_pixel' in results and 'mrae' in results['per_pixel']:
            if 'values' in results['per_pixel']['mrae']:
                fig.add_trace(
                    go.Histogram(x=results['per_pixel']['mrae']['values'],
                                nbinsx=50, name='Pixel Errors'),
                    row=2, col=1
                )
        
        # Summary table
        table_data = []
        for metric in metrics:
            if metric.lower() in results:
                val = results[metric.lower()]['mean']
                table_data.append([metric, f'{val:.6f}'])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                          fill_color='paleturquoise',
                          align='left'),
                cells=dict(values=list(zip(*table_data)) if table_data else [[], []],
                          fill_color='lavender',
                          align='left')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="MSWR-Net Test Results - Interactive Dashboard",
            showlegend=False,
            height=800
        )
        
        # Save interactive plot
        save_path = self.output_dir / 'interactive_results.html'
        fig.write_html(str(save_path))
        logger.info(f"Saved interactive plot: {save_path}")

class NTIRETestEngine:
    """Main testing engine following NTIRE 2022 protocol"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = self._load_model()
        self.data_loader = DataLoader(config)
        self.metrics_calculator = MetricsCalculator(
            calculate_sam=config.calculate_sam,
            calculate_ssim=config.calculate_ssim
        )
        self.visualizer = Visualizer(self.output_dir)
        
        # Results storage
        self.all_results = []
        self.predictions_cache = []
        
        logger.info(f"Test engine initialized on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load model for testing"""
        logger.info(f"Loading model from: {self.config.model_path}")
        
        # Load checkpoint
        # SECURITY FIX: Use weights_only=True to prevent arbitrary code execution via pickled payloads
        checkpoint = torch.load(self.config.model_path, map_location='cpu', weights_only=True)
        
        # Create model
        if 'model_config' in checkpoint:
            model_config = MSWRDualConfig(**checkpoint['model_config'])
            model = IntegratedMSWRNet(model_config)
        else:
            # Model registry
            model_registry = {
                'tiny': create_mswr_tiny,
                'small': create_mswr_small,
                'base': create_mswr_base,
                'large': create_mswr_large
            }
            model = model_registry[self.config.model_size]()
        
        # Load weights
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model = model.to(self.device)
        model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: {total_params:,} parameters")
        
        return model
    
    def test_single_image(self, rgb: torch.Tensor, gt: Optional[torch.Tensor] = None,
                         name: str = 'test') -> Dict[str, Any]:
        """Test single image"""
        rgb = rgb.to(self.device)
        
        # Inference
        with torch.no_grad():
            if self.config.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    pred = self._run_inference(rgb)
            else:
                pred = self._run_inference(rgb)
        
        result = {'name': name, 'prediction': pred}
        
        # Calculate metrics if ground truth available
        if gt is not None:
            gt = gt.to(self.device)
            
            # Ensure same spatial dimensions
            if pred.shape[-2:] != gt.shape[-2:]:
                # Center crop to match
                h_diff = gt.shape[-2] - pred.shape[-2]
                w_diff = gt.shape[-1] - pred.shape[-1]
                
                if h_diff > 0 or w_diff > 0:
                    h_start = h_diff // 2
                    w_start = w_diff // 2
                    gt = gt[:, :, h_start:h_start+pred.shape[-2], w_start:w_start+pred.shape[-1]]
            
            # Update metrics
            self.metrics_calculator.update(
                pred, gt,
                calculate_per_band=self.config.per_band_metrics,
                calculate_per_pixel=self.config.per_pixel_metrics
            )
            
            # Store for visualization
            if self.config.save_visualizations:
                result['ground_truth'] = gt
        
        return result
    
    def _run_inference(self, rgb: torch.Tensor) -> torch.Tensor:
        """Run inference with optional ensemble"""
        if self.config.ensemble_mode:
            return self._ensemble_inference(rgb)
        elif self.config.mc_dropout_samples > 0:
            return self._mc_dropout_inference(rgb)
        else:
            return self.model(rgb)
    
    def _ensemble_inference(self, rgb: torch.Tensor) -> torch.Tensor:
        """Test-time augmentation ensemble"""
        predictions = []
        
        # Define augmentations
        if self.config.ensemble_mode == 'flip':
            augs = [
                lambda x: x,
                lambda x: torch.flip(x, dims=[2]),
                lambda x: torch.flip(x, dims=[3]),
            ]
        elif self.config.ensemble_mode == 'rotate':
            augs = [
                lambda x: x,
                lambda x: torch.rot90(x, k=1, dims=[2, 3]),
                lambda x: torch.rot90(x, k=2, dims=[2, 3]),
                lambda x: torch.rot90(x, k=3, dims=[2, 3]),
            ]
        else:  # full
            augs = [
                lambda x: x,
                lambda x: torch.flip(x, dims=[2]),
                lambda x: torch.flip(x, dims=[3]),
                lambda x: torch.flip(x, dims=[2, 3]),
            ]
        
        # Apply augmentations and get predictions
        for i, aug in enumerate(augs):
            aug_input = aug(rgb)
            pred = self.model(aug_input)
            
            # Apply inverse transform
            if self.config.ensemble_mode == 'flip':
                if i == 1:
                    pred = torch.flip(pred, dims=[2])
                elif i == 2:
                    pred = torch.flip(pred, dims=[3])
            elif self.config.ensemble_mode == 'rotate':
                if i > 0:
                    pred = torch.rot90(pred, k=-i, dims=[2, 3])
            
            predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
    
    def _mc_dropout_inference(self, rgb: torch.Tensor) -> torch.Tensor:
        """Monte Carlo dropout inference for uncertainty"""
        # Enable dropout
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        
        predictions = []
        for _ in range(self.config.mc_dropout_samples):
            pred = self.model(rgb)
            predictions.append(pred)
        
        # Set back to eval
        self.model.eval()
        
        # Return mean prediction
        return torch.stack(predictions).mean(dim=0)
    
    def run_test(self):
        """Run complete test protocol"""
        logger.info("="*60)
        logger.info("Starting NTIRE 2022 Testing Protocol")
        logger.info("="*60)
        
        # Test all images
        for idx in tqdm(range(len(self.data_loader)), desc="Testing"):
            rgb, gt, name = self.data_loader[idx]
            
            result = self.test_single_image(rgb, gt, name)
            self.all_results.append(result)
            
            # Save predictions if requested
            if self.config.save_predictions:
                self._save_prediction(result['prediction'], name)
            
            # Visualize selected samples
            if (self.config.save_visualizations and 
                idx < self.config.n_visualization_samples and
                'ground_truth' in result):
                
                pred_np = result['prediction'].squeeze(0).cpu().numpy().transpose(1, 2, 0)
                gt_np = result['ground_truth'].squeeze(0).cpu().numpy().transpose(1, 2, 0)
                
                if self.config.save_error_maps:
                    self.visualizer.plot_error_maps(pred_np, gt_np, name)
                
                if self.config.save_spectral_plots:
                    self.visualizer.plot_spectral_comparison(pred_np, gt_np, name)
        
        # Get final metrics
        final_results = self.metrics_calculator.get_results()
        
        # Statistical analysis
        if self.config.statistical_analysis:
            final_results['statistical'] = self._statistical_analysis()
        
        # Baseline comparison
        if self.config.compare_baselines and self.config.baseline_results_path:
            final_results['comparison'] = self._compare_baselines(final_results)
        
        # Save results
        self._save_results(final_results)
        
        # Create visualizations
        self.visualizer.plot_metrics_summary(final_results)
        self.visualizer.create_interactive_plot(final_results)
        
        # Print summary
        self._print_summary(final_results)
        
        return final_results
    
    def _save_prediction(self, pred: torch.Tensor, name: str):
        """Save prediction to file"""
        pred_np = pred.squeeze(0).cpu().numpy()
        
        save_dir = self.output_dir / 'predictions'
        save_dir.mkdir(exist_ok=True)
        
        if self.config.save_format == 'mat':
            save_path = save_dir / f'{name}_pred.mat'
            sio.savemat(str(save_path), {'prediction': pred_np})
        elif self.config.save_format == 'npy':
            save_path = save_dir / f'{name}_pred.npy'
            np.save(str(save_path), pred_np)
        elif self.config.save_format == 'h5':
            save_path = save_dir / f'{name}_pred.h5'
            with h5py.File(str(save_path), 'w') as f:
                f.create_dataset('prediction', data=pred_np, compression='gzip')
    
    def _statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis on results"""
        analysis = {}
        
        if self.metrics_calculator.raw_errors:
            all_errors = np.concatenate(self.metrics_calculator.raw_errors, axis=0)
            
            # Error statistics
            analysis['error_stats'] = {
                'mean': float(np.mean(all_errors)),
                'std': float(np.std(all_errors)),
                'median': float(np.median(all_errors)),
                'min': float(np.min(all_errors)),
                'max': float(np.max(all_errors)),
                'percentiles': {
                    '1': float(np.percentile(all_errors, 1)),
                    '5': float(np.percentile(all_errors, 5)),
                    '25': float(np.percentile(all_errors, 25)),
                    '50': float(np.percentile(all_errors, 50)),
                    '75': float(np.percentile(all_errors, 75)),
                    '95': float(np.percentile(all_errors, 95)),
                    '99': float(np.percentile(all_errors, 99))
                }
            }
            
            # Normality test
            if all_errors.size < 1e6:  # Avoid memory issues for large arrays
                statistic, p_value = stats.normaltest(all_errors.flatten())
                analysis['normality_test'] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
        
        return analysis
    
    def _compare_baselines(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare with baseline results"""
        comparison = {}
        
        # Load baseline results
        with open(self.config.baseline_results_path, 'r') as f:
            baseline = json.load(f)
        
        # Compare main metrics
        for metric in ['mrae', 'rmse', 'psnr', 'sam', 'ssim']:
            if metric in results and metric in baseline:
                our_value = results[metric]['mean']
                baseline_value = baseline[metric]['mean']
                
                improvement = ((baseline_value - our_value) / baseline_value * 100 
                             if metric in ['mrae', 'rmse', 'sam'] else
                             (our_value - baseline_value) / baseline_value * 100)
                
                comparison[metric] = {
                    'ours': our_value,
                    'baseline': baseline_value,
                    'improvement_percent': improvement,
                    'is_better': improvement > 0
                }
        
        return comparison
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        # Save JSON
        json_path = self.output_dir / 'test_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.number) else str(x))
        
        # Save YAML
        yaml_path = self.output_dir / 'test_results.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Results saved to {json_path} and {yaml_path}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print results summary"""
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY - NTIRE 2022 Protocol")
        print("="*60)
        
        # Main metrics
        print("\nMain Metrics:")
        print("-"*30)
        for metric in ['mrae', 'rmse', 'psnr', 'sam', 'ssim']:
            if metric in results:
                value = results[metric]['mean']
                print(f"{metric.upper():8s}: {value:.6f}")
        
        # Statistical summary
        if 'statistical' in results and 'error_stats' in results['statistical']:
            print("\nError Statistics:")
            print("-"*30)
            stats = results['statistical']['error_stats']
            print(f"Mean:     {stats['mean']:.6f}")
            print(f"Std:      {stats['std']:.6f}")
            print(f"Median:   {stats['median']:.6f}")
            print(f"95% tile: {stats['percentiles']['95']:.6f}")
        
        # Baseline comparison
        if 'comparison' in results:
            print("\nBaseline Comparison:")
            print("-"*30)
            for metric, comp in results['comparison'].items():
                symbol = "↑" if comp['is_better'] else "↓"
                print(f"{metric.upper():8s}: {comp['ours']:.6f} vs {comp['baseline']:.6f} "
                     f"({comp['improvement_percent']:+.1f}% {symbol})")
        
        print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='NTIRE 2022 Testing Protocol for MSWR-Net')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to test RGB data')
    parser.add_argument('--gt_data_path', type=str, required=True,
                       help='Path to ground truth HSI data')
    
    # Model settings
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--device', type=str, default='cuda')
    
    # Data settings
    parser.add_argument('--data_format', type=str, default='mat',
                       choices=['mat', 'h5', 'npy'])
    parser.add_argument('--center_crop', action='store_true', default=True)
    parser.add_argument('--crop_size', type=int, default=256)
    
    # Metrics
    parser.add_argument('--calculate_sam', action='store_true', default=True)
    parser.add_argument('--calculate_ssim', action='store_true', default=True)
    parser.add_argument('--per_band_metrics', action='store_true', default=True)
    parser.add_argument('--per_pixel_metrics', action='store_true', default=False)
    
    # Analysis
    parser.add_argument('--statistical_analysis', action='store_true', default=True)
    parser.add_argument('--compare_baselines', action='store_true')
    parser.add_argument('--baseline_results_path', type=str)
    
    # Visualization
    parser.add_argument('--save_visualizations', action='store_true', default=True)
    parser.add_argument('--save_error_maps', action='store_true', default=True)
    parser.add_argument('--save_spectral_plots', action='store_true', default=True)
    parser.add_argument('--n_visualization_samples', type=int, default=10)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./test_results')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--save_format', type=str, default='mat',
                       choices=['mat', 'npy', 'h5'])
    
    # Advanced
    parser.add_argument('--ensemble_mode', type=str, default=None,
                       choices=['flip', 'rotate', 'full'])
    parser.add_argument('--mc_dropout_samples', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Create config
    config = TestConfig(**vars(args))
    
    # Run testing
    engine = NTIRETestEngine(config)
    _results = engine.run_test()

    logger.info("Testing completed successfully!")


if __name__ == '__main__':
    main()

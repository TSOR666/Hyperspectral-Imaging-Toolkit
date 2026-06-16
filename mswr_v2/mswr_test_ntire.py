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
from scipy import stats
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    plt = None
    gridspec = None
    sns = None
    go = None
    make_subplots = None
    PLOTTING_AVAILABLE = False

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
    AverageMeter,
    save_matv73,
)

# Reuse the same validation reader the model was trained/validated against so
# the test metrics line up with the trainer's `Val MRAE`. It handles the ARAD-1K
# layout natively: .jpg RGB via cv2, the 'cube' HSI key via h5py, and pairing
# through split_txt/valid_list.txt.
from dataloader import ValidDataset

# Configure plotting style when the optional visualization stack is installed.
if PLOTTING_AVAILABLE:
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


def _resolve_eval_amp_dtype(requested: str = 'auto') -> torch.dtype:
    """Resolve the autocast dtype for scoring/inference.

    Mirrors ``resolve_amp_dtype`` in train_mswr_v212_logging.py so the test
    engine scores under the SAME numeric regime the trainer measured
    ``best_mrae`` in (bf16 on Ampere+, else fp16). A bare ``autocast()`` would
    silently default to fp16 even on bf16-trained runs.
    """
    bf16_ok = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    if requested == 'fp16':
        return torch.float16
    if requested in ('auto', 'bf16'):
        return torch.bfloat16 if bf16_ok else torch.float16
    raise ValueError(f"Unsupported amp_dtype: {requested!r}")


@dataclass
class TestConfig:
    """NTIRE 2022 Testing Configuration"""
    # Model settings
    model_path: str
    model_size: str = 'base'
    device: str = 'cuda'
    
    # Data settings — ARAD-1K root holding Valid/Test RGB+Spec folders and split_txt.
    data_root: str = None
    split: str = 'auto'  # 'auto' prefers test_list when present, then validation splits.
    bgr2rgb: bool = True  # cv2 reads BGR; convert to RGB to match training.

    # Testing settings
    batch_size: int = 1
    # Border-crop convention used by the public NTIRE 2022 / ARAD‑1K leaderboard:
    # full-image inference, then drop `crop_border` pixels from each side of the
    # PREDICTION (and the matching GT) before computing metrics. 482x512 -> 226x256.
    crop_border: int = 128
    use_amp: bool = True
    # Autocast dtype for scoring. 'auto' resolves to bf16 on Ampere+ (matching
    # the trainer's default), else fp16. Scoring under a DIFFERENT precision than
    # the trainer measured 'best_mrae' in can shift the headline number, so this
    # must mirror resolve_amp_dtype('auto') in the trainer.
    amp_dtype: str = 'auto'
    
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
    save_hsi_viz_inputs: bool = True
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

    MAX_RAW_ERROR_VALUES = 500_000
    RAW_ERROR_VALUES_PER_SAMPLE = 10_000
    
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
            # 'mrae' is computed on the clamped[0,1] prediction (reflectance
            # reporting domain). 'mrae_unclamped' is computed on the raw model
            # output and is the MST++ / NTIRE-comparable leaderboard quantity:
            # it matches the trainer's selection_mrae (unclamped) that best
            # checkpoints are selected on. Clamping can only ever lower MRAE,
            # so reporting only the clamped value is optimistically biased and
            # not apples-to-apples with the published ARAD-1K numbers.
            'mrae': AverageMeter(),
            'mrae_unclamped': AverageMeter(),
            'rmse': AverageMeter(),
            'psnr': AverageMeter(),
            'sam': AverageMeter() if self.calculate_sam else None,
            'ssim': AverageMeter() if self.calculate_ssim else None,
        }
        
        self.per_band_metrics = defaultdict(lambda: defaultdict(list))
        self.per_pixel_metrics = defaultdict(list)
        self.raw_errors = []
        self.raw_error_values = 0
    
    def calculate_ssim_metric(self, pred: torch.Tensor, target: torch.Tensor, 
                              data_range: float = 1.0) -> torch.Tensor:
        """Calculate SSIM for hyperspectral images.

        Vectorized over all 31 bands in one pass: F.avg_pool2d is a per-channel
        box filter, so running it on the full (B, C, H, W) tensor is
        mathematically identical to the old per-band loop (every channel has the
        same H*W element count, so the global mean equals the mean of per-band
        means). This removes ~31 small-kernel launches and host syncs per image.
        """
        # Per-channel box-filter statistics in a single multi-channel pass.
        mu1 = F.avg_pool2d(pred, 11, 1, 5)
        mu2 = F.avg_pool2d(target, 11, 1, 5)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred * pred, 11, 1, 5) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, 11, 1, 5) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, 11, 1, 5) - mu1_mu2

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

        return ssim_map.mean()
    
    def update(self, pred: torch.Tensor, target: torch.Tensor,
              calculate_per_band: bool = False, calculate_per_pixel: bool = False,
              pred_unclamped: Optional[torch.Tensor] = None):
        """Update metrics with new predictions.

        Args:
            pred: prediction used for reflectance-domain metrics (typically
                clamped to [0, 1] by the caller).
            target: ground-truth cube.
            pred_unclamped: the RAW (un-clamped) prediction. When provided, an
                MST++/NTIRE-comparable unclamped MRAE is recorded under
                'mrae_unclamped'. When omitted it defaults to ``pred`` so the
                metric is always populated (degenerating to the clamped value).
        """
        # Ensure same device
        if pred.device != target.device:
            target = target.to(pred.device)
        if pred_unclamped is None:
            pred_unclamped = pred
        elif pred_unclamped.device != target.device:
            pred_unclamped = pred_unclamped.to(target.device)

        # Basic metrics
        mrae = self.mrae_fn(pred, target)
        mrae_unclamped = self.mrae_fn(pred_unclamped, target)
        rmse = self.rmse_fn(pred, target)
        psnr = self.psnr_fn(pred, target, data_range=1.0)
        sample_metrics = {
            'mrae': mrae.item(),
            'mrae_unclamped': mrae_unclamped.item(),
            'rmse': rmse.item(),
            'psnr': psnr.item(),
        }

        self.metrics['mrae'].update(mrae.item())
        self.metrics['mrae_unclamped'].update(mrae_unclamped.item())
        self.metrics['rmse'].update(rmse.item())
        self.metrics['psnr'].update(psnr.item())
        
        # SAM (Spectral Angle Mapper)
        if self.calculate_sam and self.sam_fn is not None:
            sam = self.sam_fn(pred, target)
            sam_deg = sam.item() * 180.0 / np.pi  # Convert to degrees
            sample_metrics['sam'] = sam_deg
            self.metrics['sam'].update(sam_deg)
        
        # SSIM
        if self.calculate_ssim:
            ssim = self.calculate_ssim_metric(pred, target)
            sample_metrics['ssim'] = ssim.item()
            self.metrics['ssim'].update(ssim.item())
        
        # Per-band metrics
        if calculate_per_band:
            self._calculate_per_band_metrics(pred, target)
        
        # Per-pixel metrics
        if calculate_per_pixel:
            self._calculate_per_pixel_metrics(pred, target)
        
        # Keep a bounded, deterministic sample for statistical analysis.
        # Sampling tensor elements before subtraction avoids materializing a
        # full error volume solely for diagnostics.
        remaining = self.MAX_RAW_ERROR_VALUES - self.raw_error_values
        if remaining > 0:
            pred_flat = pred.detach().reshape(-1)
            target_flat = target.detach().reshape(-1)
            take = min(remaining, self.RAW_ERROR_VALUES_PER_SAMPLE, pred_flat.numel())
            step = max(1, (pred_flat.numel() + take - 1) // take)
            error_sample = (
                pred_flat[::step][:take] - target_flat[::step][:take]
            ).abs().float().cpu().numpy()
            self.raw_errors.append(error_sample)
            self.raw_error_values += int(error_sample.size)

        return sample_metrics
    
    def _calculate_per_band_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """Calculate metrics for each spectral band.

        Vectorized over all bands at once instead of 31 separate metric-fn calls
        (each of which triggered a host sync via .item()). The per-band values
        are bit-equivalent to Loss_MRAE/Loss_RMSE/Loss_PSNR applied band-by-band:
          * MRAE/RMSE are plain spatial+batch means and reduce directly.
          * PSNR is NON-linear (10*log10 of a per-image MSE, then mean over the
            batch), so it is computed as per-image-per-band MSE -> log -> mean,
            NOT as a single mean of an error volume.
        """
        n_bands = pred.shape[1]
        eps = float(getattr(self.mrae_fn, "epsilon", 1e-6))

        p = pred.to(torch.float32)
        t = target.to(torch.float32)
        sq_err = (p - t) ** 2

        # MRAE per band: |p - t| / clamp(|t|, eps), averaged over (B, H, W).
        denom = torch.clamp_min(t.abs(), eps)
        mrae_band = (p - t).abs().div(denom).mean(dim=(0, 2, 3))           # (C,)
        # RMSE per band: sqrt of mean squared error over (B, H, W).
        rmse_band = torch.sqrt(sq_err.mean(dim=(0, 2, 3)))                 # (C,)
        # PSNR per band: clamp to [0,1] (Loss_PSNR convention, data_range=1.0),
        # per-image-per-band MSE, 10*log10(1/MSE), then mean over the batch.
        mse_bc = ((t.clamp(0.0, 1.0) - p.clamp(0.0, 1.0)) ** 2).mean(dim=(2, 3))  # (B, C)
        psnr_bc = 10.0 * torch.log10(1.0 / mse_bc.clamp_min(1e-6))         # (B, C)
        psnr_band = psnr_bc.mean(dim=0)                                    # (C,)

        # Single host transfer for all bands instead of 3*n_bands .item() syncs.
        mrae_list = mrae_band.tolist()
        rmse_list = rmse_band.tolist()
        psnr_list = psnr_band.tolist()
        for band_idx in range(n_bands):
            self.per_band_metrics['mrae'][band_idx].append(mrae_list[band_idx])
            self.per_band_metrics['rmse'][band_idx].append(rmse_list[band_idx])
            self.per_band_metrics['psnr'][band_idx].append(psnr_list[band_idx])
    
    def _calculate_per_pixel_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """Calculate metrics for each pixel"""
        # Reshape to (n_pixels, n_bands)
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
        target_flat = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])

        # Match the strict aggregate MRAE denominator so per-pixel diagnostics
        # explain the reported leaderboard-style score.
        eps = getattr(self.mrae_fn, "epsilon", 1e-6)
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

    @staticmethod
    def _require_plotting() -> None:
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Visualization requires matplotlib, seaborn, and plotly. "
                "Install mswr_v2 requirements or disable visualization output."
            )
    
    def plot_metrics_summary(self, results: Dict[str, Any]):
        """Create comprehensive metrics summary plot"""
        self._require_plotting()
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
        self._require_plotting()
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
        self._require_plotting()
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
        self._require_plotting()
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
        self.dataset = ValidDataset(
            config.data_root,
            split=config.split,
            bgr2rgb=config.bgr2rgb,
            logger=logger,
        )
        self.metrics_calculator = MetricsCalculator(
            calculate_sam=config.calculate_sam,
            calculate_ssim=config.calculate_ssim
        )
        self.visualizer = Visualizer(self.output_dir)
        
        # Results storage
        self.all_results = []
        
        logger.info(f"Test engine initialized on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load model for testing"""
        logger.info(f"Loading model from: {self.config.model_path}")

        # weights_only=True is safe but rejects non-tensor globals (numpy
        # scalars, dataclass-derived metadata) that our own trainer writes;
        # retry without it for trusted checkpoints.
        try:
            checkpoint = torch.load(self.config.model_path, map_location='cpu', weights_only=True)
        except Exception as load_exc:
            logger.warning(
                "weights_only=True load failed (%s); retrying with "
                "weights_only=False. Only do this for checkpoints you trust.",
                load_exc,
            )
            checkpoint = torch.load(self.config.model_path, map_location='cpu', weights_only=False)

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

        selected_state = checkpoint.get('selected_state_dict')
        selected_used = False

        def _load_checked(state: Dict[str, Any], label: str) -> None:
            """Load weights and FAIL LOUDLY on a large key mismatch.

            strict=False is kept (EMA/base dicts may legitimately omit a handful
            of buffers), but a *silent* partial load of a mismatched checkpoint
            yields a half- or fully-random model that still produces a
            plausible-looking dashboard (observed: MRAE ~2.7 from an unloaded
            model). Report the mismatch and refuse to score garbage.
            """
            incompat = model.load_state_dict(state, strict=False)
            missing, unexpected = list(incompat.missing_keys), list(incompat.unexpected_keys)
            n_model = len(model.state_dict())
            if missing or unexpected:
                logger.warning(
                    "%s: %d/%d model keys missing, %d unexpected keys",
                    label, len(missing), n_model, len(unexpected),
                )
                if missing:
                    logger.warning("  missing sample: %s", missing[:8])
                if unexpected:
                    logger.warning("  unexpected sample: %s", unexpected[:8])
            if len(missing) > 0.25 * max(1, n_model):
                raise RuntimeError(
                    f"{label}: {len(missing)}/{n_model} model parameters were NOT loaded "
                    f"-- checkpoint/architecture mismatch. Refusing to report metrics from a "
                    f"half-random model. Verify the checkpoint's model_config matches the model "
                    f"(e.g. use_spectral_attn) and that the EMA dict format is unwrapped."
                )

        if isinstance(selected_state, dict) and selected_state:
            _load_checked(
                selected_state,
                f"selected weights ({checkpoint.get('weights_source', 'unknown')})",
            )
            selected_used = True
            logger.info(
                "Loaded selected checkpoint weights (source=%s).",
                checkpoint.get('weights_source', 'unknown'),
            )

        # The trainer's full checkpoints wrap EMA weights as ModelEMA.state_dict()
        # = {'decay':..., 'ema_state': <weights>}; the lightweight checkpoint
        # stores the raw weights under 'ema_state_dict'. Normalize both to an
        # actual state_dict. The previous code passed the {'decay','ema_state'}
        # wrapper straight to load_state_dict, which with strict=False silently
        # loaded NOTHING and left the model fully random.
        ema_container = checkpoint.get('ema') or checkpoint.get('ema_state_dict')
        ema_weights = None
        requested_source = str(checkpoint.get('weights_source', '')).lower()
        force_base = requested_source in {'model', 'raw', 'base'}
        force_ema = requested_source == 'ema'
        if isinstance(ema_container, dict) and ema_container:
            if isinstance(ema_container.get('ema_state'), dict):
                ema_weights = ema_container['ema_state']          # full checkpoint wrapper
            elif isinstance(ema_container.get('shadow'), dict):
                ema_weights = ema_container['shadow']             # legacy 'shadow' format
            elif 'decay' not in ema_container:
                ema_weights = ema_container                       # already a raw state_dict

        ema_used = False
        if not selected_used and not force_base and ema_weights:
            _load_checked(ema_weights, "EMA weights")
            ema_used = True
            logger.info("Loaded EMA weights for testing (preferred over base state_dict).")

        if force_ema and not selected_used and not ema_used:
            raise RuntimeError("Checkpoint requested EMA weights but no loadable EMA state was found.")

        if not selected_used and not ema_used:
            if 'state_dict' in checkpoint:
                _load_checked(checkpoint['state_dict'], "base state_dict")
            else:
                _load_checked(checkpoint, "raw checkpoint")
        
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

        # Inference (full image; border crop is applied below to (pred, gt)).
        # Gate AMP on CUDA (autocast on CPU is a no-op at best, a slowdown at
        # worst) and use the trainer-matched dtype (bf16 on Ampere+) so the
        # headline MRAE is scored in the same precision best_mrae was measured.
        with torch.no_grad():
            if self.config.use_amp and self.device.type == 'cuda':
                from torch.cuda.amp import autocast
                amp_dtype = _resolve_eval_amp_dtype(getattr(self.config, 'amp_dtype', 'auto'))
                with autocast(dtype=amp_dtype):
                    pred = self._run_inference(rgb)
            else:
                pred = self._run_inference(rgb)

        # Predictions are float32 from here on for protocol compliance.
        # Under AMP the model output can be fp16; metric / save code expects fp32.
        pred = pred.float()

        result = {'name': name, 'prediction': pred}

        # Calculate metrics if ground truth available
        if gt is not None:
            gt = gt.to(self.device).float()

            # Ensure same spatial dimensions
            if pred.shape[-2:] != gt.shape[-2:]:
                # Center crop to match
                h_diff = gt.shape[-2] - pred.shape[-2]
                w_diff = gt.shape[-1] - pred.shape[-1]

                if h_diff > 0 or w_diff > 0:
                    h_start = h_diff // 2
                    w_start = w_diff // 2
                    gt = gt[:, :, h_start:h_start+pred.shape[-2], w_start:w_start+pred.shape[-1]]

            # NTIRE protocol: drop `crop_border` pixels from each side of the
            # prediction and matching GT, then compute metrics on the interior.
            # Use a metric-only view so the full-resolution prediction is still
            # available for saving (NTIRE submissions are full-image).
            b = int(getattr(self.config, "crop_border", 0))
            if b > 0 and pred.shape[-2] > 2 * b and pred.shape[-1] > 2 * b:
                pred_for_metrics = pred[:, :, b:-b, b:-b]
                gt_for_metrics = gt[:, :, b:-b, b:-b]
            else:
                pred_for_metrics, gt_for_metrics = pred, gt

            # Keep the RAW (un-clamped, border-cropped) prediction for the
            # MST++/NTIRE-comparable MRAE, which is the quantity the trainer
            # selects best checkpoints on. Then clamp to [0, 1] for the
            # reflectance-domain metrics: Loss_PSNR clamps internally but
            # Loss_MRAE/Loss_RMSE/Loss_SAM do not — out-of-range outputs would
            # otherwise inflate those reported errors relative to the leaderboard.
            pred_unclamped = pred_for_metrics
            pred_for_metrics = pred_for_metrics.clamp(0.0, 1.0)

            # Update metrics
            result['metrics'] = self.metrics_calculator.update(
                pred_for_metrics, gt_for_metrics,
                calculate_per_band=self.config.per_band_metrics,
                calculate_per_pixel=self.config.per_pixel_metrics,
                pred_unclamped=pred_unclamped,
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
        prediction_sum = None
        
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
                lambda x: torch.rot90(x, k=1, dims=[2, 3]),
                lambda x: torch.rot90(x, k=2, dims=[2, 3]),
                lambda x: torch.rot90(x, k=3, dims=[2, 3]),
                lambda x: torch.rot90(torch.flip(x, dims=[2]), k=1, dims=[2, 3]),
            ]
        
        # Inverse transforms must mirror the forward augmentation EXACTLY,
        # otherwise the averaged prediction is misaligned with the GT and the
        # ensemble becomes worse than a single forward pass. The previous
        # implementation only handled 'flip' and 'rotate'; 'full' silently
        # averaged un-inverted flipped predictions and produced garbage.
        if self.config.ensemble_mode == 'flip':
            inverses = [
                lambda y: y,
                lambda y: torch.flip(y, dims=[2]),
                lambda y: torch.flip(y, dims=[3]),
            ]
        elif self.config.ensemble_mode == 'rotate':
            inverses = [
                lambda y, k=i: torch.rot90(y, k=-k, dims=[2, 3])
                for i in range(4)
            ]
        else:  # 'full': identity, flip H, flip W, flip both — each is its own inverse
            inverses = [
                lambda y: y,
                lambda y: torch.flip(y, dims=[2]),
                lambda y: torch.flip(y, dims=[3]),
                lambda y: torch.flip(y, dims=[2, 3]),
                lambda y: torch.rot90(y, k=-1, dims=[2, 3]),
                lambda y: torch.rot90(y, k=-2, dims=[2, 3]),
                lambda y: torch.rot90(y, k=-3, dims=[2, 3]),
                lambda y: torch.flip(torch.rot90(y, k=-1, dims=[2, 3]), dims=[2]),
            ]

        # Apply augmentations and get predictions
        for aug, inv in zip(augs, inverses):
            aug_input = aug(rgb)
            pred = self.model(aug_input)
            pred = inv(pred)
            pred_fp32 = pred.float()
            if prediction_sum is None:
                prediction_sum = pred_fp32
            else:
                prediction_sum.add_(pred_fp32)

        return prediction_sum.div_(len(augs))

    @staticmethod
    def _result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
        """Return only scalar metadata safe to retain for the full test run."""
        summary = {'name': result['name']}
        if 'metrics' in result:
            summary['metrics'] = dict(result['metrics'])
        return summary
    
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
        for idx in tqdm(range(len(self.dataset)), desc="Testing"):
            rgb, gt = self.dataset[idx]
            # ValidDataset yields (C, H, W); the engine expects a leading batch dim.
            rgb = rgb.unsqueeze(0)
            gt = gt.unsqueeze(0)
            name = self.dataset.stems[idx]

            result = self.test_single_image(rgb, gt, name)
            
            # Save predictions if requested
            if self.config.save_predictions:
                self._save_prediction(result['prediction'], name)
            
            selected_for_figures = idx < self.config.n_visualization_samples
            if (
                self.config.save_hsi_viz_inputs
                and selected_for_figures
                and 'ground_truth' in result
            ):
                self._save_hsi_viz_input(
                    result['prediction'],
                    result['ground_truth'],
                    name,
                    result.get('metrics', {}),
                )

            # Visualize selected samples
            if (self.config.save_visualizations and 
                selected_for_figures and
                'ground_truth' in result):
                
                pred_np = result['prediction'].squeeze(0).cpu().numpy().transpose(1, 2, 0)
                gt_np = result['ground_truth'].squeeze(0).cpu().numpy().transpose(1, 2, 0)
                
                if self.config.save_error_maps:
                    self.visualizer.plot_error_maps(pred_np, gt_np, name)
                
                if self.config.save_spectral_plots:
                    self.visualizer.plot_spectral_comparison(pred_np, gt_np, name)

            # MetricsCalculator owns aggregate state. Retaining full prediction
            # and target tensors here kept every evaluated scene resident,
            # including on CUDA, until the entire test run completed.
            self.all_results.append(self._result_summary(result))
        
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
        
        # Aggregate plots are optional so metric-only/headless evaluation does
        # not require the plotting dependency stack.
        if self.config.save_visualizations:
            self.visualizer.plot_metrics_summary(final_results)
            self.visualizer.create_interactive_plot(final_results)
        
        # Print summary
        self._print_summary(final_results)
        
        return final_results
    
    def _save_prediction(self, pred: torch.Tensor, name: str):
        """Save prediction in NTIRE submission format.

        NTIRE 2022 / ARAD‑1K expects:
          - variable name 'cube'
          - shape (H, W, 31) (HWC), not the network's native (31, H, W)
          - dtype float32
          - values clamped to [0, 1]
        """
        pred_clamped = pred.squeeze(0).clamp(0.0, 1.0).float()
        # (31, H, W) -> (H, W, 31)
        pred_np = pred_clamped.permute(1, 2, 0).cpu().numpy().astype(np.float32)

        save_dir = self.output_dir / 'predictions'
        save_dir.mkdir(exist_ok=True)

        if self.config.save_format == 'mat':
            save_path = save_dir / f'{name}_pred.mat'
            save_matv73(str(save_path), 'cube', pred_np)
        elif self.config.save_format == 'npy':
            save_path = save_dir / f'{name}_pred.npy'
            np.save(str(save_path), pred_np)
        elif self.config.save_format == 'h5':
            save_path = save_dir / f'{name}_pred.h5'
            with h5py.File(str(save_path), 'w') as f:
                f.create_dataset('prediction', data=pred_np, compression='gzip')

    def _save_hsi_viz_input(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        name: str,
        metrics: Dict[str, float],
    ) -> None:
        """Export selected samples in the layout consumed by hsi_viz_suite."""
        hsi_dir = self.output_dir / 'hsi'
        metrics_dir = self.output_dir / 'metrics'
        hsi_dir.mkdir(exist_ok=True)
        metrics_dir.mkdir(exist_ok=True)

        pred_np = (
            pred.squeeze(0)
            .clamp(0.0, 1.0)
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        target_np = (
            target.squeeze(0)
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        np.save(hsi_dir / f'{name}.npy', pred_np)
        np.save(hsi_dir / f'{name}_target.npy', target_np)
        if metrics:
            with open(metrics_dir / f'{name}_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
    
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
        for metric in ['mrae', 'mrae_unclamped', 'rmse', 'psnr', 'sam', 'ssim']:
            if metric in results and metric in baseline:
                our_value = results[metric]['mean']
                baseline_value = baseline[metric]['mean']

                improvement = ((baseline_value - our_value) / baseline_value * 100
                             if metric in ['mrae', 'mrae_unclamped', 'rmse', 'sam'] else
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
        
        # Main metrics. mrae_unclamped is the MST++/NTIRE leaderboard-comparable
        # headline (raw output); mrae is the clamped reflectance-domain view.
        print("\nMain Metrics:")
        print("-"*30)
        for metric in ['mrae_unclamped', 'mrae', 'rmse', 'psnr', 'sam', 'ssim']:
            if metric in results:
                value = results[metric]['mean']
                label = 'MRAE*' if metric == 'mrae_unclamped' else metric.upper()
                print(f"{label:14s}: {value:.6f}")
        print("  (* MRAE* = unclamped, MST++/NTIRE-comparable headline)")
        
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
    parser.add_argument('--data_root', type=str, required=True,
                       help='ARAD-1K dataset root (holds RGB/Spec folders and split_txt)')

    # Model settings
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--device', type=str, default='cuda')

    # Data settings
    parser.add_argument('--split', type=str, default='auto',
                       choices=['auto', 'valid', 'test'],
                       help='Evaluation split. auto prefers test_list.txt when available.')
    parser.add_argument('--bgr2rgb', action='store_true', default=True,
                       help='Convert OpenCV BGR to RGB (matches training preprocessing).')
    parser.add_argument('--crop_border', type=int, default=128,
                       help='NTIRE border-crop: pixels dropped per side before metrics.')
    
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
    parser.add_argument('--save_visualizations', dest='save_visualizations',
                       action='store_true', default=True)
    parser.add_argument('--no_save_visualizations', dest='save_visualizations',
                       action='store_false',
                       help='Run metric/submission evaluation without plotting dependencies.')
    parser.add_argument('--save_error_maps', action='store_true', default=True)
    parser.add_argument('--save_spectral_plots', action='store_true', default=True)
    parser.add_argument('--save_hsi_viz_inputs', dest='save_hsi_viz_inputs',
                       action='store_true', default=True,
                       help='Export selected pred/target .npy pairs for hsi_viz_suite.')
    parser.add_argument('--no_save_hsi_viz_inputs', dest='save_hsi_viz_inputs',
                       action='store_false')
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

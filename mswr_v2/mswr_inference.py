"""
MSWR-Net v2.1.2 - Production Inference Pipeline
===============================================

High-performance inference script with comprehensive features:
- Multi-GPU support with automatic load balancing
- Memory-efficient processing with automatic tiling for large images
- Mixed precision inference
- Comprehensive logging and progress tracking
- Result visualization and export options
- Support for various input formats
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import argparse
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import yaml
from tqdm import tqdm
import warnings
import cv2
import h5py
import scipy.io as sio
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_style("whitegrid")

# Import model and utilities
from model.mswr_net_v212 import (
    IntegratedMSWRNet,
    MSWRDualConfig,
    create_mswr_tiny,
    create_mswr_small, 
    create_mswr_base,
    create_mswr_large
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Model registry
MODEL_REGISTRY = {
    'tiny': create_mswr_tiny,
    'small': create_mswr_small,
    'base': create_mswr_base,
    'large': create_mswr_large
}

@dataclass
class InferenceConfig:
    """Comprehensive inference configuration"""
    # Model settings
    model_path: str
    model_size: str = 'base'
    device: str = 'cuda'
    
    # Input/Output
    input_path: str = None
    output_dir: str = './outputs'
    save_format: str = 'mat'  # 'mat', 'npy', 'h5', 'png'
    
    # Processing settings
    batch_size: int = 1
    tile_size: int = 256
    tile_overlap: int = 32
    use_amp: bool = True
    num_workers: int = 4
    
    # Advanced settings
    ensemble_mode: str = None  # None, 'flip', 'rotate', 'full'
    post_processing: bool = False
    save_visualization: bool = True
    save_intermediate: bool = False
    
    # Memory management
    max_memory_gb: float = None
    force_cpu_offload: bool = False
    gradient_checkpointing: bool = False
    
    # Logging
    verbose: bool = True
    log_file: str = None
    profile_performance: bool = False
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class MemoryManager:
    """Advanced memory management for large-scale inference"""
    
    def __init__(self, max_memory_gb: Optional[float] = None):
        self.max_memory_gb = max_memory_gb
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.memory_stats = defaultdict(dict)
        
    def get_available_memory(self, device: int = 0) -> float:
        """Get available GPU memory in GB"""
        if not torch.cuda.is_available():
            return float('inf')
        
        torch.cuda.synchronize(device)
        free_memory = torch.cuda.mem_get_info(device)[0] / 1024**3
        
        if self.max_memory_gb:
            return min(free_memory, self.max_memory_gb)
        return free_memory
    
    def estimate_tile_size(self, model: nn.Module, channels: int = 3, 
                          target_channels: int = 31) -> int:
        """Estimate optimal tile size based on available memory"""
        available_gb = self.get_available_memory()
        
        # Estimate memory per pixel (rough approximation)
        # Forward pass typically needs 3-4x the model size
        model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        overhead_factor = 4.0
        
        # Calculate maximum feasible image area
        bytes_per_pixel = (channels + target_channels) * 4 * overhead_factor
        max_pixels = (available_gb * 1024**3 * 0.8) / bytes_per_pixel  # Use 80% of available
        
        # Convert to tile size (square tiles)
        tile_size = int(np.sqrt(max_pixels))
        
        # Round to multiple of 32 for better performance
        tile_size = (tile_size // 32) * 32
        
        return max(128, min(tile_size, 1024))  # Clamp between reasonable bounds
    
    def log_memory_usage(self, stage: str, device: int = 0):
        """Log memory usage statistics"""
        if not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        
        self.memory_stats[stage] = {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'timestamp': time.time()
        }
        
        logger.debug(f"Memory [{stage}]: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

class TiledProcessor:
    """Efficient tiled processing for large images"""
    
    def __init__(self, tile_size: int = 256, overlap: int = 32):
        self.tile_size = tile_size
        self.overlap = overlap
        
    def split_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], Dict]:
        """Split image into overlapping tiles"""
        H, W = image.shape[1:3] if len(image.shape) == 4 else image.shape[:2]
        
        tiles = []
        positions = []
        
        stride = self.tile_size - self.overlap
        
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                # Calculate tile boundaries
                y_end = min(y + self.tile_size, H)
                x_end = min(x + self.tile_size, W)
                
                # Adjust start if we're at the edge
                if y_end == H:
                    y = max(0, H - self.tile_size)
                if x_end == W:
                    x = max(0, W - self.tile_size)
                
                # Extract tile
                if len(image.shape) == 4:
                    tile = image[:, y:y+self.tile_size, x:x+self.tile_size, :]
                else:
                    tile = image[y:y+self.tile_size, x:x+self.tile_size]
                
                tiles.append(tile)
                positions.append((y, x))
        
        metadata = {
            'original_shape': (H, W),
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'positions': positions,
            'n_tiles': len(tiles)
        }
        
        return tiles, metadata
    
    def merge_tiles(self, tiles: List[np.ndarray], metadata: Dict) -> np.ndarray:
        """Merge tiles back with overlap blending"""
        H, W = metadata['original_shape']
        
        if len(tiles[0].shape) == 4:
            B, _, _, C = tiles[0].shape
            output = np.zeros((B, H, W, C), dtype=tiles[0].dtype)
            weights = np.zeros((B, H, W, 1), dtype=np.float32)
        else:
            C = tiles[0].shape[-1] if len(tiles[0].shape) == 3 else 1
            output = np.zeros((H, W, C), dtype=tiles[0].dtype)
            weights = np.zeros((H, W, 1), dtype=np.float32)
        
        # Create blending weights (cosine window)
        blend_weights = self._create_blend_weights(self.tile_size, self.overlap)
        
        for tile, (y, x) in zip(tiles, metadata['positions']):
            y_end = min(y + self.tile_size, H)
            x_end = min(x + self.tile_size, W)
            
            tile_h = y_end - y
            tile_w = x_end - x
            
            # Apply weighted blending
            if len(tile.shape) == 4:
                output[:, y:y_end, x:x_end] += tile[:, :tile_h, :tile_w] * blend_weights[:tile_h, :tile_w, None]
                weights[:, y:y_end, x:x_end] += blend_weights[:tile_h, :tile_w, None]
            else:
                if len(tile.shape) == 3:
                    output[y:y_end, x:x_end] += tile[:tile_h, :tile_w] * blend_weights[:tile_h, :tile_w, None]
                else:
                    output[y:y_end, x:x_end] += tile[:tile_h, :tile_w] * blend_weights[:tile_h, :tile_w]
                weights[y:y_end, x:x_end] += blend_weights[:tile_h, :tile_w, None]
        
        # Normalize by weights
        output = output / np.maximum(weights, 1e-8)
        
        return output
    
    def _create_blend_weights(self, size: int, overlap: int) -> np.ndarray:
        """Create cosine blending weights for smooth transitions"""
        weights = np.ones((size, size), dtype=np.float32)
        
        if overlap > 0:
            # Create 1D cosine ramp
            ramp = np.linspace(0, np.pi/2, overlap)
            ramp_weights = np.sin(ramp) ** 2
            
            # Apply to edges
            weights[:overlap, :] *= ramp_weights[:, None]
            weights[-overlap:, :] *= ramp_weights[::-1, None]
            weights[:, :overlap] *= ramp_weights[None, :]
            weights[:, -overlap:] *= ramp_weights[::-1][None, :]
        
        return weights

class EnsembleProcessor:
    """Test-time augmentation and ensemble predictions"""
    
    def __init__(self, mode: str = 'flip'):
        self.mode = mode
        self.transforms = self._get_transforms(mode)
    
    def _get_transforms(self, mode: str) -> List[callable]:
        """Get list of augmentation transforms"""
        transforms = []
        
        if mode == 'flip':
            transforms = [
                lambda x: x,  # Original
                lambda x: torch.flip(x, dims=[2]),  # Horizontal flip
                lambda x: torch.flip(x, dims=[3]),  # Vertical flip
            ]
        elif mode == 'rotate':
            transforms = [
                lambda x: x,  # Original
                lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # 90 degrees
                lambda x: torch.rot90(x, k=2, dims=[2, 3]),  # 180 degrees
                lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # 270 degrees
            ]
        elif mode == 'full':
            transforms = [
                lambda x: x,
                lambda x: torch.flip(x, dims=[2]),
                lambda x: torch.flip(x, dims=[3]),
                lambda x: torch.flip(x, dims=[2, 3]),
                lambda x: torch.rot90(x, k=1, dims=[2, 3]),
                lambda x: torch.rot90(x, k=2, dims=[2, 3]),
                lambda x: torch.rot90(x, k=3, dims=[2, 3]),
                lambda x: torch.rot90(torch.flip(x, dims=[2]), k=1, dims=[2, 3]),
            ]
        else:
            transforms = [lambda x: x]
        
        return transforms
    
    def process(self, model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply ensemble processing"""
        predictions = []
        
        for i, transform in enumerate(self.transforms):
            # Apply transform
            augmented = transform(input_tensor)
            
            # Get prediction
            with torch.no_grad():
                pred = model(augmented)
            
            # Apply inverse transform
            pred = self._inverse_transform(pred, i)
            predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
    
    def _inverse_transform(self, tensor: torch.Tensor, transform_idx: int) -> torch.Tensor:
        """Apply inverse transform to prediction"""
        if self.mode == 'flip':
            if transform_idx == 1:
                return torch.flip(tensor, dims=[2])
            elif transform_idx == 2:
                return torch.flip(tensor, dims=[3])
        elif self.mode == 'rotate':
            if transform_idx > 0:
                return torch.rot90(tensor, k=-transform_idx, dims=[2, 3])
        elif self.mode == 'full':
            # Full mode inverse transforms
            inverse_map = {
                1: lambda x: torch.flip(x, dims=[2]),
                2: lambda x: torch.flip(x, dims=[3]),
                3: lambda x: torch.flip(x, dims=[2, 3]),
                4: lambda x: torch.rot90(x, k=-1, dims=[2, 3]),
                5: lambda x: torch.rot90(x, k=-2, dims=[2, 3]),
                6: lambda x: torch.rot90(x, k=-3, dims=[2, 3]),
                7: lambda x: torch.rot90(torch.flip(x, dims=[2]), k=-1, dims=[2, 3]),
            }
            if transform_idx in inverse_map:
                return inverse_map[transform_idx](tensor)
        
        return tensor

class MSWRInference:
    """Main inference engine for MSWR-Net"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = self._setup_device()
        self.memory_manager = MemoryManager(config.max_memory_gb)
        self.tiled_processor = TiledProcessor(config.tile_size, config.tile_overlap)
        self.ensemble_processor = EnsembleProcessor(config.ensemble_mode) if config.ensemble_mode else None
        
        # Setup logging
        if config.log_file:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
            logger.addHandler(file_handler)
        
        # Load model
        self.model = self._load_model()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if self.config.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for inference")
        
        return device
    
    def _load_model(self) -> nn.Module:
        """Load and prepare model for inference"""
        logger.info(f"Loading model from: {self.config.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.config.model_path, map_location='cpu')
        
        # Extract model configuration if available
        if 'model_config' in checkpoint:
            model_config = MSWRDualConfig(**checkpoint['model_config'])
            model = IntegratedMSWRNet(model_config)
        else:
            # Use specified model size
            if self.config.model_size in MODEL_REGISTRY:
                model = MODEL_REGISTRY[self.config.model_size]()
            else:
                raise ValueError(f"Unknown model size: {self.config.model_size}")
        
        # Load state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded successfully: {total_params:,} parameters")
        
        if 'best_mrae' in checkpoint:
            logger.info(f"Model performance - Best MRAE: {checkpoint['best_mrae']:.6f}")
        
        # Optimize model for inference
        if hasattr(torch, 'compile') and not self.config.gradient_checkpointing:
            try:
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("Model compiled with torch.compile")
            except:
                logger.warning("Failed to compile model, using eager mode")
        
        return model
    
    def load_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Load image from various formats"""
        image_path = Path(image_path)
        metadata = {'source': str(image_path), 'format': image_path.suffix}
        
        if image_path.suffix in ['.png', '.jpg', '.jpeg', '.bmp']:
            # Load RGB image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            metadata['original_dtype'] = 'uint8'
            
        elif image_path.suffix == '.mat':
            # Load from MATLAB file
            mat_data = sio.loadmat(str(image_path))
            # Try common variable names
            for key in ['rgb', 'RGB', 'img', 'image', 'data']:
                if key in mat_data:
                    image = mat_data[key].astype(np.float32)
                    break
            else:
                # Use first non-metadata variable
                for key, value in mat_data.items():
                    if not key.startswith('__') and isinstance(value, np.ndarray):
                        image = value.astype(np.float32)
                        break
            
            # Normalize if needed
            if image.max() > 1.0:
                image = image / 255.0
            
            metadata['original_dtype'] = str(mat_data[key].dtype)
            
        elif image_path.suffix == '.npy':
            # Load numpy array
            image = np.load(str(image_path)).astype(np.float32)
            metadata['original_dtype'] = str(image.dtype)
            
            # Normalize if needed
            if image.max() > 1.0:
                image = image / 255.0
                
        elif image_path.suffix in ['.h5', '.hdf5']:
            # Load HDF5 file
            with h5py.File(str(image_path), 'r') as f:
                # Try common dataset names
                for key in ['rgb', 'RGB', 'img', 'image', 'data']:
                    if key in f:
                        image = f[key][:].astype(np.float32)
                        break
                else:
                    # Use first dataset
                    key = list(f.keys())[0]
                    image = f[key][:].astype(np.float32)
            
            # Normalize if needed
            if image.max() > 1.0:
                image = image / 255.0
            
            metadata['original_dtype'] = 'float32'
        
        else:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        # Ensure correct shape (H, W, C)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3:
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
        
        metadata['shape'] = image.shape
        metadata['min'] = float(image.min())
        metadata['max'] = float(image.max())
        
        return image, metadata
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Convert to tensor
        if len(image.shape) == 3:
            # H, W, C -> C, H, W
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        else:
            tensor = torch.from_numpy(image).float()
        
        # Add batch dimension if needed
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess model output"""
        # Remove batch dimension if present
        if output.dim() == 4:
            output = output.squeeze(0)
        
        # Convert to numpy
        output = output.cpu().numpy()
        
        # C, H, W -> H, W, C
        if output.shape[0] == 31:
            output = output.transpose(1, 2, 0)
        
        # Clip to valid range
        output = np.clip(output, 0, 1)
        
        # Optional post-processing
        if self.config.post_processing:
            output = self._apply_post_processing(output)
        
        return output
    
    def _apply_post_processing(self, output: np.ndarray) -> np.ndarray:
        """Apply post-processing filters"""
        # Example: Bilateral filtering for noise reduction
        if len(output.shape) == 3:
            for i in range(output.shape[2]):
                output[:, :, i] = cv2.bilateralFilter(
                    (output[:, :, i] * 255).astype(np.uint8),
                    d=5, sigmaColor=75, sigmaSpace=75
                ) / 255.0
        
        return output
    
    def process_single_image(self, image: np.ndarray) -> np.ndarray:
        """Process a single image"""
        H, W = image.shape[:2]
        
        # Check if tiling is needed
        estimated_memory = (H * W * 3 * 31 * 4) / 1024**3  # Rough estimate in GB
        available_memory = self.memory_manager.get_available_memory()
        
        if estimated_memory > available_memory * 0.5 or H > 1024 or W > 1024:
            logger.info(f"Using tiled processing for {H}x{W} image")
            return self._process_tiled(image)
        else:
            return self._process_full(image)
    
    def _process_full(self, image: np.ndarray) -> np.ndarray:
        """Process full image without tiling"""
        # Preprocess
        input_tensor = self.preprocess(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            if self.config.use_amp and self.device.type == 'cuda':
                with autocast():
                    if self.ensemble_processor:
                        output = self.ensemble_processor.process(self.model, input_tensor)
                    else:
                        output = self.model(input_tensor)
            else:
                if self.ensemble_processor:
                    output = self.ensemble_processor.process(self.model, input_tensor)
                else:
                    output = self.model(input_tensor)
        
        # Postprocess
        return self.postprocess(output)
    
    def _process_tiled(self, image: np.ndarray) -> np.ndarray:
        """Process image using tiling"""
        # Split into tiles
        tiles, metadata = self.tiled_processor.split_image(image)
        
        # Process each tile
        processed_tiles = []
        for tile in tqdm(tiles, desc="Processing tiles", disable=not self.config.verbose):
            tile_tensor = self.preprocess(tile).to(self.device)
            
            with torch.no_grad():
                if self.config.use_amp and self.device.type == 'cuda':
                    with autocast():
                        output = self.model(tile_tensor)
                else:
                    output = self.model(tile_tensor)
            
            processed_tiles.append(self.postprocess(output))
        
        # Merge tiles
        return self.tiled_processor.merge_tiles(processed_tiles, metadata)
    
    def save_output(self, output: np.ndarray, input_path: str, metadata: Dict):
        """Save output in specified format"""
        input_name = Path(input_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if self.config.save_format == 'mat':
            output_path = self.output_dir / f"{input_name}_hsi_{timestamp}.mat"
            sio.savemat(str(output_path), {'hsi': output, 'metadata': metadata})
            
        elif self.config.save_format == 'npy':
            output_path = self.output_dir / f"{input_name}_hsi_{timestamp}.npy"
            np.save(str(output_path), output)
            
            # Save metadata separately
            meta_path = self.output_dir / f"{input_name}_metadata_{timestamp}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
                
        elif self.config.save_format == 'h5':
            output_path = self.output_dir / f"{input_name}_hsi_{timestamp}.h5"
            with h5py.File(str(output_path), 'w') as f:
                f.create_dataset('hsi', data=output, compression='gzip')
                # Save metadata as attributes
                for key, value in metadata.items():
                    f.attrs[key] = str(value)
                    
        elif self.config.save_format == 'png':
            # Save as multi-channel PNG sequence
            output_dir = self.output_dir / f"{input_name}_hsi_{timestamp}"
            output_dir.mkdir(exist_ok=True)
            
            for i in range(output.shape[2]):
                channel_path = output_dir / f"channel_{i:03d}.png"
                channel_img = (output[:, :, i] * 255).astype(np.uint8)
                cv2.imwrite(str(channel_path), channel_img)
        
        logger.info(f"Output saved: {output_path}")
        
        # Save visualization if requested
        if self.config.save_visualization:
            self._save_visualization(output, input_path, input_name, timestamp)
        
        return output_path
    
    def _save_visualization(self, output: np.ndarray, input_path: str, 
                           input_name: str, timestamp: str):
        """Create and save visualization of results"""
        # Load original for comparison
        original, _ = self.load_image(input_path)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=fig)
        
        # Original RGB
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original)
        ax1.set_title('Original RGB')
        ax1.axis('off')
        
        # Reconstructed RGB (using specific bands for RGB)
        ax2 = fig.add_subplot(gs[0, 1])
        if output.shape[2] >= 31:
            # Use bands corresponding to RGB wavelengths (approximate)
            rgb_bands = [13, 9, 5]  # Roughly R, G, B bands
            rgb_output = output[:, :, rgb_bands]
        else:
            rgb_output = output[:, :, :3] if output.shape[2] >= 3 else output[:, :, 0]
        
        ax2.imshow(np.clip(rgb_output, 0, 1))
        ax2.set_title('Reconstructed RGB')
        ax2.axis('off')
        
        # Spectral profile at center
        ax3 = fig.add_subplot(gs[0, 2:])
        h, w = output.shape[:2]
        center_spectrum = output[h//2, w//2, :]
        wavelengths = np.linspace(400, 700, output.shape[2])
        ax3.plot(wavelengths, center_spectrum, 'b-', linewidth=2)
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Intensity')
        ax3.set_title('Center Pixel Spectrum')
        ax3.grid(True, alpha=0.3)
        
        # Channel montage
        n_channels = min(8, output.shape[2])
        for i in range(n_channels):
            ax = fig.add_subplot(gs[1, i%4])
            channel_idx = i * (output.shape[2] // n_channels)
            ax.imshow(output[:, :, channel_idx], cmap='viridis')
            ax.set_title(f'Band {channel_idx} (~{wavelengths[channel_idx]:.0f}nm)')
            ax.axis('off')
        
        plt.suptitle(f'MSWR-Net Hyperspectral Reconstruction - {input_name}', fontsize=16)
        plt.tight_layout()
        
        viz_path = self.output_dir / f"{input_name}_visualization_{timestamp}.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved: {viz_path}")
    
    def process_directory(self, input_dir: str):
        """Process all images in a directory"""
        input_path = Path(input_dir)
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.mat', '.npy', '.h5']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        results = []
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                result = self.process_image(str(image_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue
        
        # Save summary
        self._save_summary(results)
        
        return results
    
    def process_image(self, image_path: str) -> Dict:
        """Process a single image file"""
        start_time = time.time()
        
        # Load image
        logger.info(f"Processing: {image_path}")
        image, metadata = self.load_image(image_path)
        
        # Process
        self.memory_manager.log_memory_usage("before_inference")
        output = self.process_single_image(image)
        self.memory_manager.log_memory_usage("after_inference")
        
        # Save output
        output_path = self.save_output(output, image_path, metadata)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        
        result = {
            'input_path': image_path,
            'output_path': str(output_path),
            'input_shape': image.shape,
            'output_shape': output.shape,
            'processing_time': processing_time,
            'metadata': metadata
        }
        
        # Log performance
        self.performance_stats['processing_times'].append(processing_time)
        self.performance_stats['image_sizes'].append(image.shape)
        
        logger.info(f"Completed in {processing_time:.2f}s")
        
        # Clean up GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return result
    
    def _save_summary(self, results: List[Dict]):
        """Save processing summary"""
        summary = {
            'config': self.config.to_dict(),
            'results': results,
            'performance': {
                'avg_processing_time': np.mean(self.performance_stats['processing_times']),
                'total_processing_time': sum(self.performance_stats['processing_times']),
                'num_images': len(results),
                'memory_stats': dict(self.memory_manager.memory_stats)
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_path = self.output_dir / 'inference_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        logger.info(f"Summary saved: {summary_path}")
    
    def run(self):
        """Main inference pipeline"""
        logger.info("="*60)
        logger.info("MSWR-Net v2.1.2 Inference Pipeline")
        logger.info("="*60)
        
        if self.config.input_path:
            input_path = Path(self.config.input_path)
            
            if input_path.is_file():
                # Process single image
                result = self.process_image(str(input_path))
                logger.info("Inference completed successfully")
                
            elif input_path.is_dir():
                # Process directory
                results = self.process_directory(str(input_path))
                logger.info(f"Processed {len(results)} images successfully")
                
            else:
                raise ValueError(f"Invalid input path: {input_path}")
        else:
            logger.warning("No input path specified")
        
        # Final memory report
        if self.config.profile_performance:
            self._generate_performance_report()
        
        logger.info("="*60)
    
    def _generate_performance_report(self):
        """Generate detailed performance report"""
        report = []
        report.append("="*60)
        report.append("Performance Report")
        report.append("="*60)
        
        if self.performance_stats['processing_times']:
            times = self.performance_stats['processing_times']
            report.append(f"Average processing time: {np.mean(times):.2f}s")
            report.append(f"Min/Max processing time: {np.min(times):.2f}s / {np.max(times):.2f}s")
            report.append(f"Total processing time: {sum(times):.2f}s")
        
        if self.memory_manager.memory_stats:
            report.append("\nMemory Usage:")
            for stage, stats in self.memory_manager.memory_stats.items():
                report.append(f"  {stage}: {stats['allocated_gb']:.2f}GB allocated")
        
        report.append("="*60)
        
        report_str = "\n".join(report)
        logger.info(report_str)
        
        # Save to file
        report_path = self.output_dir / 'performance_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_str)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='MSWR-Net v2.1.2 Inference')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input image or directory')
    
    # Model settings
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model size variant')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Computation device')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--save_format', type=str, default='mat',
                       choices=['mat', 'npy', 'h5', 'png'],
                       help='Output format')
    
    # Processing settings
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    parser.add_argument('--tile_size', type=int, default=256,
                       help='Tile size for large images')
    parser.add_argument('--tile_overlap', type=int, default=32,
                       help='Overlap between tiles')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision')
    
    # Advanced settings
    parser.add_argument('--ensemble_mode', type=str, default=None,
                       choices=['flip', 'rotate', 'full'],
                       help='Test-time augmentation mode')
    parser.add_argument('--post_processing', action='store_true',
                       help='Apply post-processing filters')
    parser.add_argument('--save_visualization', action='store_true',
                       help='Save visualization plots')
    
    # Memory management
    parser.add_argument('--max_memory_gb', type=float, default=None,
                       help='Maximum GPU memory to use (GB)')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file')
    parser.add_argument('--profile_performance', action='store_true',
                       help='Enable performance profiling')
    
    args = parser.parse_args()
    
    # Create config
    config = InferenceConfig(
        model_path=args.model_path,
        model_size=args.model_size,
        device=args.device,
        input_path=args.input_path,
        output_dir=args.output_dir,
        save_format=args.save_format,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        use_amp=args.use_amp,
        ensemble_mode=args.ensemble_mode,
        post_processing=args.post_processing,
        save_visualization=args.save_visualization,
        max_memory_gb=args.max_memory_gb,
        verbose=args.verbose,
        log_file=args.log_file,
        profile_performance=args.profile_performance
    )
    
    # Run inference
    inference_engine = MSWRInference(config)
    inference_engine.run()


if __name__ == '__main__':
    main()

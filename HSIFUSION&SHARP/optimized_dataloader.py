"""
Optimized DataLoader for HSIFusion & SHARP Training
Simplified version with performance optimizations
"""

import os
import torch
import numpy as np
import h5py
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class OptimizedTrainDataset(Dataset):
    """Optimized training dataset with memory modes"""
    
    def __init__(self, 
                 data_root: str,
                 crop_size: int = 128,
                 stride: int = 8,
                 memory_mode: str = 'float16',
                 augment: bool = True,
                 cache_size: int = 100):
        
        self.data_root = data_root
        self.crop_size = crop_size
        self.stride = stride
        self.memory_mode = memory_mode
        self.augment = augment
        
        # Load file lists
        self.rgb_files = []
        self.hsi_files = []
        self._load_file_lists()
        
        # Precompute patch indices for faster access
        self.patches_per_image = []
        self.cumulative_patches = [0]
        self._compute_patch_indices()
        
        # Setup caching for lazy mode
        if memory_mode == 'lazy':
            self._setup_cache(cache_size)
        else:
            # Preload all data for standard/float16 modes
            self._preload_data()
    
    def _load_file_lists(self):
        """Load train file lists"""
        train_list_path = os.path.join(self.data_root, 'split_txt', 'train_list.txt')
        
        with open(train_list_path, 'r') as f:
            file_names = [line.strip() for line in f]
        
        rgb_dir = os.path.join(self.data_root, 'Train_RGB')
        hsi_dir = os.path.join(self.data_root, 'Train_Spec')
        
        for name in file_names:
            rgb_path = os.path.join(rgb_dir, f"{name}.jpg")
            hsi_path = os.path.join(hsi_dir, f"{name}.mat")
            
            if os.path.exists(rgb_path) and os.path.exists(hsi_path):
                self.rgb_files.append(rgb_path)
                self.hsi_files.append(hsi_path)
        
        logger.info(f"Found {len(self.rgb_files)} training images")
    
    def _compute_patch_indices(self):
        """Precompute patch counts for each image"""
        for i in range(len(self.rgb_files)):
            # Assume standard ARAD-1K size (482x512)
            h, w = 482, 512
            # Fix: ensure we don't go beyond image boundaries
            n_patches_h = max(1, (h - self.crop_size) // self.stride + 1)
            n_patches_w = max(1, (w - self.crop_size) // self.stride + 1)
            n_patches = n_patches_h * n_patches_w
            
            self.patches_per_image.append(n_patches)
            self.cumulative_patches.append(self.cumulative_patches[-1] + n_patches)
        
        self.total_patches = self.cumulative_patches[-1]
        logger.info(f"Total training patches: {self.total_patches}")
    
    def _setup_cache(self, cache_size: int):
        """Setup LRU cache for lazy loading"""
        @lru_cache(maxsize=cache_size)
        def load_image_cached(idx: int) -> Tuple[np.ndarray, np.ndarray]:
            return self._load_image_pair(idx)
        
        self._load_cached = load_image_cached
    
    def _load_image_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single RGB-HSI pair"""
        # Load RGB
        rgb = cv2.imread(self.rgb_files[idx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        
        # Load HSI
        with h5py.File(self.hsi_files[idx], 'r') as f:
            hsi = np.array(f['cube'], dtype=np.float32)
        hsi = np.transpose(hsi, [0, 2, 1])  # [31, H, W]
        
        # Apply memory mode
        if self.memory_mode == 'float16':
            rgb = rgb.astype(np.float16)
            hsi = hsi.astype(np.float16)
        
        return rgb, hsi
    
    def _preload_data(self):
        """Preload all data for standard/float16 modes"""
        self.rgb_data = []
        self.hsi_data = []
        
        logger.info(f"Preloading data in {self.memory_mode} mode...")
        
        for i in range(len(self.rgb_files)):
            rgb, hsi = self._load_image_pair(i)
            self.rgb_data.append(rgb)
            self.hsi_data.append(hsi)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Loaded {i + 1}/{len(self.rgb_files)} images")
    
    def _get_image_and_patch(self, idx: int) -> Tuple[int, int, int, int]:
        """Convert global patch index to image index and patch coordinates"""
        # Binary search to find which image this patch belongs to
        img_idx = np.searchsorted(self.cumulative_patches, idx, side='right') - 1
        patch_idx = idx - self.cumulative_patches[img_idx]
        
        # Get patch coordinates
        h, w = 482, 512
        patches_per_row = max(1, (w - self.crop_size) // self.stride + 1)
        
        patch_row = patch_idx // patches_per_row
        patch_col = patch_idx % patches_per_row
        
        y = patch_row * self.stride
        x = patch_col * self.stride
        
        # Fix: ensure we don't exceed image boundaries
        y = min(y, h - self.crop_size)
        x = min(x, w - self.crop_size)
        
        return img_idx, y, x, self.crop_size
    
    def _augment(self, rgb_patch: np.ndarray, hsi_patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        # Random horizontal flip
        if random.random() < 0.5:
            rgb_patch = np.fliplr(rgb_patch).copy()
            hsi_patch = np.flip(hsi_patch, axis=2).copy()  # Flip along width axis
        
        # Random vertical flip
        if random.random() < 0.5:
            rgb_patch = np.flipud(rgb_patch).copy()
            hsi_patch = np.flip(hsi_patch, axis=1).copy()  # Flip along height axis
        
        # Random rotation (0, 90, 180, 270)
        if random.random() < 0.5:
            k = random.randint(1, 3)
            rgb_patch = np.rot90(rgb_patch, k).copy()
            # For HSI, rotate only the spatial dimensions (axes 1 and 2)
            hsi_patch = np.rot90(hsi_patch, k, axes=(1, 2)).copy()
        
        # Ensure arrays are contiguous
        rgb_patch = np.ascontiguousarray(rgb_patch)
        hsi_patch = np.ascontiguousarray(hsi_patch)
        
        return rgb_patch, hsi_patch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training patch"""
        # Get image index and patch coordinates
        img_idx, y, x, size = self._get_image_and_patch(idx)
        
        # Load image data
        if self.memory_mode == 'lazy':
            rgb, hsi = self._load_cached(img_idx)
        else:
            rgb = self.rgb_data[img_idx]
            hsi = self.hsi_data[img_idx]
        
        # Get actual image dimensions
        h, w = rgb.shape[:2]
        
        # Ensure coordinates are valid
        y = min(y, h - size)
        x = min(x, w - size)
        y = max(y, 0)
        x = max(x, 0)
        
        # Extract patches - use copy to avoid views
        rgb_patch = rgb[y:y+size, x:x+size].copy()
        hsi_patch = hsi[:, y:y+size, x:x+size].copy()
        
        # Verify shapes before augmentation
        assert rgb_patch.shape == (size, size, 3), f"RGB shape error: {rgb_patch.shape}"
        assert hsi_patch.shape == (31, size, size), f"HSI shape error: {hsi_patch.shape}"
        
        # Verify patch sizes
        if rgb_patch.shape != (size, size, 3):
            logger.warning(f"RGB patch shape mismatch: expected ({size}, {size}, 3), got {rgb_patch.shape}")
            # Pad if necessary
            if rgb_patch.shape[0] < size or rgb_patch.shape[1] < size:
                pad_h = max(0, size - rgb_patch.shape[0])
                pad_w = max(0, size - rgb_patch.shape[1])
                rgb_patch = np.pad(rgb_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                hsi_patch = np.pad(hsi_patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
        
        # Apply augmentation
        if self.augment:
            rgb_patch, hsi_patch = self._augment(rgb_patch, hsi_patch)

        # Ensure patch uses (31, size, size) layout after augmentation
        if hsi_patch.shape != (31, size, size):
            try:
                # Fix common axis swaps introduced by augmentation
                if hsi_patch.shape == (size, 31, size):
                    hsi_patch = np.transpose(hsi_patch, (1, 0, 2))
                elif hsi_patch.shape == (size, size, 31):
                    hsi_patch = np.transpose(hsi_patch, (2, 0, 1))
                elif hsi_patch.shape == (31, size, size):
                    pass  # already correct
                else:
                    raise ValueError(
                        f"Unexpected HSI patch shape after augmentation: {hsi_patch.shape}"
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to correct HSI patch shape: {e}")

        # Ensure contiguous arrays before tensor conversion
        rgb_patch = np.ascontiguousarray(rgb_patch.transpose(2, 0, 1), dtype=np.float32)
        hsi_patch = np.ascontiguousarray(hsi_patch, dtype=np.float32)
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_patch)
        hsi_tensor = torch.from_numpy(hsi_patch)
        
        # Final shape validation
        assert rgb_tensor.shape == (3, size, size), f"Final RGB tensor shape error: {rgb_tensor.shape}"
        assert hsi_tensor.shape == (31, size, size), f"Final HSI tensor shape error: {hsi_tensor.shape}"
        
        return rgb_tensor, hsi_tensor
    
    def __len__(self) -> int:
        return self.total_patches


class OptimizedValDataset(Dataset):
    """Optimized validation dataset"""
    
    def __init__(self, data_root: str, memory_mode: str = 'float16'):
        self.data_root = data_root
        self.memory_mode = memory_mode
        
        # Load file lists
        self.rgb_files = []
        self.hsi_files = []
        self._load_file_lists()
        
        # Preload validation data (it's small)
        self._preload_data()
    
    def _load_file_lists(self):
        """Load validation file lists"""
        val_list_path = os.path.join(self.data_root, 'split_txt', 'valid_list.txt')
        
        with open(val_list_path, 'r') as f:
            file_names = [line.strip() for line in f]
        
        rgb_dir = os.path.join(self.data_root, 'Train_RGB')
        hsi_dir = os.path.join(self.data_root, 'Train_Spec')
        
        for name in file_names:
            rgb_path = os.path.join(rgb_dir, f"{name}.jpg")
            hsi_path = os.path.join(hsi_dir, f"{name}.mat")
            
            if os.path.exists(rgb_path) and os.path.exists(hsi_path):
                self.rgb_files.append(rgb_path)
                self.hsi_files.append(hsi_path)
        
        logger.info(f"Found {len(self.rgb_files)} validation images")
    
    def _preload_data(self):
        """Preload all validation data"""
        self.rgb_data = []
        self.hsi_data = []
        
        for i in range(len(self.rgb_files)):
            # Load RGB
            rgb = cv2.imread(self.rgb_files[i])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) / 255.0
            
            # Load HSI
            with h5py.File(self.hsi_files[i], 'r') as f:
                hsi = np.array(f['cube'], dtype=np.float32)
            hsi = np.transpose(hsi, [0, 2, 1])
            
            # Apply memory mode
            if self.memory_mode == 'float16':
                rgb = rgb.astype(np.float16)
                hsi = hsi.astype(np.float16)
            
            self.rgb_data.append(rgb)
            self.hsi_data.append(hsi)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a validation image"""
        rgb = self.rgb_data[idx]
        hsi = self.hsi_data[idx]
        
        # Ensure contiguous arrays
        rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
        hsi = np.ascontiguousarray(hsi, dtype=np.float32)
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb)
        hsi_tensor = torch.from_numpy(hsi)
        
        return rgb_tensor, hsi_tensor
    
    def __len__(self) -> int:
        return len(self.rgb_data)


class MSTPlusPlusLoss(torch.nn.Module):
    """MST++ style MRAE loss"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MRAE loss"""
        return torch.mean(torch.abs(pred - target) / (target + 1e-8))


def create_optimized_dataloaders(config: Dict, memory_mode: Optional[str] = None) -> Tuple[DataLoader, DataLoader]:
    """Create optimized dataloaders"""
    
    data_root = config.data_root
    batch_size = config.batch_size
    num_workers = config.num_workers
    memory_mode = memory_mode or config.memory_mode
    
    # Create datasets
    train_dataset = OptimizedTrainDataset(
        data_root=data_root,
        crop_size=config.patch_size,
        stride=config.stride,
        memory_mode=memory_mode,
        augment=True,
        cache_size=100 if memory_mode == 'lazy' else 0
    )
    
    val_dataset = OptimizedValDataset(
        data_root=data_root,
        memory_mode=memory_mode
    )
    
    # Worker init function to reduce h5py cache
    def worker_init_fn(worker_id):
        # Set h5py cache to 4MB instead of default 64MB
        try:
            import h5py._hl.base
            h5py._hl.base.phil.acquire()
            h5py._hl.base.default_file_cache_size = 4 * 1024 * 1024
            h5py._hl.base.phil.release()
        except:
            pass
        
        # Set random seed
        np.random.seed(42 + worker_id)
        random.seed(42 + worker_id)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0 and memory_mode == 'lazy'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Full images for validation
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    logger.info(f"Created dataloaders:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val samples: {len(val_loader)}")
    logger.info(f"  Memory mode: {memory_mode}")
    
    return train_loader, val_loader


def test_dataloader():
    """Test the dataloader"""
    import time
    
    class SimpleConfig:
        data_root = './dataset'
        batch_size = 20
        num_workers = 4
        memory_mode = 'float16'
        patch_size = 128
        stride = 8
    
    config = SimpleConfig()
    
    print("Testing dataloader...")
    train_loader, val_loader = create_optimized_dataloaders(config)
    
    # Test training loader
    print("\nTesting training loader...")
    start = time.time()
    for i, (rgb, hsi) in enumerate(train_loader):
        if i == 0:
            print(f"  RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
            print(f"  HSI shape: {hsi.shape}, dtype: {hsi.dtype}")
            print(f"  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
            print(f"  HSI range: [{hsi.min():.3f}, {hsi.max():.3f}]")
        if i >= 10:
            break
    elapsed = time.time() - start
    print(f"  Loaded 10 batches in {elapsed:.2f}s ({elapsed/10:.3f}s per batch)")
    
    # Test validation loader
    print("\nTesting validation loader...")
    for i, (rgb, hsi) in enumerate(val_loader):
        print(f"  Sample {i}: RGB {rgb.shape}, HSI {hsi.shape}")
        if i >= 5:
            break
    
    print("\nDataloader test completed!")


if __name__ == '__main__':
    test_dataloader()




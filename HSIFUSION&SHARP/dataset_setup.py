#!/usr/bin/env python
"""
Dataset setup utility for SHARP v3.2.2 training
Helps prepare ARAD-1K format datasets
"""

import os
import sys
import random
import argparse
from pathlib import Path
import numpy as np
import h5py
import cv2


def verify_dataset_structure(data_root: Path) -> bool:
    """Verify dataset has correct structure.

    Args:
        data_root: Path to the dataset root directory

    Returns:
        True if dataset structure is valid with at least one matched pair, False otherwise
    """
    required_dirs = ['Train_RGB', 'Train_Spec']
    
    print(f"Checking dataset structure in: {data_root}")
    
    # Check directories exist
    for dir_name in required_dirs:
        dir_path = data_root / dir_name
        if not dir_path.exists():
            print(f"ERROR Missing directory: {dir_path}")
            return False
        print(f"OK Found: {dir_path}")
    
    # Check for RGB images
    rgb_dir = data_root / 'Train_RGB'
    rgb_files = list(rgb_dir.glob('*.jpg')) + list(rgb_dir.glob('*.png'))
    print(f"OK Found {len(rgb_files)} RGB images")
    
    # Check for HSI files
    hsi_dir = data_root / 'Train_Spec'
    hsi_files = list(hsi_dir.glob('*.mat'))
    print(f"OK Found {len(hsi_files)} HSI files")
    
    # Check matching
    rgb_names = {f.stem for f in rgb_files}
    hsi_names = {f.stem for f in hsi_files}
    
    matched = rgb_names & hsi_names
    rgb_only = rgb_names - hsi_names
    hsi_only = hsi_names - rgb_names
    
    print(f"\nOK Matched pairs: {len(matched)}")
    if rgb_only:
        print(f"WARNING RGB without HSI: {len(rgb_only)}")
    if hsi_only:
        print(f"WARNING HSI without RGB: {len(hsi_only)}")
    
    return len(matched) > 0


def verify_data_format(data_root: Path, sample_count: int = 5) -> bool:
    """Verify data format by checking a few samples"""
    rgb_dir = data_root / 'Train_RGB'
    hsi_dir = data_root / 'Train_Spec'
    
    # Get matched files
    rgb_files = list(rgb_dir.glob('*.jpg')) + list(rgb_dir.glob('*.png'))
    hsi_files = {f.stem: f for f in hsi_dir.glob('*.mat')}
    
    matched_files = []
    for rgb_file in rgb_files:
        if rgb_file.stem in hsi_files:
            matched_files.append((rgb_file, hsi_files[rgb_file.stem]))
    
    if not matched_files:
        print("ERROR No matched RGB-HSI pairs found")
        return False
    
    # Sample random files
    samples = random.sample(matched_files, min(sample_count, len(matched_files)))
    
    print(f"\nVerifying data format ({len(samples)} samples)...")
    
    for i, (rgb_path, hsi_path) in enumerate(samples):
        print(f"\nSample {i+1}: {rgb_path.stem}")
        
        # Check RGB
        try:
            rgb = cv2.imread(str(rgb_path))
            if rgb is None:
                print(f"  ERROR Failed to load RGB image")
                continue
            h, w, c = rgb.shape
            print(f"  OK RGB: {w}x{h}x{c}")
        except Exception as e:
            print(f"  ERROR RGB error: {e}")
            continue
        
        # Check HSI
        try:
            with h5py.File(hsi_path, 'r') as f:
                # Check for 'cube' variable
                if 'cube' not in f:
                    print(f"  ERROR No 'cube' variable in .mat file")
                    print(f"     Available keys: {list(f.keys())}")
                    continue
                
                cube = f['cube']
                shape = cube.shape
                dtype = cube.dtype
                
                # Expected shape: (31, H, W) or (C, H, W)
                if len(shape) != 3:
                    print(f"  ERROR HSI shape {shape} is not 3D")
                    continue
                
                if shape[0] == 31:
                    print(f"  OK HSI: {shape[0]}x{shape[1]}x{shape[2]} ({dtype})")
                else:
                    print(f"  WARNING HSI: {shape} - expected 31 channels in first dimension")
                
                # Check if dimensions match RGB (allowing for transposition)
                if (shape[1] == h and shape[2] == w) or (shape[1] == w and shape[2] == h):
                    print(f"  OK Dimensions match RGB")
                else:
                    print(f"  WARNING HSI spatial dims {shape[1]}x{shape[2]} don't match RGB {h}x{w}")
                    
        except Exception as e:
            print(f"  ERROR HSI error: {e}")
            continue
    
    return True


def create_train_val_splits(data_root: Path, train_ratio: float = 0.95, 
                          shuffle: bool = True, seed: int = 42):
    """Create train/validation split files"""
    rgb_dir = data_root / 'Train_RGB'
    hsi_dir = data_root / 'Train_Spec'
    split_dir = data_root / 'split_txt'
    
    # Get matched files
    rgb_files = set(f.stem for f in rgb_dir.glob('*.jpg'))
    rgb_files.update(f.stem for f in rgb_dir.glob('*.png'))
    hsi_files = set(f.stem for f in hsi_dir.glob('*.mat'))
    
    matched = sorted(list(rgb_files & hsi_files))
    
    if not matched:
        print("ERROR No matched RGB-HSI pairs found")
        return False
    
    print(f"\nCreating train/val splits from {len(matched)} matched pairs...")
    
    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(matched)
    
    # Split
    split_idx = int(len(matched) * train_ratio)
    train_files = matched[:split_idx]
    val_files = matched[split_idx:]
    
    print(f"  Train: {len(train_files)} samples")
    print(f"  Val: {len(val_files)} samples")
    
    # Create directory and save
    split_dir.mkdir(exist_ok=True)
    
    train_list_path = split_dir / 'train_list.txt'
    with open(train_list_path, 'w') as f:
        f.write('\n'.join(train_files))
    print(f"  OK Saved: {train_list_path}")
    
    val_list_path = split_dir / 'valid_list.txt'
    with open(val_list_path, 'w') as f:
        f.write('\n'.join(val_files))
    print(f"  OK Saved: {val_list_path}")
    
    return True


def estimate_memory_usage(data_root: Path):
    """Estimate memory usage for different configurations"""
    rgb_dir = data_root / 'Train_RGB'
    
    # Get image dimensions from first file
    rgb_files = list(rgb_dir.glob('*.jpg')) + list(rgb_dir.glob('*.png'))
    if not rgb_files:
        print("No RGB files found")
        return
    
    rgb = cv2.imread(str(rgb_files[0]))
    h, w = rgb.shape[:2]
    
    print(f"\nMemory usage estimates (image size: {w}x{h}):")
    
    # Calculate patches
    patch_size = 128
    stride = 8
    patches_per_image = ((h - patch_size) // stride + 1) * ((w - patch_size) // stride + 1)
    num_images = len(rgb_files)
    total_patches = patches_per_image * num_images
    
    print(f"  Patches per image: {patches_per_image:,}")
    print(f"  Total patches: {total_patches:,}")
    
    # Memory estimates
    patch_memory_rgb = patch_size * patch_size * 3 * 4  # float32
    patch_memory_hsi = patch_size * patch_size * 31 * 4  # float32
    patch_memory_total = patch_memory_rgb + patch_memory_hsi
    
    print(f"\nPer patch memory:")
    print(f"  RGB: {patch_memory_rgb / 1024 / 1024:.2f} MB")
    print(f"  HSI: {patch_memory_hsi / 1024 / 1024:.2f} MB")
    print(f"  Total: {patch_memory_total / 1024 / 1024:.2f} MB")
    
    # Dataset memory
    for mode in ['standard', 'float16', 'lazy']:
        if mode == 'standard':
            multiplier = 1.0
            desc = "float32, all in memory"
        elif mode == 'float16':
            multiplier = 0.5
            desc = "float16, all in memory"
        else:
            multiplier = 0.0
            desc = "load on demand"
        
        if mode != 'lazy':
            dataset_mem = num_images * h * w * (3 + 31) * 4 * multiplier / 1024 / 1024 / 1024
            print(f"\n{mode} mode ({desc}):")
            print(f"  Dataset memory: {dataset_mem:.2f} GB")
        else:
            print(f"\n{mode} mode ({desc}):")
            print(f"  Dataset memory: minimal (cached on demand)")
    
    # Batch memory
    print(f"\nBatch memory (batch_size=20):")
    batch_mem = 20 * patch_memory_total / 1024 / 1024
    print(f"  Per batch: {batch_mem:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Dataset setup utility for SHARP v3.2.2'
    )
    
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to dataset root directory'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify dataset, do not create splits'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.95,
        help='Ratio of data for training (default: 0.95)'
    )
    
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle data before splitting'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling (default: 42)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of samples to verify (default: 5)'
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR Dataset root does not exist: {data_root}")
        sys.exit(1)
    
    print("="*60)
    print("SHARP v3.2.2 Dataset Setup Utility")
    print("="*60)
    
    # Step 1: Verify structure
    if not verify_dataset_structure(data_root):
        print("\nERROR Dataset structure verification failed")
        print("\nExpected structure:")
        print("dataset/")
        print("|-- Train_RGB/     # RGB images")
        print("`-- Train_Spec/    # HSI .mat files")
        sys.exit(1)
    
    # Step 2: Verify format
    verify_data_format(data_root, args.samples)
    
    # Step 3: Estimate memory
    estimate_memory_usage(data_root)
    
    # Step 4: Create splits (unless verify-only)
    if not args.verify_only:
        print("\n" + "="*60)
        create_train_val_splits(
            data_root,
            train_ratio=args.train_ratio,
            shuffle=not args.no_shuffle,
            seed=args.seed
        )
    
    print("\nOK... Dataset setup complete!")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Review the train/validation split in split_txt/")
    print("2. Start training with:")
    print(f"   python sharp_training_script_fixed.py --data_root {data_root}")


if __name__ == '__main__':
    main()





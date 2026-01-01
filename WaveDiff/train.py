import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import argparse
import json
from tqdm import tqdm
from datetime import datetime

# Import model implementations
from models.base_model import HSILatentDiffusionModel
from models.wavelet_model import WaveletHSILatentDiffusionModel
from models.adaptive_model import AdaptiveWaveletHSILatentDiffusionModel

# Import utilities for data loading and visualization
from utils.visualization import (
    visualize_reconstruction_comparison,
    visualize_training_progress
)
from utils.augmentation import RGBHSIAugmentation
from utils.progressive_training import ProgressiveTrainingManager
from losses.spectral_consistency import CombinedSpectralLoss

# Dataset class for HSI data
class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, hsi_transform=None, augmentation=None):
        """
        Dataset for RGB-HSI pairs

        Args:
            root_dir: Root directory containing RGB and HSI data
            transform: Transforms to apply to RGB images
            hsi_transform: Transforms to apply to HSI data
            augmentation: RGBHSIAugmentation instance for synchronized augmentation
        """
        self.root_dir = root_dir
        self.transform = transform
        self.hsi_transform = hsi_transform
        self.augmentation = augmentation
        
        # Assuming directory structure:
        # root_dir/
        #   - RGB/
        #     - img1.png
        #     - img2.png
        #   - HSI/
        #     - img1.mat (or .npy)
        #     - img2.mat (or .npy)
        
        self.rgb_dir = os.path.join(root_dir, 'RGB')
        self.hsi_dir = os.path.join(root_dir, 'HSI')
        
        # Get RGB filenames (assuming they match HSI filenames)
        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        from PIL import Image
        import scipy.io as sio
        
        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        rgb_img = Image.open(rgb_path).convert('RGB')
        
        # Apply transform to RGB
        if self.transform:
            rgb_img = self.transform(rgb_img)
        
        # Load HSI data (supporting both .npy and .mat formats)
        hsi_filename = os.path.splitext(self.rgb_files[idx])[0]
        hsi_path_npy = os.path.join(self.hsi_dir, f"{hsi_filename}.npy")
        hsi_path_mat = os.path.join(self.hsi_dir, f"{hsi_filename}.mat")
        
        if os.path.exists(hsi_path_npy):
            # Load as numpy array
            hsi_data = np.load(hsi_path_npy)
            # Convert to tensor if not already
            hsi_tensor = torch.from_numpy(hsi_data).float()
            # Ensure channel first format (C, H, W)
            if hsi_tensor.shape[0] != 31:
                hsi_tensor = hsi_tensor.permute(2, 0, 1)
        elif os.path.exists(hsi_path_mat):
            # Load from .mat file
            mat_data = sio.loadmat(hsi_path_mat)

            # Extract HSI data (mirrors inference key handling)
            hsi_data = None
            for key in ('cube', 'data'):
                if key in mat_data:
                    hsi_data = mat_data[key]
                    break

            if hsi_data is None:
                for value in mat_data.values():
                    if isinstance(value, np.ndarray) and value.ndim == 3:
                        hsi_data = value
                        break

            if hsi_data is None:
                raise KeyError(f"Could not find HSI cube in {hsi_path_mat}")

            # Convert to tensor
            hsi_tensor = torch.from_numpy(hsi_data).float()
            # Ensure channel first format (C, H, W)
            if hsi_tensor.shape[0] != 31:
                hsi_tensor = hsi_tensor.permute(2, 0, 1)
        else:
            raise FileNotFoundError(f"HSI data not found for {hsi_filename}")

        # Apply transform to HSI if provided
        if self.hsi_transform:
            hsi_tensor = self.hsi_transform(hsi_tensor)

        # Apply synchronized augmentation if provided
        if self.augmentation is not None:
            rgb_img, hsi_tensor = self.augmentation(rgb_img, hsi_tensor)

        return {
            'rgb': rgb_img,
            'hsi': hsi_tensor,
            'filename': self.rgb_files[idx]
        }


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, config, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load model checkpoint"""
    if not os.path.exists(path):
        return None, 0, float('inf')
    
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint, checkpoint['epoch'] + 1, checkpoint.get('val_loss', float('inf'))


def train(config):
    """Training function"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure HSI normalization parameters are available
    config.setdefault('hsi_max_value', 1.0)
    config.setdefault('hsi_normalize_to_neg_one_to_one', True)
    
    # Data transforms
    from torchvision import transforms
    
    rgb_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # HSI transform: resize and normalize to match RGB preprocessing
    hsi_resize = transforms.Resize((config['image_size'], config['image_size']))
    hsi_max_value = config.get('hsi_max_value')
    normalize_to_neg_one = config.get('hsi_normalize_to_neg_one_to_one', True)

    def hsi_transform(hsi_tensor: torch.Tensor) -> torch.Tensor:
        tensor = hsi_tensor.clone().float()
        tensor = hsi_resize(tensor)

        denominator: float
        if hsi_max_value is not None and hsi_max_value > 0:
            denominator = max(float(hsi_max_value), 1e-6)
        else:
            data_max = torch.max(tensor)
            if torch.isfinite(data_max) and data_max.item() > 0:
                denominator = data_max.item()
            else:
                denominator = 1.0

        tensor = tensor / denominator
        tensor = torch.clamp(tensor, 0.0, 1.0)

        if normalize_to_neg_one:
            tensor = tensor * 2.0 - 1.0

        return tensor

    # Initialize progressive training manager
    progressive_manager = None
    if config.get('use_progressive_training', True):
        progressive_manager = ProgressiveTrainingManager(config)

    # Initialize augmentation
    train_augmentation = None
    if config.get('use_augmentation', True):
        aug_strength = 0.5 if progressive_manager is None else progressive_manager.get_augmentation_strength(0)
        train_augmentation = RGBHSIAugmentation(
            geometric_prob=aug_strength,
            photometric_prob=aug_strength,
            noise_prob=aug_strength * 0.6,
            spectral_shift_prob=0.2,
            mixup_prob=config.get('mixup_prob', 0.0),
            training=True
        )

    # Create datasets
    train_dataset = HSIDataset(
        root_dir=config['train_dir'],
        transform=rgb_transform,
        hsi_transform=hsi_transform,
        augmentation=train_augmentation
    )

    val_dataset = HSIDataset(
        root_dir=config['val_dir'],
        transform=rgb_transform,
        hsi_transform=hsi_transform,
        augmentation=None  # No augmentation for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create masking configuration
    masking_config = {
        'mask_strategy': config.get('mask_strategy', 'curriculum'),
        'mask_ratio': config.get('mask_ratio', 0.5),
        'band_mask_ratio': config.get('band_mask_ratio', 0.3),
        'spatial_mask_ratio': config.get('spatial_mask_ratio', 0.3),
        'block_size': config.get('block_size', 32),
        'high_info_mask_prob': config.get('high_info_mask_prob', 0.7),
        'initial_mask_ratio': config.get('initial_mask_ratio', 0.1),
        'final_mask_ratio': config.get('final_mask_ratio', 0.7),
        'progression_strategy': config.get('progression_strategy', 'linear'),
        'base_mask_strategy': config.get('base_mask_strategy', 'random'),
        'curriculum_strategies': config.get('curriculum_strategies', 
                                          ['random', 'block', 'spectral', 'combined']),
        'num_epochs': config.get('num_epochs', 100),
        'low_freq_keep_ratio': config.get('low_freq_keep_ratio', 0.8),
        'high_freq_keep_ratio': config.get('high_freq_keep_ratio', 0.4)
    }
    
    # Create model based on type
    model_type = config.get('model_type', 'base')
    
    refinement_config = config.get('refinement_config')

    # Initialize spectral consistency loss
    spectral_loss_module = None
    if config.get('use_spectral_consistency', True):
        spectral_loss_module = CombinedSpectralLoss(
            num_bands=31,
            use_sam=config.get('use_sam_loss', True),
            use_spectral_grad=config.get('use_spectral_grad_loss', True),
            use_frequency=config.get('use_frequency_loss', True),
            use_perceptual=config.get('use_perceptual_loss', False),
            use_physical=config.get('use_physical_loss', True),
            sam_weight=config.get('sam_weight', 0.1),
            spectral_grad_weight=config.get('spectral_grad_weight', 0.5),
            frequency_weight=config.get('frequency_weight', 0.3),
            perceptual_weight=config.get('perceptual_weight', 0.2),
            physical_weight=config.get('physical_weight', 0.1)
        ).to(device)

    if model_type == 'base':
        print("Initializing Base HSI Latent Diffusion Model with Enhancements")
        model = HSILatentDiffusionModel(
            latent_dim=config['latent_dim'],
            out_channels=31,  # 31 HSI bands output for ARAD-1K
            timesteps=config['timesteps'],
            use_batchnorm=config['use_batchnorm'],
            masking_config=masking_config,
            refinement_config=refinement_config,
            use_enhanced_attention=config.get('use_enhanced_attention', True),
            use_domain_adaptation=config.get('use_domain_adaptation', True),
            dropout=config.get('dropout', 0.1)
        ).to(device)
    elif model_type == 'wavelet':
        print("Initializing Wavelet-enhanced HSI Latent Diffusion Model")
        model = WaveletHSILatentDiffusionModel(
            latent_dim=config['latent_dim'],
            out_channels=31,
            timesteps=config['timesteps'],
            use_batchnorm=config['use_batchnorm'],
            masking_config=masking_config,
            refinement_config=refinement_config
        ).to(device)
    elif model_type == 'adaptive_wavelet':
        print("Initializing Adaptive Wavelet HSI Latent Diffusion Model")
        model = AdaptiveWaveletHSILatentDiffusionModel(
            latent_dim=config['latent_dim'],
            out_channels=31,
            timesteps=config['timesteps'],
            use_batchnorm=config['use_batchnorm'],
            masking_config=masking_config,
            threshold_method=config.get('threshold_method', 'soft'),
            init_threshold=config.get('init_threshold', 0.1),
            trainable_threshold=config.get('trainable_threshold', True),
            refinement_config=refinement_config
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs'] * len(train_loader), 
        eta_min=config['min_lr']
    )
    
    # Create output directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['visualization_dir'], exist_ok=True)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_diffusion_loss': [],
        'train_cycle_loss': [],
        'train_l1_loss': [],
        'train_wavelet_loss': [],
        'train_pre_spectral_l1': [],
        'train_pre_pixel_l1': [],
        'val_diffusion_loss': [],
        'val_cycle_loss': [],
        'val_l1_loss': [],
        'val_wavelet_loss': [],
        'val_pre_spectral_l1': [],
        'val_pre_pixel_l1': []
    }
    
    if config['resume_from_checkpoint']:
        checkpoint_path = config['resume_from_checkpoint']
        checkpoint, start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )
        if checkpoint is not None and 'history' in checkpoint:
            history = checkpoint['history']
            for key in [
                'train_pre_spectral_l1', 'train_pre_pixel_l1',
                'val_pre_spectral_l1', 'val_pre_pixel_l1'
            ]:
                history.setdefault(key, [])
    
    # Training loop
    for epoch in range(start_epoch, config['num_epochs']):
        # Update progressive training manager
        if progressive_manager is not None:
            progressive_manager.update_epoch(epoch)

            # Update learning rate based on progressive schedule
            new_lr = progressive_manager.get_learning_rate(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            # Update augmentation strength
            if train_augmentation is not None:
                aug_strength = progressive_manager.get_augmentation_strength(epoch)
                train_augmentation.geometric_prob = aug_strength
                train_augmentation.photometric_prob = aug_strength
                train_augmentation.noise_prob = aug_strength * 0.6

        # Update masking manager with current epoch
        model.update_masking_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {
            'diffusion_loss': 0.0,
            'cycle_loss': 0.0,
            'l1_loss': 0.0,
            'wavelet_loss': 0.0,
            'total_loss': 0.0
        }
        optional_metrics = {
            'pre_spectral_l1': {'sum': 0.0, 'count': 0},
            'pre_pixel_l1': {'sum': 0.0, 'count': 0}
        }
        
        # Training epoch
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, batch in enumerate(train_pbar):
            # Move data to device
            rgb_imgs = batch['rgb'].to(device)
            hsi_data = batch['hsi'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with masking
            outputs = model(rgb_imgs, use_masking=config['use_masking'])
            
            # Calculate losses
            losses = model.calculate_losses(outputs, rgb_imgs, hsi_data)
            
            # Get loss weights (potentially progressive)
            if progressive_manager is not None:
                loss_weights = progressive_manager.get_loss_weights(epoch)
            else:
                loss_weights = config

            # Combine losses with weights
            total_loss = (
                losses['diffusion_loss'] * loss_weights.get('diffusion_loss_weight', config['diffusion_loss_weight']) +
                losses['cycle_loss'] * loss_weights.get('cycle_loss_weight', config['cycle_loss_weight']) +
                losses['l1_loss'] * loss_weights.get('l1_loss_weight', config['l1_loss_weight'])
            )

            # Add wavelet loss if available
            if 'wavelet_loss' in losses:
                total_loss += losses['wavelet_loss'] * loss_weights.get('wavelet_loss_weight', config.get('wavelet_loss_weight', 0.5))

            # Add threshold regularization if available
            if 'threshold_reg' in losses:
                total_loss += losses['threshold_reg'] * config.get('threshold_reg_weight', 1e-4)

            # Add spectral consistency loss if enabled
            if spectral_loss_module is not None:
                spectral_loss = spectral_loss_module(outputs['hsi_output'], hsi_data)
                spectral_weight = loss_weights.get('spectral_consistency_weight', config.get('spectral_consistency_weight', 0.3))
                total_loss += spectral_loss * spectral_weight
                if 'spectral_consistency' not in epoch_metrics:
                    epoch_metrics['spectral_consistency'] = 0.0
                epoch_metrics['spectral_consistency'] += spectral_loss.item()
            
            # Backward pass and optimization
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['diffusion_loss'] += losses['diffusion_loss'].item()
            epoch_metrics['cycle_loss'] += losses['cycle_loss'].item()
            epoch_metrics['l1_loss'] += losses['l1_loss'].item()
            if 'wavelet_loss' in losses:
                epoch_metrics['wavelet_loss'] += losses['wavelet_loss'].item()

            for key in optional_metrics:
                if key in losses:
                    optional_metrics[key]['sum'] += losses[key].item()
                    optional_metrics[key]['count'] += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': epoch_loss / (batch_idx + 1),
                'diff_loss': epoch_metrics['diffusion_loss'] / (batch_idx + 1),
                'cycle_loss': epoch_metrics['cycle_loss'] / (batch_idx + 1)
            })
            
            # Save training example periodically
            if batch_idx % config['log_interval'] == 0:
                with torch.no_grad():
                    # Visualize training results
                    save_path = os.path.join(
                        config['visualization_dir'], 
                        f"epoch_{epoch+1}_batch_{batch_idx}_train.png"
                    )
                    
                    visualize_reconstruction_comparison(
                        rgb_imgs[0].cpu(),
                        hsi_data[0].cpu(),
                        outputs['hsi_output'][0].cpu(),
                        save_path=save_path
                    )
        
        # Calculate average losses for training epoch
        for k in epoch_metrics:
            epoch_metrics[k] /= len(train_loader)
        
        # Store training metrics
        history['train_loss'].append(epoch_metrics['total_loss'])
        history['train_diffusion_loss'].append(epoch_metrics['diffusion_loss'])
        history['train_cycle_loss'].append(epoch_metrics['cycle_loss'])
        history['train_l1_loss'].append(epoch_metrics['l1_loss'])
        history['train_wavelet_loss'].append(epoch_metrics['wavelet_loss'])
        for key in ['pre_spectral_l1', 'pre_pixel_l1']:
            metric = optional_metrics[key]
            if metric['count'] > 0:
                history[f'train_{key}'].append(metric['sum'] / metric['count'])
            else:
                history[f'train_{key}'].append(None)
        
        print(f"Epoch {epoch+1} - Train Losses: "
              f"Total={epoch_metrics['total_loss']:.4f}, "
              f"Diffusion={epoch_metrics['diffusion_loss']:.4f}, "
              f"Cycle={epoch_metrics['cycle_loss']:.4f}, "
              f"L1={epoch_metrics['l1_loss']:.4f}, "
              f"Wavelet={epoch_metrics['wavelet_loss']:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {
            'diffusion_loss': 0.0,
            'cycle_loss': 0.0,
            'l1_loss': 0.0,
            'wavelet_loss': 0.0,
            'total_loss': 0.0
        }
        val_optional_metrics = {
            'pre_spectral_l1': {'sum': 0.0, 'count': 0},
            'pre_pixel_l1': {'sum': 0.0, 'count': 0}
        }
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(val_pbar):
                # Move data to device
                rgb_imgs = batch['rgb'].to(device)
                hsi_data = batch['hsi'].to(device)
                
                # Forward pass without masking
                outputs = model(rgb_imgs, use_masking=False)
                
                # Calculate losses
                losses = model.calculate_losses(outputs, rgb_imgs, hsi_data)
                
                # Combine losses with weights
                total_loss = (
                    losses['diffusion_loss'] * config['diffusion_loss_weight'] +
                    losses['cycle_loss'] * config['cycle_loss_weight'] + 
                    losses['l1_loss'] * config['l1_loss_weight']
                )
                
                # Add wavelet loss if available
                if 'wavelet_loss' in losses:
                    total_loss += losses['wavelet_loss'] * config.get('wavelet_loss_weight', 0.5)
                
                # Update metrics
                val_loss += total_loss.item()
                val_metrics['total_loss'] += total_loss.item()
                val_metrics['diffusion_loss'] += losses['diffusion_loss'].item()
                val_metrics['cycle_loss'] += losses['cycle_loss'].item()
                val_metrics['l1_loss'] += losses['l1_loss'].item()
                if 'wavelet_loss' in losses:
                    val_metrics['wavelet_loss'] += losses['wavelet_loss'].item()

                for key in val_optional_metrics:
                    if key in losses:
                        val_optional_metrics[key]['sum'] += losses[key].item()
                        val_optional_metrics[key]['count'] += 1
                
                # Visualize validation results (first batch only)
                if batch_idx == 0:
                    save_path = os.path.join(
                        config['visualization_dir'], 
                        f"epoch_{epoch+1}_validation.png"
                    )
                    
                    visualize_reconstruction_comparison(
                        rgb_imgs[0].cpu(),
                        hsi_data[0].cpu(),
                        outputs['hsi_output'][0].cpu(),
                        save_path=save_path
                    )
        
        # Calculate average validation metrics
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        # Store validation metrics
        history['val_loss'].append(val_metrics['total_loss'])
        history['val_diffusion_loss'].append(val_metrics['diffusion_loss'])
        history['val_cycle_loss'].append(val_metrics['cycle_loss'])
        history['val_l1_loss'].append(val_metrics['l1_loss'])
        history['val_wavelet_loss'].append(val_metrics['wavelet_loss'])
        for key in ['pre_spectral_l1', 'pre_pixel_l1']:
            metric = val_optional_metrics[key]
            if metric['count'] > 0:
                history[f'val_{key}'].append(metric['sum'] / metric['count'])
            else:
                history[f'val_{key}'].append(None)
        
        print(f"Validation Losses: "
              f"Total={val_metrics['total_loss']:.4f}, "
              f"Diffusion={val_metrics['diffusion_loss']:.4f}, "
              f"Cycle={val_metrics['cycle_loss']:.4f}, "
              f"L1={val_metrics['l1_loss']:.4f}, "
              f"Wavelet={val_metrics['wavelet_loss']:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            checkpoint_path = os.path.join(config['checkpoint_dir'], f"best_model_epoch_{epoch+1}.pt")
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                epoch_metrics['total_loss'], val_metrics['total_loss'], 
                config, checkpoint_path
            )
            
            # Save training history
            history_path = os.path.join(config['checkpoint_dir'], 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        
        # Always save latest model
        latest_path = os.path.join(config['checkpoint_dir'], "latest_model.pt")
        save_checkpoint(
            model, optimizer, scheduler, epoch, 
            epoch_metrics['total_loss'], val_metrics['total_loss'], 
            config, latest_path
        )
        
        # Visualize training progress
        visualize_training_progress(
            history,
            save_path=os.path.join(config['visualization_dir'], 'training_progress.png')
        )
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train HSI Latent Diffusion Model")
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='base', 
                        choices=['base', 'wavelet', 'adaptive_wavelet'],
                        help='Type of model to train')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--use_batchnorm', dest='use_batchnorm', action='store_true',
                        help='Enable batch normalization')
    parser.add_argument('--no_batchnorm', dest='use_batchnorm', action='store_false',
                        help='Disable batch normalization')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Loss weights
    parser.add_argument('--diffusion_loss_weight', type=float, default=1.0,
                        help='Weight for diffusion loss')
    parser.add_argument('--l1_loss_weight', type=float, default=1.0,
                        help='Weight for L1 loss')
    parser.add_argument('--cycle_loss_weight', type=float, default=0.8,
                        help='Weight for cycle consistency loss')
    parser.add_argument('--wavelet_loss_weight', type=float, default=0.5,
                        help='Weight for wavelet loss (if applicable)')
    parser.add_argument('--spectral_consistency_weight', type=float, default=0.3,
                        help='Weight for spectral consistency loss')

    # Generalization enhancements
    parser.add_argument('--use_enhanced_attention', action='store_true', default=True,
                        help='Use enhanced multi-head spectral attention')
    parser.add_argument('--use_domain_adaptation', action='store_true', default=True,
                        help='Use domain-adaptive attention for cross-dataset generalization')
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                        help='Use data augmentation for better generalization')
    parser.add_argument('--use_progressive_training', action='store_true', default=True,
                        help='Use progressive training strategies')
    parser.add_argument('--use_spectral_consistency', action='store_true', default=True,
                        help='Use spectral consistency losses')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, default='data/ARAD1K/train',
                        help='Directory containing training data')
    parser.add_argument('--val_dir', type=str, default='data/ARAD1K/val',
                        help='Directory containing validation data')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images to')
    
    # Masking parameters
    parser.add_argument('--use_masking', dest='use_masking', action='store_true',
                        help='Enable masking for training')
    parser.add_argument('--no_masking', dest='use_masking', action='store_false',
                        help='Disable masking during training')

    parser.add_argument('--hsi_max_value', type=float, default=1.0,
                        help='Maximum expected value for raw HSI data before normalization')
    parser.add_argument('--normalize_hsi_to_neg_one_to_one', dest='hsi_normalize_to_neg_one_to_one',
                        action='store_true', default=True,
                        help='Normalize HSI tensors to [-1, 1] after scaling to [0, 1]')
    parser.add_argument('--keep_hsi_unit_range', dest='hsi_normalize_to_neg_one_to_one',
                        action='store_false',
                        help='Keep normalized HSI tensors within [0, 1]')
    parser.add_argument('--mask_strategy', type=str, default='curriculum',
                        choices=['random', 'block', 'spectral', 'combined', 'curriculum'],
                        help='Masking strategy')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--visualization_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Interval for logging training progress')
    
    # Ensure sensible defaults for boolean flags
    parser.set_defaults(use_batchnorm=True, use_masking=True)

    # Parse arguments
    args = parser.parse_args()
    
    # Create config dictionary
    config = vars(args)
    
    # Add timestamp to checkpoint and visualization directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], f"{config['model_type']}_{timestamp}")
    config['visualization_dir'] = os.path.join(config['visualization_dir'], f"{config['model_type']}_{timestamp}")
    
    # Train model
    model, history = train(config)
    
    # Save final model
    final_path = os.path.join(config['checkpoint_dir'], "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()

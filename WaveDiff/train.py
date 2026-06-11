import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import argparse
import json
import math
from collections import OrderedDict
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
from utils.ema import ModelEMA


def _torch_load_checkpoint(path, map_location):
    """Load WaveDiff checkpoints without enabling arbitrary pickle payloads."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_hsi_chw(hsi_tensor, num_bands=31):
    """Validate and convert a 3D HSI cube to channel-first layout."""
    if hsi_tensor.ndim != 3:
        raise ValueError(
            f"Expected a 3D HSI cube, got shape {tuple(hsi_tensor.shape)}"
        )
    if hsi_tensor.shape[0] == num_bands:
        return hsi_tensor
    if hsi_tensor.shape[-1] == num_bands:
        return hsi_tensor.permute(2, 0, 1)
    raise ValueError(
        f"Could not identify {num_bands} spectral bands in shape "
        f"{tuple(hsi_tensor.shape)}"
    )


def get_loss_weight(loss_weights, config, loss_name, default):
    """Read either ``name`` or ``name_weight`` curriculum conventions."""
    weight_key = f"{loss_name}_weight"
    if weight_key in loss_weights:
        return loss_weights[weight_key]
    if loss_name in loss_weights:
        return loss_weights[loss_name]
    return config.get(weight_key, default)


def combine_weighted_losses(losses, config, loss_weights=None):
    """Combine all configured training losses with consistent key handling."""
    loss_weights = loss_weights or config
    total = (
        losses['diffusion_loss']
        * get_loss_weight(loss_weights, config, 'diffusion_loss', 1.0)
        + losses['cycle_loss']
        * get_loss_weight(loss_weights, config, 'cycle_loss', 0.8)
        + losses['l1_loss']
        * get_loss_weight(loss_weights, config, 'l1_loss', 1.0)
    )
    optional_defaults = {
        'wavelet_loss': 0.5,
        'threshold_reg': 1e-4,
        'spectral_consistency': 0.3,
        'latent_reconstruction_loss': 0.5,
    }
    for name, default in optional_defaults.items():
        if name in losses:
            total = total + losses[name] * get_loss_weight(
                loss_weights, config, name, default
            )
    return total


def build_optimizer(model, learning_rate, weight_decay):
    """Create AdamW groups without decaying bias and normalization scales."""
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith('.bias'):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return optim.AdamW(
        [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ],
        lr=learning_rate,
    )


def build_lr_scheduler(
    optimizer,
    total_steps,
    warmup_steps,
    learning_rate,
    min_lr,
):
    """Single warmup-plus-cosine schedule stepped once per optimizer update."""
    min_ratio = min_lr / max(learning_rate, 1e-12)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# Dataset class for HSI data
class HSIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        hsi_transform=None,
        augmentation=None,
        npy_cache_size=0,
    ):
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
        self.npy_cache_size = max(int(npy_cache_size), 0)
        self._npy_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
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
        hsi_cache_key = None
        hsi_is_cached = False

        if os.path.exists(hsi_path_npy):
            cached = self._npy_cache.get(hsi_path_npy)
            if cached is not None:
                self.cache_hits += 1
                self._npy_cache.move_to_end(hsi_path_npy)
                hsi_tensor = cached.clone()
                hsi_is_cached = True
            else:
                self.cache_misses += 1
                hsi_data = np.load(hsi_path_npy)
                hsi_tensor = ensure_hsi_chw(
                    torch.from_numpy(hsi_data).float()
                )
                hsi_cache_key = hsi_path_npy
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
            hsi_tensor = ensure_hsi_chw(hsi_tensor)
        else:
            raise FileNotFoundError(f"HSI data not found for {hsi_filename}")

        # Apply transform to HSI if provided
        if self.hsi_transform and not hsi_is_cached:
            hsi_tensor = self.hsi_transform(hsi_tensor)
        if hsi_cache_key is not None and self.npy_cache_size > 0:
            self._npy_cache[hsi_cache_key] = hsi_tensor.clone()
            while len(self._npy_cache) > self.npy_cache_size:
                self._npy_cache.popitem(last=False)

        # Apply synchronized augmentation if provided
        if self.augmentation is not None:
            rgb_img, hsi_tensor = self.augmentation(rgb_img, hsi_tensor)

        return {
            'rgb': rgb_img,
            'hsi': hsi_tensor,
            'filename': self.rgb_files[idx]
        }


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    val_loss,
    config,
    path,
    history=None,
    scaler=None,
    ema=None,
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }
    if history is not None:
        checkpoint['history'] = history
    if scaler is not None and scaler.is_enabled():
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    path,
    device,
    scaler=None,
    ema=None,
):
    """Load model checkpoint"""
    if not os.path.exists(path):
        return None, 0, float('inf')
    
    print(f"Loading checkpoint from {path}")
    checkpoint = _torch_load_checkpoint(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    except ValueError as exc:
        print(f"Optimizer layout changed; resuming with fresh optimizer state: {exc}")
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
    elif ema is not None:
        ema.module.load_state_dict(model.state_dict())
        ema.num_updates = 0
    
    return checkpoint, checkpoint['epoch'] + 1, checkpoint.get('val_loss', float('inf'))


def train(config):
    """Training function"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp = bool(config.get('use_amp', True) and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Ensure HSI normalization parameters are available
    config.setdefault('hsi_max_value', 1.0)
    config.setdefault('hsi_normalize_to_neg_one_to_one', True)
    config.setdefault('enhanced_attention_mode', 'channel')
    config.setdefault('norm_type', 'group')
    config.setdefault('norm_groups', 8)
    config.setdefault('cross_attention_mode', 'channel')
    config.setdefault('attention_window_size', 8)
    config.setdefault('conditional_residual_diffusion', True)
    config.setdefault('residual_scale', 1.0)
    config.setdefault('gradient_accumulation_steps', 1)
    config.setdefault('use_ema', True)
    config.setdefault('ema_decay', 0.999)
    config.setdefault('use_torch_compile', False)
    config.setdefault('compile_mode', 'default')
    config.setdefault('npy_cache_size', 8)
    
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
        tensor = hsi_tensor.float()
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
            spectral_shift_prob=config.get('spectral_shift_prob', 0.0),
            mixup_prob=config.get('mixup_prob', 0.0),
            training=True
        )

    # Create datasets
    train_dataset = HSIDataset(
        root_dir=config['train_dir'],
        transform=rgb_transform,
        hsi_transform=hsi_transform,
        augmentation=train_augmentation,
        npy_cache_size=config['npy_cache_size'],
    )

    val_dataset = HSIDataset(
        root_dir=config['val_dir'],
        transform=rgb_transform,
        hsi_transform=hsi_transform,
        augmentation=None,
        npy_cache_size=config.get(
            'val_npy_cache_size', config['npy_cache_size']
        ),
    )
    
    # Create data loaders
    pin_memory = device.type == 'cuda'
    loader_worker_options = {}
    if config['num_workers'] > 0:
        loader_worker_options['persistent_workers'] = True
        loader_worker_options['prefetch_factor'] = config.get('prefetch_factor', 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=pin_memory,
        **loader_worker_options,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=pin_memory,
        **loader_worker_options,
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
            physical_weight=config.get('physical_weight', 0.1),
            normalized_to_neg_one_to_one=normalize_to_neg_one,
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
            dropout=config.get('dropout', 0.1),
            enhanced_attention_mode=config['enhanced_attention_mode'],
            norm_type=config['norm_type'],
            norm_groups=config['norm_groups'],
            cross_attention_mode=config['cross_attention_mode'],
            attention_window_size=config['attention_window_size'],
            conditional_residual_diffusion=config[
                'conditional_residual_diffusion'
            ],
            residual_scale=config['residual_scale'],
        ).to(device)
    elif model_type == 'wavelet':
        print("Initializing Wavelet-enhanced HSI Latent Diffusion Model")
        model = WaveletHSILatentDiffusionModel(
            latent_dim=config['latent_dim'],
            out_channels=31,
            timesteps=config['timesteps'],
            use_batchnorm=config['use_batchnorm'],
            masking_config=masking_config,
            refinement_config=refinement_config,
            norm_type=config['norm_type'],
            norm_groups=config['norm_groups'],
            cross_attention_mode=config['cross_attention_mode'],
            attention_window_size=config['attention_window_size'],
            conditional_residual_diffusion=config[
                'conditional_residual_diffusion'
            ],
            residual_scale=config['residual_scale'],
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
            refinement_config=refinement_config,
            norm_type=config['norm_type'],
            norm_groups=config['norm_groups'],
            cross_attention_mode=config['cross_attention_mode'],
            attention_window_size=config['attention_window_size'],
            conditional_residual_diffusion=config[
                'conditional_residual_diffusion'
            ],
            residual_scale=config['residual_scale'],
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Setup optimizer and scheduler
    optimizer = build_optimizer(
        model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    accumulation_steps = max(
        int(config['gradient_accumulation_steps']),
        1,
    )
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = config['num_epochs'] * updates_per_epoch
    warmup_steps = config.get('lr_warmup_epochs', 5) * updates_per_epoch
    scheduler = build_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        learning_rate=config['learning_rate'],
        min_lr=config['min_lr'],
    )
    ema = ModelEMA(model, decay=config['ema_decay']) if config['use_ema'] else None
    forward_model = model
    if config['use_torch_compile']:
        if not hasattr(torch, 'compile'):
            print("torch.compile is unavailable; continuing without compilation")
        else:
            try:
                forward_model = torch.compile(
                    model,
                    mode=config['compile_mode'],
                )
                print(f"Enabled torch.compile mode={config['compile_mode']}")
            except Exception as exc:
                print(f"torch.compile setup failed; continuing eagerly: {exc}")
    
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
            model,
            optimizer,
            scheduler,
            checkpoint_path,
            device,
            scaler=scaler,
            ema=ema,
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

            # Update augmentation strength
            if train_augmentation is not None:
                aug_strength = progressive_manager.get_augmentation_strength(epoch)
                train_augmentation.geometric_prob = aug_strength
                train_augmentation.photometric_prob = aug_strength
                train_augmentation.noise_prob = aug_strength * 0.6

        # Update masking manager with current epoch
        model.update_masking_epoch(epoch)
        
        model.train()
        forward_model.train()
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
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_pbar):
            # Move data to device
            rgb_imgs = batch['rgb'].to(device, non_blocking=pin_memory)
            hsi_data = batch['hsi'].to(device, non_blocking=pin_memory)

            # Get loss weights (potentially progressive)
            if progressive_manager is not None:
                loss_weights = progressive_manager.get_loss_weights(epoch)
            else:
                loss_weights = config

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = forward_model(
                    rgb_imgs,
                    use_masking=config['use_masking'],
                    hsi_target=hsi_data,
                )
                losses = model.calculate_losses(outputs, rgb_imgs, hsi_data)
                if spectral_loss_module is not None:
                    losses['spectral_consistency'] = spectral_loss_module(
                        outputs['hsi_output'], hsi_data
                    )
                total_loss = combine_weighted_losses(
                    losses, config, loss_weights=loss_weights
                )

            remainder = len(train_loader) % accumulation_steps
            final_group_start = len(train_loader) - (remainder or accumulation_steps)
            group_size = (
                remainder
                if remainder and batch_idx >= final_group_start
                else accumulation_steps
            )
            scaler.scale(total_loss / group_size).backward()
            should_step = (
                (batch_idx + 1) % accumulation_steps == 0
                or batch_idx + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['max_grad_norm'],
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_ran = (
                    not use_amp or scaler.get_scale() >= previous_scale
                )
                if optimizer_ran:
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)
                optimizer.zero_grad(set_to_none=True)
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['diffusion_loss'] += losses['diffusion_loss'].item()
            epoch_metrics['cycle_loss'] += losses['cycle_loss'].item()
            epoch_metrics['l1_loss'] += losses['l1_loss'].item()
            if 'wavelet_loss' in losses:
                epoch_metrics['wavelet_loss'] += losses['wavelet_loss'].item()
            if 'spectral_consistency' in losses:
                if 'spectral_consistency' not in epoch_metrics:
                    epoch_metrics['spectral_consistency'] = 0.0
                epoch_metrics['spectral_consistency'] += losses[
                    'spectral_consistency'
                ].item()

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
        evaluation_model = ema.module if ema is not None else model
        evaluation_model.eval()
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
        
        with torch.inference_mode():
            val_pbar = tqdm(val_loader, desc="Validation")

            for batch_idx, batch in enumerate(val_pbar):
                # Move data to device
                rgb_imgs = batch['rgb'].to(device, non_blocking=pin_memory)
                hsi_data = batch['hsi'].to(device, non_blocking=pin_memory)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = evaluation_model(
                        rgb_imgs,
                        use_masking=False,
                        hsi_target=hsi_data,
                    )
                    losses = evaluation_model.calculate_losses(
                        outputs,
                        rgb_imgs,
                        hsi_data,
                    )
                    if spectral_loss_module is not None:
                        losses['spectral_consistency'] = spectral_loss_module(
                            outputs['hsi_output'], hsi_data
                        )
                    total_loss = combine_weighted_losses(
                        losses, config, loss_weights=loss_weights
                    )
                
                # Update metrics
                val_loss += total_loss.item()
                val_metrics['total_loss'] += total_loss.item()
                val_metrics['diffusion_loss'] += losses['diffusion_loss'].item()
                val_metrics['cycle_loss'] += losses['cycle_loss'].item()
                val_metrics['l1_loss'] += losses['l1_loss'].item()
                if 'wavelet_loss' in losses:
                    val_metrics['wavelet_loss'] += losses['wavelet_loss'].item()
                if 'spectral_consistency' in losses:
                    if 'spectral_consistency' not in val_metrics:
                        val_metrics['spectral_consistency'] = 0.0
                    val_metrics['spectral_consistency'] += losses[
                        'spectral_consistency'
                    ].item()

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
                config,
                checkpoint_path,
                history=history,
                scaler=scaler,
                ema=ema,
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
            config,
            latest_path,
            history=history,
            scaler=scaler,
            ema=ema,
        )
        
        # Visualize training progress
        visualize_training_progress(
            history,
            save_path=os.path.join(config['visualization_dir'], 'training_progress.png')
        )
    
    if ema is not None:
        model._wavediff_ema_state = ema.state_dict()
    return model, history


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='JSON configuration file; explicit CLI flags take precedence',
    )
    config_args, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        description="Train HSI Latent Diffusion Model",
        parents=[config_parser],
    )
    
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
    parser.add_argument(
        '--norm_type',
        choices=['batch', 'group', 'none'],
        default='group',
        help='Normalization used by residual and refinement blocks',
    )
    parser.add_argument('--norm_groups', type=int, default=8)
    parser.add_argument(
        '--cross_attention_mode',
        choices=['spatial', 'channel', 'windowed'],
        default='channel',
        help='Attention axis for encoder, denoiser, and decoder attention',
    )
    parser.add_argument('--attention_window_size', type=int, default=8)
    parser.add_argument(
        '--conditional_residual_diffusion',
        dest='conditional_residual_diffusion',
        action='store_true',
        help='Diffuse an HSI residual latent conditioned on the RGB latent',
    )
    parser.add_argument(
        '--legacy_latent_diffusion',
        dest='conditional_residual_diffusion',
        action='store_false',
        help='Use the legacy unconditional latent diffusion objective',
    )
    parser.add_argument('--residual_scale', type=float, default=1.0)
    
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
    parser.add_argument('--use_amp', dest='use_amp', action='store_true',
                        help='Enable CUDA automatic mixed precision')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                        help='Disable CUDA automatic mixed precision')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--npy_cache_size', type=int, default=8)
    parser.add_argument('--val_npy_cache_size', type=int, default=8)
    parser.add_argument('--use_ema', dest='use_ema', action='store_true')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument(
        '--use_torch_compile',
        dest='use_torch_compile',
        action='store_true',
    )
    parser.add_argument(
        '--no_torch_compile',
        dest='use_torch_compile',
        action='store_false',
    )
    parser.add_argument(
        '--compile_mode',
        choices=['default', 'reduce-overhead', 'max-autotune'],
        default='default',
    )
    
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
    parser.add_argument(
        '--enhanced_attention_mode',
        type=str,
        choices=['spatial', 'channel'],
        default='channel',
        help='Attention axis for the optional enhanced latent attention',
    )
    
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
    parser.set_defaults(
        use_batchnorm=True,
        use_masking=True,
        use_amp=True,
        use_ema=True,
        use_torch_compile=False,
        conditional_residual_diffusion=True,
    )

    if config_args.config:
        with open(config_args.config, 'r') as config_file:
            file_config = json.load(config_file)
        if not isinstance(file_config, dict):
            raise ValueError("Training config JSON must contain an object")
        parser.set_defaults(**file_config)

    # Parse arguments
    args = parser.parse_args()
    
    # Create config dictionary
    config = vars(args)
    config.pop('config', None)
    
    # Add timestamp to checkpoint and visualization directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['checkpoint_dir'] = os.path.join(config['checkpoint_dir'], f"{config['model_type']}_{timestamp}")
    config['visualization_dir'] = os.path.join(config['visualization_dir'], f"{config['model_type']}_{timestamp}")
    
    # Train model
    model, history = train(config)
    
    # Save final model
    final_path = os.path.join(config['checkpoint_dir'], "final_model.pt")
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }
    if hasattr(model, '_wavediff_ema_state'):
        final_checkpoint['ema_state_dict'] = model._wavediff_ema_state
    torch.save(final_checkpoint, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()

"""
Progressive training strategies for improved generalization
Implements curriculum learning and progressive difficulty scheduling
"""
import torch
import numpy as np
from typing import Dict, Any, Optional


class ProgressiveTrainingScheduler:
    """
    Manages progressive training strategies including:
    - Curriculum learning with increasing difficulty
    - Progressive loss weighting
    - Adaptive learning rate scheduling
    - Dynamic augmentation strength
    """
    def __init__(
        self,
        total_epochs,
        warmup_epochs=10,
        strategy='linear'
    ):
        """
        Args:
            total_epochs: Total number of training epochs
            warmup_epochs: Number of warmup epochs
            strategy: Progression strategy ('linear', 'exponential', 'cosine')
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.strategy = strategy
        self.current_epoch = 0

    def get_progress(self, epoch=None):
        """Get training progress [0, 1]"""
        if epoch is None:
            epoch = self.current_epoch
        return min(1.0, max(0.0, epoch / self.total_epochs))

    def get_warmup_progress(self, epoch=None):
        """Get warmup progress [0, 1]"""
        if epoch is None:
            epoch = self.current_epoch
        return min(1.0, max(0.0, epoch / self.warmup_epochs))

    def get_difficulty(self, epoch=None):
        """
        Get current difficulty level [0, 1]
        0 = easy, 1 = hard
        """
        progress = self.get_progress(epoch)

        if self.strategy == 'linear':
            difficulty = progress
        elif self.strategy == 'exponential':
            difficulty = progress ** 2
        elif self.strategy == 'cosine':
            difficulty = (1 - np.cos(progress * np.pi)) / 2
        else:
            difficulty = progress

        return difficulty

    def update_epoch(self, epoch):
        """Update current epoch"""
        self.current_epoch = epoch


class CurriculumLossWeighting:
    """
    Progressive loss weighting strategy
    Starts with simple losses and gradually introduces complex ones
    """
    def __init__(
        self,
        initial_weights: Dict[str, float],
        final_weights: Dict[str, float],
        warmup_epochs: int = 20
    ):
        """
        Args:
            initial_weights: Initial loss weights (simpler)
            final_weights: Final loss weights (more complex)
            warmup_epochs: Epochs to transition from initial to final
        """
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.warmup_epochs = warmup_epochs

    def get_weights(self, epoch):
        """Get loss weights for current epoch"""
        if epoch >= self.warmup_epochs:
            return self.final_weights

        # Linear interpolation
        alpha = epoch / self.warmup_epochs
        weights = {}

        for key in self.initial_weights:
            initial = self.initial_weights[key]
            final = self.final_weights.get(key, initial)
            weights[key] = initial + alpha * (final - initial)

        # Add any new losses that only appear in final_weights
        for key in self.final_weights:
            if key not in weights:
                weights[key] = alpha * self.final_weights[key]

        return weights


class DifficultyBasedSampling:
    """
    Sample training data based on difficulty
    Gradually introduces harder examples
    """
    def __init__(
        self,
        initial_easy_ratio=0.8,
        final_easy_ratio=0.3,
        warmup_epochs=30
    ):
        """
        Args:
            initial_easy_ratio: Initial ratio of easy samples
            final_easy_ratio: Final ratio of easy samples
            warmup_epochs: Epochs to transition
        """
        self.initial_easy_ratio = initial_easy_ratio
        self.final_easy_ratio = final_easy_ratio
        self.warmup_epochs = warmup_epochs

    def get_easy_ratio(self, epoch):
        """Get current easy sample ratio"""
        if epoch >= self.warmup_epochs:
            return self.final_easy_ratio

        alpha = epoch / self.warmup_epochs
        return self.initial_easy_ratio + alpha * (
            self.final_easy_ratio - self.initial_easy_ratio
        )


class AdaptiveAugmentationScheduler:
    """
    Progressively increases augmentation strength
    Helps model generalize better as training progresses
    """
    def __init__(
        self,
        initial_strength=0.2,
        final_strength=0.8,
        warmup_epochs=20
    ):
        """
        Args:
            initial_strength: Initial augmentation strength [0, 1]
            final_strength: Final augmentation strength [0, 1]
            warmup_epochs: Epochs to reach final strength
        """
        self.initial_strength = initial_strength
        self.final_strength = final_strength
        self.warmup_epochs = warmup_epochs

    def get_strength(self, epoch):
        """Get augmentation strength for current epoch"""
        if epoch >= self.warmup_epochs:
            return self.final_strength

        # Smooth transition
        alpha = epoch / self.warmup_epochs
        alpha_smooth = (1 - np.cos(alpha * np.pi)) / 2  # Cosine interpolation

        return self.initial_strength + alpha_smooth * (
            self.final_strength - self.initial_strength
        )


class MultiScaleTrainingScheduler:
    """
    Progressive multi-scale training
    Starts with lower resolution and gradually increases
    """
    def __init__(
        self,
        base_size=256,
        scales=(0.5, 0.75, 1.0),
        epochs_per_scale=None
    ):
        """
        Args:
            base_size: Base image size
            scales: List of scale factors
            epochs_per_scale: Epochs to spend at each scale
        """
        self.base_size = base_size
        self.scales = sorted(scales)
        self.epochs_per_scale = epochs_per_scale

    def get_current_size(self, epoch, total_epochs):
        """Get current training image size"""
        if self.epochs_per_scale is None:
            # Automatic scheduling
            progress = epoch / total_epochs
            scale_idx = min(
                int(progress * len(self.scales)),
                len(self.scales) - 1
            )
        else:
            # Fixed epochs per scale
            scale_idx = min(
                epoch // self.epochs_per_scale,
                len(self.scales) - 1
            )

        scale = self.scales[scale_idx]
        size = int(self.base_size * scale)

        return size, scale


class LearningRateWarmup:
    """
    Learning rate warmup and decay scheduling
    """
    def __init__(
        self,
        base_lr,
        warmup_epochs=5,
        total_epochs=100,
        warmup_strategy='linear',
        decay_strategy='cosine',
        min_lr=1e-6
    ):
        """
        Args:
            base_lr: Target learning rate after warmup
            warmup_epochs: Warmup duration
            total_epochs: Total training epochs
            warmup_strategy: 'linear' or 'exponential'
            decay_strategy: 'cosine', 'linear', or 'step'
            min_lr: Minimum learning rate
        """
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_strategy = warmup_strategy
        self.decay_strategy = decay_strategy
        self.min_lr = min_lr

    def get_lr(self, epoch):
        """Get learning rate for current epoch"""
        # Warmup phase
        if epoch < self.warmup_epochs:
            if self.warmup_strategy == 'linear':
                return self.base_lr * (epoch / self.warmup_epochs)
            elif self.warmup_strategy == 'exponential':
                return self.base_lr * ((epoch / self.warmup_epochs) ** 2)
            else:
                return self.base_lr * (epoch / self.warmup_epochs)

        # Decay phase
        progress = (epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )

        if self.decay_strategy == 'cosine':
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(progress * np.pi)
            )
        elif self.decay_strategy == 'linear':
            lr = self.base_lr - (self.base_lr - self.min_lr) * progress
        elif self.decay_strategy == 'step':
            # Step decay at 50%, 75%, 90%
            if progress < 0.5:
                lr = self.base_lr
            elif progress < 0.75:
                lr = self.base_lr * 0.1
            elif progress < 0.9:
                lr = self.base_lr * 0.01
            else:
                lr = self.min_lr
        else:
            lr = self.base_lr

        return max(lr, self.min_lr)


class EarlyStopping:
    """
    Early stopping with patience
    Monitors validation loss and stops if no improvement
    """
    def __init__(
        self,
        patience=10,
        min_delta=1e-4,
        mode='min',
        restore_best=True
    ):
        """
        Args:
            patience: Epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            restore_best: Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, score, epoch):
        """
        Check if training should stop

        Args:
            score: Current validation score
            epoch: Current epoch

        Returns:
            should_stop: Whether to stop training
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        # Check improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best score: {self.best_score:.6f} at epoch {self.best_epoch}")

        return self.should_stop


class ProgressiveTrainingManager:
    """
    Comprehensive progressive training manager
    Coordinates all progressive training strategies
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary with progressive training settings
        """
        self.config = config
        total_epochs = config.get('num_epochs', 100)

        # Initialize schedulers
        self.scheduler = ProgressiveTrainingScheduler(
            total_epochs=total_epochs,
            warmup_epochs=config.get('warmup_epochs', 10),
            strategy=config.get('progression_strategy', 'cosine')
        )

        # Loss weighting
        if config.get('use_curriculum_loss', True):
            initial_weights = {
                'diffusion_loss': 1.0,
                'l1_loss': 1.0,
                'cycle_loss': 0.5,
                'wavelet_loss': 0.0,
                'spectral_consistency': 0.0
            }
            final_weights = {
                'diffusion_loss': 1.0,
                'l1_loss': 1.0,
                'cycle_loss': 0.8,
                'wavelet_loss': 0.5,
                'spectral_consistency': 0.3
            }
            self.loss_weighting = CurriculumLossWeighting(
                initial_weights,
                final_weights,
                warmup_epochs=config.get('loss_warmup_epochs', 20)
            )
        else:
            self.loss_weighting = None

        # Augmentation scheduling
        if config.get('use_progressive_augmentation', True):
            self.aug_scheduler = AdaptiveAugmentationScheduler(
                initial_strength=config.get('initial_aug_strength', 0.2),
                final_strength=config.get('final_aug_strength', 0.8),
                warmup_epochs=config.get('aug_warmup_epochs', 20)
            )
        else:
            self.aug_scheduler = None

        # Learning rate scheduling
        self.lr_scheduler = LearningRateWarmup(
            base_lr=config.get('learning_rate', 1e-4),
            warmup_epochs=config.get('lr_warmup_epochs', 5),
            total_epochs=total_epochs,
            min_lr=config.get('min_lr', 1e-6)
        )

        # Early stopping
        if config.get('use_early_stopping', False):
            self.early_stopping = EarlyStopping(
                patience=config.get('early_stopping_patience', 10),
                min_delta=config.get('early_stopping_delta', 1e-4)
            )
        else:
            self.early_stopping = None

    def update_epoch(self, epoch):
        """Update all schedulers for new epoch"""
        self.scheduler.update_epoch(epoch)

    def get_loss_weights(self, epoch):
        """Get loss weights for current epoch"""
        if self.loss_weighting is None:
            return self.config  # Return original config weights

        return self.loss_weighting.get_weights(epoch)

    def get_augmentation_strength(self, epoch):
        """Get augmentation strength for current epoch"""
        if self.aug_scheduler is None:
            return 0.5  # Default medium strength

        return self.aug_scheduler.get_strength(epoch)

    def get_learning_rate(self, epoch):
        """Get learning rate for current epoch"""
        return self.lr_scheduler.get_lr(epoch)

    def check_early_stopping(self, val_loss, epoch):
        """Check if training should stop early"""
        if self.early_stopping is None:
            return False

        return self.early_stopping(val_loss, epoch)

    def get_training_state(self, epoch):
        """Get complete training state for logging"""
        return {
            'epoch': epoch,
            'progress': self.scheduler.get_progress(epoch),
            'difficulty': self.scheduler.get_difficulty(epoch),
            'learning_rate': self.get_learning_rate(epoch),
            'augmentation_strength': self.get_augmentation_strength(epoch),
            'loss_weights': self.get_loss_weights(epoch)
        }

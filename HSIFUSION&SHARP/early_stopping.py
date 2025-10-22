"""
Early Stopping Implementation for HSIFusionNet and SHARP Training
Author: Thierry Silvio Claude Soreze
Date: 2025-10-21

This module provides early stopping functionality to prevent overfitting and
save compute resources by terminating training when validation metrics plateau.
"""

import numpy as np
from typing import Optional


class EarlyStopping:
    """Early stopping to prevent overfitting and save compute resources.

    Monitors a validation metric and stops training when the metric stops
    improving for a specified number of validation checks (patience).

    Example:
        >>> early_stopping = EarlyStopping(patience=20, min_delta=1e-4, mode='min')
        >>> for epoch in range(num_epochs):
        >>>     train_loss = train_one_epoch()
        >>>     val_loss = validate()
        >>>
        >>>     if early_stopping(val_loss):
        >>>         print(f"Early stopping triggered at epoch {epoch}")
        >>>         break
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = 'min',
        baseline: Optional[float] = None,
        restore_best_weights: bool = False,
        verbose: bool = True
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of validation checks with no improvement after which
                     training will be stopped. For example, if val_interval=10 and
                     patience=20, training stops after 200 epochs with no improvement.
            min_delta: Minimum change in monitored metric to qualify as improvement.
                      For example, if min_delta=1e-4, then val_loss must decrease
                      by at least 0.0001 to be considered an improvement.
            mode: One of {'min', 'max'}. In 'min' mode, training stops when metric
                  stops decreasing. In 'max' mode, training stops when metric stops
                  increasing.
            baseline: Baseline value for the monitored metric. Training will stop if
                     metric doesn't show improvement over baseline.
            restore_best_weights: Whether to restore model weights from the epoch with
                                 the best value of the monitored metric. Requires
                                 external handling.
            verbose: Whether to print messages when early stopping occurs.
        """
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        # Internal state
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

        # Validation
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        # Set comparison operator based on mode
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1

    def __call__(self, current: float, epoch: Optional[int] = None) -> bool:
        """
        Check if training should be stopped.

        Args:
            current: Current value of the monitored metric
            epoch: Current epoch number (optional, for logging)

        Returns:
            True if training should be stopped, False otherwise
        """
        if self.best_score is None:
            # First call - initialize best score
            self.best_score = current
            self.best_epoch = epoch if epoch is not None else 0
            if self.verbose:
                print(f"Early stopping initialized with {self.mode}={current:.6f}")
            return False

        # Check if baseline is met
        if self.baseline is not None:
            if self.mode == 'min' and current >= self.baseline:
                if self.verbose:
                    print(f"Early stopping: metric {current:.6f} did not improve over baseline {self.baseline:.6f}")
                self.early_stop = True
                return True
            elif self.mode == 'max' and current <= self.baseline:
                if self.verbose:
                    print(f"Early stopping: metric {current:.6f} did not improve over baseline {self.baseline:.6f}")
                self.early_stop = True
                return True

        # Check if current value is better than best score
        if self.monitor_op(current - self.min_delta, self.best_score):
            # Improvement detected
            improvement = abs(current - self.best_score)
            self.best_score = current
            self.best_epoch = epoch if epoch is not None else self.best_epoch
            self.counter = 0

            if self.verbose:
                epoch_str = f" (epoch {epoch})" if epoch is not None else ""
                print(f"Early stopping: metric improved by {improvement:.6f} to {current:.6f}{epoch_str}")
        else:
            # No improvement
            self.counter += 1

            if self.verbose:
                epoch_str = f" (epoch {epoch})" if epoch is not None else ""
                print(f"Early stopping: no improvement for {self.counter}/{self.patience} checks{epoch_str}")

            if self.counter >= self.patience:
                if self.verbose:
                    print(f"\nEarly stopping triggered!")
                    print(f"  Best {self.mode}={self.best_score:.6f} at epoch {self.best_epoch}")
                    print(f"  Current {self.mode}={current:.6f}")
                    print(f"  No improvement for {self.patience} validation checks")
                self.early_stop = True
                return True

        return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

    @property
    def stopped(self) -> bool:
        """Check if early stopping has been triggered."""
        return self.early_stop

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'early_stop': self.early_stop
        }

    def load_state_dict(self, state_dict: dict):
        """Load state dict from checkpoint."""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.best_epoch = state_dict['best_epoch']
        self.early_stop = state_dict['early_stop']


# Example usage with HSIFusionNet training
def example_hsifusion_integration():
    """
    Example of how to integrate early stopping into HSIFusionNet training.

    Add this to hsifusion_training.py:
    """
    from early_stopping import EarlyStopping

    # In HSIFusionTrainingConfig:
    # early_stopping_patience: int = 30
    # early_stopping_min_delta: float = 1e-4

    # In HSIFusionTrainer.__init__:
    # self.early_stopping = EarlyStopping(
    #     patience=config.early_stopping_patience,
    #     min_delta=config.early_stopping_min_delta,
    #     mode='min',
    #     verbose=True
    # )

    # In HSIFusionTrainer.train():
    # for epoch in range(self.start_epoch, self.config.epochs):
    #     # ... training code ...
    #
    #     if (epoch + 1) % self.config.val_interval == 0:
    #         metrics = self.validate()
    #
    #         # Check early stopping
    #         if self.early_stopping(metrics["mrae"], epoch=epoch+1):
    #             print(f"Training stopped early at epoch {epoch+1}")
    #             print(f"Best MRAE: {self.best_mrae:.6f}")
    #             break


# Example usage with SHARP training
def example_sharp_integration():
    """
    Example of how to integrate early stopping into SHARP training.

    Add this to sharp_training_script_fixed.py:
    """
    from early_stopping import EarlyStopping

    # In SHARPTrainingConfig:
    # early_stopping_patience: int = 20
    # early_stopping_min_delta: float = 1e-4

    # In DedicatedSHARPTrainer.__init__:
    # self.early_stopping = EarlyStopping(
    #     patience=config.early_stopping_patience,
    #     min_delta=config.early_stopping_min_delta,
    #     mode='min',
    #     verbose=True
    # )

    # In DedicatedSHARPTrainer.train():
    # for epoch in range(self.start_epoch, self.config.epochs):
    #     # ... training code ...
    #
    #     if (epoch + 1) % self.config.val_interval == 0:
    #         val_metrics = self._validate(epoch)
    #
    #         # Check early stopping
    #         if self.early_stopping(val_metrics['mrae'], epoch=epoch+1):
    #             print(f"Training stopped early at epoch {epoch+1}")
    #             print(f"Best MRAE: {self.best_mrae:.6f}")
    #             break


if __name__ == "__main__":
    # Test early stopping
    print("Testing Early Stopping implementation...\n")

    # Test case 1: Early stopping with improvement
    print("=== Test 1: Early stopping with improvement ===")
    es = EarlyStopping(patience=3, min_delta=1e-4, mode='min', verbose=True)

    test_losses = [0.1, 0.09, 0.08, 0.081, 0.082, 0.083, 0.084]  # Should stop after 0.084
    for epoch, loss in enumerate(test_losses):
        if es(loss, epoch=epoch+1):
            print(f"Stopped at epoch {epoch+1}")
            break

    print(f"\nBest score: {es.best_score:.6f} at epoch {es.best_epoch}")

    # Test case 2: Mode='max'
    print("\n=== Test 2: Mode='max' (accuracy) ===")
    es_max = EarlyStopping(patience=3, min_delta=1e-4, mode='max', verbose=True)

    test_accs = [0.8, 0.85, 0.87, 0.869, 0.868, 0.867, 0.866]  # Should stop after 0.866
    for epoch, acc in enumerate(test_accs):
        if es_max(acc, epoch=epoch+1):
            print(f"Stopped at epoch {epoch+1}")
            break

    print(f"\nBest score: {es_max.best_score:.6f} at epoch {es_max.best_epoch}")

    # Test case 3: State dict save/load
    print("\n=== Test 3: State dict save/load ===")
    es_save = EarlyStopping(patience=5, mode='min')
    es_save(0.1, epoch=1)
    es_save(0.09, epoch=2)

    state = es_save.state_dict()
    print(f"Saved state: {state}")

    es_load = EarlyStopping(patience=5, mode='min')
    es_load.load_state_dict(state)
    print(f"Loaded state: {es_load.state_dict()}")

    print("\n✅ All tests passed!")

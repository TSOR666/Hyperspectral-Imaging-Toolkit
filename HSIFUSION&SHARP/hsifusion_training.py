#!/usr/bin/env python
"""
Training script for HSIFusionNet v2.5.3 ("Lightning Pro").
Provides a modern training loop with AMP, torch.compile optional support,
and metric tracking for reconstruction quality.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hsifusion_v252_complete import create_hsifusion_lightning_pro
from optimized_dataloader import MSTPlusPlusLoss, create_optimized_dataloaders


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class HSIFusionTrainingConfig:
    """Configuration for HSIFusion training."""

    # Model
    model_size: str = "base"
    in_channels: int = 3
    out_channels: int = 31

    # Data
    data_root: str = "./dataset"
    batch_size: int = 12
    num_workers: int = 4
    memory_mode: str = "float16"  # standard | float16 | lazy
    patch_size: int = 128
    stride: int = 8
    augment: bool = True

    # Optimisation
    epochs: int = 300
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    accumulate_steps: int = 1
    warmup_epochs: int = 5

    # Runtime features
    use_amp: bool = True
    compile_model: bool = True
    use_channels_last: bool = True

    # Validation & logging
    val_interval: int = 10
    log_interval: int = 50
    save_interval: int = 50

    # Output
    output_dir: str = "./experiments/hsifusion"
    experiment_name: Optional[str] = None
    resume_from: Optional[str] = None

    # Misc
    device: str = "cuda"
    seed: int = 42

    def experiment_path(self) -> Path:
        root = Path(self.output_dir)
        name = self.experiment_name or f"hsifusion_{self.model_size}"
        return root / name


class HSIFusionTrainer:
    """Trainer that handles optimisation and evaluation for HSIFusion."""

    def __init__(self, config: HSIFusionTrainingConfig):
        self.config = config
        set_random_seed(config.seed)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.exp_dir = config.experiment_path()
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.exp_dir / "tensorboard")
        self.criterion = MSTPlusPlusLoss()
        self.scaler = GradScaler(enabled=config.use_amp)

        self.model = self._build_model()
        self.train_loader, self.val_loader = self._build_dataloaders()
        steps_per_epoch = max(1, len(self.train_loader))
        self.optimizer, self.scheduler = self._build_optimisers(steps_per_epoch)
        self.start_epoch = 0
        self.iteration = 0
        self.best_mrae = math.inf

        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    def _build_model(self) -> nn.Module:
        """Create HSIFusion model and move to device."""
        model = create_hsifusion_lightning_pro(
            model_size=self.config.model_size,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            compile_mode="reduce-overhead" if self.config.compile_model else None,
            expected_min_size=self.config.patch_size,
            lazy_compile=False,
            force_compile=self.config.compile_model,
        )
        model = model.to(self.device)
        if self.config.use_channels_last and self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        return model

    def _build_optimisers(
        self, steps_per_epoch: int
    ) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Create optimiser and scheduler."""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )

        warmup_epochs = max(1, self.config.warmup_epochs)
        effective_steps = max(1, math.ceil(steps_per_epoch / self.config.accumulate_steps))
        total_steps = effective_steps * self.config.epochs
        warmup_steps = effective_steps * warmup_epochs

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(1e-6, step / max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, scheduler

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        return create_optimized_dataloaders(self.config)

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "iteration": self.iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_mrae": self.best_mrae,
            "config": self.config,
        }
        suffix = "best.pth" if is_best else f"epoch_{epoch:04d}.pth"
        path = self.exp_dir / "checkpoints" / suffix
        torch.save(checkpoint, path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0)
        self.iteration = ckpt.get("iteration", 0)
        self.best_mrae = ckpt.get("best_mrae", math.inf)
        print(f"Resumed training from epoch {self.start_epoch}")

    def _step_scheduler(self) -> None:
        if isinstance(self.scheduler, optim.lr_scheduler.LambdaLR):
            self.scheduler.step()
        else:
            self.scheduler.step()

    def _compute_metrics(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        pred = prediction.detach()
        tgt = target.detach()
        mse = F.mse_loss(pred, tgt).item()
        rmse = math.sqrt(mse)
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-8))
        mrae = torch.mean(torch.abs(pred - tgt) / (tgt + 1e-8)).item()
        return {"mrae": mrae, "rmse": rmse, "psnr": psnr}

    def train(self) -> None:
        """Main training loop."""
        total_steps = max(1, len(self.train_loader))
        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            for batch_idx, (rgb, hsi) in enumerate(self.train_loader):
                rgb = rgb.to(self.device, non_blocking=True)
                hsi = hsi.to(self.device, non_blocking=True)

                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(rgb)
                    if isinstance(outputs, tuple):
                        outputs, _ = outputs
                    loss = self.criterion(outputs, hsi)
                    aux_loss = self.model.get_auxiliary_loss()
                    if torch.is_tensor(aux_loss):
                        loss = loss + aux_loss
                loss = loss / self.config.accumulate_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.accumulate_steps == 0:
                    if self.config.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._step_scheduler()

                running_loss += loss.item()
                self.iteration += 1

                if self.iteration % self.config.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/loss", loss.item(), self.iteration)
                    self.writer.add_scalar("train/lr", current_lr, self.iteration)

            avg_loss = running_loss / total_steps
            print(f"Epoch {epoch+1}/{self.config.epochs} - train loss: {avg_loss:.6f}")

            if (epoch + 1) % self.config.val_interval == 0:
                metrics = self.validate()
                self.writer.add_scalar("val/mrae", metrics["mrae"], epoch + 1)
                self.writer.add_scalar("val/rmse", metrics["rmse"], epoch + 1)
                self.writer.add_scalar("val/psnr", metrics["psnr"], epoch + 1)

                is_best = metrics["mrae"] < self.best_mrae
                if is_best:
                    self.best_mrae = metrics["mrae"]
                    print(f"New best MRAE: {self.best_mrae:.6f}")
                self._save_checkpoint(epoch + 1, is_best=is_best)
            elif (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch + 1, is_best=False)

        print("Training complete.")
        print(f"Best validation MRAE: {self.best_mrae:.6f}")

    def validate(self) -> Dict[str, float]:
        """Run full-image validation."""
        self.model.eval()
        metrics_agg = {"mrae": 0.0, "rmse": 0.0, "psnr": 0.0}
        count = 0

        with torch.no_grad():
            for rgb, hsi in self.val_loader:
                rgb = rgb.to(self.device, non_blocking=True)
                hsi = hsi.to(self.device, non_blocking=True)

                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(rgb)
                    if isinstance(outputs, tuple):
                        outputs, _ = outputs

                metrics = self._compute_metrics(outputs, hsi)
                for key in metrics_agg:
                    metrics_agg[key] += metrics[key]
                count += 1

        if count == 0:
            return {k: 0.0 for k in metrics_agg}
        return {k: v / count for k, v in metrics_agg.items()}


def parse_args() -> HSIFusionTrainingConfig:
    parser = argparse.ArgumentParser(description="Train HSIFusionNet v2.5.3 Lightning Pro")
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["tiny", "small", "base", "large", "xlarge"])
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--memory_mode", type=str, default="float16",
                        choices=["standard", "float16", "lazy"])
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./experiments/hsifusion")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return HSIFusionTrainingConfig(
        model_size=args.model_size,
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        compile_model=not args.no_compile,
        memory_mode=args.memory_mode,
        patch_size=args.patch_size,
        stride=args.stride,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        resume_from=args.resume,
        device=args.device,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    trainer = HSIFusionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

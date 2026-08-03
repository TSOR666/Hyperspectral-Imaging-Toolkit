#!/usr/bin/env python
"""Unified MST++/NTIRE-faithful trainer for HSIFusion v2.5.3 and SHARP v3.2.2.

One trainer, one protocol, switched by ``--model {hsifusion,sharp}``. Replaces the
per-model scripts (hsifusion_training.py / sharp_training_script_fixed.py) as the
canonical entrypoint; those are retained only because the pass-3/4/5 regression
tests pin their internals.

MST++/ARAD-1K protocol defaults (NTIRE 2022 spectral reconstruction):
  - data: ARAD-1K layout (split_txt/{train,valid}_list.txt, Train_RGB/, Train_Spec/),
    128x128 patches on a stride-8 grid, random flips + k*90-degree rotations,
    RGB / 255, HSI cubes pre-normalized to [0, 1]
  - objective: MRAE = mean(|pred - gt| / max(|gt|, eps)); the training floor
    anneals from 1e-2 to 1e-3 so dark pixels cannot explode mixed-precision
    gradients, while validation/selection retain the exact 1e-6 protocol floor
  - recipe: Adam(beta1=0.9, beta2=0.999), lr 4e-4, batch 20, 300 epochs,
    cosine annealing to eta_min = 1e-6 stepped PER ITERATION, no weight decay,
    no warmup (both available as opt-in deviations and reported in the config dump)
  - validation: full scenes at batch 1; the SELECTION metric is MRAE on the MST++
    center region (crop_border=128: 482x512 -> 226x256, i.e. [..., 128:-128, 128:-128]);
    full-frame metrics are reported alongside for NTIRE-style comparison
  - metrics: MRAE, RMSE, PSNR, SAM (degrees), SSIM (+ MAE) from hsi_benchmark.metrics,
    computed per scene in fp32 and averaged over scenes

Inputs are reflect-padded to a multiple of 8 before the forward pass and cropped
back afterwards: SHARP's exact-doubling decoder otherwise CRASHES on the real
ARAD validation frames (482 is not divisible by 8), and HSIFusion enforces a
64-pixel minimum side.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# hsi_benchmark (repo root) is the single home of the metric implementations;
# importing it here instead of re-implementing keeps trainer and benchmark numbers
# bit-identical.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:
    from hsi_benchmark.metrics import compute_hsi_metrics
except ImportError as exc:  # pragma: no cover - depends on repo layout
    raise ImportError(
        "unified_training requires the repo-root hsi_benchmark package for its "
        "metric implementations (MRAE/RMSE/PSNR/SAM/SSIM). Run from the "
        "Hyperspectral-Imaging-Toolkit checkout."
    ) from exc

from optimized_dataloader import MSTPlusPlusLoss, create_optimized_dataloaders
from training_stability import (
    annealed_mrae_epsilon,
    autocast_context,
    make_grad_scaler,
    resolve_amp_dtype,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard optional
    SummaryWriter = None

MODEL_CHOICES = ("hsifusion", "sharp")
METRIC_KEYS = ("mrae", "rmse", "psnr", "sam", "ssim", "mae")
PAD_MULTIPLE = 8


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class UnifiedTrainingConfig:
    """Every field is echoed to stdout and persisted to config.json at startup."""

    # Model selection
    model: str = "sharp"                  # hsifusion | sharp
    model_size: str = "base"
    # Extra factory kwargs (JSON on the CLI), e.g. '{"key_rbf_mode": "linear"}' for
    # SHARP or '{"spectral_min_bands_per_group": 4}' for HSIFusion.
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    compile_model: bool = False

    # Data (MST++ defaults)
    data_root: str = "./dataset"
    batch_size: int = 20
    patch_size: int = 128
    stride: int = 8
    augment: bool = True
    num_workers: int = 4
    memory_mode: str = "float16"          # standard | float16 | lazy
    cache_size: int = 4

    # Recipe (MST++ defaults; deviations are opt-in and visible in the config dump)
    epochs: int = 300
    learning_rate: float = 4e-4
    eta_min: float = 1e-6                 # cosine floor, stepped per iteration
    optimizer: str = "adam"               # adam (MST++) | adamw (deviation)
    weight_decay: float = 0.0             # only applied by adamw, norms/bias exempt
    warmup_epochs: float = 0.0            # 0 = MST++ faithful
    accumulate_steps: int = 1
    gradient_clip: float = 1.0            # stability guard; 0 disables
    ema_decay: float = 0.0                # 0 = off (MST++ has no EMA)

    # Precision / stability
    amp: str = "auto"                     # auto | bf16 | fp16 | off
    fp16_init_scale: float = 1024.0         # conservative for relative-error gradients
    max_consecutive_nonfinite: int = 8
    train_mrae_eps_start: float = 1e-2      # stable optimization floor
    train_mrae_eps_end: float = 1e-3        # exact 1e-6 remains the validation metric
    train_mrae_eps_anneal_steps: int = 50_000

    # Validation / selection
    val_interval: int = 10
    val_crop_border: int = 128            # MST++ [..., 128:-128, 128:-128]; auto-skipped
                                          # (with a warning) for scenes too small to crop
    mrae_eps: float = 1e-6

    # Logging / output
    output_dir: str = "./experiments/unified"
    experiment_name: Optional[str] = None
    resume_from: Optional[str] = None
    checkpoint_interval_steps: int = 5_000  # rolling mid-epoch recovery; 0 disables
    log_interval: int = 100
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.model not in MODEL_CHOICES:
            raise ValueError(f"model must be one of {MODEL_CHOICES}, got {self.model!r}")
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be adam or adamw")
        if self.amp not in {"auto", "bf16", "fp16", "off"}:
            raise ValueError("amp must be auto/bf16/fp16/off")
        if not math.isfinite(self.fp16_init_scale) or self.fp16_init_scale <= 0:
            raise ValueError("fp16_init_scale must be a finite positive number")
        # Validate all schedule values eagerly instead of failing after model construction.
        annealed_mrae_epsilon(
            self.train_mrae_eps_start,
            self.train_mrae_eps_end,
            0,
            self.train_mrae_eps_anneal_steps,
        )
        if not math.isfinite(self.mrae_eps) or self.mrae_eps <= 0:
            raise ValueError("mrae_eps must be a finite positive number")
        if self.accumulate_steps <= 0:
            raise ValueError("accumulate_steps must be > 0")
        if self.val_interval <= 0:
            raise ValueError("val_interval must be > 0")
        if self.max_consecutive_nonfinite <= 0:
            raise ValueError("max_consecutive_nonfinite must be > 0")
        if self.checkpoint_interval_steps < 0:
            raise ValueError("checkpoint_interval_steps must be >= 0")
        if self.val_crop_border < 0:
            raise ValueError("val_crop_border must be >= 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if not isinstance(self.model_kwargs, dict):
            raise ValueError("model_kwargs must be a JSON object")

    # Attributes create_optimized_dataloaders reads via getattr
    distributed: bool = field(default=False, init=False)
    rank: int = field(default=0, init=False)
    world_size: int = field(default=1, init=False)

    def experiment_path(self) -> Path:
        name = self.experiment_name or f"{self.model}_{self.model_size}"
        return Path(self.output_dir) / name


# ============================================================================
# Shared helpers (also imported by unified_inference.py)
# ============================================================================

def build_model(
    model: str,
    model_size: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    compile_model: bool = False,
) -> nn.Module:
    """Single construction point for both families (pass-5 defaults ON)."""
    kwargs = dict(model_kwargs or {})
    if model == "hsifusion":
        from hsifusion_v252_complete import create_hsifusion_lightning_pro

        kwargs.setdefault("standard_attn_rope", True)
        kwargs.setdefault("spectral_min_bands_per_group", 4)
        kwargs.setdefault("cross_attention_max_tokens", 1024)
        return create_hsifusion_lightning_pro(
            model_size=model_size,
            compile_mode="reduce-overhead" if compile_model else None,
            force_compile=compile_model,
            rank=0 if compile_model else 1,  # rank!=0 silences the factory banner
            **kwargs,
        )
    if model == "sharp":
        from sharp_v322_hardened import create_sharp_v32

        return create_sharp_v32(
            model_size=model_size,
            compile_model=compile_model,
            verbose=False,
            **kwargs,
        )
    raise ValueError(f"Unknown model family {model!r}")


def unwrap_model(model: nn.Module) -> nn.Module:
    model = getattr(model, "module", model)     # DDP
    return getattr(model, "_orig_mod", model)   # torch.compile


def pad_to_multiple(
    x: torch.Tensor, multiple: int = PAD_MULTIPLE, min_size: int = 64
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Reflect-pad H/W up to a multiple (and floor) so both models accept the frame.

    SHARP's decoder doubles exactly while its encoder floors, so any side not
    divisible by 8 crashes (the real ARAD 482x512 frames included); HSIFusion
    rejects sides < 64. Returns the padded tensor and the original (H, W).
    """
    h, w = int(x.shape[-2]), int(x.shape[-1])
    target_h = max(math.ceil(h / multiple) * multiple, min_size)
    target_w = max(math.ceil(w / multiple) * multiple, min_size)
    if (target_h, target_w) == (h, w):
        return x, (h, w)
    pad = (0, target_w - w, 0, target_h - h)
    # reflect requires pad < dim; fall back for very small inputs
    mode = "reflect" if pad[1] < w and pad[3] < h else "replicate"
    return F.pad(x, pad, mode=mode), (h, w)


def crop_back(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return x[..., : size[0], : size[1]]


def forward_reconstruction(model: nn.Module, rgb: torch.Tensor) -> torch.Tensor:
    """Forward with padding handled; normalizes tuple outputs (uncertainty heads)."""
    padded, original = pad_to_multiple(rgb)
    out = model(padded)
    if isinstance(out, tuple):
        out = out[0]
    return crop_back(out, original)


def evaluate_scene(
    prediction: torch.Tensor,
    target: torch.Tensor,
    crop_border: int,
    eps: float,
) -> Dict[str, Dict[str, float]]:
    """Full-frame and MST++-crop metrics for one scene (fp32, on CPU)."""
    pred = prediction.detach().float().cpu()
    truth = target.detach().float().cpu()
    rows: Dict[str, Dict[str, float]] = {}
    full, _ = compute_hsi_metrics(pred, truth, epsilon=eps)
    rows["full"] = {k: full[k] for k in METRIC_KEYS}
    h, w = truth.shape[-2:]
    if crop_border > 0 and h > 2 * crop_border and w > 2 * crop_border:
        cropped, _ = compute_hsi_metrics(pred, truth, epsilon=eps, crop_border=crop_border)
        rows["crop"] = {k: cropped[k] for k in METRIC_KEYS}
    return rows


def format_metric_table(protocols: Dict[str, Dict[str, float]], indent: str = "  ") -> str:
    header = f"{indent}{'protocol':<12}" + "".join(f"{k.upper():>10}" for k in METRIC_KEYS)
    lines = [header]
    for name, row in protocols.items():
        lines.append(
            f"{indent}{name:<12}"
            + "".join(f"{row[k]:>10.4f}" for k in METRIC_KEYS)
        )
    return "\n".join(lines)


# ============================================================================
# Trainer
# ============================================================================

class UnifiedTrainer:
    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self._set_seed(config.seed)

        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        self.exp_dir = config.experiment_path()
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = MSTPlusPlusLoss(eps=config.train_mrae_eps_start)
        self.amp_dtype = resolve_amp_dtype(config.amp, self.device)
        self.use_amp = self.amp_dtype is not None
        self.scaler = make_grad_scaler(
            self.device, self.amp_dtype, init_scale=config.fp16_init_scale
        )

        self.model = build_model(
            config.model, config.model_size, config.model_kwargs, config.compile_model
        ).to(self.device)
        self._orig_model = unwrap_model(self.model)

        self.train_loader, self.val_loader = create_optimized_dataloaders(config)
        steps_per_epoch = max(1, len(self.train_loader))
        self.optimizer_steps_per_epoch = math.ceil(steps_per_epoch / config.accumulate_steps)
        self.total_optimizer_steps = self.optimizer_steps_per_epoch * config.epochs

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.ema_state: Optional[Dict[str, torch.Tensor]] = (
            {
                name: p.detach().float().cpu().clone()
                for name, p in self._orig_model.named_parameters()
                if p.requires_grad
            }
            if config.ema_decay > 0
            else None
        )

        self.writer = (
            SummaryWriter(log_dir=str(self.exp_dir / "tensorboard"))
            if SummaryWriter is not None
            else None
        )

        self.start_epoch = 0
        self.iteration = 0
        self.optimizer_step = 0
        self.best_mrae = math.inf
        self.consecutive_nonfinite = 0

        self._dump_config()
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _autocast(self, enabled: bool = True):
        return autocast_context(
            self.device, self.amp_dtype if enabled and self.use_amp else None
        )

    def _set_training_mrae_epsilon(self) -> float:
        eps = annealed_mrae_epsilon(
            self.config.train_mrae_eps_start,
            self.config.train_mrae_eps_end,
            self.optimizer_step,
            self.config.train_mrae_eps_anneal_steps,
        )
        self.criterion.eps = eps
        return eps

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        if cfg.optimizer == "adam":
            # MST++: plain Adam, no weight decay
            return torch.optim.Adam(
                self._orig_model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999)
            )
        norm_types = tuple(
            t for t in (
                nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            )
        )
        decay, no_decay = [], []
        for module in self._orig_model.modules():
            for name, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if isinstance(module, norm_types) or name.endswith("bias") or p.dim() <= 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """MST++ cosine annealing to eta_min, stepped per optimizer step, with an
        optional linear warmup prefix (warmup_epochs=0 keeps the exact MST++ schedule)."""
        cfg = self.config
        total = max(1, self.total_optimizer_steps)
        warmup = int(round(cfg.warmup_epochs * self.optimizer_steps_per_epoch))
        floor = cfg.eta_min / cfg.learning_rate

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(1.0, max(0.0, progress))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _dump_config(self) -> None:
        cfg_dict = dataclasses.asdict(self.config)
        params = sum(p.numel() for p in self._orig_model.parameters() if p.requires_grad)
        banner = {
            "model_family": self.config.model,
            "trainable_parameters": params,
            "device": str(self.device),
            "amp_dtype": str(self.amp_dtype),
            "rolling_checkpoint": str(self.exp_dir / "last.pth"),
            "train_batches_per_epoch": len(self.train_loader),
            "val_scenes": len(self.val_loader.dataset),
            "total_optimizer_steps": self.total_optimizer_steps,
            "selection_metric": (
                f"MRAE on MST++ center crop (border {self.config.val_crop_border})"
                if self.config.val_crop_border > 0
                else "MRAE on full frames"
            ),
        }
        print("=" * 78)
        print(f"Unified trainer -- {self.config.model} ({self.config.model_size})")
        print("=" * 78)
        for key, value in banner.items():
            print(f"  {key:<28}: {value}")
        print("-" * 78)
        print("  Resolved configuration:")
        for key in sorted(cfg_dict):
            print(f"    {key:<28}: {cfg_dict[key]}")
        print("=" * 78, flush=True)
        with (self.exp_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump({**cfg_dict, **banner}, handle, indent=2, default=str)

    # ----------------------------------------------------------------- train
    def train(self) -> Dict[str, float]:
        cfg = self.config
        last_metrics: Dict[str, float] = {}
        try:
            for epoch in range(self.start_epoch, cfg.epochs):
                try:
                    epoch_loss = self._train_epoch(epoch)
                    print(
                        f"Epoch {epoch + 1}/{cfg.epochs} - loss {epoch_loss:.6f} - "
                        f"lr {self.optimizer.param_groups[0]['lr']:.2e}"
                    )
                    is_best = False
                    if (epoch + 1) % cfg.val_interval == 0 or (epoch + 1) == cfg.epochs:
                        metrics = self.validate(epoch + 1)
                        last_metrics = metrics
                        selection = metrics.get(
                            "crop/mrae", metrics.get("full/mrae", math.inf)
                        )
                        is_best = math.isfinite(selection) and selection < self.best_mrae
                        if is_best:
                            self.best_mrae = selection
                            print(f"  New best selection MRAE: {self.best_mrae:.6f}")
                        elif not math.isfinite(selection):
                            warnings.warn(
                                "Validation selection MRAE is non-finite; retaining the "
                                "previous best checkpoint."
                            )
                except Exception:
                    # Keep the most recent finite model/optimizer state even when a run
                    # aborts during its first epoch. Resume repeats the interrupted epoch.
                    try:
                        self._save_checkpoint(epoch, is_best=False)
                        print(f"  Saved recovery checkpoint after failure in epoch {epoch + 1}")
                    except Exception as save_exc:  # pragma: no cover - filesystem failure
                        warnings.warn(f"Could not save recovery checkpoint: {save_exc}")
                    raise

                # A rolling checkpoint is written every epoch; validation cadence only
                # controls expensive metrics and best-model selection.
                self._save_checkpoint(epoch + 1, is_best=is_best)
            print(f"Training complete. Best selection MRAE: {self.best_mrae:.6f}")
        finally:
            if self.writer is not None:
                self.writer.close()
        return last_metrics

    def _train_epoch(self, epoch: int) -> float:
        cfg = self.config
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_batches = max(1, len(self.train_loader))
        running, counted = 0.0, 0

        for batch_idx, (rgb, hsi) in enumerate(self.train_loader):
            rgb = rgb.to(self.device, non_blocking=True)
            hsi = hsi.to(self.device, non_blocking=True)
            active_mrae_eps = self._set_training_mrae_epsilon()

            with self._autocast():
                outputs = self.model(rgb)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, hsi)
                aux_fn = getattr(self._orig_model, "get_auxiliary_loss", None)
                if callable(aux_fn):
                    loss = loss + aux_fn()

            if not bool(torch.isfinite(loss).item()):
                self.consecutive_nonfinite += 1
                self.optimizer.zero_grad(set_to_none=True)
                warnings.warn(
                    f"Non-finite loss (epoch {epoch + 1}, batch {batch_idx + 1}, "
                    f"train_mrae_eps={active_mrae_eps:.3g}); skipping."
                )
                if self.consecutive_nonfinite >= cfg.max_consecutive_nonfinite:
                    raise RuntimeError("Too many consecutive non-finite losses; aborting.")
                continue

            # Weight each micro-batch by its own accumulation-group size so the final
            # (possibly partial) group is not under-weighted.
            accum = cfg.accumulate_steps
            n_full = (total_batches // accum) * accum
            group = accum if batch_idx < n_full else max(1, total_batches - n_full)
            self.scaler.scale(loss / group).backward()

            step_now = ((batch_idx + 1) % accum == 0) or (batch_idx + 1 == total_batches)
            if step_now:
                self.scaler.unscale_(self.optimizer)
                grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                finite = (
                    bool(torch.stack([torch.isfinite(g).all() for g in grads]).all().item())
                    if grads else False
                )
                if not finite:
                    self.optimizer.zero_grad(set_to_none=True)
                    # unscale_ already ran: update() resets scaler state (pass-4 fix).
                    self.scaler.update()
                    self.consecutive_nonfinite += 1
                    warnings.warn(
                        f"Non-finite gradients (epoch {epoch + 1}, batch {batch_idx + 1}, "
                        f"train_mrae_eps={active_mrae_eps:.3g}, "
                        f"loss_scale={self.scaler.get_scale():.3g}); skipping optimizer step."
                    )
                    if self.consecutive_nonfinite >= cfg.max_consecutive_nonfinite:
                        raise RuntimeError("Too many consecutive non-finite grads; aborting.")
                    continue
                if cfg.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self._update_ema()
                self.optimizer_step += 1
                self.consecutive_nonfinite = 0
                if (
                    cfg.checkpoint_interval_steps > 0
                    and self.optimizer_step % cfg.checkpoint_interval_steps == 0
                ):
                    # `epoch` is the count of fully completed epochs, so resuming this
                    # mid-epoch checkpoint safely repeats the interrupted epoch.
                    self._save_checkpoint(epoch, is_best=False)

            running += loss.item()
            counted += 1
            self.iteration += 1
            if self.iteration % cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  iter {self.iteration}: loss {loss.item():.6f}, lr {lr:.2e}",
                    flush=True,
                )
                if self.writer is not None:
                    self.writer.add_scalar("train/loss", loss.item(), self.iteration)
                    self.writer.add_scalar("train/lr", lr, self.iteration)
                    self.writer.add_scalar(
                        "train/mrae_epsilon", active_mrae_eps, self.iteration
                    )
        return running / max(1, counted)

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.ema_state is None:
            return
        d = self.config.ema_decay
        for name, p in self._orig_model.named_parameters():
            if name in self.ema_state:
                self.ema_state[name].mul_(d).add_(p.detach().float().cpu(), alpha=1 - d)

    def _ema_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        """Full state_dict with EMA parameters and live buffers (BN stats etc.)."""
        if self.ema_state is None:
            return None
        state = {k: v.detach().cpu().clone() for k, v in self._orig_model.state_dict().items()}
        for name, value in self.ema_state.items():
            state[name] = value.to(state[name].dtype).clone()
        return state

    # -------------------------------------------------------------- validate
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        cfg = self.config
        eval_model = self._orig_model
        used_ema = False
        backup: Optional[Dict[str, torch.Tensor]] = None
        if self.ema_state is not None:
            backup = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
            eval_model.load_state_dict(self._ema_state_dict())
            used_ema = True
        eval_model.eval()

        per_scene: Dict[str, List[Dict[str, float]]] = {"full": [], "crop": []}
        crop_skipped = 0
        start = time.time()
        for rgb, hsi in self.val_loader:
            rgb = rgb.to(self.device, non_blocking=True)
            # fp32 validation: the selection metric must be free of AMP rounding.
            with self._autocast(enabled=False):
                pred = forward_reconstruction(eval_model, rgb)
            rows = evaluate_scene(pred, hsi, cfg.val_crop_border, cfg.mrae_eps)
            per_scene["full"].append(rows["full"])
            if "crop" in rows:
                per_scene["crop"].append(rows["crop"])
            else:
                crop_skipped += 1
        elapsed = time.time() - start

        if backup is not None:
            eval_model.load_state_dict(backup)

        if crop_skipped and cfg.val_crop_border > 0:
            warnings.warn(
                f"{crop_skipped} validation scene(s) too small for the MST++ "
                f"{cfg.val_crop_border}-pixel border crop; falling back to full-frame "
                "metrics for selection on those scenes."
            )

        protocols: Dict[str, Dict[str, float]] = {}
        for name, rows in per_scene.items():
            if rows:
                protocols[name] = {
                    key: float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS
                }
        n_scenes = len(per_scene["full"])
        print(
            f"Validation @ epoch {epoch} ({n_scenes} scenes, {elapsed:.1f}s, "
            f"EMA={'on' if used_ema else 'off'})"
        )
        print(format_metric_table(protocols))

        flat = {
            f"{proto}/{key}": value
            for proto, row in protocols.items()
            for key, value in row.items()
        }
        if self.writer is not None:
            for key, value in flat.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
        with (self.exp_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "ema": used_ema, **flat}) + "\n")
        return flat

    # ------------------------------------------------------------ checkpoint
    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        payload = {
            "unified_version": 1,
            "model": self.config.model,
            "model_size": self.config.model_size,
            "model_kwargs": self.config.model_kwargs,
            "config": dataclasses.asdict(self.config),
            "model_state_dict": self._orig_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": epoch,
            "iteration": self.iteration,
            "optimizer_step": self.optimizer_step,
            "best_mrae": self.best_mrae,
            "train_mrae_eps": float(self.criterion.eps),
        }
        ema_sd = self._ema_state_dict()
        if ema_sd is not None:
            payload["ema_model_state_dict"] = ema_sd
            payload["ema_shadow"] = self.ema_state
        torch.save(payload, self.exp_dir / "last.pth")
        if is_best:
            torch.save(payload, self.exp_dir / "best.pth")
            print(f"  Saved best checkpoint (epoch {epoch})")

    def _load_checkpoint(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        if ckpt.get("model") != self.config.model:
            raise ValueError(
                f"Checkpoint model family {ckpt.get('model')!r} does not match "
                f"--model {self.config.model!r}"
            )
        self._orig_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.iteration = int(ckpt.get("iteration", 0))
        self.optimizer_step = int(
            ckpt.get("optimizer_step", max(0, self.scheduler.last_epoch))
        )
        self.best_mrae = float(ckpt.get("best_mrae", math.inf))
        if self.ema_state is not None and "ema_shadow" in ckpt:
            self.ema_state = {k: v.float().cpu() for k, v in ckpt["ema_shadow"].items()}
        print(f"Resumed from {path} at epoch {self.start_epoch}")


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: Optional[List[str]] = None) -> UnifiedTrainingConfig:
    parser = argparse.ArgumentParser(
        description="Unified MST++/NTIRE-faithful trainer for HSIFusion and SHARP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="sharp", choices=list(MODEL_CHOICES))
    parser.add_argument("--model_size", type=str, default="base")
    parser.add_argument("--model_kwargs", type=str, default="{}",
                        help="JSON object of extra model-factory kwargs")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--memory_mode", type=str, default="float16",
                        choices=["standard", "float16", "lazy"])
    parser.add_argument("--cache_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=float, default=0.0)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA decay (e.g. 0.999); 0 disables (MST++ faithful)")
    parser.add_argument("--amp", type=str, default="auto",
                        choices=["auto", "bf16", "fp16", "off"])
    parser.add_argument("--fp16_init_scale", type=float, default=1024.0,
                        help="Initial GradScaler scale used only for FP16 AMP")
    parser.add_argument("--train_mrae_eps_start", type=float, default=1e-2,
                        help="Initial denominator floor for the training MRAE")
    parser.add_argument("--train_mrae_eps_end", type=float, default=1e-3,
                        help="Final denominator floor for the training MRAE")
    parser.add_argument("--train_mrae_eps_anneal_steps", type=int, default=50_000,
                        help="Optimizer steps for log-linear MRAE-floor annealing")
    parser.add_argument("--mrae_eps", type=float, default=1e-6,
                        help="Exact validation/selection MRAE denominator floor")
    parser.add_argument("--max_consecutive_nonfinite", type=int, default=8,
                        help="Abort after this many failed optimizer attempts")
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--val_crop_border", type=int, default=128,
                        help="MST++ selection crop border (0 = full-frame selection)")
    parser.add_argument("--output_dir", type=str, default="./experiments/unified")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint_interval_steps", type=int, default=5_000,
                        help="Rolling mid-epoch checkpoint cadence (0 disables)")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--model_kwargs is not valid JSON: {exc}")

    return UnifiedTrainingConfig(
        model=args.model,
        model_size=args.model_size,
        model_kwargs=model_kwargs,
        compile_model=args.compile_model,
        data_root=args.data_root,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        augment=not args.no_augment,
        num_workers=args.num_workers,
        memory_mode=args.memory_mode,
        cache_size=args.cache_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        eta_min=args.eta_min,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        accumulate_steps=args.accumulate_steps,
        gradient_clip=args.gradient_clip,
        ema_decay=args.ema_decay,
        amp=args.amp,
        fp16_init_scale=args.fp16_init_scale,
        train_mrae_eps_start=args.train_mrae_eps_start,
        train_mrae_eps_end=args.train_mrae_eps_end,
        train_mrae_eps_anneal_steps=args.train_mrae_eps_anneal_steps,
        mrae_eps=args.mrae_eps,
        max_consecutive_nonfinite=args.max_consecutive_nonfinite,
        val_interval=args.val_interval,
        val_crop_border=args.val_crop_border,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        resume_from=args.resume,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device,
    )


def main(argv: Optional[List[str]] = None) -> None:
    config = parse_args(argv)
    trainer = UnifiedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

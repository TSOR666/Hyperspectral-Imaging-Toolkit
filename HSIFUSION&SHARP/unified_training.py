#!/usr/bin/env python
"""Unified optimizer-update trainer for HSIFusion v2.5.3 and SHARP v3.2.2.

One trainer, one protocol, switched by ``--model {hsifusion,sharp}``. Replaces the
per-model scripts (hsifusion_training.py / sharp_training_script_fixed.py) as the
canonical entrypoint; those are retained only because the pass-3/4/5 regression
tests pin their internals.

MST++/ARAD-1K protocol defaults (NTIRE 2022 spectral reconstruction):
  - data: ARAD-1K layout (split_txt/{train,valid}_list.txt, Train_RGB/, Train_Spec/),
    128x128 patches on a stride-8 grid, random flips + k*90-degree rotations,
    explicit RGB normalization/patch-grid/augmentation policies; HSI unchanged
  - objective: MRAE = mean(|pred - gt| / max(|gt|, eps)); the training floor
    anneals from 1e-2 to 1e-3 so dark pixels cannot explode mixed-precision
    gradients, while validation/selection use an explicitly selected floor; exact mode uses zero
  - recipe: Adam(beta1=0.9, beta2=0.999), lr 4e-4, batch 20, 300,000 successful optimizer updates,
    cosine annealing to eta_min = 1e-6 stepped PER OPTIMIZER UPDATE, no weight decay,
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
import hashlib
import math
import random
import sys
import subprocess
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
    from hsi_benchmark.metrics import compute_hsi_metrics, mrae_breakdown
    from hsi_benchmark.training_health import TrainingHealthMonitor
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
    augmentation_policy: str = 'legacy'
    num_workers: int = 4
    memory_mode: str = "float16"          # standard | float16 | lazy
    cache_size: int = 4
    rgb_normalization: str = "divide_255"
    patch_grid: str = "ceil"
    strict_files: bool = True
    exclude_samples: List[str] = field(default_factory=list)

    # Recipe (MST++ defaults; deviations are opt-in and visible in the config dump)
    epochs: int = 300
    updates_per_epoch: int = 1000         # logical epoch, independent of loader length
    max_optimizer_steps: Optional[int] = None  # defaults to 300 * 1000
    warmup_steps: Optional[int] = None
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
    train_mrae_eps_end: float = 1e-3        # validation floor is independently configured
    train_mrae_eps_anneal_steps: int = 50_000
    loss_mode: str = "floored"            # exact: strictly positive targets, epsilon=0
    auxiliary_loss: bool = True
    health_interval_steps: int = 0        # 0 disables instrumentation

    # Validation / selection
    val_interval: Optional[int] = None    # deprecated logical-epoch cadence
    val_interval_steps: int = 1000
    strict_selection_crop: bool = False
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
        if self.loss_mode not in {"exact", "floored"}:
            raise ValueError("loss_mode must be exact/floored")
        annealed_mrae_epsilon(
            self.train_mrae_eps_start,
            self.train_mrae_eps_end,
            0,
            self.train_mrae_eps_anneal_steps,
        )
        if not math.isfinite(self.mrae_eps) or self.mrae_eps < 0:
            raise ValueError("mrae_eps must be finite and nonnegative")
        if self.accumulate_steps <= 0:
            raise ValueError("accumulate_steps must be > 0")
        if self.updates_per_epoch <= 0 or self.epochs <= 0:
            raise ValueError("epochs and updates_per_epoch must be > 0")
        if self.max_optimizer_steps is None:
            self.max_optimizer_steps = self.epochs * self.updates_per_epoch
        if self.max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be > 0")
        if self.val_interval is not None:
            if self.val_interval <= 0:
                raise ValueError("val_interval must be > 0")
            self.val_interval_steps = self.val_interval * self.updates_per_epoch
        if self.val_interval_steps <= 0:
            raise ValueError("val_interval_steps must be > 0")
        if self.warmup_steps is None:
            self.warmup_steps = round(self.warmup_epochs * self.updates_per_epoch)
        if not 0 <= self.warmup_steps < self.max_optimizer_steps:
            raise ValueError("warmup_steps must be >= 0 and below max_optimizer_steps")
        if self.health_interval_steps < 0 or self.log_interval <= 0:
            raise ValueError("health interval must be >= 0 and log_interval > 0")
        if self.rgb_normalization not in {"divide_255", "scene_minmax"} or self.patch_grid not in {"floor", "ceil"}:
            raise ValueError("Unknown RGB normalization or patch grid")
        if self.augmentation_policy not in {'legacy', 'mst'}:
            raise ValueError("augmentation_policy must be legacy/mst")
        if self.loss_mode == "exact" and self.memory_mode == "float16":
            raise ValueError("Exact MRAE requires standard or lazy FP32 target storage")
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
        if not (0 < self.eta_min <= self.learning_rate < math.inf):
            raise ValueError("Learning rates must satisfy 0 < eta_min <= learning_rate")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip < 0:
            raise ValueError("gradient_clip must be finite and nonnegative")

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


def build_model_from_config(family: str, resolved: Dict[str, Any]) -> nn.Module:
    """Reconstruct every saved architecture field, including output-changing controls."""
    if family == 'hsifusion':
        from hsifusion_v252_complete import HSIFusionNetV25LightningPro, LightningProConfig
        return HSIFusionNetV25LightningPro(LightningProConfig(**resolved))
    if family == 'sharp':
        from sharp_v322_hardened import SHARPv32, SHARPv32Config
        return SHARPv32(SHARPv32Config(**resolved))
    raise ValueError(f"Unknown model family {family!r}")


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

        self.criterion = MSTPlusPlusLoss(eps=0.0 if config.loss_mode == "exact" else config.train_mrae_eps_start)
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
        if len(self.train_loader) == 0:
            raise ValueError("Training loader has no batches; reduce batch_size")
        self.optimizer_steps_per_epoch = config.updates_per_epoch
        self.total_optimizer_steps = config.max_optimizer_steps
        self._data_iterator = None
        self.loader_cycles = 0
        self.resolved_model_config = json.loads(json.dumps(dataclasses.asdict(self._orig_model.config), default=str))
        self.run_manifest = self._build_manifest()
        self.health = TrainingHealthMonitor(self._orig_model) if config.health_interval_steps else None
        self._last_metrics = {}
        self._last_validation_step = -1

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

        if config.resume_from:
            self._load_checkpoint(config.resume_from)
        elif (self.exp_dir / 'last.pth').exists():
            raise FileExistsError("Experiment already has a checkpoint; use --resume or a new experiment_name")
        self._dump_config()

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
        if getattr(self.config, 'loss_mode', 'floored') == 'exact':
            self.criterion.eps = 0.0
            return 0.0
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
        optional linear warmup prefix (warmup_steps=0 has no warmup)."""
        cfg = self.config
        total = max(1, self.total_optimizer_steps)
        warmup = cfg.warmup_steps
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
            json.dump({**cfg_dict, **banner, 'resolved_model_config': self.resolved_model_config,
                       'run_manifest': self.run_manifest}, handle, indent=2, default=str)

    def _build_manifest(self):
        splits = {}
        for split, loader in [('train', self.train_loader), ('valid', self.val_loader)]:
            path = Path(self.config.data_root) / 'split_txt' / (split + '_list.txt')
            scenes = [Path(path).stem for path in loader.dataset.hsi_files]
            splits[split] = {'split_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'effective_scenes': scenes,
                             'effective_sha256': hashlib.sha256('\n'.join(scenes).encode()).hexdigest()}
        def git(*args):
            result = subprocess.run(['git', *args], cwd=_REPO_ROOT, capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        return {'code_revision': git('rev-parse', 'HEAD'),
                'code_dirty': bool(git('status', '--porcelain')),
                'source_sha256': {str(path.relative_to(_REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(__file__), Path(__file__).with_name('optimized_dataloader.py'),
                                               Path(__file__).with_name('reconstruction_layers.py'),
                                               Path(__file__).with_name('common_utils_v32.py'),
                                               Path(__file__).with_name('hsifusion_v252_complete.py'),
                                               Path(__file__).with_name('sharp_v322_hardened.py'),
                                               _REPO_ROOT / 'hsi_benchmark' / 'metrics.py']},
                'data': {'splits': splits, 'rgb_normalization': self.config.rgb_normalization,
                         'patch_grid': self.config.patch_grid, 'patch_size': self.config.patch_size,
                         'stride': self.config.stride, 'augment': self.config.augment,
                         'augmentation_policy': self.config.augmentation_policy,
                         'strict_files': self.config.strict_files,
                         'exclude_samples': self.config.exclude_samples,
                         'training_storage': 'float16' if self.config.memory_mode == 'float16' else 'float32',
                         'validation_storage': 'float32'},
                'protocol': {'loss_mode': self.config.loss_mode, 'selection_epsilon': self.config.mrae_eps,
                             'effective_amp': str(self.amp_dtype) if self.use_amp else 'off',
                             'optimizer': self.config.optimizer,
                             'effective_weight_decay': self.config.weight_decay if self.config.optimizer == 'adamw' else 0.,
                             'strict_selection_crop': self.config.strict_selection_crop,
                             'val_crop_border': self.config.val_crop_border,
                             'selection_output': 'raw', 'counter': 'successful_optimizer_updates'}}

    # ----------------------------------------------------------------- train
    def _next_batch(self):
        if self._data_iterator is None:
            self._data_iterator = iter(self.train_loader)
        try:
            return next(self._data_iterator)
        except StopIteration:
            self.loader_cycles += 1
            sampler = getattr(self.train_loader, "sampler", None)
            if callable(getattr(sampler, "set_epoch", None)):
                sampler.set_epoch(self.loader_cycles)
            self._data_iterator = iter(self.train_loader)
            return next(self._data_iterator)

    def _validate_and_checkpoint(self):
        metrics = self.validate(self.optimizer_step)
        self._last_metrics = metrics
        self._last_validation_step = self.optimizer_step
        # Require the same selection population at every validation.
        selection = metrics.get("crop/mrae", metrics["full/mrae"])
        if not math.isfinite(selection):
            raise FloatingPointError("Non-finite validation selection MRAE")
        is_best = selection < self.best_mrae
        if is_best:
            self.best_mrae = selection
        self._save_checkpoint(self.optimizer_step // self.config.updates_per_epoch, is_best)

    def train(self) -> Dict[str, float]:
        try:
            while self.optimizer_step < self.total_optimizer_steps:
                logical_epoch = self.optimizer_step // self.config.updates_per_epoch
                loss = self._train_epoch(logical_epoch)
                print(f"Update {self.optimizer_step}/{self.total_optimizer_steps}: "
                      f"loss {loss:.6f}, lr {self.optimizer.param_groups[0]['lr']:.2e}", flush=True)
            if self._last_validation_step != self.optimizer_step:
                self._validate_and_checkpoint()
            print(f"Training complete. Best selection MRAE: {self.best_mrae:.6f}")
            return self._last_metrics
        except Exception:
            try:
                self._save_checkpoint(self.optimizer_step // self.config.updates_per_epoch, False)
                print(f"Saved recovery checkpoint at optimizer update {self.optimizer_step}")
            except Exception as exc:
                warnings.warn(f"Could not save recovery checkpoint: {exc}")
            raise
        finally:
            if self.writer is not None:
                self.writer.close()
            if self.health is not None:
                self.health.close()

    def _train_epoch(self, epoch: int) -> float:
        """Train one logical block of successful updates; recycle data as needed."""
        cfg = self.config
        goal = min((epoch + 1) * cfg.updates_per_epoch, self.total_optimizer_steps)
        self.model.train()
        running, counted = 0.0, 0
        while self.optimizer_step < goal:
            self.optimizer.zero_grad(set_to_none=True)
            active_mrae_eps = self._set_training_mrae_epsilon()
            sampled = self.health is not None and (
                self.optimizer_step == 0 or (self.optimizer_step + 1) % cfg.health_interval_steps == 0)
            if self.health is not None:
                self.health.enabled = sampled
                self.health.activations.clear()
            failed = False
            group_loss = 0.0
            for _ in range(cfg.accumulate_steps):
                rgb, hsi = self._next_batch()
                self.iteration += 1
                rgb, hsi = rgb.to(self.device), hsi.to(self.device)
                with self._autocast():
                    output = self.model(rgb)
                    if isinstance(output, tuple):
                        output = output[0]
                    loss = self.criterion(output, hsi)
                    aux_fn = getattr(self._orig_model, "get_auxiliary_loss", None)
                    if cfg.auxiliary_loss and callable(aux_fn):
                        loss = loss + aux_fn()
                if not torch.isfinite(loss).item():
                    failed = True
                    break
                self.scaler.scale(loss / cfg.accumulate_steps).backward()
                group_loss += loss.detach().item() / cfg.accumulate_steps
            if not failed:
                self.scaler.unscale_(self.optimizer)
                grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                failed = not grads or not all(torch.isfinite(g).all().item() for g in grads)
                # GradScaler state must be reset after unscale even when the step fails.
                if failed:
                    self.scaler.update()
            if failed:
                self.optimizer.zero_grad(set_to_none=True)
                self.consecutive_nonfinite += 1
                warnings.warn(f"Non-finite loss/gradient at update {self.optimizer_step}; discarded accumulation group")
                if self.consecutive_nonfinite >= cfg.max_consecutive_nonfinite:
                    raise RuntimeError("Too many consecutive non-finite grads/losses; aborting.")
                continue
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.gradient_clip if cfg.gradient_clip > 0 else float('inf'),
                error_if_nonfinite=True)
            if sampled:
                self.health.before_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer_step += 1
            self.scheduler.step()
            self._update_ema()
            self.consecutive_nonfinite = 0
            running += group_loss
            counted += 1
            if sampled:
                record = {"optimizer_step": self.optimizer_step, "loss": group_loss,
                          "pre_clip_gradient_norm": float(grad_norm), "health": self.health.after_step()}
                with (self.exp_dir / 'training_health.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record) + '\n')
            self.optimizer.zero_grad(set_to_none=True)
            if self.optimizer_step % cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  update {self.optimizer_step}: loss {group_loss:.6f}, lr {lr:.2e}", flush=True)
                if self.writer is not None:
                    self.writer.add_scalar('train/loss', group_loss, self.optimizer_step)
                    self.writer.add_scalar('train/lr', lr, self.optimizer_step)
                    self.writer.add_scalar('train/mrae_epsilon', active_mrae_eps, self.optimizer_step)
            if (self.optimizer_step % cfg.val_interval_steps == 0
                    or self.optimizer_step == self.total_optimizer_steps):
                if self.health is not None:
                    self.health.enabled = False
                self._validate_and_checkpoint()
                self.model.train()
            elif cfg.checkpoint_interval_steps and self.optimizer_step % cfg.checkpoint_interval_steps == 0:
                self._save_checkpoint(epoch, False)
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
    def validate(self, step: int) -> Dict[str, float]:
        cfg = self.config
        eval_model = self._orig_model
        was_training = eval_model.training
        health_enabled = self.health.enabled if self.health is not None else False
        backup = None
        per_scene = {"full": [], "crop": []}
        diagnostics = []
        crop_skipped = 0
        start = time.time()
        try:
            if self.ema_state is not None:
                backup = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
                eval_model.load_state_dict(self._ema_state_dict())
            eval_model.eval()
            for index, (rgb, hsi) in enumerate(self.val_loader):
                if self.health is not None:
                    self.health.enabled = True
                    self.health.activations.clear()
                with self._autocast(enabled=False):
                    pred = forward_reconstruction(eval_model, rgb.float().to(self.device))
                rows = evaluate_scene(pred, hsi, cfg.val_crop_border, cfg.mrae_eps)
                per_scene["full"].append(rows["full"])
                if "crop" in rows:
                    per_scene["crop"].append(rows["crop"])
                else:
                    crop_skipped += 1
                name = Path(self.val_loader.dataset.hsi_files[index]).stem
                for protocol in rows:
                    prediction, target = pred.detach().float().cpu(), hsi.float()
                    if protocol == 'crop':
                        border = cfg.val_crop_border
                        prediction = prediction[..., border:-border, border:-border]
                        target = target[..., border:-border, border:-border]
                    diagnostics.append({'scene': name, 'protocol': protocol,
                                        'operators': {name: value for name, value in self.health.activations.items()
                                                      if 'attention_operator' in value} if self.health is not None else {},
                                        **mrae_breakdown(prediction, target, epsilon=cfg.mrae_eps)})
        finally:
            if backup is not None:
                eval_model.load_state_dict(backup)
            eval_model.train(was_training)
            if self.health is not None:
                self.health.enabled = health_enabled
        if not per_scene['full']:
            raise ValueError("Validation split has no scenes")
        if crop_skipped and cfg.val_crop_border > 0:
            if cfg.strict_selection_crop:
                raise ValueError("Validation scene is too small for the required selection crop")
            # A partial crop average would select checkpoints on a different population.
            per_scene['crop'].clear()
            warnings.warn(f"{crop_skipped} scene(s) too small for the {cfg.val_crop_border}-pixel "
                          "selection crop; using full-frame selection for the entire split.")
        protocols = {name: {key: float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS}
                     for name, rows in per_scene.items() if rows}
        print(f"Validation @ update {step} ({len(per_scene['full'])} scenes, "
              f"{time.time() - start:.1f}s, EMA={'on' if backup is not None else 'off'})")
        print(format_metric_table(protocols))
        flat = {f"{proto}/{key}": value for proto, row in protocols.items() for key, value in row.items()}
        if self.writer is not None:
            for key, value in flat.items():
                self.writer.add_scalar(f"val/{key}", value, step)
        with (self.exp_dir / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'optimizer_step': step, 'ema': backup is not None, **flat}) + '\n')
        with (self.exp_dir / 'validation_diagnostics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'optimizer_step': step, 'scenes': diagnostics}) + '\n')
        return flat

    # ------------------------------------------------------------ checkpoint
    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        payload = {
            "unified_version": 2,
            "model": self.config.model,
            "model_size": self.config.model_size,
            "model_kwargs": self.config.model_kwargs,
            "config": dataclasses.asdict(self.config),
            "resolved_model_config": self.resolved_model_config,
            "run_manifest": self.run_manifest,
            "model_state_dict": self._orig_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": epoch,
            "iteration": self.iteration,
            "optimizer_step": self.optimizer_step,
            "best_mrae": self.best_mrae,
            "train_mrae_eps": float(self.criterion.eps),
            "total_optimizer_steps": self.total_optimizer_steps,
            "loader_cycles": self.loader_cycles,
            "last_validation_step": self._last_validation_step,
            "last_metrics": self._last_metrics,
        }
        ema_sd = self._ema_state_dict()
        if ema_sd is not None:
            payload["ema_model_state_dict"] = ema_sd
            payload["ema_shadow"] = self.ema_state
        def save_atomic(name):
            destination = self.exp_dir / name
            temporary = destination.with_suffix('.pth.tmp')
            torch.save(payload, temporary)
            temporary.replace(destination)
        save_atomic("last.pth")
        if is_best:
            save_atomic("best.pth")
            print(f"  Saved best checkpoint (update {self.optimizer_step})")

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
        if ckpt.get('unified_version') != 2:
            raise ValueError("Legacy checkpoints use a different schedule; load them for inference or start a fresh run")
        canonical = lambda value: json.dumps(value, sort_keys=True, default=str)
        if canonical(ckpt.get('resolved_model_config')) != canonical(self.resolved_model_config):
            raise ValueError("Resume architecture semantics differ from the saved resolved_model_config")
        if canonical(ckpt.get('run_manifest', {}).get('data')) != canonical(self.run_manifest['data']):
            raise ValueError("Resume data split/preprocessing/storage policy differs from checkpoint")
        if canonical(ckpt.get('run_manifest', {}).get('protocol')) != canonical(self.run_manifest['protocol']):
            raise ValueError("Resume effective precision/objective/selection policy differs from checkpoint")
        fields = ('max_optimizer_steps', 'updates_per_epoch', 'accumulate_steps', 'batch_size',
                  'learning_rate', 'eta_min', 'warmup_steps', 'optimizer', 'weight_decay',
                  'gradient_clip', 'amp', 'ema_decay', 'loss_mode', 'auxiliary_loss',
                  'train_mrae_eps_start', 'train_mrae_eps_end', 'train_mrae_eps_anneal_steps',
                  'mrae_eps', 'val_crop_border', 'val_interval_steps', 'strict_selection_crop', 'seed')
        current = dataclasses.asdict(self.config)
        different = [name for name in fields if ckpt['config'].get(name) != current[name]]
        if different:
            raise ValueError("Resume training policy differs: " + ', '.join(different))
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
        self.loader_cycles = int(ckpt.get('loader_cycles', 0))
        self._last_validation_step = int(ckpt.get('last_validation_step', -1))
        self._last_metrics = ckpt.get('last_metrics', {})
        if self.ema_state is not None and "ema_shadow" in ckpt:
            self.ema_state = {k: v.float().cpu() for k, v in ckpt["ema_shadow"].items()}
        if self.scheduler.last_epoch != self.optimizer_step:
            raise ValueError("Checkpoint scheduler and optimizer update counters disagree")
        if self.optimizer_step > self.total_optimizer_steps:
            raise ValueError("Checkpoint exceeds the requested optimizer budget")
        print(f"Resumed from {path} at optimizer update {self.optimizer_step}")


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: Optional[List[str]] = None) -> UnifiedTrainingConfig:
    parser = argparse.ArgumentParser(
        description="Unified optimizer-update trainer for HSIFusion and SHARP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="sharp", choices=list(MODEL_CHOICES))
    parser.add_argument("--recipe", type=Path, help="JSON training recipe; explicit CLI arguments override it")
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
    parser.add_argument("--rgb_normalization", choices=['divide_255', 'scene_minmax'], default='divide_255')
    parser.add_argument("--patch_grid", choices=['ceil', 'floor'], default='ceil')
    parser.add_argument('--augmentation_policy', choices=['legacy', 'mst'], default='legacy')
    parser.add_argument("--exclude_samples", nargs='*', default=[])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--updates_per_epoch", type=int, default=1000,
                        help="Successful updates per logical epoch, independent of loader length")
    parser.add_argument("--max_optimizer_steps", type=int, default=None,
                        help="Total successful optimizer updates; default epochs * 1000")
    parser.add_argument("--warmup_steps", type=int, default=None)
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
                        help="Validation/selection MRAE denominator floor; 0 requires positive targets")
    parser.add_argument("--max_consecutive_nonfinite", type=int, default=8,
                        help="Abort after this many failed optimizer attempts")
    parser.add_argument("--loss_mode", choices=['floored', 'exact'], default='floored')
    parser.add_argument("--no_auxiliary_loss", action='store_false', dest='auxiliary_loss', default=True)
    parser.add_argument("--health_interval_steps", type=int, default=0)
    parser.add_argument("--val_interval", type=int, default=None, help="Deprecated cadence in logical epochs")
    parser.add_argument("--val_interval_steps", type=int, default=1000)
    parser.add_argument('--strict_selection_crop', action='store_true')
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
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.recipe:
        recipe = json.loads(preliminary.recipe.read_text(encoding='utf-8'))
        known = {f.name for f in dataclasses.fields(UnifiedTrainingConfig) if f.init}
        unknown = set(recipe) - known
        if unknown:
            parser.error('Unknown recipe fields: ' + ', '.join(sorted(unknown)))
        rename = {'learning_rate': 'lr', 'resume_from': 'resume', 'augment': 'no_augment'}
        defaults = {rename.get(k, k): (not v if k == 'augment' else v) for k, v in recipe.items()}
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)

    try:
        model_kwargs = json.loads(args.model_kwargs) if isinstance(args.model_kwargs, str) else args.model_kwargs
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--model_kwargs is not valid JSON: {exc}")

    values = vars(args).copy()
    values.pop('recipe')
    values['model_kwargs'] = model_kwargs
    values['learning_rate'] = values.pop('lr')
    values['resume_from'] = values.pop('resume')
    values['augment'] = not values.pop('no_augment')
    return UnifiedTrainingConfig(**values)



def main(argv: Optional[List[str]] = None) -> None:
    config = parse_args(argv)
    trainer = UnifiedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

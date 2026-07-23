#!/usr/bin/env python3
"""MST++/NTIRE-style trainer for CAS-HSI on ARAD-1K.

Recipe (the MST++ defaults every ARAD-1K number in this repo is measured against):

    loss        MRAE, eps 1e-6, on the UNCLAMPED prediction
    data        128x128 patches on a stride-8 grid, rot90 + h/v flips
    optimizer   Adam(0.9, 0.999), lr 4e-4, no weight decay or gradient clipping
    schedule    cosine to 1e-6, stepped PER OPTIMIZER STEP, no warmup
    batch       20
    epochs      300
    validation  full scenes at batch 1, in fp32
    selection   MRAE on the MST++ centre crop (128-px border)

Logged every epoch, for BOTH train and validation: the loss plus MRAE, PSNR, RMSE,
SSIM (and SAM/MAE, which come free from the same pass). Train metrics are computed
on the same batches the loss is computed on, using definitions numerically identical
to the validation ones, so the two curves are directly comparable and the
train-minus-val gap means what you think it means.

    python train_cas_hsi.py --config configs/cas_hsi_tiny.yaml --data_root /path/to/ARAD_1K

Anything in the YAML can be overridden on the command line.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, get_type_hints

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cas_hsi import build_cas_hsi  # noqa: E402
from dataloader import EvalDataset, TrainDataset  # noqa: E402
from evaluation import (  # noqa: E402
    METRIC_KEYS,
    METRICS_SOURCE,
    MST_CROP_BORDER,
    average_rows,
    evaluate_scene,
    forward_scene,
    format_metric_table,
)
from metrics import AverageMeter, batch_metrics, build_criterion  # noqa: E402

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover - torch < 2.4
    from torch.cuda.amp import GradScaler, autocast  # type: ignore


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TrainConfig:
    """Every field is echoed to stdout and persisted to config.json at startup."""

    # --- model ---
    variant: str = "tiny"                 # tiny | base
    backend: str = "research"             # research | edge
    model_overrides: Dict[str, Any] = field(default_factory=dict)

    # --- data (MST++ defaults) ---
    data_root: str = "./data/ARAD_1K"
    patch_size: int = 128
    stride: int = 8
    batch_size: int = 20
    num_workers: int = 4
    pin_memory: bool = True
    rgb_norm: str = "minmax"              # MST++ normalizes each image by its own min/max
    cache_dtype: str = "float32"
    augment: bool = True

    # --- objective ---
    loss: str = "mrae"                    # the ARAD-1K/NTIRE objective
    mrae_eps: float = 1e-6

    # --- optimization (MST++ defaults) ---
    epochs: int = 300
    steps_per_epoch: int = 1000           # <=0 consumes the full loader
    optimizer: str = "adam"               # adam | adamw
    lr: float = 4e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.0             # MST++ uses none
    warmup_epochs: int = 0                # MST++ uses none
    scheduler: str = "cosine"
    gradient_clip: float = 0.0             # MST++ uses none; >0 opts into norm clipping
    accumulate_steps: int = 1

    # --- precision / speed ---
    amp: bool = True
    amp_dtype: str = "auto"               # auto | bf16 | fp16
    channels_last: bool = True

    # --- EMA ---
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_start_epoch: int = 5

    # --- validation ---
    val_interval: int = 1                 # epochs
    val_crop_border: int = MST_CROP_BORDER
    val_tile_size: int = 0                # >0 uses tiled inference (spec 8.7)
    clamp_eval: bool = False              # NTIRE convention is unclamped
    selection_protocol: str = "crop"      # crop | full
    selection_metric: str = "mrae"

    # --- logging / checkpoints ---
    experiment_name: str = "cas_hsi"
    output_root: str = "./experiments"
    log_interval: int = 100               # optimizer steps
    train_metric_interval: int = 1        # compute train metrics every N batches
    save_interval: int = 0                # extra periodic snapshots; 0 = off
    resume: str = ""
    seed: int = 42
    deterministic: bool = False
    max_consecutive_nonfinite: int = 20

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(argv: Optional[List[str]] = None) -> TrainConfig:
    """YAML first, then explicit CLI flags win (a flag left at its default does not)."""

    def parse_bool(value: str) -> bool:
        key = str(value).strip().lower()
        if key in {"1", "true", "yes", "y", "on"}:
            return True
        if key in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(
            f"expected a boolean value (true/false, yes/no, 1/0), got {value!r}"
        )

    parser = argparse.ArgumentParser(
        description="Train CAS-HSI on ARAD-1K (MST++/NTIRE protocol)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file")

    known = {f.name: f for f in fields(TrainConfig)}
    type_hints = get_type_hints(TrainConfig)
    for name, spec in known.items():
        if name == "model_overrides":
            parser.add_argument(
                "--model_overrides", type=str, default=None,
                help='JSON dict of CASHSIConfig overrides, e.g. \'{"drop_path": 0.1}\'',
            )
            continue
        field_type = type_hints[name]
        if field_type is bool:
            parser.add_argument(
                f"--{name}", type=parse_bool, default=None, help=f"(bool) default {spec.default}"
            )
        elif field_type in {str, int, float}:
            parser.add_argument(f"--{name}", type=field_type, default=None)
        else:  # pragma: no cover - guard for future TrainConfig fields
            raise TypeError(f"No CLI converter registered for TrainConfig.{name}: {field_type!r}")

    args = parser.parse_args(argv)

    payload: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{args.config} must contain a YAML mapping")

        # A config may carry a `model:` block (a CASHSIConfig) alongside training keys.
        model_block = loaded.pop("model", None)
        unknown = sorted(set(loaded) - set(known))
        if unknown:
            raise ValueError(
                f"{args.config} contains key(s) the trainer does not understand: "
                f"{', '.join(unknown)}. A typo here would otherwise be a silent no-op."
            )
        payload.update(loaded)
        if model_block:
            overrides = dict(payload.get("model_overrides") or {})
            overrides.update(model_block)
            payload["model_overrides"] = overrides

    for name in known:
        value = getattr(args, name, None)
        if value is None:
            continue
        if name == "model_overrides":
            payload["model_overrides"] = {
                **(payload.get("model_overrides") or {}),
                **json.loads(value),
            }
        else:
            payload[name] = value

    return TrainConfig(**payload)


# ============================================================================
# Helpers
# ============================================================================


def create_logger(log_dir: Path, name: str = "cas_hsi") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_dir / "train.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def capture_rng_state() -> Dict[str, Any]:
    """Capture every RNG stream that affects data order and augmentation."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a checkpointed RNG state, tolerating a CPU/CUDA environment change."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        saved_cuda = state["cuda"]
        if len(saved_cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved_cuda)


def resolve_amp_dtype(requested: str) -> torch.dtype:
    """bf16 where supported: fp16 overflows the attention logits of this family.

    (The SSTrans project in this repo lost a full training run to exactly that.)
    """
    bf16_ok = (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16 if bf16_ok else torch.float16
    if requested == "auto":
        return torch.bfloat16 if bf16_ok else torch.float16
    raise ValueError(f"amp_dtype must be auto|bf16|fp16, got {requested!r}")


class ModelEMA:
    """Exponential moving average of the model weights."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = dict(model.named_parameters())
        for name, ema_param in self.module.named_parameters():
            ema_param.mul_(self.decay).add_(source[name].detach(), alpha=1.0 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, ema_buffer in self.module.named_buffers():
            if name in source_buffers:
                ema_buffer.copy_(source_buffers[name])

    @torch.no_grad()
    def refresh(self, model: nn.Module) -> None:
        """Re-snapshot from the live model (used when EMA starts late)."""
        self.module.load_state_dict(model.state_dict())


# ============================================================================
# Trainer
# ============================================================================


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(config.output_root) / f"{config.experiment_name}_{stamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

        self.logger = create_logger(self.exp_dir)
        set_seed(config.seed, config.deterministic)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channels_last = bool(config.channels_last and self.device.type == "cuda")
        self.amp_enabled = bool(config.amp and self.device.type == "cuda")
        self.amp_dtype = resolve_amp_dtype(config.amp_dtype) if self.amp_enabled else torch.float32

        self._build_model()
        self._build_data()
        self._build_optimization()

        # A GradScaler is only meaningful for fp16; bf16 has fp32's exponent range.
        self.scaler = GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        self.epoch = 0
        self.step = 0
        self.best_selection = float("inf")
        self.best_epoch = -1
        self.consecutive_nonfinite = 0

        self.ema: Optional[ModelEMA] = None
        if config.use_ema:
            self.ema = ModelEMA(self.model, config.ema_decay)
            self._ema_started = False

        if config.resume:
            self._load_checkpoint(config.resume)

        self._dump_config()
        self._history_path = self.exp_dir / "history.csv"
        self._metrics_path = self.exp_dir / "metrics.jsonl"

    # ------------------------------------------------------------------ setup
    def _build_model(self) -> None:
        cfg = self.config
        overrides = dict(cfg.model_overrides or {})
        overrides.setdefault("backend", cfg.backend)
        self.model = build_cas_hsi(cfg.variant, **overrides).to(self.device)
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        info = self.model.get_model_info()
        self.logger.info("=" * 78)
        self.logger.info("CAS-HSI %s (%s backend)", cfg.variant, info["backend"])
        self.logger.info("  parameters      : %s", f"{info['total_parameters']:,}")
        self.logger.info("  widths          : %s (head_dim=%d)", info["widths"], self.model.config.head_dim)
        self.logger.info("  depths          : %s", info["depths"])
        self.logger.info("  half mixer      : %s", info["half_mixer"])
        self.logger.info("  bottleneck      : %s", info["bottleneck_mixers"])
        self.logger.info("  device / amp    : %s / %s", self.device, self.amp_dtype)
        self.logger.info("  metrics source  : %s", METRICS_SOURCE)
        self.logger.info("=" * 78)

    def _build_data(self) -> None:
        cfg = self.config
        self.train_set = TrainDataset(
            cfg.data_root,
            crop_size=cfg.patch_size,
            stride=cfg.stride,
            rgb_norm=cfg.rgb_norm,
            augment=cfg.augment,
            cache_dtype=cfg.cache_dtype,
            expected_bands=self.model.config.output_bands,
            logger=self.logger,
        )
        self.val_set = EvalDataset(
            cfg.data_root,
            split="valid",
            rgb_norm=cfg.rgb_norm,
            cache_dtype=cfg.cache_dtype,
            expected_bands=self.model.config.output_bands,
            logger=self.logger,
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=min(2, cfg.num_workers),
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
        )
        self.logger.info(
            "Data: %d train patches (%d scenes) | %d val scenes",
            len(self.train_set), len(self.train_set.rgb_images), len(self.val_set),
        )

    def _batches_per_epoch(self) -> int:
        available = len(self.train_loader)
        if self.config.steps_per_epoch <= 0:
            return available
        return max(1, min(self.config.steps_per_epoch, available))

    def _build_optimization(self) -> None:
        cfg = self.config
        params = [p for p in self.model.parameters() if p.requires_grad]

        if cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999))
        elif cfg.optimizer == "adamw":
            # Scales, biases and norms are excluded from decay -- decaying a LayerScale
            # gamma pulls the residual branch toward zero, i.e. toward doing nothing.
            decay, no_decay = [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                leaf = name.split(".")[-1]
                if param.ndim <= 1 or "norm" in name or leaf in {
                    "scale", "temperature", "relative_position_bias_table", "residual_scale"
                }:
                    no_decay.append(param)
                else:
                    decay.append(param)
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": cfg.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=cfg.lr,
                betas=(0.9, 0.999),
            )
        else:
            raise ValueError(f"optimizer must be adam|adamw, got {cfg.optimizer!r}")

        # The schedule advances per OPTIMIZER STEP, so accumulation must be divided out
        # or the cosine would only travel 1/accumulate_steps of its arc.
        steps_per_epoch = max(1, math.ceil(self._batches_per_epoch() / max(1, cfg.accumulate_steps)))
        total_steps = max(1, steps_per_epoch * cfg.epochs)
        warmup_steps = max(0, min(total_steps, steps_per_epoch * cfg.warmup_epochs))
        min_factor = min(1.0, max(0.0, cfg.min_lr / cfg.lr)) if cfg.lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = min(1.0, max(0, step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
            if cfg.scheduler == "cosine":
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.criterion = build_criterion(cfg.loss, cfg.mrae_eps).to(self.device)
        self.total_steps = total_steps

        self.logger.info(
            "Optim: %s lr=%.1e -> %.1e (%s, %d steps/epoch, %d total), "
            "warmup=%d steps, wd=%g, clip=%s, loss=%s",
            cfg.optimizer, cfg.lr, cfg.min_lr, cfg.scheduler,
            steps_per_epoch, total_steps, warmup_steps, cfg.weight_decay,
            f"{cfg.gradient_clip:g}" if cfg.gradient_clip > 0 else "off", cfg.loss,
        )

    def _dump_config(self) -> None:
        payload = {
            "train": self.config.to_dict(),
            "model": self.model.config.to_dict(),
            "model_info": self.model.get_model_info(),
            "metrics_source": METRICS_SOURCE,
            "device": str(self.device),
            "amp_dtype": str(self.amp_dtype),
        }
        with (self.exp_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _autocast(self, enabled: Optional[bool] = None):
        active = self.amp_enabled if enabled is None else (enabled and self.amp_enabled)
        return autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=active)

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.channels_last and tensor.dim() == 4:
            return tensor.to(self.device, non_blocking=True, memory_format=torch.channels_last)
        return tensor.to(self.device, non_blocking=True)

    # ------------------------------------------------------------------ train
    def train_epoch(self) -> Dict[str, float]:
        cfg = self.config
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss_meter = AverageMeter()
        metric_meters: Dict[str, AverageMeter] = defaultdict(AverageMeter)

        total_batches = self._batches_per_epoch()
        accumulate = max(1, cfg.accumulate_steps)
        full_groups = (total_batches // accumulate) * accumulate

        progress = tqdm(total=total_batches, desc=f"epoch {self.epoch + 1}/{cfg.epochs}", leave=False)

        for index, (rgb, hsi) in enumerate(self.train_loader):
            if index >= total_batches:
                break

            rgb = self._to_device(rgb)
            hsi = self._to_device(hsi)

            with self._autocast():
                prediction = self.model(rgb)
                # NEVER clamp before the loss: clamp has zero gradient outside its
                # range, so an out-of-range prediction would stop receiving any
                # corrective signal at all (spec 7.4).
                loss = self.criterion(prediction, hsi)

            is_boundary = (index + 1) % accumulate == 0 or (index + 1) == total_batches
            loss_finite = bool(torch.isfinite(loss))

            if loss_finite:
                self.consecutive_nonfinite = 0
                # Weight each micro-batch by its own group size so a trailing partial
                # group is not under-weighted relative to the full ones.
                group = accumulate if index < full_groups else max(1, total_batches - full_groups)
                self.scaler.scale(loss / group).backward()
            else:
                self.consecutive_nonfinite += 1
                self.logger.warning(
                    "Non-finite loss at epoch %d batch %d; dropping this micro-batch's "
                    "gradient (%d in a row)",
                    self.epoch + 1, index, self.consecutive_nonfinite,
                )
                if self.consecutive_nonfinite >= cfg.max_consecutive_nonfinite:
                    raise RuntimeError(
                        f"{self.consecutive_nonfinite} consecutive non-finite losses; aborting. "
                        "Try --amp_dtype bf16, a lower --lr, or --amp false."
                    )
                # Do NOT zero the whole group here: valid gradients from earlier finite
                # micro-batches in this accumulation group are still worth stepping on.

            if is_boundary:
                # Step only if some finite micro-batch actually contributed a gradient.
                # A group that was entirely non-finite has nothing to step on.
                stepped = False
                if any(p.grad is not None for p in self.model.parameters()):
                    if cfg.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    stepped = True
                self.optimizer.zero_grad(set_to_none=True)

                # Advance the schedule once per accumulation boundary, ALWAYS -- even when
                # the group was dropped. The cosine tracks training progress, not the
                # count of successful steps; skipping scheduler.step() on a dropped group
                # (the old `continue` did) left the LR ending above min_lr.
                self.scheduler.step()
                self.step += 1

                if stepped and self.ema is not None:
                    if not self._ema_started and self.epoch >= cfg.ema_start_epoch:
                        # Re-snapshot rather than let a stale epoch-0 copy bleed in.
                        self.ema.refresh(self.model)
                        self._ema_started = True
                        self.logger.info("EMA started at epoch %d", self.epoch + 1)
                    if self._ema_started:
                        self.ema.update(self.model)

                if cfg.log_interval > 0 and self.step % cfg.log_interval == 0:
                    self.logger.info(
                        "  step %d | loss %.6f | lr %.3e",
                        self.step, float(loss.detach()) if loss_finite else float("nan"),
                        self.optimizer.param_groups[0]["lr"],
                    )

            # A non-finite loss must not pollute the reported train averages.
            if loss_finite:
                loss_meter.update(float(loss.detach()), rgb.size(0))
                if cfg.train_metric_interval > 0 and index % cfg.train_metric_interval == 0:
                    for key, value in batch_metrics(prediction.float(), hsi.float(), cfg.mrae_eps).items():
                        metric_meters[key].update(value, rgb.size(0))

            progress.update(1)
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

        progress.close()

        result = {"loss": loss_meter.avg}
        result.update({key: meter.avg for key, meter in metric_meters.items()})
        return result

    # --------------------------------------------------------------- validate
    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        cfg = self.config
        model = self.ema.module if (self.ema is not None and self._ema_started) else self.model
        model.eval()

        rows: Dict[str, List[Dict[str, float]]] = {"full": [], "crop": []}
        loss_meter = AverageMeter()
        started = time.time()

        for rgb, hsi in self.val_loader:
            rgb = rgb.to(self.device, non_blocking=True)
            hsi_device = hsi.to(self.device, non_blocking=True)

            # fp32: the selection metric must not carry AMP rounding.
            with self._autocast(enabled=False):
                prediction = forward_scene(
                    model, rgb, tile_size=cfg.val_tile_size
                )
                loss_meter.update(float(self.criterion(prediction, hsi_device)))

            scene = evaluate_scene(
                prediction,
                hsi,
                crop_border=cfg.val_crop_border,
                epsilon=cfg.mrae_eps,
                clamp=cfg.clamp_eval,
            )
            for protocol, values in scene.items():
                rows[protocol].append(values)

        self.model.train()

        protocols = {name: average_rows(values) for name, values in rows.items() if values}
        selection_protocol = cfg.selection_protocol if cfg.selection_protocol in protocols else "full"
        selection = protocols[selection_protocol][cfg.selection_metric]

        return {
            "loss": loss_meter.avg,
            "protocols": protocols,
            "selection": selection,
            "selection_protocol": selection_protocol,
            "seconds": time.time() - started,
            "ema": self.ema is not None and self._ema_started,
        }

    # ------------------------------------------------------------ checkpoints
    def _state(self, epoch: int) -> Dict[str, Any]:
        state = {
            "cas_hsi_version": 1,
            "epoch": epoch,
            "step": self.step,
            "train_config": self.config.to_dict(),
            "model_config": self.model.config.to_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "best_selection": self.best_selection,
            "best_epoch": self.best_epoch,
        }
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.module.state_dict()
            state["ema_started"] = self._ema_started
        return state

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        state = self._state(epoch)
        torch.save(state, self.ckpt_dir / "last.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")
        if self.config.save_interval > 0 and epoch % self.config.save_interval == 0:
            torch.save(state, self.ckpt_dir / f"epoch_{epoch:04d}.pth")

    def _load_checkpoint(self, path: str) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info("Resuming from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        saved = checkpoint.get("model_config")
        if saved:
            saved = dict(saved)
            saved.pop("channels", None)
            current = self.model.config.to_dict()
            current.pop("channels", None)
            structural = (
                "base_width", "head_dim", "output_bands", "input_channels", "depths",
                "backend", "spectral_head", "ffn_expansion", "spatial_kernel",
                "dilations_half", "dilations_quarter", "enable_stripe_attention",
                "stripe_frequency",
            )
            differences = {
                key: (saved.get(key), current.get(key))
                for key in structural
                if saved.get(key) != current.get(key)
            }
            if differences:
                detail = "; ".join(f"{k}: ckpt={a!r} now={b!r}" for k, (a, b) in differences.items())
                raise RuntimeError(
                    "Checkpoint architecture does not match the requested model; a resume "
                    f"would load weights that mean something else. Differences: {detail}"
                )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if isinstance(checkpoint.get("rng_state"), Mapping):
            restore_rng_state(checkpoint["rng_state"])

        self.epoch = int(checkpoint.get("epoch", 0))
        self.step = int(checkpoint.get("step", 0))
        self.best_selection = float(checkpoint.get("best_selection", float("inf")))
        self.best_epoch = int(checkpoint.get("best_epoch", -1))

        if self.ema is not None and checkpoint.get("ema_state_dict"):
            self.ema.module.load_state_dict(checkpoint["ema_state_dict"])
            self._ema_started = bool(checkpoint.get("ema_started", False))

        self.logger.info(
            "Resumed at epoch %d, step %d (best %s=%.6f)",
            self.epoch, self.step, self.config.selection_metric, self.best_selection,
        )

    # -------------------------------------------------------------- reporting
    def _history_fieldnames(self) -> List[str]:
        """The COMPLETE, fixed CSV column order -- independent of any single row.

        The header is written once, on the first epoch. If val_interval > 1 that first
        epoch has no validation, so deriving the header from its keys would omit every
        val column, and later validation rows would append unlabeled trailing columns
        (a malformed CSV whose val metrics are unrecoverable by name). Enumerating the
        full schema up front makes early rows leave the val cells blank instead.
        """
        columns = ["epoch", "lr", "train/loss"]
        columns += [f"train/{key}" for key in METRIC_KEYS]
        columns += ["val/loss"]
        for protocol in ("full", "crop"):
            columns += [f"val/{protocol}/{key}" for key in METRIC_KEYS]
        columns += ["val/selection", "val/ema"]
        return columns

    def _record(self, epoch: int, train: Dict[str, float], val: Optional[Dict[str, Any]]) -> None:
        row: Dict[str, Any] = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}
        row.update({f"train/{key}": value for key, value in train.items()})
        if val:
            row["val/loss"] = val["loss"]
            for protocol, values in val["protocols"].items():
                row.update({f"val/{protocol}/{k}": v for k, v in values.items()})
            row["val/selection"] = val["selection"]
            row["val/ema"] = val["ema"]

        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        fieldnames = self._history_fieldnames()
        write_header = not self._history_path.exists()
        with self._history_path.open("a", newline="", encoding="utf-8") as handle:
            # restval="" fills the val cells on train-only epochs; extrasaction="ignore"
            # tolerates a stray key (e.g. a crop protocol absent on tiny scenes) without
            # corrupting the fixed header.
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="", extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _log_epoch(self, epoch: int, train: Dict[str, float], val: Optional[Dict[str, Any]]) -> None:
        self.logger.info(
            "epoch %d/%d | train loss %.6f | MRAE %.4f  PSNR %.2f  RMSE %.4f  SSIM %.4f  SAM %.2f",
            epoch, self.config.epochs, train["loss"],
            train.get("mrae", float("nan")), train.get("psnr", float("nan")),
            train.get("rmse", float("nan")), train.get("ssim", float("nan")),
            train.get("sam", float("nan")),
        )
        if val:
            self.logger.info(
                "           | val   loss %.6f | %d scenes in %.1fs | EMA=%s | selection: %s %s = %.6f",
                val["loss"], len(self.val_set), val["seconds"],
                "on" if val["ema"] else "off",
                val["selection_protocol"], self.config.selection_metric.upper(), val["selection"],
            )
            for line in format_metric_table(val["protocols"]).splitlines():
                self.logger.info("%s", line)

    # ------------------------------------------------------------------- loop
    def fit(self) -> Dict[str, Any]:
        cfg = self.config
        self.logger.info(
            "Selection: %s on the %s protocol (%s)",
            cfg.selection_metric.upper(), cfg.selection_protocol,
            f"MST++ centre crop, {cfg.val_crop_border}px border"
            if cfg.selection_protocol == "crop" else "full frame",
        )
        if cfg.clamp_eval:
            self.logger.warning(
                "clamp_eval=True: predictions are clamped to [0,1] before metrics. This "
                "flatters MRAE relative to the unclamped NTIRE convention used by the rest "
                "of this repo. Do not compare these numbers to unclamped ones."
            )

        last_val: Optional[Dict[str, Any]] = None
        for epoch in range(self.epoch, cfg.epochs):
            self.epoch = epoch
            train = self.train_epoch()

            val = None
            if cfg.val_interval > 0 and ((epoch + 1) % cfg.val_interval == 0 or epoch + 1 == cfg.epochs):
                val = self.validate()
                last_val = val

                is_best = val["selection"] < self.best_selection
                if is_best:
                    self.best_selection = val["selection"]
                    self.best_epoch = epoch + 1
                self._save_checkpoint(epoch + 1, is_best=is_best)
                if is_best:
                    self.logger.info(
                        "  new best: %s = %.6f (epoch %d)",
                        cfg.selection_metric.upper(), self.best_selection, self.best_epoch,
                    )
            else:
                self._save_checkpoint(epoch + 1, is_best=False)

            self._log_epoch(epoch + 1, train, val)
            self._record(epoch + 1, train, val)

        self.logger.info(
            "Training complete. Best %s = %.6f at epoch %d. Checkpoints in %s",
            cfg.selection_metric.upper(), self.best_selection, self.best_epoch, self.ckpt_dir,
        )
        return {"best_selection": self.best_selection, "best_epoch": self.best_epoch, "last_val": last_val}


def main(argv: Optional[List[str]] = None) -> int:
    config = load_config(argv)
    trainer = Trainer(config)
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint_payload
from .data import ARAD1KDataset
from .losses import SpectralReconstructionLoss
from .ntire import autocast_dtype, evaluate_loader, resolve_device
from .presets import build_model, get_config


@dataclass(frozen=True)
class TrainingStage:
    patch_size: int
    iterations: int
    batch_size: int
    learning_rate: float

    def __post_init__(self) -> None:
        if min(self.patch_size, self.iterations, self.batch_size) < 1:
            raise ValueError("Stage sizes and iteration count must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("Stage learning_rate must be positive.")


@dataclass(frozen=True)
class LossConfig:
    mrae_weight: float = 0.0
    l1_weight: float = 1.0
    sam_weight: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    data_root: str
    output_dir: str = "runs/hsiformer_arad1k"
    preset: str = "recommended_retrain"
    model: dict[str, Any] = field(default_factory=dict)
    stages: tuple[TrainingStage, ...] = (
        TrainingStage(128, 300_000, 32, 4e-4),
        TrainingStage(256, 50_000, 8, 4e-5),
        TrainingStage(512, 50_000, 1, 4e-5),
    )
    loss: LossConfig = LossConfig()
    min_learning_rate: float = 1e-6
    crops_per_scene: int = 16
    num_workers: int = 8
    validation_every: int = 2_000
    checkpoint_every: int = 2_000
    log_every: int = 50
    validation_tile_size: int | None = None
    validation_overlap: int = 16
    rgb_normalization: str = "scale_255"
    train_manifest: str | None = None
    validation_manifest: str | None = None
    seed: int = 42
    amp: bool = True
    amp_dtype: str = "bf16"
    grad_clip_norm: float | None = 1.0
    warmup_steps: int = 0
    max_consecutive_nonfinite: int = 100

    def __post_init__(self) -> None:
        if not self.data_root:
            raise ValueError("data_root is required.")
        if self.crops_per_scene < 1:
            raise ValueError("crops_per_scene must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.log_every < 1:
            raise ValueError("log_every must be positive.")
        if self.validation_every < 0 or self.checkpoint_every < 0:
            raise ValueError("Validation and checkpoint intervals cannot be negative.")
        if self.validation_tile_size is not None and self.validation_tile_size < 1:
            raise ValueError("validation_tile_size must be positive.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive.")
        if self.amp_dtype not in {"bf16", "fp16"}:
            raise ValueError("amp_dtype must be 'bf16' or 'fp16'.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")
        if self.max_consecutive_nonfinite < 1:
            raise ValueError("max_consecutive_nonfinite must be positive.")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> TrainingConfig:
        data = dict(values)
        data["stages"] = tuple(
            stage
            if isinstance(stage, TrainingStage)
            else TrainingStage(**stage)
            for stage in data.get("stages", cls.stages)
        )
        loss = data.get("loss", {})
        data["loss"] = loss if isinstance(loss, LossConfig) else LossConfig(**loss)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train(
    config: TrainingConfig,
    *,
    resume: str | Path | None = None,
    device: str | torch.device = "auto",
) -> Path:
    """Run MST++-style iteration training and return the latest checkpoint."""
    if not config.stages:
        raise ValueError("At least one training stage is required.")
    if config.min_learning_rate < 0:
        raise ValueError("min_learning_rate cannot be negative.")

    selected_device = (
        resolve_device(device) if isinstance(device, str) else device
    )
    _seed_everything(config.seed)
    if selected_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.resolved.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    model_config = asdict(get_config(config.preset))
    model_config.update(config.model)
    model = build_model(config.preset, **config.model).to(selected_device)
    criterion = SpectralReconstructionLoss(
        l1_weight=config.loss.l1_weight,
        mrae_weight=config.loss.mrae_weight,
        sam_weight=config.loss.sam_weight,
    )
    amp_enabled = config.amp and selected_device.type == "cuda"
    amp_compute_dtype = (
        autocast_dtype(selected_device, config.amp_dtype)
        if amp_enabled
        else torch.float32
    )
    # GradScaler only matters for fp16, whose narrow range needs gradients
    # rescaled around it. bf16 keeps the float32 exponent range, so the scaler
    # stays disabled (a passthrough) yet remains in the checkpoint for resume.
    use_grad_scaler = amp_enabled and amp_compute_dtype == torch.float16
    scaler = _make_grad_scaler(enabled=use_grad_scaler)

    start_stage = 0
    start_stage_step = 0
    global_step = 0
    best_mrae = math.inf
    resume_payload: dict[str, Any] | None = None
    if resume is not None:
        loaded = load_checkpoint_payload(resume, map_location=selected_device)
        if not isinstance(loaded, dict):
            raise TypeError("Training resume checkpoint must be a dictionary.")
        resume_payload = loaded
        saved_config = loaded.get("model_config")
        if saved_config is not None and dict(saved_config) != model_config:
            raise ValueError(
                "Resume checkpoint architecture does not match this training config."
            )
        model.load_state_dict(loaded["model"])
        # Refuse to resume a NaN/Inf-poisoned checkpoint: a run that already
        # collapsed saves non-finite weights (and Adam moments), and resuming
        # them re-poisons the model on the first step with no path to recover.
        # The user must restart from a pre-collapse checkpoint (best.pt or an
        # earlier step_*.pt) instead of the post-collapse latest.pt.
        _require_finite_resume_state(model, loaded.get("optimizer"))
        scaler_state = loaded.get("scaler")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        start_stage = int(loaded.get("stage_index", 0))
        start_stage_step = int(loaded.get("stage_step", 0))
        global_step = int(loaded.get("global_step", 0))
        best_mrae = float(loaded.get("best_mrae", math.inf))

    validation_loader = _build_validation_loader(config, selected_device)
    metrics_path = output_dir / "metrics.jsonl"
    latest_path = checkpoint_dir / "latest.pt"

    for stage_index, stage in enumerate(config.stages):
        if stage_index < start_stage:
            continue
        stage_step = start_stage_step if stage_index == start_stage else 0
        optimizer = Adam(
            model.parameters(),
            lr=stage.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = _build_scheduler(optimizer, stage, config)
        if resume_payload is not None and stage_index == start_stage:
            if resume_payload.get("optimizer"):
                optimizer.load_state_dict(resume_payload["optimizer"])
            if resume_payload.get("scheduler"):
                scheduler.load_state_dict(resume_payload["scheduler"])

        train_loader = _build_train_loader(
            config,
            stage,
            selected_device,
            stage_index,
        )
        train_iterator = iter(train_loader)
        running_loss = torch.zeros((), device=selected_device)
        running_count = 0
        nonfinite_skips = 0
        consecutive_skips = 0
        model.train()
        print(
            f"stage={stage_index + 1}/{len(config.stages)} "
            f"patch={stage.patch_size} batch={stage.batch_size} "
            f"start={stage_step} target={stage.iterations}"
        )

        while stage_step < stage.iterations:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            rgb = batch["cond"].to(selected_device, non_blocking=True)
            target = batch["label"].to(selected_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=amp_compute_dtype if amp_enabled else None,
                enabled=amp_enabled,
            ):
                prediction = model(rgb)
                loss = criterion(prediction, target)

            stepped = _optimization_step(
                loss,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip_norm=config.grad_clip_norm,
                use_grad_scaler=use_grad_scaler,
            )
            scheduler.step()

            stage_step += 1
            global_step += 1
            if stepped:
                running_loss.add_(loss.detach())
                running_count += 1
                consecutive_skips = 0
            else:
                nonfinite_skips += 1
                consecutive_skips += 1
                print(
                    f"warning: non-finite loss/gradient at step {global_step}; "
                    f"optimizer update skipped (cumulative skips={nonfinite_skips})"
                )
                if consecutive_skips >= config.max_consecutive_nonfinite:
                    raise RuntimeError(
                        f"Aborting: {consecutive_skips} consecutive non-finite "
                        f"optimizer steps at global step {global_step}. Training "
                        "is not recovering (e.g. a corrupt sample or an unstable "
                        "configuration). Inspect the data and reduce the learning "
                        "rate before retrying."
                    )

            if global_step % config.log_every == 0:
                mean_loss = (
                    float((running_loss / running_count).item())
                    if running_count
                    else float("nan")
                )
                train_record = {
                    "type": "train",
                    "global_step": global_step,
                    "stage_index": stage_index,
                    "stage_step": stage_step,
                    "loss": mean_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "nonfinite_skips": nonfinite_skips,
                }
                _append_jsonl(metrics_path, train_record)
                print(
                    f"step={global_step} stage_step={stage_step} "
                    f"loss={train_record['loss']:.6f} "
                    f"lr={train_record['learning_rate']:.3e}"
                )
                running_loss.zero_()
                running_count = 0

            should_validate = (
                config.validation_every > 0
                and global_step % config.validation_every == 0
            ) or stage_step == stage.iterations
            should_checkpoint = (
                config.checkpoint_every > 0
                and global_step % config.checkpoint_every == 0
            )
            if should_validate:
                summary, _ = evaluate_loader(
                    model,
                    validation_loader,
                    device=selected_device,
                    tile_size=config.validation_tile_size,
                    overlap=config.validation_overlap,
                    amp=config.amp,
                )
                validation_record = {
                    "type": "validation",
                    "global_step": global_step,
                    "stage_index": stage_index,
                    "stage_step": stage_step,
                    **summary,
                }
                _append_jsonl(metrics_path, validation_record)
                print(
                    "validation "
                    f"mrae={summary['mrae']:.6f} rmse={summary['rmse']:.6f} "
                    f"psnr={summary['psnr']:.4f} sam={summary['sam']:.6f}"
                )
                improved = summary["mrae"] < best_mrae
                if improved:
                    best_mrae = summary["mrae"]
                payload = _training_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    model_config,
                    stage_index,
                    stage_step,
                    global_step,
                    best_mrae,
                )
                _atomic_torch_save(payload, latest_path)
                if improved:
                    _atomic_torch_save(payload, checkpoint_dir / "best.pt")
                model.train()

            if should_checkpoint and not should_validate:
                payload = _training_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    model_config,
                    stage_index,
                    stage_step,
                    global_step,
                    best_mrae,
                )
                _atomic_torch_save(payload, latest_path)
            if should_checkpoint:
                numbered = checkpoint_dir / f"step_{global_step:09d}.pt"
                _atomic_torch_save(
                    _training_payload(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        model_config,
                        stage_index,
                        stage_step,
                        global_step,
                        best_mrae,
                    ),
                    numbered,
                )

        resume_payload = None
        start_stage_step = 0

    return latest_path


def _build_train_loader(
    config: TrainingConfig,
    stage: TrainingStage,
    device: torch.device,
    stage_index: int,
) -> DataLoader:
    dataset = ARAD1KDataset(
        config.data_root,
        split="train",
        manifest_path=config.train_manifest,
        crop_size=stage.patch_size,
        random_crop=True,
        crops_per_scene=config.crops_per_scene,
        augment=True,
        rgb_normalization=config.rgb_normalization,
    )
    generator = torch.Generator().manual_seed(config.seed + stage_index)
    return DataLoader(
        dataset,
        batch_size=stage.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        drop_last=False,
        generator=generator,
    )


def _build_validation_loader(
    config: TrainingConfig,
    device: torch.device,
) -> DataLoader:
    dataset = ARAD1KDataset(
        config.data_root,
        split="validation",
        manifest_path=config.validation_manifest,
        augment=False,
        rgb_normalization=config.rgb_normalization,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )


def _training_payload(
    model: nn.Module,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    scaler: Any,
    config: TrainingConfig,
    model_config: dict[str, Any],
    stage_index: int,
    stage_step: int,
    global_step: int,
    best_mrae: float,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "preset": config.preset,
        "model_config": model_config,
        "training_config": config.to_dict(),
        "stage_index": stage_index,
        "stage_step": stage_step,
        "global_step": global_step,
        "best_mrae": best_mrae,
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_grad_scaler(*, enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _build_scheduler(
    optimizer: Adam,
    stage: TrainingStage,
    config: TrainingConfig,
) -> Any:
    """Cosine decay, optionally preceded by a short linear LR warmup.

    Warmup is off by default (``warmup_steps == 0``) so the schedule and its
    ``state_dict`` shape are identical to the published recipe and remain
    resume-compatible with existing checkpoints.
    """
    warmup = min(config.warmup_steps, max(0, stage.iterations - 1))
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, stage.iterations - warmup),
        eta_min=config.min_learning_rate,
    )
    if warmup <= 0:
        return cosine
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=warmup,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine],
        milestones=[warmup],
    )


def _optimization_step(
    loss: torch.Tensor,
    *,
    model: nn.Module,
    optimizer: Adam,
    scaler: Any,
    grad_clip_norm: float | None,
    use_grad_scaler: bool,
) -> bool:
    """Run one scaled backward + optimizer step, skipping non-finite updates.

    Returns ``True`` if the optimizer stepped and ``False`` if the step was
    skipped because the loss or gradients were non-finite. Skipping a bad batch
    (rather than stepping on NaN/Inf gradients) is what keeps a single overflow
    from poisoning the Adam moments and weights — the failure that turned a
    transient spike into the permanent, unrecoverable collapse in the logs.
    """
    if not torch.isfinite(loss):
        return False

    scaler.scale(loss).backward()

    grads_finite = True
    if grad_clip_norm is not None:
        # unscale_ is a no-op when the scaler is disabled (bf16); the grads are
        # already at scale 1, so clipping operates on the true gradient norm.
        scaler.unscale_(optimizer)
        total_norm = clip_grad_norm_(model.parameters(), grad_clip_norm)
        grads_finite = bool(torch.isfinite(total_norm))
    elif not use_grad_scaler:
        # No clipping and no scaler to guard the step — check explicitly.
        grads_finite = _gradients_are_finite(model)

    if grads_finite or use_grad_scaler:
        # Under fp16 the GradScaler skips the step itself on inf/NaN grads and
        # update() lowers the loss scale; always drive both for it.
        scaler.step(optimizer)
    scaler.update()
    return grads_finite


def _gradients_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        grad = param.grad
        if grad is not None and not bool(torch.isfinite(grad).all()):
            return False
    return True


def _require_finite_resume_state(
    model: nn.Module,
    optimizer_state: dict[str, Any] | None,
) -> None:
    """Reject resuming from a checkpoint whose weights or Adam moments are NaN.

    A collapsed run persists non-finite parameters (and optimizer moments); the
    fp32 master weights and Adam state would re-poison a fresh run on step one.
    Fail loudly so the user resumes from a pre-collapse checkpoint instead.
    """
    for name, param in model.named_parameters():
        if not bool(torch.isfinite(param).all()):
            raise ValueError(
                f"Resume checkpoint parameter '{name}' is non-finite. This "
                "checkpoint is from a collapsed run; resume from an earlier, "
                "finite checkpoint (best.pt or a pre-collapse step_*.pt)."
            )
    if not optimizer_state:
        return
    for state in optimizer_state.get("state", {}).values():
        for key, value in state.items():
            if (
                isinstance(value, torch.Tensor)
                and not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"Resume checkpoint optimizer moment '{key}' is non-finite. "
                    "This checkpoint is from a collapsed run; resume from an "
                    "earlier, finite checkpoint."
                )

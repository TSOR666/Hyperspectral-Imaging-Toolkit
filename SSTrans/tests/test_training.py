from __future__ import annotations

import h5py
import json
import numpy as np
import pytest
import torch
from importlib.resources import files
from pathlib import Path
from PIL import Image

from hsiformer import (
    TrainingConfig,
    TrainingStage,
    build_model_from_checkpoint,
    train,
)
from hsiformer.ntire import autocast_dtype


def _write_tiny_dataset(tmp_path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    rgb_root = data_root / "Train_RGB"
    spectral_root = data_root / "Train_spectral"
    rgb_root.mkdir(parents=True)
    spectral_root.mkdir()
    scene_id = "ARAD_1K_0001"
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(rgb_root / f"{scene_id}.jpg")
    cube = np.full((31, 8, 8), 0.5, dtype=np.float32)
    with h5py.File(spectral_root / f"{scene_id}.mat", "w") as handle:
        handle.create_dataset("cube", data=cube.transpose(0, 2, 1))
    manifest = tmp_path / "split.txt"
    manifest.write_text(f"{scene_id}\n", encoding="utf-8")
    return data_root, manifest


_TINY_MODEL = {
    "hidden_dim": 8,
    "input_resolution": (8, 8),
    "n_blocks": (1,),
    "bottle_depth": 1,
    "n_refine": 1,
    "patch_size": 2,
    "use_checkpoint": False,
}


def test_default_training_config_enables_stability_guards() -> None:
    config = TrainingConfig.from_json("configs/train_arad1k.json")
    assert config.grad_clip_norm == 1.0
    assert config.amp_dtype == "bf16"
    assert config.warmup_steps == 0


def test_autocast_dtype_is_float32_on_cpu() -> None:
    assert autocast_dtype(torch.device("cpu")) == torch.float32
    assert autocast_dtype(torch.device("cpu"), "fp16") == torch.float32


def test_nonfinite_loss_is_skipped_without_corrupting_weights(
    tmp_path, monkeypatch
) -> None:
    import hsiformer.training as training_module
    from hsiformer.losses import SpectralReconstructionLoss as RealLoss

    data_root, manifest = _write_tiny_dataset(tmp_path)

    class FlakyLoss(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self._real = RealLoss(**kwargs)
            self._calls = 0

        def forward(self, prediction, target):
            self._calls += 1
            if self._calls <= 2:
                # A non-finite loss that, if back-propagated, would poison every
                # weight; the trainer must skip the step instead.
                return prediction.sum() * float("nan")
            return self._real(prediction, target)

    monkeypatch.setattr(training_module, "SpectralReconstructionLoss", FlakyLoss)

    config = TrainingConfig(
        data_root=str(data_root),
        output_dir=str(tmp_path / "run"),
        model=_TINY_MODEL,
        stages=(TrainingStage(8, 6, 1, 1e-4),),
        crops_per_scene=1,
        num_workers=0,
        validation_every=6,
        checkpoint_every=6,
        log_every=1,
        train_manifest=str(manifest),
        validation_manifest=str(manifest),
        amp=False,
    )
    latest = train(config, device="cpu")

    restored, _ = build_model_from_checkpoint(latest)
    for param in restored.parameters():
        assert torch.isfinite(param).all()
    with torch.inference_mode():
        prediction = restored(torch.rand(1, 3, 8, 8))
    assert torch.isfinite(prediction).all()

    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    train_records = [record for record in records if record["type"] == "train"]
    assert max(record.get("nonfinite_skips", 0) for record in train_records) >= 2


def test_warmup_scheduler_ramps_then_decays(tmp_path) -> None:
    data_root, manifest = _write_tiny_dataset(tmp_path)
    config = TrainingConfig(
        data_root=str(data_root),
        output_dir=str(tmp_path / "run"),
        model=_TINY_MODEL,
        stages=(TrainingStage(8, 12, 1, 1e-3),),
        crops_per_scene=12,
        num_workers=0,
        validation_every=12,
        checkpoint_every=12,
        log_every=1,
        train_manifest=str(manifest),
        validation_manifest=str(manifest),
        amp=False,
        warmup_steps=4,
    )
    latest = train(config, device="cpu")

    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    lrs = [record["learning_rate"] for record in records if record["type"] == "train"]
    peak = lrs.index(max(lrs))
    assert 0 < peak < len(lrs) - 1  # ramps up, then decays
    assert lrs[0] < max(lrs)
    assert lrs[-1] < max(lrs)

    restored, _ = build_model_from_checkpoint(latest)
    for param in restored.parameters():
        assert torch.isfinite(param).all()


def test_persistent_nonfinite_loss_trips_circuit_breaker(
    tmp_path, monkeypatch
) -> None:
    import hsiformer.training as training_module

    data_root, manifest = _write_tiny_dataset(tmp_path)

    class AlwaysNaNLoss(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()

        def forward(self, prediction, target):
            return prediction.sum() * float("nan")

    monkeypatch.setattr(training_module, "SpectralReconstructionLoss", AlwaysNaNLoss)

    config = TrainingConfig(
        data_root=str(data_root),
        output_dir=str(tmp_path / "run"),
        model=_TINY_MODEL,
        stages=(TrainingStage(8, 50, 1, 1e-4),),
        crops_per_scene=1,
        num_workers=0,
        validation_every=0,
        checkpoint_every=0,
        log_every=1,
        train_manifest=str(manifest),
        validation_manifest=str(manifest),
        amp=False,
        max_consecutive_nonfinite=3,
    )
    with pytest.raises(RuntimeError, match="consecutive non-finite"):
        train(config, device="cpu")


def test_resume_rejects_nonfinite_checkpoint(tmp_path) -> None:
    from hsiformer.checkpoint import load_checkpoint_payload

    data_root, manifest = _write_tiny_dataset(tmp_path)
    config = TrainingConfig(
        data_root=str(data_root),
        output_dir=str(tmp_path / "run"),
        model=_TINY_MODEL,
        stages=(TrainingStage(8, 1, 1, 1e-4),),
        crops_per_scene=1,
        num_workers=0,
        validation_every=1,
        checkpoint_every=1,
        log_every=1,
        train_manifest=str(manifest),
        validation_manifest=str(manifest),
        amp=False,
    )
    latest = train(config, device="cpu")

    payload = load_checkpoint_payload(latest)
    poisoned_key = next(iter(payload["model"]))
    tensor = payload["model"][poisoned_key].clone()
    tensor.view(-1)[0] = float("nan")
    payload["model"][poisoned_key] = tensor
    torch.save(payload, latest)

    with pytest.raises(ValueError, match="non-finite"):
        train(config, resume=str(latest), device="cpu")


def test_packaged_training_config_matches_repository_config() -> None:
    packaged = json.loads(
        files("hsiformer")
        .joinpath("resources", "train_arad1k.json")
        .read_text(encoding="utf-8")
    )
    repository = json.loads(
        Path("configs/train_arad1k.json").read_text(encoding="utf-8")
    )
    assert packaged == repository


def test_published_training_config_uses_l1_loss() -> None:
    config = TrainingConfig.from_json("configs/train_arad1k.json")
    assert config.loss.l1_weight == 1.0
    assert config.loss.mrae_weight == 0.0
    assert config.loss.sam_weight == 0.0


def test_one_iteration_training_cycle_saves_reusable_checkpoint(tmp_path) -> None:
    data_root = tmp_path / "data"
    rgb_root = data_root / "Train_RGB"
    spectral_root = data_root / "Train_spectral"
    rgb_root.mkdir(parents=True)
    spectral_root.mkdir()
    scene_id = "ARAD_1K_0001"

    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(rgb_root / f"{scene_id}.jpg")
    cube = np.full((31, 8, 8), 0.5, dtype=np.float32)
    with h5py.File(spectral_root / f"{scene_id}.mat", "w") as handle:
        handle.create_dataset("cube", data=cube.transpose(0, 2, 1))
    manifest = tmp_path / "split.txt"
    manifest.write_text(f"{scene_id}\n", encoding="utf-8")

    config = TrainingConfig(
        data_root=str(data_root),
        output_dir=str(tmp_path / "run"),
        model={
            "hidden_dim": 8,
            "input_resolution": (8, 8),
            "n_blocks": (1,),
            "bottle_depth": 1,
            "n_refine": 1,
            "patch_size": 2,
            "use_checkpoint": False,
        },
        stages=(TrainingStage(8, 1, 1, 1e-4),),
        crops_per_scene=1,
        num_workers=0,
        validation_every=1,
        checkpoint_every=1,
        log_every=1,
        train_manifest=str(manifest),
        validation_manifest=str(manifest),
        amp=False,
    )
    latest = train(config, device="cpu")

    assert latest.is_file()
    assert (latest.parent / "best.pt").is_file()
    restored, payload = build_model_from_checkpoint(latest)
    assert payload["global_step"] == 1
    with torch.inference_mode():
        prediction = restored(torch.rand(1, 3, 8, 8))
    assert prediction.shape == (1, 31, 8, 8)

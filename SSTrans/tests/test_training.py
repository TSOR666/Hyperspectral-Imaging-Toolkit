from __future__ import annotations

import h5py
import json
import numpy as np
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

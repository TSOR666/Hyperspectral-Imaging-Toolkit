import shutil
import uuid
from pathlib import Path

import pytest
import torch

from hsi_model.utils.training_setup import (
    resolve_resume_stage_position,
    resume_training_state,
    setup_paths,
)


ROOT = Path(__file__).resolve().parents[1] / ".tmp_manual_dataset"


def _case_dir(prefix: str) -> Path:
    path = ROOT / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_setup_paths_skips_local_data_validation_for_huggingface():
    case_dir = _case_dir("setup_hf")
    cfg = {
        "dataset_source": "huggingface",
        "data_dir": str(case_dir / "not_required_locally"),
        "checkpoint_dir": str(case_dir / "ckpt"),
        "log_dir": str(case_dir / "logs"),
    }

    try:
        out = setup_paths(cfg)

        assert Path(out["checkpoint_dir"]).exists()
        assert Path(out["log_dir"]).exists()
        assert out["data_dir"] == str(case_dir / "not_required_locally")
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_setup_paths_accepts_validation_directory_aliases():
    case_dir = _case_dir("setup_mst")
    for name in ("Train_RGB", "Train_Spec", "Validation_RGB", "Validation_Spec", "split_txt"):
        (case_dir / name).mkdir()

    cfg = {
        "dataset_source": "mst",
        "data_dir": str(case_dir),
        "checkpoint_dir": str(case_dir / "ckpt"),
        "log_dir": str(case_dir / "logs"),
    }

    try:
        out = setup_paths(cfg)

        assert Path(out["checkpoint_dir"]).exists()
        assert Path(out["log_dir"]).exists()
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_setup_paths_still_requires_mst_data_dir():
    case_dir = _case_dir("setup_missing")
    cfg = {
        "dataset_source": "mst",
        "data_dir": str(case_dir / "missing"),
        "checkpoint_dir": str(case_dir / "ckpt"),
        "log_dir": str(case_dir / "logs"),
    }

    try:
        with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
            setup_paths(cfg)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_resume_training_state_normalizes_cuda_rng_state_all(monkeypatch):
    case_dir = _case_dir("resume_cuda_rng")
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    ckpt_path = case_dir / "latest_checkpoint.pth"
    captured = {}

    try:
        torch.save(
            {
                "state_dict": source.state_dict(),
                "cuda_rng_state_all": [[1, 2, 3, 4]],
                "iter": 17,
                "epoch": 3,
                "best_mrae": 0.25,
            },
            ckpt_path,
        )

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def fake_set_rng_state_all(states):
            for state in states:
                if not isinstance(state, torch.Tensor):
                    raise TypeError("RNG state must be a torch.ByteTensor")
                if state.device.type != "cpu" or state.dtype != torch.uint8:
                    raise TypeError("RNG state must be a torch.ByteTensor")
            captured["states"] = states

        monkeypatch.setattr(torch.cuda, "set_rng_state_all", fake_set_rng_state_all)

        info = resume_training_state(
            checkpoint_path=str(ckpt_path),
            model=target,
            optimizers={},
            schedulers={},
            scalers={},
            device=torch.device("cpu"),
        )

        assert info["iteration"] == 17
        assert info["epoch"] == 3
        assert info["best_mrae"] == pytest.approx(0.25)
        assert captured["states"][0].tolist() == [1, 2, 3, 4]
        for key, value in target.state_dict().items():
            assert torch.allclose(value, source.state_dict()[key])
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_resume_training_state_does_not_synthesize_missing_stage_position():
    case_dir = _case_dir("resume_legacy_stage")
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    ckpt_path = case_dir / "legacy_checkpoint.pth"

    try:
        torch.save(
            {
                "state_dict": source.state_dict(),
                "iter": 50_000,
                "epoch": 50,
                "best_mrae": 0.5,
            },
            ckpt_path,
        )

        info = resume_training_state(
            checkpoint_path=str(ckpt_path),
            model=target,
            optimizers={},
            schedulers={},
            scalers={},
            device=torch.device("cpu"),
        )

        assert info["iteration"] == 50_000
        assert "stage_idx" not in info
        assert "stage_iter" not in info
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_resume_training_state_preserves_explicit_stage_position():
    case_dir = _case_dir("resume_progressive_stage")
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    ckpt_path = case_dir / "progressive_checkpoint.pth"

    try:
        torch.save(
            {
                "state_dict": source.state_dict(),
                "iter": 125,
                "stage_idx": 1,
                "stage_iter": 25,
            },
            ckpt_path,
        )

        info = resume_training_state(
            checkpoint_path=str(ckpt_path),
            model=target,
            optimizers={},
            schedulers={},
            scalers={},
            device=torch.device("cpu"),
        )

        assert info["iteration"] == 125
        assert info["stage_idx"] == 1
        assert info["stage_iter"] == 25
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_resume_training_state_preserves_early_stopping_state():
    case_dir = _case_dir("resume_early_stopping")
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    ckpt_path = case_dir / "early_stopping_checkpoint.pth"

    try:
        torch.save(
            {
                "state_dict": source.state_dict(),
                "iter": 26_000,
                "best_mrae": 0.2786,
                "early_stopping_best_mrae": 0.2786,
                "early_stopping_bad_epochs": 10,
            },
            ckpt_path,
        )

        info = resume_training_state(
            checkpoint_path=str(ckpt_path),
            model=target,
            optimizers={},
            schedulers={},
            scalers={},
            device=torch.device("cpu"),
        )

        assert info["early_stopping_best_mrae"] == pytest.approx(0.2786)
        assert info["early_stopping_bad_epochs"] == 10
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_resolve_resume_stage_position_derives_single_stage_legacy_iteration():
    stages = [{"iterations": 100_000}]

    assert resolve_resume_stage_position(stages, {"iteration": 50_000}) == (
        0,
        50_000,
    )


def test_resolve_resume_stage_position_maps_legacy_iteration_to_progressive_stage():
    stages = [
        {"iterations": 100},
        {"iterations": 200},
        {"iterations": 300},
    ]

    assert resolve_resume_stage_position(stages, {"iteration": 150}) == (1, 50)
    assert resolve_resume_stage_position(stages, {"iteration": 300}) == (2, 0)
    assert resolve_resume_stage_position(stages, {"iteration": 999}) == (2, 300)


def test_resolve_resume_stage_position_prefers_explicit_stage_position():
    stages = [{"iterations": 100}, {"iterations": 200}]

    assert resolve_resume_stage_position(
        stages, {"iteration": 150, "stage_idx": 0, "stage_iter": 75}
    ) == (0, 75)

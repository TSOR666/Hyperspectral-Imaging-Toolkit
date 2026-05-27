import shutil
import uuid
from pathlib import Path

import pytest
import torch

from hsi_model.utils.training_setup import resume_training_state, setup_paths


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

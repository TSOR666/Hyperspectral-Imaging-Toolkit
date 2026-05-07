import shutil
import uuid
from pathlib import Path

import pytest
import torch

from hsi_model.utils.checkpoint import load_checkpoint


ROOT = Path(__file__).resolve().parents[1] / ".tmp_manual_dataset"


def _case_dir(prefix: str) -> Path:
    path = ROOT / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_load_checkpoint_accepts_training_script_schema():
    case_dir = _case_dir("ckpt_schema")
    model = torch.nn.Linear(3, 2)
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ckpt_path = case_dir / "net_1epoch.pth"

    try:
        torch.save(
            {
                "epoch": 1,
                "iter": 25,
                "state_dict": expected,
                "best_mrae": 0.123,
            },
            ckpt_path,
        )

        target = torch.nn.Linear(3, 2)
        loaded, info = load_checkpoint(target, checkpoint_path=str(ckpt_path), strict=True)

        assert loaded is target
        assert info["iteration"] == 25
        assert info["best_mrae"] == pytest.approx(0.123)
        for key, value in target.state_dict().items():
            assert torch.allclose(value, expected[key])
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_load_checkpoint_strict_mismatch_raises():
    case_dir = _case_dir("ckpt_bad")
    ckpt_path = case_dir / "bad.pth"
    try:
        torch.save({"model_state_dict": torch.nn.Linear(3, 2).state_dict()}, ckpt_path)

        with pytest.raises(RuntimeError):
            load_checkpoint(torch.nn.Linear(4, 2), checkpoint_path=str(ckpt_path), strict=True)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

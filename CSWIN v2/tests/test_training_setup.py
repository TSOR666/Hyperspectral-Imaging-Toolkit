import shutil
import uuid
from pathlib import Path

import pytest

from hsi_model.utils.training_setup import setup_paths


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

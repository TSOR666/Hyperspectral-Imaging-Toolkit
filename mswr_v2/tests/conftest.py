"""Pytest configuration and shared fixtures for MSWR v2 tests."""

import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

# Add parent directory to path for imports
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

def _select_temp_root() -> Path:
    candidates = []
    if os.name == "nt":
        candidates.append(Path("C:/Temp/mswr_pytest"))
    candidates.append(_parent / ".tmp")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue

    raise RuntimeError("No writable temporary directory available for tests.")


_tmp_root = _select_temp_root()
for env_key in ("TMP", "TEMP", "TMPDIR"):
    os.environ[env_key] = str(_tmp_root)
tempfile.tempdir = str(_tmp_root)
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DIR", str(_parent / ".wandb"))


@pytest.fixture
def device():
    """Get available device (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_rgb_tensor():
    """Create a sample RGB tensor for testing."""
    return torch.randn(1, 3, 128, 128)


@pytest.fixture
def sample_hsi_tensor():
    """Create a sample HSI tensor for testing."""
    return torch.randn(1, 31, 128, 128)


@pytest.fixture
def sample_rgb_array():
    """Create a sample RGB numpy array for testing."""
    return np.random.rand(3, 128, 128).astype(np.float32)


@pytest.fixture
def sample_hsi_array():
    """Create a sample HSI numpy array for testing."""
    return np.random.rand(31, 128, 128).astype(np.float32)


@pytest.fixture
def small_rgb_array():
    """Create a small RGB array that's smaller than typical crop size."""
    return np.random.rand(3, 64, 64).astype(np.float32)


@pytest.fixture
def small_hsi_array():
    """Create a small HSI array that's smaller than typical crop size."""
    return np.random.rand(31, 64, 64).astype(np.float32)


@pytest.fixture
def workspace_tmp_dir():
    """Create a temporary directory inside the repo without relying on pytest tmp_path."""
    base_dir = _parent / ".test_runs"
    base_dir.mkdir(exist_ok=True)
    path = base_dir / f"mswr_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

"""Pytest configuration and shared fixtures for MSWR v2 tests."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add parent directory to path for imports
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))


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

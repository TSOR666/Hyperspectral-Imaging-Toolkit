"""Pytest configuration for CAS-HSI."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


def _select_temp_root() -> Path:
    """pytest's default tmp root is not writable on this machine.

    `%LOCALAPPDATA%\\Temp\\pytest-of-<user>` raises PermissionError here (the same
    reason mswr_v2/tests/conftest.py carries this shim), which turns every `tmp_path`
    test into a collection ERROR rather than a failure. Redirect to somewhere we can
    actually write, in-repo as a last resort.
    """
    candidates = []
    if os.name == "nt":
        candidates.append(Path("C:/Temp/cas_hsi_pytest"))
    candidates.append(_PROJECT / ".tmp")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise RuntimeError("No writable temporary directory available for the test suite.")


_TMP_ROOT = _select_temp_root()
for _key in ("TMP", "TEMP", "TMPDIR"):
    os.environ[_key] = str(_TMP_ROOT)
tempfile.tempdir = str(_TMP_ROOT)

from cas_hsi import CASHSIConfig, build_cas_hsi  # noqa: E402
from cas_hsi.config import Depths  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: full-resolution runs; slow on CPU")


@pytest.fixture(scope="session")
def small_config() -> CASHSIConfig:
    """A structurally complete but shallow model: every stage and mixer, one block each.

    Deep enough to exercise the encoder, both downsamples, the bottleneck's stripe
    block, both skip fusions and the head -- shallow enough to run in a test suite.
    """
    return CASHSIConfig(
        name="cas_hsi_test",
        base_width=32,
        head_dim=32,
        depths=Depths(
            encoder_full=1,
            encoder_half=1,
            bottleneck=3,   # stripe_frequency=3 -> block 3 is the hybrid one
            decoder_half=1,
            decoder_full=1,
            refinement=1,
        ),
    )


@pytest.fixture(scope="session")
def small_model(small_config: CASHSIConfig):
    torch.manual_seed(0)
    return build_cas_hsi(small_config).eval()


@pytest.fixture(scope="session")
def small_edge_model(small_config: CASHSIConfig):
    torch.manual_seed(0)
    return build_cas_hsi(small_config.as_edge()).eval()


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    return build_cas_hsi("tiny").eval()

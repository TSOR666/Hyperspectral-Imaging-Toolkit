"""Pytest configuration for hsi_viz_suite tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Add package paths to sys.path for imports."""
    package_root = Path(__file__).parent.parent
    scripts_dir = package_root / "scripts"

    # Add paths if not already present
    for path in [str(package_root), str(scripts_dir)]:
        if path not in sys.path:
            sys.path.insert(0, path)

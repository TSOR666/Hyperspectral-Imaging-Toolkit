"""Tests for hsi_viz_suite result layout discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from result_layout import (
    candidate_samples,
    find_sample_pairs,
    no_sample_pairs_message,
    prediction_path,
    target_path,
)


def _save_cube(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros((31, 4, 5), dtype=np.float32))


def test_find_sample_pairs_supports_standard_hsi_layout(tmp_path: Path) -> None:
    _save_cube(tmp_path / "hsi" / "sample_a.npy")
    _save_cube(tmp_path / "hsi" / "sample_a_target.npy")

    assert find_sample_pairs(tmp_path) == ["sample_a"]
    assert prediction_path(tmp_path, "sample_a") == tmp_path / "hsi" / "sample_a.npy"
    assert target_path(tmp_path, "sample_a") == tmp_path / "hsi" / "sample_a_target.npy"


def test_find_sample_pairs_supports_pred_suffix_and_predictions_dir(tmp_path: Path) -> None:
    _save_cube(tmp_path / "predictions" / "ARAD_1K_0950_pred.npy")
    _save_cube(tmp_path / "targets" / "ARAD_1K_0950.npy")
    _save_cube(tmp_path / "hsi" / "ARAD_1K_0951_pred.npy")
    _save_cube(tmp_path / "hsi" / "ARAD_1K_0951_target.npy")

    assert candidate_samples(tmp_path) == ["ARAD_1K_0950", "ARAD_1K_0951"]
    assert find_sample_pairs(tmp_path) == ["ARAD_1K_0950", "ARAD_1K_0951"]


def test_no_sample_pairs_message_mentions_mswr_aggregate_output(tmp_path: Path) -> None:
    (tmp_path / "test_results.json").write_text("{}", encoding="utf-8")

    message = no_sample_pairs_message(tmp_path)

    assert "No visualizable HSI prediction/target pairs" in message
    assert "MSWR aggregate test output" in message

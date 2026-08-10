"""Regression tests for SSTrans/NTIRE output compatibility."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from hsi_io import load_hsi
import generate_all_visualizations as visualization_pipeline
from generate_all_visualizations import (
    _normalize_results_directory,
    _reported_target_directory,
)
from metrics_io import load_metric_for_sample, load_metric_rows
from result_layout import find_prediction_samples, find_sample_pairs, prediction_path


def _write_sstrans_cube(path: Path, cube_chw: np.ndarray) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    cube_hwc = np.moveaxis(cube_chw, 0, -1)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cube", data=cube_hwc.T)
        handle.create_dataset("bands", data=np.array([400.0, 500.0], dtype=np.float32))
        handle.create_dataset("norm_factor", data=np.array(1.0, dtype=np.float32))


def test_sstrans_mat_is_decoded_with_rectangular_orientation(tmp_path: Path) -> None:
    original = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "cubes" / "scene_01.mat"
    _write_sstrans_cube(path, original)

    loaded = load_hsi(path)

    np.testing.assert_array_equal(loaded.cube, original)
    np.testing.assert_array_equal(loaded.wavelengths, np.array([400.0, 500.0]))
    assert loaded.metadata["format"] == "ntire_hdf5"
    assert loaded.metadata["norm_factor"] == pytest.approx(1.0)


def test_prediction_only_sstrans_folder_is_discoverable(tmp_path: Path) -> None:
    cube = np.zeros((2, 3, 4), dtype=np.float32)
    path = tmp_path / "cubes" / "scene_01.mat"
    _write_sstrans_cube(path, cube)

    assert find_prediction_samples(tmp_path) == ["scene_01"]
    assert prediction_path(tmp_path, "scene_01") == path
    assert find_sample_pairs(tmp_path) == []


def test_explicit_target_directory_pairs_with_sstrans_prediction(tmp_path: Path) -> None:
    cube = np.zeros((2, 3, 4), dtype=np.float32)
    _write_sstrans_cube(tmp_path / "cubes" / "scene_01.mat", cube)
    target_dir = tmp_path / "Train_spectral"
    target_dir.mkdir()
    np.save(target_dir / "scene_01.npy", cube)

    assert find_sample_pairs(tmp_path, target_dir=target_dir) == ["scene_01"]


def test_arad_matlab_hdf5_target_keeps_rectangular_orientation(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    original = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "Train_spectral" / "scene_01.mat"
    path.parent.mkdir()
    with h5py.File(path, "w") as handle:
        # Standard ARAD MATLAB/HDF5 storage: CHW with the spatial axes reversed.
        handle.create_dataset("rad", data=original.transpose(0, 2, 1))

    loaded = load_hsi(path)

    np.testing.assert_array_equal(loaded.cube, original)
    assert loaded.metadata["format"] == "matlab_hdf5"


def test_sstrans_metrics_csv_is_available_per_sample(tmp_path: Path) -> None:
    with (tmp_path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_id", "mrae", "rmse", "psnr", "sam"])
        writer.writeheader()
        writer.writerow({"scene_id": "scene_01", "mrae": "0.12", "rmse": "0.03", "psnr": "31.5", "sam": "2.4"})

    rows = load_metric_rows(tmp_path)
    row = load_metric_for_sample(tmp_path, "scene_01")
    assert rows[0]["sample"] == "scene_01"
    assert row is not None
    assert float(row["psnr"]) == pytest.approx(31.5)


def test_sstrans_radian_sam_is_normalized_for_paper_figures(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        '{"sam_unit": "radians"}\n',
        encoding="utf-8",
    )
    with (tmp_path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene_id",
                "mrae",
                "rmse",
                "psnr",
                "sam",
                "sam_degrees",
                "sam_unit",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scene_id": "scene_01",
                "mrae": "0.12",
                "rmse": "0.03",
                "psnr": "31.5",
                "sam": str(np.pi / 2),
                "sam_degrees": "90.0",
                "sam_unit": "radians",
            }
        )

    row = load_metric_for_sample(tmp_path, "scene_01")

    assert row is not None
    assert float(row["sam"]) == pytest.approx(90.0)
    assert row["sam_unit"] == "degrees"
    assert float(row["sam_radians"]) == pytest.approx(np.pi / 2)


def test_sstrans_summary_discovers_reference_directory(tmp_path: Path) -> None:
    target_dir = tmp_path / "Train_spectral"
    target_dir.mkdir()
    (tmp_path / "summary.json").write_text(
        '{"target_dir": "' + str(target_dir).replace("\\", "\\\\") + '"}\n',
        encoding="utf-8",
    )

    assert _reported_target_directory(tmp_path) == target_dir.resolve()


def test_results_argument_may_point_at_sstrans_cubes_child(tmp_path: Path) -> None:
    cubes_dir = tmp_path / "cubes"
    cubes_dir.mkdir()
    (tmp_path / "metrics.csv").write_text("scene_id,mrae\n", encoding="utf-8")

    assert _normalize_results_directory(cubes_dir) == tmp_path.resolve()


def test_one_shot_pipeline_autodiscovers_sstrans_pairs_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "evaluation"
    cubes_dir = run_dir / "cubes"
    target_dir = tmp_path / "Train_spectral"
    cubes_dir.mkdir(parents=True)
    target_dir.mkdir()
    np.save(cubes_dir / "scene_01.npy", np.zeros((2, 3, 4), dtype=np.float32))
    np.save(target_dir / "scene_01.npy", np.zeros((2, 3, 4), dtype=np.float32))
    (run_dir / "summary.json").write_text(
        json.dumps({"target_dir": str(target_dir.resolve())}) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_id", "mrae"])
        writer.writeheader()
        writer.writerow({"scene_id": "scene_01", "mrae": "0.12"})

    calls: list[tuple[str, list[str]]] = []

    def fake_run_script(script: str, args: list[str]) -> None:
        calls.append((script, args))

    monkeypatch.setattr(visualization_pipeline, "run_script", fake_run_script)
    monkeypatch.setattr(
        visualization_pipeline.sys,
        "argv",
        [
            "generate_all_visualizations.py",
            "--results",
            str(cubes_dir),
            "--output",
            str(tmp_path / "figures"),
        ],
    )

    visualization_pipeline.main()

    scripts = [script for script, _ in calls]
    assert "generate_error_maps.py" in scripts
    assert "plot_metrics_statistics.py" in scripts
    error_args = dict(zip(calls[scripts.index("generate_error_maps.py")][1][::2], calls[scripts.index("generate_error_maps.py")][1][1::2]))
    assert error_args["--targets"] == str(target_dir.resolve())
    stats_args = calls[scripts.index("plot_metrics_statistics.py")][1]
    assert str(run_dir.resolve()) in stats_args

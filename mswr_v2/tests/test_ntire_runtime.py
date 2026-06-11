"""Runtime regression tests for the NTIRE evaluation path."""

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import mswr_test_ntire
from mswr_inference import EnsembleProcessor
from mswr_test_ntire import MetricsCalculator, NTIRETestEngine


def test_raw_error_statistics_are_strictly_bounded():
    calculator = MetricsCalculator(calculate_sam=False, calculate_ssim=False)
    calculator.MAX_RAW_ERROR_VALUES = 101
    calculator.RAW_ERROR_VALUES_PER_SAMPLE = 17

    pred = torch.ones(1, 3, 16, 16)
    target = torch.zeros_like(pred)
    for _ in range(20):
        calculator.update(pred, target)

    stored = sum(sample.size for sample in calculator.raw_errors)
    assert stored == 101
    assert calculator.raw_error_values == 101
    assert all(sample.ndim == 1 for sample in calculator.raw_errors)


def test_result_summary_drops_full_resolution_tensors():
    result = {
        "name": "scene_01",
        "metrics": {"mrae": 0.1},
        "prediction": torch.zeros(1, 31, 8, 8),
        "ground_truth": torch.zeros(1, 31, 8, 8),
    }

    summary = NTIRETestEngine._result_summary(result)

    assert summary == {"name": "scene_01", "metrics": {"mrae": 0.1}}
    assert "prediction" not in summary
    assert "ground_truth" not in summary


def test_full_ensemble_is_eight_way_spatially_aligned_and_streamed(monkeypatch):
    engine = object.__new__(NTIRETestEngine)
    engine.config = SimpleNamespace(ensemble_mode="full")

    class CountingIdentity(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x

    engine.model = CountingIdentity()
    rgb = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(1, 3, 5, 7)
    monkeypatch.setattr(
        torch,
        "stack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ensemble should not stack all predictions")
        ),
    )

    output = engine._ensemble_inference(rgb)

    assert engine.model.calls == 8
    assert torch.equal(output, rgb)


def test_production_ensemble_streams_predictions(monkeypatch):
    processor = EnsembleProcessor(mode="full")
    model = nn.Identity()
    image = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(1, 3, 5, 7)
    monkeypatch.setattr(
        torch,
        "stack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ensemble should not stack all predictions")
        ),
    )

    output = processor.process(model, image)

    assert torch.equal(output, image)


def test_mat_prediction_uses_v73_writer(monkeypatch, workspace_tmp_dir):
    captured = {}

    def fake_save_matv73(path, variable, value):
        captured["path"] = path
        captured["variable"] = variable
        captured["value"] = value

    monkeypatch.setattr(mswr_test_ntire, "save_matv73", fake_save_matv73)
    engine = object.__new__(NTIRETestEngine)
    engine.output_dir = workspace_tmp_dir
    engine.config = SimpleNamespace(save_format="mat")

    pred = torch.rand(1, 31, 8, 6)
    engine._save_prediction(pred, "scene")

    assert captured["path"].endswith("scene_pred.mat")
    assert captured["variable"] == "cube"
    assert captured["value"].shape == (8, 6, 31)
    assert captured["value"].dtype == np.float32

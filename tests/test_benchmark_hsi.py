from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from hsi_benchmark.data import (
    DatasetOptions,
    discover_samples,
    load_sample,
    resample_spectra,
)
from hsi_benchmark.metrics import compute_hsi_metrics, summarize_metric_rows
from hsi_benchmark.models import ModelAdapter, predict_tiled
from hsi_benchmark.models import _load_state_checked
from hsi_benchmark.report import write_paper_tables


def test_perfect_metrics() -> None:
    target = np.linspace(0.1, 1.0, 31 * 8 * 9, dtype=np.float32).reshape(31, 8, 9)
    metrics, details = compute_hsi_metrics(target, target)

    assert metrics["mrae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["psnr"] == 100.0
    assert metrics["sam"] < 1e-4
    assert metrics["ssim"] > 0.9999
    assert details["rmse"].shape == (31,)
    assert details["map_sam"].shape == (8, 9)


def test_linear_spectral_resampling() -> None:
    wavelengths = np.array([400.0, 500.0, 600.0, 700.0], dtype=np.float32)
    cube = wavelengths[:, None, None] / 700.0
    target_wavelengths = np.array([450.0, 550.0, 650.0], dtype=np.float32)

    result = resample_spectra(cube, wavelengths, target_wavelengths)

    np.testing.assert_allclose(
        result[:, 0, 0], target_wavelengths / 700.0, rtol=1e-6
    )


def test_descending_wavelengths_are_reordered(tmp_path: Path) -> None:
    wavelengths = np.array([700.0, 600.0, 500.0, 400.0], dtype=np.float32)
    cube = np.broadcast_to(
        (wavelengths / 700.0)[:, None, None], (4, 8, 9)
    ).copy()
    np.savez(tmp_path / "scene.npz", cube=cube, wavelengths=wavelengths)
    options = DatasetOptions(
        preset="custom",
        root=tmp_path,
        rgb_source="cie",
        target_range=(450.0, 650.0),
        target_bands=3,
    )

    sample = load_sample(discover_samples(options)[0], options)

    np.testing.assert_allclose(
        sample.target[:, 0, 0],
        np.array([450.0, 550.0, 650.0], dtype=np.float32) / 700.0,
        rtol=1e-6,
    )


def test_cave_band_stack_and_paired_rgb(tmp_path: Path) -> None:
    scene_dir = tmp_path / "balloons"
    stack_dir = scene_dir / "balloons_ms"
    stack_dir.mkdir(parents=True)
    for band in range(31):
        array = np.full((12, 10), band * 8, dtype=np.uint8)
        Image.fromarray(array).save(stack_dir / f"band_{band + 1:02d}.png")
    rgb = np.zeros((12, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 128
    Image.fromarray(rgb).save(scene_dir / "balloons_RGB.bmp")

    options = DatasetOptions(preset="cave", root=tmp_path, rgb_source="paired")
    records = discover_samples(options)
    sample = load_sample(records[0], options)

    assert len(records) == 1
    assert sample.target.shape == (31, 12, 10)
    assert sample.rgb.shape == (3, 12, 10)
    assert sample.metadata["rgb_protocol"].startswith("paired:")
    np.testing.assert_allclose(sample.wavelengths[[0, -1]], [400.0, 700.0])


class _RepeatModel(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :1].repeat(1, 31, 1, 1)


def test_tiled_prediction_matches_full_image() -> None:
    adapter = ModelAdapter(
        _RepeatModel(),
        torch.device("cpu"),
        name="repeat",
        kind="test",
        normalization="unit",
        use_amp=False,
    )
    rgb = np.random.default_rng(0).random((3, 35, 41), dtype=np.float32)

    result = predict_tiled(
        adapter,
        rgb,
        tile_size=16,
        overlap=4,
        tile_batch_size=3,
    )

    expected = np.repeat(rgb[:1], 31, axis=0)
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)


def test_ema_shadow_overlays_exact_base_state() -> None:
    model = nn.Linear(3, 2)
    base = {
        "weight": torch.zeros_like(model.weight),
        "bias": torch.zeros_like(model.bias),
    }
    checkpoint = {
        "state_dict": base,
        "ema": {"shadow": {"weight": torch.ones_like(model.weight)}},
    }

    source = _load_state_checked(
        model, checkpoint, prefer_ema=True, allow_partial=False
    )

    assert "+" in source
    torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
    torch.testing.assert_close(model.bias, torch.zeros_like(model.bias))


def test_summary_and_paper_tables(tmp_path: Path) -> None:
    summary = summarize_metric_rows(
        [
            {"mrae": 0.1, "psnr": 30.0},
            {"mrae": 0.2, "psnr": 32.0},
        ],
        bootstrap_samples=100,
    )
    results = [
        {
            "dataset": "cave",
            "method": "model_a",
            "model": {"parameters": 123},
            "summary": summary,
        }
    ]

    write_paper_tables(tmp_path, results)

    assert (tmp_path / "paper_table.csv").exists()
    assert "model\\_a" in (tmp_path / "paper_table.tex").read_text(encoding="utf-8")

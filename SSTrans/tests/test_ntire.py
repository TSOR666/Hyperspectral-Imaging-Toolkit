from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from hsiformer import (
    RGBImageDataset,
    build_model,
    build_model_from_checkpoint,
    evaluate_loader,
    get_config,
    load_ntire_cube,
    predict_hsi,
    save_ntire_cube,
)


class _RepeatModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :1].repeat(1, 31, 1, 1)


def test_ntire_cube_round_trip_matches_official_transpose_layout(tmp_path) -> None:
    cube = np.arange(5 * 7 * 31, dtype=np.float32).reshape(5, 7, 31)
    path = tmp_path / "scene.mat"
    save_ntire_cube(path, cube)
    loaded, bands = load_ntire_cube(path)
    np.testing.assert_array_equal(loaded, cube)
    np.testing.assert_array_equal(bands, np.arange(400, 701, 10))


def test_tiled_prediction_matches_full_prediction_for_pointwise_model() -> None:
    model = _RepeatModel()
    rgb = torch.rand(1, 3, 11, 13)
    full = predict_hsi(model, rgb)
    tiled = predict_hsi(model, rgb, tile_size=6, overlap=2)
    torch.testing.assert_close(tiled, full)


def test_rgb_inference_dataset_respects_manifest_order(tmp_path) -> None:
    for scene_id, value in (("scene_b", 64), ("scene_a", 128)):
        image = np.full((4, 5, 3), value, dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(tmp_path / f"{scene_id}.jpg")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("scene_a\nscene_b\n", encoding="utf-8")

    dataset = RGBImageDataset(tmp_path, manifest_path=manifest)
    assert [dataset[index]["scene_id"] for index in range(2)] == [
        "scene_a",
        "scene_b",
    ]


def test_evaluation_writes_ntire_cubes_and_zero_error_metrics(tmp_path) -> None:
    rgb = torch.full((3, 4, 4), 0.5)
    label = rgb[:1].repeat(31, 1, 1)
    loader = DataLoader(
        [{"cond": rgb, "label": label, "scene_id": "scene_1"}],
        batch_size=1,
    )
    summary, rows = evaluate_loader(
        _RepeatModel(),
        loader,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    assert summary["mrae"] == 0.0
    assert summary["rmse"] == 0.0
    assert rows[0]["scene_id"] == "scene_1"
    assert (tmp_path / "scene_1.mat").is_file()


def test_checkpoint_metadata_reconstructs_tiny_model(tmp_path) -> None:
    overrides = {
        "hidden_dim": 8,
        "input_resolution": (16, 16),
        "n_blocks": (1,),
        "bottle_depth": 1,
        "n_refine": 1,
        "patch_size": 2,
        "use_checkpoint": False,
    }
    model = build_model("ablation_no_rpe", **overrides)
    model_config = asdict(get_config("ablation_no_rpe"))
    model_config.update(overrides)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"model": model.state_dict(), "model_config": model_config},
        checkpoint,
    )

    restored, _ = build_model_from_checkpoint(checkpoint)
    with torch.inference_mode():
        output = restored(torch.rand(1, 3, 9, 11))
    assert output.shape == (1, 31, 9, 11)

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
    checkpoint_rgb_normalization,
    evaluate_loader,
    get_config,
    load_ntire_cube,
    mean_relative_absolute_error,
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


def test_evaluation_uses_ntire_center_crop_but_exports_full_cube(tmp_path) -> None:
    rgb = torch.zeros(3, 6, 8)
    rgb[0, 1:-1, 1:-1] = 1.0
    label = torch.ones(31, 6, 8)
    loader = DataLoader(
        [{"cond": rgb, "label": label, "scene_id": "scene_1"}],
        batch_size=1,
    )

    full_summary, _ = evaluate_loader(
        _RepeatModel(),
        loader,
        device=torch.device("cpu"),
    )
    cropped_summary, _ = evaluate_loader(
        _RepeatModel(),
        loader,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        crop_border=1,
    )

    assert full_summary["mrae"] > 0.0
    assert cropped_summary["mrae"] == 0.0
    assert cropped_summary["crop_border"] == 1.0
    exported, _ = load_ntire_cube(tmp_path / "scene_1.mat")
    assert exported.shape == (6, 8, 31)


def test_clip_only_changes_export_not_raw_mrae(tmp_path) -> None:
    rgb = torch.full((3, 4, 4), 2.0)
    label = torch.ones(31, 4, 4)
    loader = DataLoader(
        [{"cond": rgb, "label": label, "scene_id": "scene_1"}],
        batch_size=1,
    )

    summary, _ = evaluate_loader(
        _RepeatModel(),
        loader,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        clip=True,
    )

    assert summary["mrae"] == 1.0
    exported, _ = load_ntire_cube(tmp_path / "scene_1.mat")
    np.testing.assert_array_equal(exported, np.ones_like(exported))


def test_source_mrae_reproduces_additive_zero_denominator() -> None:
    prediction = torch.tensor([[[[0.25, 0.75]]]])
    target = torch.tensor([[[[0.0, 0.5]]]])

    actual = mean_relative_absolute_error(
        prediction,
        target,
        eps=1e-5,
        denominator="source_additive",
    )
    expected = (
        (prediction - target).abs() / (target + 1e-5)
    ).mean()
    torch.testing.assert_close(actual, expected)


def test_source_arad_metric_protocol_is_full_frame_and_raw_psnr() -> None:
    rgb = torch.full((3, 4, 4), 2.0)
    label = torch.ones(31, 4, 4)
    loader = DataLoader(
        [{"cond": rgb, "label": label, "scene_id": "scene_1"}],
        batch_size=1,
    )

    summary, _ = evaluate_loader(
        _RepeatModel(),
        loader,
        device=torch.device("cpu"),
        crop_border=0,
        mrae_denominator="source_additive",
        mrae_epsilon=1e-5,
        psnr_clip=False,
    )

    assert summary["mrae"] == 1.0
    assert summary["psnr"] == 0.0


def test_checkpoint_rgb_normalization_uses_training_metadata() -> None:
    payload = {"training_config": {"rgb_normalization": "per_image"}}
    assert checkpoint_rgb_normalization(payload) == "per_image"


def test_checkpoint_rgb_normalization_requires_legacy_override() -> None:
    with np.testing.assert_raises_regex(
        ValueError,
        "no RGB preprocessing metadata",
    ):
        checkpoint_rgb_normalization({"model": {}})
    assert (
        checkpoint_rgb_normalization({"model": {}}, default="scale_255")
        == "scale_255"
    )


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

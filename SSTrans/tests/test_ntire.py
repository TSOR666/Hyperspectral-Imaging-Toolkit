from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

import hsiformer.cli as cli
import hsiformer.ntire as ntire_module

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
    run_hsi_viz_suite,
    save_ntire_cube,
    write_metric_reports,
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
        include_ssim=True,
    )
    assert summary["mrae"] == 0.0
    assert summary["rmse"] == 0.0
    assert summary["ssim"] == 1.0
    assert rows[0]["scene_id"] == "scene_1"
    assert (tmp_path / "scene_1.mat").is_file()


def test_evaluation_exports_target_less_scenes_without_scoring_them(
    tmp_path,
) -> None:
    rgb = torch.full((3, 4, 4), 0.5)
    label = rgb[:1].repeat(31, 1, 1)
    loader = DataLoader(
        [
            {"cond": rgb, "label": label, "scene_id": "scene_1"},
            {"cond": rgb, "scene_id": "scene_2"},
        ],
        batch_size=1,
    )

    summary, rows = evaluate_loader(
        _RepeatModel(),
        loader,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )

    assert summary["count"] == 1.0
    assert summary["skipped"] == 1.0
    assert "ssim" not in summary
    assert [row["scene_id"] for row in rows] == ["scene_1"]
    assert (tmp_path / "scene_2.mat").is_file()


def test_evaluation_without_any_ground_truth_is_an_actionable_error(
    tmp_path,
) -> None:
    loader = DataLoader(
        [{"cond": torch.full((3, 4, 4), 0.5), "scene_id": "scene_1"}],
        batch_size=1,
    )
    with np.testing.assert_raises_regex(ValueError, "infer_loader"):
        evaluate_loader(
            _RepeatModel(),
            loader,
            device=torch.device("cpu"),
            output_dir=tmp_path,
        )


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


def test_metric_reports_preserve_native_sam_and_add_degrees(tmp_path) -> None:
    write_metric_reports(
        tmp_path,
        {"sam": float(np.pi / 2)},
        [
            {
                "scene_id": "scene_1",
                "mrae": 0.1,
                "rmse": 0.02,
                "psnr": 30.0,
                "sam": float(np.pi / 2),
                "ssim": 0.9,
            }
        ],
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert summary["sam"] == np.pi / 2
    assert summary["sam_degrees"] == 90.0
    assert summary["sam_unit"] == "radians"
    assert summary["metric_units"]["sam_degrees"] == "degrees"
    assert float(row["sam"]) == np.pi / 2
    assert float(row["sam_degrees"]) == 90.0
    assert row["sam_unit"] == "radians"
    assert float(row["ssim"]) == 0.9


def test_hsi_viz_handoff_uses_shared_suite_with_paired_targets(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    results_dir = Path("evaluation")
    target_dir = Path("targets")
    suite_dir = tmp_path / "hsi_viz_suite"
    entrypoint = suite_dir / "scripts" / "generate_all_visualizations.py"
    results_dir.mkdir()
    target_dir.mkdir()
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# test entrypoint\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs) -> None:
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(ntire_module.subprocess, "run", fake_run)
    figures = run_hsi_viz_suite(
        results_dir,
        target_dir=target_dir,
        suite_path=suite_dir,
        max_samples=3,
        dpi=240,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[1] == str(entrypoint.resolve())
    assert command[command.index("--results") + 1] == str(results_dir.resolve())
    assert command[command.index("--targets") + 1] == str(target_dir.resolve())
    assert command[command.index("--max-samples") + 1] == "3"
    assert command[command.index("--dpi") + 1] == "240"
    assert captured["kwargs"] == {"check": True, "cwd": str(suite_dir)}
    assert figures == results_dir.resolve() / "figures"


def test_test_cli_runs_paired_metrics_then_visualization(tmp_path, monkeypatch) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()

    class SingleSceneDataset:
        scene_ids = ("scene_1",)
        scene_ids_with_targets = ("scene_1",)
        unusable_targets: dict[str, str] = {}
        spectral_root = target_dir
        rgb_root = tmp_path / "rgb"

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            assert index == 0
            rgb = torch.full((3, 4, 4), 0.5)
            return {
                "cond": rgb,
                "label": rgb[:1].repeat(31, 1, 1),
                "scene_id": "scene_1",
            }

    dataset = SingleSceneDataset()
    observed: dict[str, object] = {}

    def fake_visualizer(results_dir, **kwargs):
        observed["results_dir"] = results_dir
        observed.update(kwargs)
        return Path(results_dir) / "figures"

    monkeypatch.setattr(cli, "ARAD1KDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(
        cli,
        "build_model_from_checkpoint",
        lambda *args, **kwargs: (_RepeatModel(), {}),
    )
    monkeypatch.setattr(
        cli,
        "checkpoint_rgb_normalization",
        lambda payload: "scale_255",
    )
    monkeypatch.setattr(cli, "run_hsi_viz_suite", fake_visualizer)

    output_dir = tmp_path / "evaluation"
    cli.test_main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--data-root",
            str(tmp_path / "arad"),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--workers",
            "0",
            "--visualize",
            "--viz-max-samples",
            "2",
        ]
    )

    assert observed["results_dir"] == output_dir
    assert observed["target_dir"] == target_dir
    assert observed["max_samples"] == 2
    assert (output_dir / "cubes" / "scene_1.mat").is_file()
    assert (output_dir / "metrics.csv").is_file()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == 1.0
    assert summary["sam_unit"] == "radians"
    assert summary["ssim"] == 1.0
    assert summary["target_dir"] == str(target_dir.resolve())


def test_test_cli_visualizes_blind_export_without_claiming_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    class BlindSceneDataset:
        scene_ids = ("scene_1",)
        scene_ids_with_targets: tuple[str, ...] = ()
        unusable_targets = {"scene_1": "Test_Spec contains only mosaic"}
        spectral_root = None
        rgb_root = tmp_path / "rgb"

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            assert index == 0
            return {
                "cond": torch.full((3, 4, 4), 0.5),
                "scene_id": "scene_1",
            }

    observed: dict[str, object] = {}

    def fake_visualizer(results_dir, **kwargs):
        observed["results_dir"] = results_dir
        observed.update(kwargs)
        return Path(results_dir) / "figures"

    monkeypatch.setattr(
        cli,
        "ARAD1KDataset",
        lambda *args, **kwargs: BlindSceneDataset(),
    )
    monkeypatch.setattr(
        cli,
        "build_model_from_checkpoint",
        lambda *args, **kwargs: (_RepeatModel(), {}),
    )
    monkeypatch.setattr(
        cli,
        "checkpoint_rgb_normalization",
        lambda payload: "scale_255",
    )
    monkeypatch.setattr(cli, "run_hsi_viz_suite", fake_visualizer)

    output_dir = tmp_path / "blind_evaluation"
    cli.test_main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--data-root",
            str(tmp_path / "arad"),
            "--output-dir",
            str(output_dir),
            "--split",
            "test",
            "--device",
            "cpu",
            "--workers",
            "0",
            "--visualize",
        ]
    )

    assert observed["results_dir"] == output_dir
    assert observed["target_dir"] is None
    assert (output_dir / "cubes" / "scene_1.mat").is_file()
    assert (output_dir / "inference.json").is_file()
    inference = json.loads((output_dir / "inference.json").read_text(encoding="utf-8"))
    assert inference["target_dir"] is None
    assert not (output_dir / "metrics.csv").exists()
    assert not (output_dir / "summary.json").exists()

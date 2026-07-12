"""Tests for the unified MST++/NTIRE training + inference infrastructure.

Covers, on a synthetic mini ARAD-1K dataset:
  - end-to-end training (1 epoch) for BOTH model families through one trainer
  - detailed config dump (config.json) and metric logging (all 5 metrics x 2 protocols)
  - checkpoint -> unified inference round trip with model family auto-detection
  - the pad-to-multiple-of-8 fix (SHARP crashes on raw 482-style frames without it)
  - MST++ recipe faithfulness: cosine-to-eta_min schedule, Adam defaults, selection crop

Run:

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
      python -m pytest -q test_unified_infra.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

torch.manual_seed(0)

SCENE = 64  # HSIFusion's minimum input side


# --------------------------------------------------------------------------------------
# Synthetic mini ARAD-1K dataset (2 train scenes, 1 valid scene)
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mini_arad(tmp_path_factory) -> Path:
    import cv2
    import h5py

    root = tmp_path_factory.mktemp("mini_arad")
    (root / "Train_RGB").mkdir()
    (root / "Train_Spec").mkdir()
    (root / "split_txt").mkdir()

    rng = np.random.default_rng(0)
    names = ["ARAD_1K_0001", "ARAD_1K_0002", "ARAD_1K_0003"]
    for name in names:
        rgb = rng.uniform(0, 255, size=(SCENE, SCENE, 3)).astype(np.uint8)
        cv2.imwrite(str(root / "Train_RGB" / f"{name}.jpg"), rgb)
        cube = rng.uniform(0.05, 1.0, size=(31, SCENE, SCENE)).astype(np.float32)
        # Loader does np.transpose(cube, [0, 2, 1]) after reading, so store transposed.
        with h5py.File(root / "Train_Spec" / f"{name}.mat", "w") as f:
            f.create_dataset("cube", data=cube.transpose(0, 2, 1))

    (root / "split_txt" / "train_list.txt").write_text("\n".join(names[:2]) + "\n")
    (root / "split_txt" / "valid_list.txt").write_text(names[2] + "\n")
    return root


def _make_config(model: str, mini_arad: Path, out_dir: Path, **overrides):
    from unified_training import UnifiedTrainingConfig

    kwargs = dict(
        model=model,
        model_size="tiny",
        data_root=str(mini_arad),
        batch_size=1,
        patch_size=SCENE,
        stride=SCENE,
        num_workers=0,
        memory_mode="standard",
        epochs=1,
        val_interval=1,
        val_crop_border=128,  # too large for 64x64 scenes -> exercises the fallback
        amp="off",
        device="cpu",
        log_interval=1,
        output_dir=str(out_dir),
        experiment_name=f"test_{model}",
    )
    kwargs.update(overrides)
    return UnifiedTrainingConfig(**kwargs)


# --------------------------------------------------------------------------------------
# End-to-end training for both families through the SAME trainer
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["sharp", "hsifusion"])
def test_unified_trainer_end_to_end(model, mini_arad, tmp_path):
    from unified_training import METRIC_KEYS, UnifiedTrainer

    config = _make_config(model, mini_arad, tmp_path, ema_decay=0.99 if model == "sharp" else 0.0)
    trainer = UnifiedTrainer(config)
    metrics = trainer.train()

    # All five requested metrics reported (full-frame protocol; crop skipped at 64px)
    for key in METRIC_KEYS:
        assert f"full/{key}" in metrics, f"missing metric {key}"
        assert math.isfinite(metrics[f"full/{key}"])

    exp_dir = config.experiment_path()
    # Detailed config dump
    with (exp_dir / "config.json").open() as f:
        dumped = json.load(f)
    assert dumped["model"] == model
    assert dumped["trainable_parameters"] > 0
    assert "selection_metric" in dumped
    # Metric history log carries every metric
    lines = (exp_dir / "metrics.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[-1])
    for key in METRIC_KEYS:
        assert f"full/{key}" in record
    # Checkpoints written and self-describing
    ckpt = torch.load(exp_dir / "best.pth", map_location="cpu", weights_only=False)
    assert ckpt["model"] == model
    assert ckpt["unified_version"] == 1
    if config.ema_decay > 0:
        assert "ema_model_state_dict" in ckpt


# --------------------------------------------------------------------------------------
# Checkpoint -> unified inference round trip with auto-detection
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["sharp", "hsifusion"])
def test_unified_inference_roundtrip(model, mini_arad, tmp_path):
    from unified_training import METRIC_KEYS, UnifiedTrainer
    from unified_inference import detect_model_family, evaluate, load_checkpoint_model

    config = _make_config(model, mini_arad, tmp_path)
    UnifiedTrainer(config).train()
    ckpt_path = config.experiment_path() / "best.pth"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert detect_model_family(ckpt) == model  # no --model flag needed

    loaded, info = load_checkpoint_model(str(ckpt_path))
    assert info["model"] == model

    out_dir = tmp_path / f"eval_{model}"
    summary = evaluate(
        loaded, str(mini_arad), torch.device("cpu"), crop_border=128, out_dir=out_dir
    )
    for key in METRIC_KEYS:
        assert math.isfinite(summary["full"][key])
    assert (out_dir / "per_scene_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()


def test_detect_model_family_from_bare_state_dict():
    from unified_inference import detect_model_family

    assert detect_model_family({"model_state_dict": {"stages.0.0.norm.weight": 0}}) == "sharp"
    assert detect_model_family(
        {"state_dict": {"encoder_stages.0.0.norm1.weight": 0}}
    ) == "hsifusion"
    with pytest.raises(ValueError):
        detect_model_family({"model_state_dict": {"foo.bar": 0}})


# --------------------------------------------------------------------------------------
# Padding fix: SHARP must survive ARAD-style frames (H not divisible by 8)
# --------------------------------------------------------------------------------------
def test_forward_reconstruction_pads_arad_frames():
    from sharp_v322_hardened import create_sharp_v32
    from unified_training import forward_reconstruction, pad_to_multiple

    model = create_sharp_v32("tiny", compile_model=False, verbose=False).eval()
    x = torch.rand(1, 3, 122, 128)  # 122 % 8 != 0: raw forward crashes (see pass 5 notes)
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(x)
    with torch.no_grad():
        y = forward_reconstruction(model, x)
    assert y.shape == (1, 31, 122, 128)
    assert torch.isfinite(y).all()

    padded, original = pad_to_multiple(torch.rand(1, 3, 482, 512))
    assert original == (482, 512)
    assert padded.shape[-2] % 8 == 0 and padded.shape[-1] % 8 == 0

    # HSIFusion's 64px floor is honoured too
    small, orig = pad_to_multiple(torch.rand(1, 3, 32, 40), min_size=64)
    assert small.shape[-2:] == (64, 64) and orig == (32, 40)


# --------------------------------------------------------------------------------------
# MST++ recipe faithfulness
# --------------------------------------------------------------------------------------
def test_defaults_match_mstpp_recipe():
    from unified_training import UnifiedTrainingConfig

    cfg = UnifiedTrainingConfig()
    assert cfg.batch_size == 20
    assert cfg.patch_size == 128 and cfg.stride == 8
    assert cfg.learning_rate == 4e-4 and cfg.eta_min == 1e-6
    assert cfg.optimizer == "adam" and cfg.weight_decay == 0.0
    assert cfg.warmup_epochs == 0.0 and cfg.ema_decay == 0.0
    assert cfg.epochs == 300
    assert cfg.val_crop_border == 128  # [..., 128:-128, 128:-128] selection
    assert cfg.mrae_eps == 1e-6


def test_scheduler_is_cosine_to_eta_min(mini_arad, tmp_path):
    from unified_training import UnifiedTrainer

    config = _make_config("sharp", mini_arad, tmp_path, epochs=5)
    trainer = UnifiedTrainer(config)
    lr0 = trainer.optimizer.param_groups[0]["lr"]
    assert lr0 == pytest.approx(config.learning_rate)
    for _ in range(trainer.total_optimizer_steps):
        trainer.optimizer.step()
        trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(config.eta_min, rel=1e-3)


def test_selection_crop_matches_mstpp_slice():
    """crop_border=128 must equal MST++'s [..., 128:-128, 128:-128] on 482x512."""
    from unified_training import evaluate_scene

    pred = torch.rand(1, 31, 482, 512)
    truth = torch.rand(1, 31, 482, 512).clamp_min(0.05)
    rows = evaluate_scene(pred, truth, crop_border=128, eps=1e-6)
    assert "crop" in rows

    manual = (
        (pred[..., 128:-128, 128:-128] - truth[..., 128:-128, 128:-128]).abs()
        / truth[..., 128:-128, 128:-128].abs().clamp_min(1e-6)
    ).mean()
    assert rows["crop"]["mrae"] == pytest.approx(manual.item(), rel=1e-5)
    assert pred[..., 128:-128, 128:-128].shape[-2:] == (226, 256)  # MST++ val region


def test_evaluate_scene_reports_all_requested_metrics():
    from unified_training import METRIC_KEYS, evaluate_scene

    rows = evaluate_scene(
        torch.rand(1, 31, 64, 64), torch.rand(1, 31, 64, 64).clamp_min(0.05),
        crop_border=0, eps=1e-6,
    )
    assert set(METRIC_KEYS) <= set(rows["full"])
    assert {"mrae", "rmse", "psnr", "sam", "ssim"} <= set(rows["full"])

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

from hsi_benchmark.metrics import compute_hsi_metrics, mrae_breakdown, relative_absolute_error
from hsi_benchmark.training_health import TrainingHealthMonitor
from diagnose_reconstruction import run_probe
from train_reconstruction_reference import validate_reference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CSWIN v2" / "src"))
from hsi_model.models.losses_consolidated import MRAELoss
from hsi_model.utils.metrics import compute_mrae
from mswr_v2.utils import Loss_MRAE


def test_all_exact_objectives_agree_below_old_floors_with_gradients():
    target = torch.tensor([1e-9, 1e-5, 1e-3, 0.1]).reshape(1, 1, 2, 2)
    pred = (2 * target).requires_grad_()
    reference = relative_absolute_error(pred, target).mean()
    assert reference.item() == pytest.approx(1.0)
    for loss in (Loss_MRAE(epsilon=0), MRAELoss(epsilon=0), lambda p, t: compute_mrae(p, t, epsilon=0)):
        actual = loss(pred, target)
        torch.testing.assert_close(actual, reference)
        grad, = torch.autograd.grad(actual, pred)
        torch.testing.assert_close(grad, 1 / (4 * target))
    metrics, details = compute_hsi_metrics(pred.detach(), target, epsilon=0)
    assert metrics["mrae"] == pytest.approx(1.0)
    assert torch.isfinite(torch.tensor(metrics["sam"]))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_exact_objective_rejects_undefined_targets(bad):
    target = torch.full((1, 1, 2, 2), bad)
    with pytest.raises((ValueError, FloatingPointError)):
        relative_absolute_error(torch.ones_like(target), target)
    for loss in (Loss_MRAE(0), MRAELoss(0)):
        with pytest.raises((ValueError, FloatingPointError)):
            loss(torch.ones_like(target), target)


def test_bucket_contributions_sum_to_total_and_expose_dark_failures():
    target = torch.tensor([1e-5, 0.05, 0.5, 0.5]).reshape(1, 1, 2, 2)
    pred = target.clone()
    pred[..., 0, 0] *= 10
    report = mrae_breakdown(pred, target)
    contributions = [bucket["contribution"] for bucket in report["intensity_buckets"]]
    assert sum(contributions) == pytest.approx(report["raw_mrae"])
    assert contributions[0] == pytest.approx(report["raw_mrae"])
    assert report["intensity_buckets"][0]["fraction"] == 0.25


def test_monitor_measures_actual_update_and_removes_hooks():
    model = nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    monitor = TrainingHealthMonitor(model)
    before = model.weight.detach().clone()
    model(torch.ones(1, 2)).sum().backward()
    monitor.before_step()
    optimizer.step()
    report = monitor.after_step()
    expected = (model.weight - before).norm() / (before.norm() + 1e-12)
    assert report["groups"]["model"]["update_ratio"] == pytest.approx(expected.item())
    monitor.close()
    assert not model._forward_hooks


def test_shared_probe_can_overfit_and_reports_exact_fixed_batch():
    torch.manual_seed(3)
    model = nn.Conv2d(3, 1, 1)
    rgb = torch.rand(4, 3, 4, 4)
    target = 0.2 + 0.1 * rgb.sum(dim=1, keepdim=True)
    history = run_probe(model, rgb, target, steps=400, lr=0.01, log_every=200)
    assert history[-1]["mrae"] < 0.02
    assert history[-1]["mrae"] < history[0]["mrae"] / 10
    assert history[0]["health"]["groups"]["model"]["update_ratio"] > 0


def test_shared_validation_scores_raw_predictions_and_restores_mode():
    class Negative(nn.Module):
        def forward(self, x):
            return -x
    model = Negative().train()
    target = torch.full((1, 3, 4, 4), 0.1)
    report = validate_reference(model, [(target, target)], device="cpu", border=1)
    assert report["raw_mrae"] == pytest.approx(2.0)
    assert report["clamped_mrae"] == pytest.approx(1.0)
    assert model.training
    with pytest.raises(ValueError, match="empty"):
        validate_reference(model, [(target, target)], device="cpu", border=2)
    assert model.training


@pytest.mark.parametrize("kind", ["mswr", "cswin"])
def test_shared_reference_trainer_runs_real_loader_and_checkpoint_cycle(tmp_path, monkeypatch, kind):
    import json
    import cv2
    import h5py
    import yaml
    from train_reconstruction_reference import main
    data = tmp_path / "data"
    for directory in ("Train_RGB", "Train_Spec", "split_txt"):
        (data / directory).mkdir(parents=True)
    for split in ("train_list.txt", "valid_list.txt"):
        (data / "split_txt" / split).write_text("scene001\n", encoding="utf-8")
    rgb = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    assert cv2.imwrite(str(data / "Train_RGB" / "scene001.jpg"), rgb)
    with h5py.File(data / "Train_Spec" / "scene001.mat", "w") as handle:
        handle.create_dataset("cube", data=np.full((31, 32, 32), 0.3, dtype=np.float32))
    config = {
        "model_size": "tiny", "wavelet_levels": [1, 1], "use_checkpoint": False,
        "spectral_output_block": True, "spectral_attn_heads": 1, "drop_path": 0.0,
    } if kind == "mswr" else {
        "base_channels": 8, "num_heads": 2, "split_sizes": [2, 2, 2],
        "stage_depths": [1] * 5, "sampling": "pixelshuffle", "use_feature_norm": False,
        "activation_checkpointing": False, "cswin_attention_mode": "local",
        "sstb_residual_mode": "correction", "sstb_outer_residual_scale": 0.1,
        "smsa_output_norm": False, "clamp_after_iters": -1,
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train_reconstruction_reference.py", "--model", kind,
                       "--config", str(path), "--data-root", str(data), "--output", str(output),
                       "--steps", "2", "--batch-size", "2", "--patch-size", "16", "--stride", "16",
                       "--workers", "0", "--validate-every", "1", "--health-every", "1",
                       "--crop-border", "1", "--device", "cpu"])
    main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"]["epsilon"] == 0
    assert len(manifest["split_sha256"]) == 2
    records = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert len(records) == 2 and records[-1]["validation"]["raw_mrae"] >= 0
    checkpoint = torch.load(output / "best.pth", weights_only=True)
    assert checkpoint["ema_applied"] is False
    assert checkpoint["protocol"]["checkpoint_source"] == "raw"
    assert checkpoint["config"]["objective"] == "exact_mrae"
    with pytest.raises(FileExistsError, match="new output"):
        main()

import pytest
import torch

from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
from hsi_model.utils.data.transforms import compute_mst_center_crop_metrics
from hsi_model.utils.metrics import (
    compute_metrics,
    compute_mrae,
    validate_model_architecture,
)


def test_compute_metrics_rejects_broadcastable_shape_mismatch():
    pred = torch.rand(1, 31, 4, 4)
    target = torch.rand(1, 1, 4, 4)

    with pytest.raises(ValueError, match="Metric shape mismatch"):
        compute_metrics(pred, target)


def test_compute_mrae_is_nonnegative_for_signed_targets():
    pred = torch.tensor([[[[0.0, 0.5]]]])
    target = torch.tensor([[[[-0.5, 0.5]]]])
    value = compute_mrae(pred, target)
    assert torch.isfinite(value)
    assert value.item() >= 0.0


def test_validate_model_architecture_uses_configured_output_channels():
    config = {
        "in_channels": 3,
        "out_channels": 16,
        "base_channels": 8,
        "split_sizes": [2, 2, 2],
        "num_heads": 2,
        "norm_groups": 4,
        "output_activation": "none",
    }
    model = NoiseRobustCSWinGenerator(config)
    assert validate_model_architecture(model, (1, 3, 8, 8), strict=True)


def test_mst_metrics_can_match_clamped_ntire_output_domain():
    target = torch.full((1, 1, 482, 512), 0.5)
    pred = target.clone()
    pred[:, :, 128:180, 128:180] = 1.5
    pred[:, :, 180:232, 180:232] = -0.5

    raw = compute_mst_center_crop_metrics(pred, target)
    deployed = compute_mst_center_crop_metrics(
        pred,
        target,
        clamp_prediction=True,
        report_raw_mrae=True,
    )

    assert deployed["mrae"] < raw["mrae"]
    assert deployed["raw_mrae"] == pytest.approx(raw["mrae"])
    assert deployed["out_of_range_fraction"] > 0.0


def test_mst_metrics_use_fixed_arad_center_window_on_noncanonical_size():
    target = torch.ones(1, 1, 512, 512)
    pred = target.clone()
    # This strip lies inside the legacy 128:-128 border crop but outside the
    # centered 226x256 ARAD scoring window (rows 143:369 for a 512px image).
    pred[:, :, 128:143, 128:384] = 0.0

    metrics = compute_mst_center_crop_metrics(pred, target)

    assert metrics["mrae"] == pytest.approx(0.0)


def test_mst_metrics_score_small_aligned_inputs_instead_of_sentinels():
    target = torch.rand(1, 3, 16, 16)
    metrics = compute_mst_center_crop_metrics(target.clone(), target)

    assert metrics["mrae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)

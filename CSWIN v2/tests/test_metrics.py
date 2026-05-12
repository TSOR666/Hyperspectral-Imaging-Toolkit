import pytest
import torch

from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
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

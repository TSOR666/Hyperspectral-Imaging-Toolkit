"""Regression tests for the 2026-06-12 MSWR fixes."""

import math

import pytest
import torch

import model.mswr_net_v212 as model_module
from model.mswr_net_v212 import (
    EnhancedDualAttention2D,
    IntegratedMSWRNet,
    MSWRDualConfig,
    create_mswr_tiny,
)
from train_mswr_v212_logging import (
    EnhancedTrainer,
    _accumulation_group_size,
    _is_accumulation_boundary,
    _validate_checkpoint_compatibility,
)


@pytest.mark.parametrize(
    ("attention_type", "has_window", "has_landmark", "has_spectral"),
    [
        ("window", True, False, False),
        ("landmark", False, True, False),
        ("dual", True, True, False),
        ("hybrid", True, True, True),
    ],
)
def test_attention_type_controls_constructed_branches(
    attention_type,
    has_window,
    has_landmark,
    has_spectral,
):
    config = MSWRDualConfig(
        base_channels=16,
        num_heads=4,
        num_stages=1,
        attention_type=attention_type,
        use_spectral_attn=False,
        use_checkpoint=False,
    )
    module = EnhancedDualAttention2D(16, config)

    assert hasattr(module, "window_attn") is has_window
    assert hasattr(module, "landmark_attn") is has_landmark
    assert hasattr(module, "spectral_attn") is has_spectral

    output = module(torch.randn(1, 16, 8, 8))
    assert output.shape == (1, 16, 8, 8)


def test_attention_modes_are_not_identical_noops():
    outputs = {}
    x = torch.randn(1, 3, 32, 32)
    for attention_type in ("window", "landmark", "dual", "hybrid"):
        torch.manual_seed(123)
        model = create_mswr_tiny(
            attention_type=attention_type,
            use_checkpoint=False,
            use_flash_attn=False,
            performance_monitoring=False,
        ).eval()
        with torch.no_grad():
            outputs[attention_type] = model(x)

    assert not torch.equal(outputs["window"], outputs["landmark"])
    assert not torch.equal(outputs["dual"], outputs["hybrid"])


def test_use_checkpoint_false_performs_no_checkpoint_calls(monkeypatch):
    calls = []
    original = model_module.checkpoint.checkpoint

    def spy(function, *args, **kwargs):
        calls.append(function)
        return original(function, *args, **kwargs)

    monkeypatch.setattr(model_module.checkpoint, "checkpoint", spy)
    model = create_mswr_tiny(
        use_checkpoint=False,
        use_flash_attn=False,
        performance_monitoring=False,
    ).train()
    output = model(torch.randn(1, 3, 32, 32))
    output.mean().backward()

    assert calls == []


def test_factory_allows_memory_efficient_override():
    model = create_mswr_tiny(memory_efficient=False, use_checkpoint=False)
    assert model.config.memory_efficient is False


def test_invalid_expanded_stage_width_fails_at_config_time():
    with pytest.raises(ValueError, match="stage 1 channels"):
        MSWRDualConfig(
            base_channels=32,
            num_heads=8,
            num_stages=3,
            channel_expansion=1.3,
        )


def test_source_best_prefers_unclamped_mrae():
    metrics = {
        "evaluation_model": "ema",
        "ema_mrae": 0.20,
        "ema_mrae_unclamped": 0.27,
        "mrae": 0.20,
        "mrae_unclamped": 0.27,
        "selection_mrae": 0.27,
    }

    assert EnhancedTrainer._source_mrae(metrics, "ema") == pytest.approx(0.27)


def test_training_resume_rejects_large_checkpoint_mismatch():
    class Incompatible:
        missing_keys = [f"missing_{i}" for i in range(30)]
        unexpected_keys = []

    with pytest.raises(RuntimeError, match="checkpoint and architecture are incompatible"):
        _validate_checkpoint_compatibility(Incompatible(), 100, "test checkpoint")


def test_wavelet_gate_starts_near_identity():
    model = IntegratedMSWRNet(
        MSWRDualConfig(
            base_channels=16,
            num_heads=4,
            num_stages=1,
            wavelet_levels=[1],
            use_checkpoint=False,
            performance_monitoring=False,
        )
    )
    gate = model.encoder_stages[0].wavelet_gate

    assert torch.count_nonzero(gate[-2].weight) == 0
    expected = torch.sigmoid(torch.tensor(4.0))
    actual = gate(torch.randn(2, 16, 8, 8))
    assert torch.allclose(actual, torch.full_like(actual, expected), atol=1e-6)


@pytest.mark.parametrize(
    ("total_batches", "accumulation_steps", "expected_sizes", "expected_boundaries"),
    [
        (5, 4, [4, 4, 4, 4, 1], [False, False, False, True, True]),
        (3, 4, [3, 3, 3], [False, False, True]),
        (10, 4, [4, 4, 4, 4, 4, 4, 4, 4, 2, 2],
         [False, False, False, True, False, False, False, True, False, True]),
    ],
)
def test_partial_accumulation_groups_are_scaled_and_stepped(
    total_batches,
    accumulation_steps,
    expected_sizes,
    expected_boundaries,
):
    sizes = [
        _accumulation_group_size(i, total_batches, accumulation_steps)
        for i in range(total_batches)
    ]
    boundaries = [
        _is_accumulation_boundary(i, total_batches, accumulation_steps)
        for i in range(total_batches)
    ]

    assert sizes == expected_sizes
    assert boundaries == expected_boundaries
    assert sum(boundaries) == math.ceil(total_batches / accumulation_steps)

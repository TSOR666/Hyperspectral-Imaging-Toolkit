import logging

import torch

from hsi_model.models.losses_consolidated import (
    CharbonnierLoss,
    SAMLoss,
    SinkhornDivergence,
    NoiseRobustLoss,
)


def test_cost_matrix_matches_cdist():
    x = torch.randn(4, 3)
    y = torch.randn(5, 3)
    cost = SinkhornDivergence._cost_matrix(x, y)
    expected = torch.cdist(x, y, p=2) ** 2
    assert cost.shape == (4, 5)
    assert torch.all(cost >= 0)
    assert torch.allclose(cost, expected)


def test_sinkhorn_divergence_non_negative():
    sinkhorn = SinkhornDivergence(epsilon=0.1, n_iters=5)
    x = torch.randn(6, 2)
    divergence = sinkhorn(x, x.clone())
    assert torch.isfinite(divergence)
    assert divergence.item() >= 0.0


def test_charbonnier_loss_finite():
    loss = CharbonnierLoss()
    pred = torch.zeros(2, 3, 4, 4)
    target = torch.zeros(2, 3, 4, 4)
    value = loss(pred, target)
    assert torch.isfinite(value)
    assert value.item() >= 0.0


def test_sam_loss_finite():
    loss = SAMLoss()
    pred = torch.zeros(1, 3, 2, 2)
    target = torch.zeros(1, 3, 2, 2)
    value = loss(pred, target)
    assert torch.isfinite(value)
    assert 0.0 <= value.item() <= 3.2


def test_adaptive_weights_warn_on_decrease(caplog):
    loss = NoiseRobustLoss({"use_adaptive_weights": True})
    with caplog.at_level(logging.WARNING):
        loss.get_adaptive_weights(10)
        loss.get_adaptive_weights(5)
    assert any("Iteration decreased" in record.message for record in caplog.records)

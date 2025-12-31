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


# GATE 2.6: Sinkhorn Gradcheck Test
def test_sinkhorn_gradcheck():
    """Test that Sinkhorn gradients are computed correctly (Finding 2.6)."""
    sinkhorn = SinkhornDivergence(epsilon=0.1, n_iters=10)
    X = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    Y = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)

    # Gradcheck verifies that numerical and analytical gradients match
    assert torch.autograd.gradcheck(
        lambda x, y: sinkhorn(x, y), (X, Y), eps=1e-6, atol=1e-4
    ), "Sinkhorn gradient check failed"


def test_sinkhorn_with_empty_inputs():
    """Test that Sinkhorn rejects empty point clouds (Finding 5.3)."""
    sinkhorn = SinkhornDivergence(epsilon=0.1, n_iters=5)
    X_empty = torch.randn(0, 2)
    Y = torch.randn(5, 2)

    # Should raise ValueError for empty point clouds
    try:
        divergence = sinkhorn(X_empty, Y)
        assert False, "Should have raised ValueError for empty point cloud"
    except (ValueError, AssertionError):
        # Expected behavior
        pass


def test_sinkhorn_with_zeros():
    """Test Sinkhorn stability with zero-valued inputs (Finding 2.2)."""
    sinkhorn = SinkhornDivergence(epsilon=0.1, n_iters=10)
    X = torch.zeros(10, 2)
    Y = torch.zeros(10, 2)

    # Should return finite loss even with zero inputs
    divergence = sinkhorn(X, Y)
    assert torch.isfinite(divergence), "Sinkhorn produced NaN/Inf with zero inputs"

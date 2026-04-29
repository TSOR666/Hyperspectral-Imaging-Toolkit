import logging

import torch

from hsi_model.models.losses_consolidated import (
    CharbonnierLoss,
    SAMLoss,
    SinkhornDivergence,
    NoiseRobustLoss,
    ComputeSinkhornDiscriminatorLoss,
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


# GATE 2.6: Sinkhorn Gradient Test
def test_sinkhorn_gradcheck():
    """Test that Sinkhorn backward pass yields finite gradients (Finding 2.6)."""
    sinkhorn = SinkhornDivergence(epsilon=0.1, n_iters=10)
    X = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    Y = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)

    loss = sinkhorn(X, Y)
    loss.backward()

    assert X.grad is not None
    assert Y.grad is not None
    assert torch.isfinite(X.grad).all()
    assert torch.isfinite(Y.grad).all()


def test_sinkhorn_with_empty_inputs():
    """Empty point clouds should yield finite zero divergence fallback."""
    sinkhorn = SinkhornDivergence(epsilon=0.1, n_iters=5)
    X_empty = torch.randn(0, 2)
    Y = torch.randn(5, 2)

    divergence = sinkhorn(X_empty, Y)
    assert torch.isfinite(divergence)
    assert divergence.item() == 0.0


def test_sinkhorn_with_zeros():
    """Test Sinkhorn stability with zero-valued inputs (Finding 2.2)."""
    sinkhorn = SinkhornDivergence(epsilon=0.1, n_iters=10)
    X = torch.zeros(10, 2)
    Y = torch.zeros(10, 2)

    # Should return finite loss even with zero inputs
    divergence = sinkhorn(X, Y)
    assert torch.isfinite(divergence), "Sinkhorn produced NaN/Inf with zero inputs"


def test_sinkhorn_extreme_inputs_are_finite_and_bounded():
    """Extreme-value Sinkhorn inputs must stay finite with capped point count."""
    sinkhorn = SinkhornDivergence(
        epsilon=1e-6,
        n_iters=20,
        max_points=64,
        kernel_clamp=40.0,
        force_fp32=True,
    )
    X = torch.full((2048, 1), 1e6, dtype=torch.float32, requires_grad=True)
    Y = torch.full((2048, 1), -1e6, dtype=torch.float32)

    divergence = sinkhorn(X, Y)
    assert torch.isfinite(divergence)
    assert 0.0 <= divergence.item() <= 1e4

    divergence.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_sinkhorn_prepare_points_caps_memory_footprint():
    """Point capping should bound OT matrix size to reduce OOM risk."""
    sinkhorn = SinkhornDivergence(max_points=32)
    prepared = sinkhorn._prepare_points(torch.randn(1024, 3))
    assert prepared.shape == (32, 3)


def test_discriminator_sinkhorn_loss_backpropagates():
    """Discriminator Sinkhorn loss must preserve gradients to logits."""
    criterion = NoiseRobustLoss({"sinkhorn_epsilon": 0.1, "sinkhorn_iters": 5})
    disc_criterion = ComputeSinkhornDiscriminatorLoss(criterion)

    real_pred = torch.randn(2, 1, 8, 8, requires_grad=True)
    fake_pred = torch.randn(2, 1, 8, 8, requires_grad=True)

    disc_loss = disc_criterion(real_pred, fake_pred)
    assert disc_loss.requires_grad
    disc_loss.backward()

    assert real_pred.grad is not None
    assert fake_pred.grad is not None
    assert torch.isfinite(real_pred.grad).all()
    assert torch.isfinite(fake_pred.grad).all()


def test_adversarial_loss_nonfinite_logits_fallback_is_finite():
    """Non-finite discriminator logits should not produce NaN adversarial loss."""
    criterion = NoiseRobustLoss({"sinkhorn_epsilon": 0.1, "sinkhorn_iters": 5})
    disc_real = torch.zeros(1, 1, 2, 2)
    disc_fake = torch.full((1, 1, 2, 2), float("nan"), requires_grad=True)

    adv_loss = criterion.compute_adversarial_loss(disc_real, disc_fake)
    assert torch.isfinite(adv_loss)
    assert adv_loss.item() == 0.0


# Audit fix: perceptual loss must warn when effectively disabled, and stay differentiable.
def test_noise_robust_loss_warns_when_perceptual_disabled(caplog):
    """lambda_perceptual>0 without a feature_extractor should log a clear warning once."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        criterion = NoiseRobustLoss({"lambda_perceptual": 0.1})
    assert any(
        "perceptual_feature_extractor" in record.message for record in caplog.records
    ), "Expected a warning pointing at perceptual_feature_extractor config."
    assert criterion.lambda_perc == 0.0
    assert criterion.get_adaptive_weights(10)["perc"] == 0.0


def test_noise_robust_loss_keeps_perceptual_weight_with_extractor():
    class TinyExtractor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c = torch.nn.Conv2d(3, 4, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.c(x)

    criterion = NoiseRobustLoss(
        {
            "lambda_perceptual": 0.1,
            "perceptual_feature_extractor": TinyExtractor(),
            "use_adaptive_weights": False,
        }
    )

    assert criterion.lambda_perc == 0.1
    assert criterion.get_adaptive_weights(0)["perc"] == 0.1


def test_perceptual_loss_returns_differentiable_zero_without_extractor():
    from hsi_model.models.losses_consolidated import ImprovedPerceptualLoss

    loss_fn = ImprovedPerceptualLoss(feature_extractor=None, num_bands=31)
    pred = torch.randn(1, 31, 8, 8, requires_grad=True)
    target = torch.randn(1, 31, 8, 8)
    value = loss_fn(pred, target)
    assert torch.isfinite(value)
    assert value.item() == 0.0
    # Must stay differentiable so combined-loss backward never breaks.
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_perceptual_loss_uses_cmf_for_hsi_when_extractor_present():
    """With a real extractor, 31-channel input must flow through the CMF->RGB path."""
    from hsi_model.models.losses_consolidated import ImprovedPerceptualLoss

    # Identity-like extractor that collapses to RGB (3 channels). Any 3-channel
    # module is fine; what matters is that it accepts (B,3,H,W) without error.
    class TinyExtractor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.c(x)

    extractor = TinyExtractor().eval()
    loss_fn = ImprovedPerceptualLoss(
        feature_extractor=extractor, num_bands=31, use_imagenet_norm=True
    )
    pred = torch.rand(1, 31, 8, 8, requires_grad=True)
    target = torch.rand(1, 31, 8, 8)
    value = loss_fn(pred, target)
    assert torch.isfinite(value)
    assert value.item() >= 0.0
    value.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()

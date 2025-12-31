"""Integration tests for CSWIN v2 model."""

import torch
import torch.optim as optim

from hsi_model.model import NoiseRobustCSWinModel
from hsi_model.models.losses_consolidated import NoiseRobustLoss


# GATE 4.3: Integration/Overfit Test
def test_single_batch_overfit():
    """
    Test that the model can overfit to a single batch (Finding 4.3).

    This is a sanity check for gradient flow and architecture correctness.
    If the model cannot overfit to a tiny dataset, it indicates a fundamental issue.
    """
    config = {
        "in_channels": 3,
        "out_channels": 31,
        "base_channels": 16,
        "split_sizes": [2, 2, 2],
        "num_heads": 2,
        "norm_groups": 4,
        "output_activation": "none",
        "lambda_rec": 1.0,
        "lambda_perceptual": 0.0,
        "lambda_adversarial": 0.0,
        "lambda_sam": 0.0,
        "discriminator_base_dim": 8,
        "discriminator_num_heads": 2,
        "discriminator_num_blocks": [1, 1],
    }

    model = NoiseRobustCSWinModel(config)
    criterion = NoiseRobustLoss(config)
    optimizer = optim.Adam(model.generator.parameters(), lr=1e-3)

    # Single batch
    rgb = torch.randn(2, 3, 16, 16)
    hsi = torch.randn(2, 31, 16, 16)

    # Train for 100 steps
    initial_loss = None
    model.train()

    for i in range(100):
        optimizer.zero_grad()
        pred = model.generator(rgb)

        # Only use reconstruction loss for this test (no discriminator)
        loss, _ = criterion(pred, hsi, None, None)

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # Model should reduce loss significantly if gradients flow correctly
    assert final_loss < initial_loss * 0.1, (
        f"Model didn't overfit: {initial_loss:.4f} → {final_loss:.4f}. "
        f"This suggests broken gradients or architecture issues."
    )

    # Verify output is reasonable
    model.eval()
    with torch.no_grad():
        final_pred = model.generator(rgb)
        assert final_pred.shape == hsi.shape
        assert torch.isfinite(final_pred).all()


def test_model_forward_backward():
    """Test that full model supports forward and backward passes."""
    config = {
        "in_channels": 3,
        "out_channels": 31,
        "base_channels": 16,
        "split_sizes": [2, 2],
        "num_heads": 2,
        "norm_groups": 4,
        "output_activation": "none",
        "lambda_rec": 1.0,
        "lambda_perceptual": 0.1,
        "lambda_adversarial": 0.01,
        "lambda_sam": 0.1,
        "discriminator_base_dim": 8,
        "discriminator_num_heads": 2,
        "discriminator_num_blocks": [1, 1],
    }

    model = NoiseRobustCSWinModel(config)
    model.train()

    rgb = torch.randn(1, 3, 16, 16, requires_grad=True)
    hsi = torch.randn(1, 31, 16, 16)

    # Forward pass
    pred_hsi = model.generator(rgb)
    disc_real = model.discriminator(rgb, hsi)
    disc_fake = model.discriminator(rgb, pred_hsi)

    # Compute loss
    criterion = NoiseRobustLoss(config)
    loss, components = criterion(pred_hsi, hsi, disc_real, disc_fake)

    # Backward pass
    loss.backward()

    # Check that gradients exist
    assert rgb.grad is not None
    assert torch.isfinite(rgb.grad).all()

    # Check loss components are valid
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in components.values())

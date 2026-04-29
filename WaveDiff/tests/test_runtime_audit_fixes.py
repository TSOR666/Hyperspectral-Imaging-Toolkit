"""Runtime audit regressions for WaveDiff inference and sampler behavior."""

import os
import sys
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockMatplotlib:
    def __getattr__(self, name):
        return MockMatplotlib()

    def __call__(self, *args, **kwargs):
        return MockMatplotlib()

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0


sys.modules.setdefault("matplotlib", MockMatplotlib())
sys.modules.setdefault("matplotlib.pyplot", MockMatplotlib())
sys.modules.setdefault("matplotlib.colors", MockMatplotlib())
sys.modules.setdefault("matplotlib.cm", MockMatplotlib())


def test_postprocess_hsi_output_restores_data_domain():
    from inference import postprocess_hsi_output

    model_domain = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    config = {
        "hsi_normalize_to_neg_one_to_one": True,
        "hsi_max_value": 2.0,
    }

    restored = postprocess_hsi_output(model_domain, config)

    assert torch.allclose(restored, torch.tensor([[[[0.0, 1.0, 2.0]]]]))


def test_postprocess_hsi_output_skips_when_metadata_absent():
    from inference import postprocess_hsi_output

    model_domain = torch.tensor([[[[-1.0, 0.0, 1.0]]]])

    restored = postprocess_hsi_output(model_domain, config={})

    assert torch.equal(restored, model_domain)


def test_evaluate_metrics_compares_prediction_in_data_domain():
    from inference import evaluate_metrics

    pred_model_domain = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    gt_data_domain = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    config = {
        "hsi_normalize_to_neg_one_to_one": True,
        "hsi_max_value": 1.0,
    }

    metrics = evaluate_metrics(pred_model_domain, gt_data_domain, config=config)

    assert metrics["rmse"] < 1e-6
    assert metrics["mrae"] < 1e-6


def test_dpm_solver_sampling_avoids_tensor_item_sync(monkeypatch):
    from diffusion.dpm_ot import DPMOT
    from diffusion.noise_schedule import BaseNoiseSchedule

    class TinyDenoiser(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = torch.nn.Conv2d(channels, channels, 3, padding=1)

        def forward(self, x, t):
            return self.net(x)

    def fail_item(self):
        raise AssertionError("Tensor.item() should not be called in the sampler loop")

    dpm = DPMOT(
        denoiser=TinyDenoiser(4),
        spectral_schedule=BaseNoiseSchedule(timesteps=16),
        timesteps=16,
    )

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    sample = dpm.sample(
        shape=(1, 4, 8, 8),
        device="cpu",
        use_dpm_solver=True,
        steps=3,
    )

    assert sample.shape == (1, 4, 8, 8)
    assert torch.isfinite(sample).all()


def test_training_checkpoint_round_trip_uses_safe_loader():
    from train import load_checkpoint, save_checkpoint

    model = torch.nn.Conv2d(3, 31, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

    checkpoint_path = os.path.abspath("checkpoint_runtime_test.pt")
    try:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            train_loss=0.4,
            val_loss=0.3,
            config={"model_type": "base", "latent_dim": 8},
            path=checkpoint_path,
        )

        new_model = torch.nn.Conv2d(3, 31, kernel_size=1)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=2)

        checkpoint, start_epoch, best_val_loss = load_checkpoint(
            new_model,
            new_optimizer,
            new_scheduler,
            checkpoint_path,
            torch.device("cpu"),
        )
    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    assert checkpoint["epoch"] == 3
    assert start_epoch == 4
    assert best_val_loss == 0.3
    for original, loaded in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(original, loaded)

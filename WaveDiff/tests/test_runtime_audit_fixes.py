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

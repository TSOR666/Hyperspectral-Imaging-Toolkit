"""Regression tests for confirmed WaveDiff bottleneck fixes."""

import json
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


def test_masking_keeps_full_target_supervision():
    from models.base_model import HSILatentDiffusionModel

    model = HSILatentDiffusionModel(
        latent_dim=16,
        out_channels=31,
        timesteps=8,
        use_batchnorm=False,
    )
    rgb_target = torch.zeros(1, 3, 16, 16)
    hsi_target = torch.zeros(1, 31, 16, 16)
    hsi_output = torch.zeros_like(hsi_target)
    rgb_output = torch.zeros_like(rgb_target)
    hsi_output[:, :, :, 8:] = 1.0
    rgb_output[:, :, :, 8:] = 1.0
    mask = torch.ones(1, 1, 16, 16)
    mask[:, :, :, 8:] = 0.0

    losses = model.calculate_losses(
        {
            "diffusion_loss": torch.tensor(0.0),
            "hsi_output": hsi_output,
            "rgb_from_hsi": rgb_output,
            "mask": mask,
        },
        rgb_target,
        hsi_target,
    )

    assert torch.allclose(losses["l1_loss"], torch.tensor(0.5))
    assert torch.allclose(losses["cycle_loss"], torch.tensor(0.5))


def test_direct_inference_bypasses_unsupervised_sampler(monkeypatch):
    from models.base_model import HSILatentDiffusionModel

    model = HSILatentDiffusionModel(
        latent_dim=16,
        out_channels=31,
        timesteps=8,
        use_batchnorm=False,
    ).eval()

    def fail_sample(*args, **kwargs):
        raise AssertionError("direct inference must not call the diffusion sampler")

    monkeypatch.setattr(model.dpm_ot, "sample", fail_sample)
    with torch.inference_mode():
        output = model.rgb_to_hsi(
            torch.randn(1, 3, 16, 16),
            latent_mode="direct",
        )

    assert output.shape == (1, 31, 16, 16)
    assert torch.isfinite(output).all()


def test_enhanced_base_checkpoint_loads(tmp_path):
    from inference import load_model
    from models.base_model import HSILatentDiffusionModel

    model = HSILatentDiffusionModel(
        latent_dim=16,
        out_channels=31,
        timesteps=8,
        use_batchnorm=False,
        use_enhanced_attention=True,
        use_domain_adaptation=True,
    )
    path = tmp_path / "enhanced.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_type": "base",
                "latent_dim": 16,
                "timesteps": 8,
                "use_batchnorm": False,
                "use_enhanced_attention": True,
                "use_domain_adaptation": True,
            },
        },
        path,
    )

    loaded, _ = load_model(str(path), torch.device("cpu"))

    assert loaded.enhanced_attention is not None
    assert loaded.domain_attention is not None


def test_ddim_terminal_update_recovers_x0_with_exact_noise():
    from diffusion.dpm_ot import DPMOT
    from diffusion.noise_schedule import BaseNoiseSchedule

    class IdentityDenoiser(torch.nn.Module):
        def forward(self, x, t):
            return x

    schedule = BaseNoiseSchedule(timesteps=16)
    dpm = DPMOT(IdentityDenoiser(), schedule, timesteps=16)
    x0 = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([15, 7])
    alpha_t = schedule.extract("alphas_cumprod", t, x0.shape)
    x_t = alpha_t.sqrt() * x0 + (1.0 - alpha_t).sqrt() * noise

    recovered = dpm._ddim_update(
        x_t,
        noise,
        t,
        torch.full_like(t, -1),
    )

    assert torch.allclose(recovered, x0, atol=1e-5, rtol=1e-5)


def test_ddpm_reverse_uses_cached_schedule_not_content_beta():
    from diffusion.dpm_ot import DPMOT
    from diffusion.noise_schedule import BaseNoiseSchedule

    class FixedSchedule(BaseNoiseSchedule):
        def forward(self, x, t):
            raise AssertionError("content-dependent beta must not be used")

    class ZeroDenoiser(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    dpm = DPMOT(ZeroDenoiser(), FixedSchedule(timesteps=2), timesteps=2)
    sample = dpm.sample_ddpm((1, 2, 4, 4), "cpu")

    assert sample.shape == (1, 2, 4, 4)
    assert torch.isfinite(sample).all()


def test_adaptive_schedule_weights_are_not_dead_parameters():
    from diffusion.noise_schedule import (
        SpectralNoiseSchedule,
        WaveletSpectralNoiseSchedule,
    )

    spectral_names = dict(SpectralNoiseSchedule(8).named_parameters())
    wavelet_names = dict(
        WaveletSpectralNoiseSchedule(8, latent_dim=4).named_parameters()
    )

    assert "spectral_weights" not in spectral_names
    assert "ll_weight" not in wavelet_names
    assert "detail_weights" not in wavelet_names


def test_curriculum_loss_keys_are_applied():
    from train import combine_weighted_losses

    losses = {
        "diffusion_loss": torch.tensor(1.0),
        "cycle_loss": torch.tensor(1.0),
        "l1_loss": torch.tensor(1.0),
        "wavelet_loss": torch.tensor(1.0),
        "spectral_consistency": torch.tensor(1.0),
    }
    curriculum = {
        "diffusion_loss": 1.0,
        "cycle_loss": 0.2,
        "l1_loss": 0.3,
        "wavelet_loss": 0.4,
        "spectral_consistency": 0.5,
    }

    total = combine_weighted_losses(losses, {}, curriculum)

    assert torch.allclose(total, torch.tensor(2.4))


def test_hsi_layout_validation():
    from train import ensure_hsi_chw

    chw = torch.randn(31, 12, 10)
    hwc = torch.randn(12, 10, 31)

    assert ensure_hsi_chw(chw).shape == (31, 12, 10)
    assert ensure_hsi_chw(hwc).shape == (31, 12, 10)

    try:
        ensure_hsi_chw(torch.randn(12, 10, 7))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid HSI layout should be rejected")


def test_json_config_and_cli_override(monkeypatch, tmp_path):
    import train

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "batch_size": 3,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "visualization_dir": str(tmp_path / "visualizations"),
            }
        ),
        encoding="ascii",
    )
    captured = {}

    def fake_train(config):
        captured.update(config)
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        return torch.nn.Linear(1, 1), {}

    monkeypatch.setattr(train, "train", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--batch_size",
            "5",
        ],
    )

    train.main()

    assert captured["batch_size"] == 5
    assert captured["checkpoint_dir"].startswith(str(tmp_path / "checkpoints"))


def test_ground_truth_alignment():
    from inference import align_hsi_ground_truth

    gt = torch.randn(1, 31, 20, 18)
    prediction = torch.randn(1, 31, 16, 16)

    aligned = align_hsi_ground_truth(gt, prediction)

    assert aligned.shape == prediction.shape
    assert torch.isfinite(aligned).all()

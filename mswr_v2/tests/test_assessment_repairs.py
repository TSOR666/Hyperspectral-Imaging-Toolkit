import pytest
import torch

from model.mswr_net_v212 import (
    IntegratedMSWRNet, MSWRDualConfig,
    OptimizedCNNWaveletTransform, OptimizedCNNInverseWaveletTransform,
)


def small_model(**overrides):
    config = dict(base_channels=8, num_heads=2, num_stages=2, window_size=4,
                  num_landmarks=4, spectral_attn_heads=1, wavelet_levels=[1, 1],
                  wavelet_type="db2", wavelet_detail_gain_mode="identity",
                  use_checkpoint=False, performance_monitoring=False, drop_path=0)
    config.update(overrides)
    return IntegratedMSWRNet(MSWRDualConfig(**config))


def test_output_spectral_branch_receives_first_step_updates_at_full_resolution():
    torch.manual_seed(4)
    model = small_model(spectral_output_block=True)
    shapes = []
    handle = model.spectral_output.register_forward_pre_hook(
        lambda module, inputs: shapes.append(inputs[0].shape[-2:])
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    before = {name: param.detach().clone() for name, param in model.spectral_output.named_parameters()}
    pred = model(torch.rand(2, 3, 17, 19))
    assert pred.shape == (2, 31, 17, 19)
    assert shapes == [(20, 20)]  # padded full image, rather than wavelet LL
    loss = (pred - torch.rand_like(pred)).abs().mean()
    loss.backward()
    for name, param in model.spectral_output.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    for branch in (model.spectral_output.attn, model.spectral_output.ffn.net):
        assert sum(float(param.grad.abs().sum()) for param in branch.parameters()) > 0
    optimizer.step()
    for prefix in ("attn.", "ffn.net."):
        assert any(not torch.equal(before[name], param) for name, param in model.spectral_output.named_parameters()
                   if name.startswith(prefix))
    handle.remove()


def test_zero_gate_output_branch_preserves_legacy_weights_and_predictions():
    base = small_model().eval()
    new = small_model(spectral_output_block=True, spectral_output_gate_init=0).eval()
    incompatible = new.load_state_dict(base.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert all(name.startswith("spectral_output.") for name in incompatible.missing_keys)
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(new(x), base(x), rtol=0, atol=0)


@pytest.mark.parametrize("wave", ["haar", "db2"])
@pytest.mark.parametrize("levels", [1, 2, 3])
@pytest.mark.parametrize("pattern", ["constant", "impulse", "ramp", "random"])
def test_wavelet_roundtrip_boundaries_and_gradients(wave, levels, pattern):
    torch.manual_seed(5)
    x = torch.ones(1, 2, 32, 32)
    if pattern == "impulse":
        x.zero_()
        x[..., 0, 0] = 1
        x[..., -1, -1] = 1
    elif pattern == "ramp":
        x = torch.linspace(0, 1, 32 * 32).reshape(1, 1, 32, 32).expand(1, 2, 32, 32).clone()
    elif pattern == "random":
        x.normal_()
    x.requires_grad_()
    reconstructed = OptimizedCNNInverseWaveletTransform(wave=wave)(
        OptimizedCNNWaveletTransform(J=levels, wave=wave)(x)
    )
    torch.testing.assert_close(reconstructed, x, rtol=1e-5, atol=2e-6)
    reconstructed.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=1e-5, atol=2e-6)

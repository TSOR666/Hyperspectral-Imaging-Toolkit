"""Tests for the SOTA-push levers: weight-tied cascade (``cascade_stages``)
and the S-MSA output-norm toggle (``smsa_output_norm``).

Contract under test (matches the established checkpoint-safety convention):
- ``cascade_stages`` defaults to 1 and adds NO parameters when off, so legacy
  checkpoints keep loading strictly.
- With ``cascade_stages > 1`` the refinement is an EXACT identity at init
  (zero-init feedback conv + zero-init gate), and a legacy state_dict loads
  under strict=False with no unexpected keys and unchanged outputs.
- ``smsa_output_norm: false`` swaps the non-standard post-projection GroupNorm
  for Identity (fresh-run option, matches Restormer/MST++ S-MSA).
"""

import pytest
import torch

from hsi_model.models.attention import SpectralMSA
from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator


def _config(**overrides):
    config = {
        "in_channels": 3,
        "out_channels": 31,
        "base_channels": 16,
        "split_sizes": [2, 2, 2],
        "num_heads": 4,
        "norm_groups": 4,
        "output_activation": "none",
        "spectral_attention_type": "s_msa",
    }
    config.update(overrides)
    return config


def _build(**overrides):
    torch.manual_seed(1234)
    gen = NoiseRobustCSWinGenerator(_config(**overrides))
    gen.eval()
    return gen


class TestCascade:
    def test_disabled_adds_no_parameters(self):
        gen = _build()
        assert gen.cascade_stages == 1
        assert not any("cascade" in k for k in gen.state_dict().keys())

    def test_identity_at_init(self):
        """With zero-init gate, a 2-stage cascade must reproduce the 1-stage
        output exactly on the same weights."""
        gen = _build(cascade_stages=2)
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out2 = gen(x)
            gen.cascade_stages = 1  # bypass the refinement pass
            out1 = gen(x)
        torch.testing.assert_close(out2, out1, rtol=0.0, atol=0.0)

    def test_legacy_checkpoint_loads_and_output_unchanged(self):
        base = _build()
        cascade = _build(cascade_stages=2)

        result = cascade.load_state_dict(base.state_dict(), strict=False)
        assert result.unexpected_keys == []
        assert all(k.startswith("cascade_") for k in result.missing_keys)

        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            torch.testing.assert_close(cascade(x), base(x), rtol=0.0, atol=0.0)

    def test_cascade_gradients_flow(self):
        gen = _build(cascade_stages=2)
        gen.train()
        x = torch.randn(1, 3, 16, 16)
        gen(x).mean().backward()
        assert gen.cascade_gate.grad is not None
        assert torch.isfinite(gen.cascade_gate.grad).all()
        # The gate gradient is nonzero at init (d out / d gate = refined head
        # output), so the refinement pass can start learning immediately.
        assert gen.cascade_gate.grad.abs().sum() > 0


class TestSmsaOutputNorm:
    def test_norm_replaced_by_identity(self):
        attn = SpectralMSA(16, num_heads=4, config=_config(smsa_output_norm=False))
        assert isinstance(attn.norm, torch.nn.Identity)

    def test_default_keeps_groupnorm(self):
        attn = SpectralMSA(16, num_heads=4, config=_config())
        assert isinstance(attn.norm, torch.nn.GroupNorm)

    def test_generator_runs_without_output_norm(self):
        gen = _build(smsa_output_norm=False, cascade_stages=2)
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = gen(x)
        assert out.shape == (1, 31, 16, 16)
        assert torch.isfinite(out).all()

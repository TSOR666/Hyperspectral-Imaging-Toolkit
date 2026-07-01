"""Tests for the Rung-3 SOTA levers: full-resolution spectral MSAB pre-layer
(``spectral_prelayer``) and per-stage depth (``blocks_per_stage``).

Contract under test (matches the established checkpoint-safety convention):
- Both levers are OFF by default and change nothing when off.
- ``spectral_prelayer`` is an EXACT identity at init (zero-init gates), and a
  legacy state_dict loads into an enabled model under strict=False with no
  unexpected keys and unchanged outputs.
- ``blocks_per_stage=1`` preserves the legacy module tree (state_dict keys);
  ``blocks_per_stage>1`` stacks blocks and remains trainable.
"""

import pytest
import torch

try:
    from model.mswr_net_v212 import (
        MSWRDualConfig,
        IntegratedMSWRNet,
        SpectralMSABlock,
        create_mswr_tiny,
    )
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    MSWRDualConfig = None
    IntegratedMSWRNet = None
    SpectralMSABlock = None
    create_mswr_tiny = None

pytestmark = pytest.mark.skipif(
    not MODEL_AVAILABLE, reason="model module dependencies not available"
)


def _tiny(**kwargs):
    torch.manual_seed(1234)
    model = create_mswr_tiny(performance_monitoring=False, **kwargs)
    model.eval()
    return model


class TestConfigFields:
    def test_defaults_off(self):
        config = MSWRDualConfig()
        assert config.spectral_prelayer is False
        assert config.blocks_per_stage == 1

    def test_blocks_per_stage_validated(self):
        with pytest.raises(AssertionError):
            MSWRDualConfig(blocks_per_stage=0)


class TestSpectralPrelayer:
    def test_block_is_exact_identity_at_init(self):
        torch.manual_seed(0)
        block = SpectralMSABlock(dim=32, num_heads=1)
        x = torch.randn(2, 32, 16, 16)
        with torch.no_grad():
            y = block(x)
        torch.testing.assert_close(y, x, rtol=0.0, atol=0.0)

    def test_legacy_checkpoint_loads_and_output_unchanged(self):
        """A legacy (prelayer-off) state_dict must load into a prelayer-on
        model with no unexpected keys, and — because the new gates are zero —
        produce the same output."""
        base = _tiny()
        upgraded = _tiny(spectral_prelayer=True, spectral_attn_heads=1)

        result = upgraded.load_state_dict(base.state_dict(), strict=False)
        assert result.unexpected_keys == []
        assert all("spectral_pre" in k for k in result.missing_keys)

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(upgraded(x), base(x), rtol=0.0, atol=0.0)

    def test_prelayer_gradients_flow(self):
        model = _tiny(spectral_prelayer=True, spectral_attn_heads=1)
        model.train()
        x = torch.randn(1, 3, 64, 64)
        model(x).mean().backward()
        grads = [
            p.grad
            for n, p in model.named_parameters()
            if "spectral_pre" in n and p.grad is not None
        ]
        assert grads, "spectral prelayer received no gradients"
        assert any(g.abs().sum() > 0 for g in grads)


class TestBlocksPerStage:
    def test_default_keeps_legacy_key_layout(self):
        model = _tiny()
        keys = model.state_dict().keys()
        assert any(k.startswith("encoder_stages.0.attn") for k in keys)
        assert not any(k.startswith("encoder_stages.0.0.") for k in keys)

    def test_stacked_blocks_build_and_run(self):
        model = _tiny(blocks_per_stage=2)
        keys = model.state_dict().keys()
        assert any(k.startswith("encoder_stages.0.0.") for k in keys)
        assert any(k.startswith("encoder_stages.0.1.") for k in keys)
        assert any(k.startswith("decoder_stages.0.1.") for k in keys)

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 31, 64, 64)
        assert torch.isfinite(out).all()

    def test_stacked_blocks_trainable(self):
        model = _tiny(blocks_per_stage=2)
        model.train()
        x = torch.randn(1, 3, 64, 64)
        model(x).mean().backward()
        grad = model.encoder_stages[0][1].attn.proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all()


class TestCombinedSotaConfig:
    def test_full_rung3_model_forward(self):
        """The sota_spectral_depth.yaml architecture combination builds and
        produces a finite reconstruction."""
        model = _tiny(
            spectral_prelayer=True,
            blocks_per_stage=2,
            use_spectral_attn=True,
            spectral_attn_heads=1,
            spectral_ffn=True,
            multistage_refine=True,
            wavelet_detail_processing=True,
        )
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 31, 64, 64)
        assert torch.isfinite(out).all()

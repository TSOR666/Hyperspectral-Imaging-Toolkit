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


class TestSstbOuterResidualScale:
    def test_default_preserves_legacy_unit_scale(self):
        gen = _build(stage_depths=[1, 1, 1, 1, 1])

        assert gen.encoder1[0].outer_residual_scale == pytest.approx(1.0)

    def test_configured_scale_reaches_every_stage(self):
        gen = _build(
            stage_depths=[1, 1, 1, 1, 1],
            sstb_outer_residual_scale=0.1,
        )
        blocks = [
            gen.encoder1[0],
            gen.encoder2[0],
            gen.bottleneck[0],
            gen.decoder1[0],
            gen.decoder2[0],
        ]

        assert [block.outer_residual_scale for block in blocks] == pytest.approx(
            [0.1] * 5
        )

    @pytest.mark.parametrize("scale", [-0.1, float("nan"), float("inf")])
    def test_invalid_scale_fails_loudly(self, scale):
        with pytest.raises(ValueError, match="sstb_outer_residual_scale"):
            _build(sstb_outer_residual_scale=scale)


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


class TestSpectralPriorAndRefinement:
    def test_disabled_by_default_adds_no_parameters(self):
        gen = _build()
        state_keys = list(gen.state_dict().keys())

        assert not any(key.startswith("spectral_input_skip") for key in state_keys)
        assert not any(key.startswith("refinement") for key in state_keys)

    def test_refinement_is_identity_at_init(self):
        base = _build()
        refined = _build(refinement_blocks=2, refinement_channels=16)

        result = refined.load_state_dict(base.state_dict(), strict=False)
        assert result.unexpected_keys == []
        assert all(key.startswith("refinement.") for key in result.missing_keys)

        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            torch.testing.assert_close(refined(x), base(x), rtol=0.0, atol=0.0)

    def test_spectral_input_skip_and_refinement_gradients_flow(self):
        gen = _build(
            use_spectral_input_skip=True,
            spectral_input_skip_init=0.03,
            refinement_blocks=1,
            refinement_channels=16,
        )
        gen.train()
        x = torch.randn(1, 3, 16, 16)

        out = gen(x)
        out.square().mean().backward()

        skip_grads = [
            parameter.grad
            for parameter in gen.spectral_input_skip.parameters()
            if parameter.requires_grad
        ]
        refinement_out = gen.refinement[0].out_proj

        assert out.shape == (1, 31, 16, 16)
        assert skip_grads
        assert all(grad is not None and torch.isfinite(grad).all() for grad in skip_grads)
        assert refinement_out.weight.grad is not None
        assert torch.isfinite(refinement_out.weight.grad).all()


class TestFeatureNormToggle:
    def test_default_keeps_feature_groupnorm(self):
        gen = _build()

        assert isinstance(gen.embedding[1], torch.nn.GroupNorm)
        assert isinstance(gen.down1.norm, torch.nn.GroupNorm)
        assert isinstance(gen.up1.norm, torch.nn.GroupNorm)

    def test_fresh_run_can_disable_feature_groupnorm(self):
        gen = _build(use_feature_norm=False, refinement_blocks=1)

        assert isinstance(gen.embedding[1], torch.nn.Identity)
        assert isinstance(gen.down1.norm, torch.nn.Identity)
        assert isinstance(gen.down2.norm, torch.nn.Identity)
        assert isinstance(gen.up1.norm, torch.nn.Identity)
        assert isinstance(gen.up2.norm, torch.nn.Identity)
        assert isinstance(gen.refinement[0].in_proj[1], torch.nn.Identity)

        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = gen(x)
        assert out.shape == (1, 31, 16, 16)
        assert torch.isfinite(out).all()


class TestInputDenoisingToggle:
    def test_default_keeps_legacy_input_denoising(self):
        gen = _build()

        assert gen.denoising is not None
        assert any(key.startswith("denoising.") for key in gen.state_dict())

    def test_fresh_run_can_disable_input_denoising(self):
        gen = _build(use_input_denoising=False)

        assert gen.denoising is None
        assert not any(key.startswith("denoising.") for key in gen.state_dict())

        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = gen(x)
        assert out.shape == (1, 31, 16, 16)
        assert torch.isfinite(out).all()

    def test_legacy_noise_block_flag_fails_loudly(self):
        with pytest.raises(ValueError, match="use_noise_block"):
            _build(use_noise_block=True)


class TestStageHeadSchedule:
    def test_default_reuses_scalar_num_heads(self):
        gen = _build(stage_depths=[1, 1, 1, 1, 1], num_heads=4)

        assert gen._stage_num_heads == [4, 4, 4, 4, 4]

    def test_stage_num_heads_reaches_nested_attention_blocks(self):
        gen = _build(
            cswin_attention_mode="cswin",
            stage_depths=[1, 1, 1, 1, 1],
            stage_num_heads=[2, 4, 8, 8, 2],
            cswin_bias_mode="window_cyclic",
        )
        blocks = [
            gen.encoder1[0],
            gen.encoder2[0],
            gen.bottleneck[0],
            gen.decoder1[0],
            gen.decoder2[0],
        ]

        assert gen._stage_num_heads == [2, 4, 8, 8, 2]
        assert [
            block.spectral_attn.attention.num_heads for block in blocks
        ] == [2, 4, 8, 8, 2]
        assert [
            block.spatial_attn.attention.num_heads for block in blocks
        ] == [2, 4, 8, 8, 2]
        assert [
            block.spatial_attn.attention.head_dim for block in blocks
        ] == [8, 8, 8, 8, 8]

        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = gen(x)
        assert out.shape == (1, 31, 16, 16)
        assert torch.isfinite(out).all()


class TestCswinGradientCoverage:
    def test_true_cswin_mode_has_no_unreachable_trainable_parameters(self):
        gen = _build(
            cswin_attention_mode="cswin",
            split_sizes=[7, 7, 7],
            stage_depths=[1, 1, 1, 1, 1],
            stage_num_heads=[2, 4, 8, 8, 2],
            cswin_bias_mode="window_cyclic",
            use_feature_norm=False,
            use_input_denoising=False,
            cascade_stages=3,
            use_spectral_input_skip=True,
            spectral_input_skip_init=0.03,
            smsa_output_norm=False,
            sampling="pixelshuffle",
        )
        gen.train()

        x = torch.randn(1, 3, 16, 16)
        out = gen(x)
        out.square().mean().backward()

        missing_grads = [
            name
            for name, parameter in gen.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        frozen_legacy = [
            name
            for name, parameter in gen.named_parameters()
            if (
                ".qkv_h." in name
                or ".qkv_v." in name
                or ".lepe_h." in name
                or ".lepe_v." in name
                or ".relative_position_bias_table_" in name
            )
            and not parameter.requires_grad
        ]

        assert missing_grads == []
        assert frozen_legacy

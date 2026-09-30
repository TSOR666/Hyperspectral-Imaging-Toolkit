from pathlib import Path

import pytest
import torch
from torch import nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from hsi_model.models.attention import CSWinAttentionBlock
from hsi_model.models.generator_v3 import DualTransformerBlock, NoiseRobustCSWinGenerator
from hsi_model.train_generator import build_criterion, validate_generator
from hsi_model.utils.data.transforms import compute_mst_center_crop_metrics
from hsi_model.utils.training_setup import load_finetune_weights
from hsi_model.utils.onnx_export import recover_architecture


class Zero(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def config(**overrides):
    result = dict(base_channels=8, num_heads=2, stage_depths=[1] * 5,
                  split_sizes=[2] * 3, use_feature_norm=False, smsa_output_norm=False,
                  cswin_attention_mode="local", sampling="pixelshuffle", activation_checkpointing=False,
                  sstb_outer_residual_scale=0.1, output_head_init_scale=1.0)
    result.update(overrides)
    return result


@pytest.mark.parametrize("mode", ["legacy", "correction"])
def test_residual_identity_contribution(mode):
    block = DualTransformerBlock(8, num_heads=2, config=config(sstb_residual_mode=mode))
    block.spectral_attn = Zero()
    block.spatial_attn = Zero()
    block.gdfn = Zero()
    block.sgfn = Zero()
    x = torch.rand(1, 8, 8, 8)
    expected = x if mode == "correction" else x + 0.1 * block.gate(x)
    torch.testing.assert_close(block(x), expected, rtol=0, atol=0)


def test_correction_model_updates_spectral_and_spatial_weights():
    torch.manual_seed(8)
    model = NoiseRobustCSWinGenerator(config(sstb_residual_mode="correction"))
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    block = model.encoder1[0]
    branches = (block.spectral_attn.attention, block.spatial_attn.attention)
    before = [[p.detach().clone() for p in branch.parameters()] for branch in branches]
    pred = model(torch.rand(1, 3, 9, 11))
    assert pred.shape == (1, 31, 9, 11)
    build_criterion({"objective": "exact_mrae"})(pred, torch.rand_like(pred) + 0.1).backward()
    optimizer.step()
    for branch, old in zip(branches, before):
        assert any(not torch.equal(p, value) for p, value in zip(branch.parameters(), old))


def test_fixed_local_operator_does_not_change_with_resolution():
    fixed = CSWinAttentionBlock(8, num_heads=2, config=config())
    switched = CSWinAttentionBlock(8, num_heads=2, config=config(cswin_attention_mode="local_global"))
    assert [fixed.attention_operator(n, n) for n in (16, 32, 64, 128, 256, 512)] == ["local"] * 6
    assert switched.attention_operator(32, 32) == "global"
    assert switched.attention_operator(64, 64) == "local"


def test_exact_raw_border_metric_matches_reference_and_clamp_differs():
    target = torch.full((1, 31, 258, 260), 1e-9)
    pred = 2 * target
    pred[..., 128:130, 128:132] = -target[..., 128:130, 128:132]
    result = compute_mst_center_crop_metrics(pred, target, mrae_epsilon=0, crop_border=128)
    assert result["mrae"] == pytest.approx(2.0)
    clipped = compute_mst_center_crop_metrics(pred, target, mrae_epsilon=0, crop_border=128, clamp_prediction=True)
    assert clipped["mrae"] == pytest.approx(1.0)


def test_residual_semantics_are_saved_for_export_and_rejected_on_finetune(tmp_path):
    cfg = config(sstb_residual_mode="correction")
    model = NoiseRobustCSWinGenerator(cfg)
    recovery = recover_architecture(model.state_dict(), cfg)
    assert recovery.config["sstb_residual_mode"] == "correction"
    path = tmp_path / "model.pth"
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)
    legacy = NoiseRobustCSWinGenerator(config())
    with pytest.raises(ValueError, match="residual mode"):
        load_finetune_weights(str(path), legacy, torch.device("cpu"))


@pytest.mark.parametrize("name", ["assessment_control", "assessment_correction", "assessment_fixed_local", "assessment_candidate"])
def test_assessment_recipe_is_exact_fp32_full_image_control(name):
    folder = Path(__file__).resolve().parents[1] / "src" / "configs"
    with initialize_config_dir(config_dir=str(folder), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name=name), resolve=True)
    assert build_criterion(cfg).epsilon == 0
    assert cfg["validation_mrae_epsilon"] == 0
    assert not cfg["mixed_precision"] and not cfg["use_ema"]
    assert not cfg["validation_clamp_output"] and not cfg["validation_tiled_inference"]
    assert cfg["gradient_clip_norm"] == 0
    assert cfg["progressive_stages"][0]["iterations"] == 300000
    assert cfg["progressive_stages"][0]["warmup_steps"] == 0


def test_validation_restores_training_mode_after_failure():
    model = nn.Conv2d(3, 31, 1).train()
    with pytest.raises(RuntimeError, match="zero samples"):
        validate_generator(model, [], nn.L1Loss(), torch.device("cpu"), 0,
                           {"num_workers": 0}, False, 42, 0)
    assert model.training

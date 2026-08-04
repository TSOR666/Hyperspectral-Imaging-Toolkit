import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
from hsi_model.train_generator import (
    _persist_resolved_config,
    _validate_training_contract,
    build_criterion,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"


@pytest.mark.parametrize(
    ("config_name", "objective", "compress_first", "decoder2_depth"),
    [
        ("ablation_stable_mrae", "mrae_annealed", False, 4),
        ("ablation_decoder_lite", "mrae_annealed", True, 2),
        ("ablation_stable_lite", "mrae_annealed", True, 2),
        ("finetune_128_polish_annealed", "mrae_annealed", False, 4),
        ("finetune_128_exact_mrae", "mrae_annealed", False, 4),
        ("finetune_radiometric_exact_mrae", "mrae_annealed", False, 4),
        ("finetune_progressive_annealed", "mrae_annealed", False, 4),
    ],
)
def test_ablation_config_composes(
    config_name,
    objective,
    compress_first,
    decoder2_depth,
):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(config_name=config_name)

    assert config.objective == objective
    assert bool(config.decoder1_compress_first) is compress_first
    assert int(config.stage_depths[4]) == decoder2_depth


def test_stable_lite_config_builds_finite_train_step():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(
            config_name="ablation_stable_lite",
            overrides=[
                "base_channels=8",
                "num_heads=2",
                "stage_depths=[1,1,1,1,1]",
                "split_sizes=[2,2,2]",
            ],
        )

    model = NoiseRobustCSWinGenerator(config).train()
    criterion = build_criterion(config)
    prediction = model(torch.rand(1, 3, 8, 8))
    loss = criterion(prediction, torch.rand_like(prediction))
    loss.backward()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_sota_cascade_config_pins_stable_recovery_architecture():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(config_name="sota_cascade")

    assert config.cswin_attention_mode == "local_global"
    assert list(config.split_sizes) == [7, 7, 7]
    assert config.stage_num_heads is None
    assert int(config.cswin_global_tokens) == 1024
    assert bool(config.smsa_output_norm) is False
    assert bool(config.use_feature_norm) is False
    assert bool(config.use_input_denoising) is False
    assert int(config.cascade_stages) == 1
    assert float(config.sstb_outer_residual_scale) == pytest.approx(0.1)
    assert float(config.output_head_init_scale) == pytest.approx(1.0)
    assert bool(config.use_spectral_input_skip) is True
    assert int(config.refinement_blocks) == 0
    assert int(config.base_channels) == 48
    assert int(config.batch_size) == 20
    assert int(config.gradient_accumulation_steps) == 1
    assert len(config.progressive_stages) == 1
    assert int(config.progressive_stages[0].iterations) == 300_000
    assert float(config.mrae_epsilon_start) == pytest.approx(1.0e-2)
    assert float(config.mrae_epsilon_end) == pytest.approx(1.0e-3)
    assert float(config.gradient_clip_norm) == pytest.approx(1.0)
    assert int(config.early_stopping_patience) == 0
    assert config.training_contract.name == "residual_balanced_mstpp_300k_v2"
    assert str(config.log_dir).endswith("sota_residual_balanced_mstpp_300k")
    assert str(config.checkpoint_dir).endswith(
        "sota_residual_balanced_mstpp_300k"
    )


def test_sota_training_contract_rejects_short_or_unstable_recipe():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(config_name="sota_cascade")

    resolved = OmegaConf.to_container(config, resolve=True)
    stages = _validate_training_contract(resolved)
    assert [stage["iterations"] for stage in stages] == [300_000]

    wrong_horizon = copy.deepcopy(resolved)
    wrong_horizon["progressive_stages"][0]["iterations"] = 70_000
    with pytest.raises(ValueError, match=r"stage\[0\]\.iterations=70000"):
        _validate_training_contract(wrong_horizon)

    wrong_floor = copy.deepcopy(resolved)
    wrong_floor["mrae_epsilon_end"] = 1.0e-8
    with pytest.raises(ValueError, match="mrae_epsilon_end"):
        _validate_training_contract(wrong_floor)

    wrong_anneal = copy.deepcopy(resolved)
    wrong_anneal["mrae_epsilon_anneal_iters"] = 300_000
    with pytest.raises(ValueError, match="mrae_epsilon_anneal_iters"):
        _validate_training_contract(wrong_anneal)

    premature_stop = copy.deepcopy(resolved)
    premature_stop["early_stopping_patience"] = 8
    with pytest.raises(ValueError, match="early_stopping_patience"):
        _validate_training_contract(premature_stop)


def test_resolved_config_is_persisted_with_fingerprint(tmp_path):
    config = {
        "log_dir": str(tmp_path),
        "objective": "mrae_annealed",
        "progressive_stages": [{"patch_size": 128, "iterations": 300_000}],
    }

    fingerprint, output_path = _persist_resolved_config(config)
    persisted = OmegaConf.to_container(OmegaConf.load(output_path), resolve=True)

    assert output_path == tmp_path / "resolved_config.yaml"
    assert len(fingerprint) == 12
    assert persisted["config_fingerprint"] == fingerprint
    assert persisted["progressive_stages"][0]["iterations"] == 300_000
    assert not output_path.with_suffix(".yaml.tmp").exists()

    repeated_fingerprint, _ = _persist_resolved_config(config)
    assert repeated_fingerprint == fingerprint


def test_sota_residual_balanced_config_has_finite_initial_train_step():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(
            config_name="sota_cascade",
            overrides=[
                "base_channels=8",
                "num_heads=2",
                "stage_depths=[1,1,1,1,1]",
                "split_sizes=[2,2,2]",
                "activation_checkpointing=false",
            ],
        )

    model = NoiseRobustCSWinGenerator(config).train()
    criterion = build_criterion(config)
    prediction = model(torch.rand(1, 3, 8, 8))
    loss = criterion(prediction, torch.rand_like(prediction))
    loss.backward()

    assert torch.isfinite(loss)
    assert float(loss.detach()) < float(config.max_initial_train_loss)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


def test_progressive_finetune_config_switches_after_saturated_128_stage():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(config_name="finetune_progressive_annealed")

    stages = list(config.progressive_stages)
    assert [int(stage.patch_size) for stage in stages] == [128, 256, 512]
    assert int(stages[0].iterations) == 70_000
    assert int(stages[1].batch_size) == 5
    assert int(stages[2].batch_size) == 1
    assert int(config.early_stopping_patience) == 3
    assert bool(config.early_stopping_final_stage_only) is False


def test_128_polish_config_does_not_switch_patch_geometry():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(config_name="finetune_128_polish_annealed")

    stages = list(config.progressive_stages)
    assert [int(stage.patch_size) for stage in stages] == [128]
    assert int(stages[0].iterations) == 72_000
    assert int(config.validation_patch_size) == 128
    assert int(config.early_stopping_patience) == 3


def test_exact_mrae_polish_restarts_with_low_lr_and_matching_architecture():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(config_name="finetune_128_exact_mrae")

    stages = list(config.progressive_stages)
    assert [int(stage.patch_size) for stage in stages] == [128]
    assert int(stages[0].iterations) == 40_000
    assert float(stages[0].init_lr) == pytest.approx(2.0e-5)
    assert int(stages[0].warmup_steps) == 0
    assert float(config.mrae_epsilon_start) == pytest.approx(1.0e-3)
    assert float(config.mrae_epsilon_end) == pytest.approx(1.0e-8)
    assert int(config.mrae_epsilon_anneal_iters) == 20_000
    assert float(config.sstb_outer_residual_scale) == pytest.approx(1.0)
    assert float(config.gradient_clip_norm) == pytest.approx(1.0)
    assert config.training_contract is None
    assert bool(config.use_spectral_input_skip) is True
    assert bool(config.use_feature_norm) is True
    assert bool(config.use_input_denoising) is True
    assert int(config.early_stopping_warmup_epochs) == 20


def test_radiometric_rescue_preserves_checkpoint_graph_and_restarts_schedule():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(config_name="finetune_radiometric_exact_mrae")

    stages = list(config.progressive_stages)
    assert [int(stage.patch_size) for stage in stages] == [128]
    assert int(stages[0].iterations) == 30_000
    assert float(stages[0].init_lr) == pytest.approx(1.0e-5)
    assert float(config.sstb_outer_residual_scale) == pytest.approx(1.0)
    assert float(config.gradient_clip_norm) == pytest.approx(1.0)
    assert config.training_contract is None
    assert bool(config.use_feature_norm) is False
    assert bool(config.use_input_denoising) is False
    assert bool(config.smsa_output_norm) is False
    assert bool(config.use_spectral_input_skip) is True
    assert int(config.early_stopping_warmup_epochs) == 20

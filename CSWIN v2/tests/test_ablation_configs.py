from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
from hsi_model.train_generator import build_criterion


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"


@pytest.mark.parametrize(
    ("config_name", "objective", "compress_first", "decoder2_depth"),
    [
        ("ablation_stable_mrae", "mrae_annealed", False, 4),
        ("ablation_decoder_lite", "mrae_annealed", True, 2),
        ("ablation_stable_lite", "mrae_annealed", True, 2),
        ("finetune_128_polish_annealed", "mrae_annealed", False, 4),
        ("finetune_128_exact_mrae", "mrae_annealed", False, 4),
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
    assert float(config.output_head_init_scale) == pytest.approx(0.01)
    assert bool(config.use_spectral_input_skip) is True
    assert int(config.refinement_blocks) == 0
    assert int(config.base_channels) == 48
    assert int(config.batch_size) == 20
    assert int(config.gradient_accumulation_steps) == 1
    assert str(config.log_dir).endswith("sota_radiometric_fresh")
    assert str(config.checkpoint_dir).endswith("sota_radiometric_fresh")


def test_sota_radiometric_fresh_config_has_finite_initial_train_step():
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
    assert bool(config.use_spectral_input_skip) is True
    assert bool(config.use_feature_norm) is True
    assert bool(config.use_input_denoising) is True
    assert int(config.early_stopping_warmup_epochs) == 20

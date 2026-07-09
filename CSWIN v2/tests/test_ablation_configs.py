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


def test_sota_cascade_config_uses_recovery_architecture():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(config_name="sota_cascade")

    assert config.cswin_attention_mode == "cswin"
    assert list(config.split_sizes) == [1, 2, 7]
    assert list(config.stage_num_heads) == [2, 4, 8, 8, 2]
    assert bool(config.use_feature_norm) is False
    assert bool(config.use_input_denoising) is False
    assert int(config.cascade_stages) == 3
    assert bool(config.use_spectral_input_skip) is True
    assert int(config.refinement_blocks) == 0
    assert int(config.base_channels) == 64
    assert int(config.gradient_accumulation_steps) == 1


def test_progressive_finetune_config_switches_after_saturated_128_stage():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(config_name="finetune_progressive_annealed")

    stages = list(config.progressive_stages)
    assert [int(stage.patch_size) for stage in stages] == [128, 256, 512]
    assert int(stages[0].iterations) == 70_000
    assert int(stages[1].batch_size) == 8
    assert int(stages[2].batch_size) == 2
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

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

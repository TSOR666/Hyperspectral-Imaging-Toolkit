import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_preflight_train import (  # noqa: E402
    CheckResult,
    assert_training_step,
    hydra_override_value,
    parse_args,
    print_results,
    tiny_config,
    trainer_script,
    training_command,
)
from hsi_model.models import (  # noqa: E402
    ComputeSinkhornDiscriminatorLoss,
    NoiseRobustCSWinModel,
    NoiseRobustLoss,
)
from hsi_model.train_generator import build_criterion  # noqa: E402


def test_parse_args_keeps_hydra_overrides_after_separator():
    args = parse_args(
        [
            "--trainer",
            "optimized",
            "--dry-run",
            "--",
            "--config-name",
            "config",
            "data_dir=/datasets/ARAD_1K",
            "batch_size=4",
        ]
    )

    assert args.trainer == "optimized"
    assert args.dry_run is True
    assert args.training_args == [
        "--config-name",
        "config",
        "data_dir=/datasets/ARAD_1K",
        "batch_size=4",
    ]


def test_hydra_override_value_extracts_simple_override():
    overrides = ["--config-name", "config", "dataset_source=huggingface", "data_dir=/data"]

    assert hydra_override_value(overrides, "dataset_source") == "huggingface"
    assert hydra_override_value(overrides, "data_dir") == "/data"
    assert hydra_override_value(overrides, "missing", "fallback") == "fallback"


def test_trainer_script_selects_expected_entrypoint():
    assert trainer_script("generator").name == "train_generator.py"
    assert trainer_script("sinkhorn").name == "training_script_fixed.py"
    assert trainer_script("optimized").name == "train_optimized.py"


def test_training_command_single_process():
    args = parse_args(["--", "data_dir=/data"])
    cmd = training_command(args)

    assert args.trainer == "generator"
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("train_generator.py")
    assert cmd[-1] == "data_dir=/data"


def test_training_command_distributed():
    args = parse_args(["--trainer", "optimized", "--nproc-per-node", "2", "--", "batch_size=8"])
    cmd = training_command(args)

    assert cmd[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=2" in cmd
    assert any(part.endswith("train_optimized.py") for part in cmd)
    assert cmd[-1] == "batch_size=8"


def test_print_results_includes_pass_fail_rows():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_results(
            [
                CheckResult("ok", True, "fine", 0.01),
                CheckResult("bad", False, "broken", 0.02),
            ]
        )

    output = buffer.getvalue()
    assert "GPU PRE-FLIGHT RESULTS" in output
    assert "PASS" in output
    assert "FAIL" in output
    assert "broken" in output


def test_generator_preflight_training_step_is_finite_on_cpu():
    config = tiny_config()
    model = NoiseRobustCSWinModel(config)
    legacy_criterion = NoiseRobustLoss(config)
    context = {
        "model": model,
        "criterion": legacy_criterion,
        "disc_criterion": ComputeSinkhornDiscriminatorLoss(legacy_criterion),
        "generator_criterion": build_criterion(config),
        "generator_only": True,
        "rgb": torch.rand(1, 3, 8, 8),
        "hsi": torch.rand(1, 31, 8, 8),
    }

    details = assert_training_step(context)

    assert details.startswith("generator_loss=")
    assert torch.isfinite(context["pred_train"]).all()

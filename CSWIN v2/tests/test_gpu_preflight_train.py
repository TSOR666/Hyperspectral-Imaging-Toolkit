import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_preflight_train import (  # noqa: E402
    CheckResult,
    hydra_override_value,
    parse_args,
    print_results,
    trainer_script,
    training_command,
)


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
    assert trainer_script("sinkhorn").name == "training_script_fixed.py"
    assert trainer_script("optimized").name == "train_optimized.py"


def test_training_command_single_process():
    args = parse_args(["--trainer", "sinkhorn", "--", "data_dir=/data"])
    cmd = training_command(args)

    assert cmd[0] == sys.executable
    assert cmd[1].endswith("training_script_fixed.py")
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

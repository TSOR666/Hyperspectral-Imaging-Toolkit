#!/usr/bin/env python3
"""GPU preflight gate that starts CSWIN v2 training only after all checks pass."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hsi_model.models import (  # noqa: E402
    ComputeSinkhornDiscriminatorLoss,
    NoiseRobustCSWinModel,
    NoiseRobustLoss,
)
from hsi_model.train_generator import build_criterion  # noqa: E402
from hsi_model.utils.metrics import compute_metrics  # noqa: E402
from hsi_model.utils.training_setup import setup_paths  # noqa: E402


GB = 1024**3
HF_SOURCES = {"huggingface", "hf", "hf_arad", "arad_hsdb"}


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str
    seconds: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run required GPU preflight checks and launch CSWIN v2 training only "
            "if every item passes. Put Hydra overrides after '--'."
        )
    )
    parser.add_argument(
        "--trainer",
        choices=("generator", "sinkhorn", "optimized"),
        default="generator",
        help="Training entrypoint to launch after preflight passes.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index to test.")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=2.0,
        help="Minimum free GPU memory required before training starts.",
    )
    parser.add_argument(
        "--preflight-size",
        type=int,
        default=16,
        help="Synthetic square patch size for forward/backward GPU checks.",
    )
    parser.add_argument(
        "--preflight-batch-size",
        type=int,
        default=2,
        help="Synthetic batch size for preflight checks.",
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip local dataset path validation. Use only when training config resolves data elsewhere.",
    )
    parser.add_argument(
        "--no-amp-check",
        action="store_true",
        help="Disable the mixed-precision CUDA check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run checks and print the training command, but do not start training.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Use torch.distributed.run with this many processes after checks pass.",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected Hydra training script.",
    )
    args = parser.parse_args(argv)
    if args.training_args and args.training_args[0] == "--":
        args.training_args = args.training_args[1:]
    return args


def hydra_override_value(args: Sequence[str], key: str, default: str | None = None) -> str | None:
    prefix = f"{key}="
    for item in args:
        if item.startswith(prefix):
            return item.split("=", 1)[1]
    return default


def trainer_script(trainer: str) -> Path:
    filenames = {
        "generator": "train_generator.py",
        "sinkhorn": "training_script_fixed.py",
        "optimized": "train_optimized.py",
    }
    filename = filenames[trainer]
    return ROOT / "src" / "hsi_model" / filename


def training_command(args: argparse.Namespace) -> list[str]:
    script = trainer_script(args.trainer)
    if args.nproc_per_node > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.nproc_per_node}",
            str(script),
            *args.training_args,
        ]
    return [sys.executable, str(script), *args.training_args]


def tiny_config() -> dict[str, object]:
    return {
        "in_channels": 3,
        "out_channels": 31,
        "base_channels": 16,
        "split_sizes": [2, 2, 2],
        "num_heads": 2,
        "norm_groups": 4,
        "output_activation": "none",
        "objective": "mrae",
        "mrae_epsilon": 1e-8,
        "sampling": "pixelshuffle",
        "spectral_attention_type": "s_msa",
        "cswin_attention_mode": "local_global",
        "cswin_global_tokens": 1024,
        "stage_depths": [1, 1, 1, 1, 1],
        "lambda_rec": 1.0,
        "lambda_perceptual": 0.0,
        "lambda_adversarial": 0.1,
        "lambda_sam": 0.05,
        "sinkhorn_epsilon": 0.1,
        "sinkhorn_iters": 5,
        "sinkhorn_max_points": 64,
        "sinkhorn_kernel_clamp": 40.0,
        "sinkhorn_force_fp32": True,
        "sinkhorn_loss_clip": 5.0,
        "use_adaptive_weights": False,
        "discriminator_base_dim": 8,
        "discriminator_num_heads": 2,
        "discriminator_num_blocks": [1, 1, 1],
    }


def run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    start = time.perf_counter()
    try:
        details = fn()
        return CheckResult(name, True, details, time.perf_counter() - start)
    except Exception as exc:
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)


def assert_cuda_available(args: argparse.Namespace) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    count = torch.cuda.device_count()
    if args.device < 0 or args.device >= count:
        raise RuntimeError(f"requested cuda:{args.device}, but only {count} CUDA device(s) are visible")
    torch.cuda.set_device(args.device)
    name = torch.cuda.get_device_name(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    return f"cuda:{args.device} {name}, capability={capability[0]}.{capability[1]}"


def assert_memory_budget(args: argparse.Namespace) -> str:
    free, total = torch.cuda.mem_get_info(args.device)
    free_gb = free / GB
    total_gb = total / GB
    if free_gb < args.min_free_gb:
        raise RuntimeError(f"free memory {free_gb:.2f} GB < required {args.min_free_gb:.2f} GB")
    return f"free={free_gb:.2f} GB total={total_gb:.2f} GB"


def assert_data_paths(args: argparse.Namespace) -> str:
    if args.skip_data_check:
        return "skipped by --skip-data-check"

    dataset_source = hydra_override_value(args.training_args, "dataset_source", "mst") or "mst"
    data_dir = hydra_override_value(
        args.training_args,
        "data_dir",
        os.environ.get("HSI_DATA_DIR", "./data/ARAD_1K"),
    )
    log_dir = hydra_override_value(
        args.training_args,
        "log_dir",
        os.environ.get("HSI_LOG_DIR", "./artifacts/logs"),
    )
    checkpoint_dir = hydra_override_value(
        args.training_args,
        "checkpoint_dir",
        os.environ.get("HSI_CKPT_DIR", "./artifacts/checkpoints"),
    )
    config = {
        "dataset_source": dataset_source,
        "data_dir": data_dir,
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
    }
    setup_paths(config)
    if dataset_source.strip().lower() in HF_SOURCES:
        return f"dataset_source={dataset_source}; local path validation skipped"
    return f"data_dir={Path(str(data_dir)).resolve()}"


def build_gpu_context(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    config = tiny_config()
    model = NoiseRobustCSWinModel(config).to(device)
    criterion = NoiseRobustLoss(config).to(device)
    disc_criterion = ComputeSinkhornDiscriminatorLoss(criterion)
    generator_criterion = build_criterion(config).to(device)
    batch_size = int(args.preflight_batch_size)
    size = int(args.preflight_size)
    rgb = torch.rand(batch_size, 3, size, size, device=device)
    hsi = torch.rand(batch_size, 31, size, size, device=device)
    return {
        "device": device,
        "model": model,
        "criterion": criterion,
        "disc_criterion": disc_criterion,
        "generator_criterion": generator_criterion,
        "generator_only": args.trainer == "generator",
        "rgb": rgb,
        "hsi": hsi,
    }


def assert_model_instantiates(context: dict[str, object]) -> str:
    model = context["model"]
    assert isinstance(model, NoiseRobustCSWinModel)
    params = sum(param.numel() for param in model.parameters())
    return f"parameters={params:,}"


def assert_forward_finite(context: dict[str, object]) -> str:
    model = context["model"]
    rgb = context["rgb"]
    hsi = context["hsi"]
    assert isinstance(model, NoiseRobustCSWinModel)
    assert isinstance(rgb, torch.Tensor)
    assert isinstance(hsi, torch.Tensor)
    model.eval()
    with torch.no_grad():
        pred = model(rgb)
    if pred.shape != hsi.shape:
        raise RuntimeError(f"prediction shape {tuple(pred.shape)} != target shape {tuple(hsi.shape)}")
    if not torch.isfinite(pred).all():
        raise RuntimeError("generator output contains NaN/Inf")
    context["pred_eval"] = pred
    return f"shape={tuple(pred.shape)}"


def assert_training_step(context: dict[str, object]) -> str:
    model = context["model"]
    criterion = context["criterion"]
    disc_criterion = context["disc_criterion"]
    rgb = context["rgb"]
    hsi = context["hsi"]
    assert isinstance(model, NoiseRobustCSWinModel)
    assert isinstance(criterion, NoiseRobustLoss)
    assert isinstance(disc_criterion, ComputeSinkhornDiscriminatorLoss)
    assert isinstance(rgb, torch.Tensor)
    assert isinstance(hsi, torch.Tensor)

    model.train()
    if context.get("generator_only", False):
        generator_criterion = context["generator_criterion"]
        assert isinstance(generator_criterion, torch.nn.Module)
        optimizer = torch.optim.Adam(model.generator.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        pred = model.generator(rgb)
        loss = generator_criterion(pred.float(), hsi.float())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite generator-only loss: {loss}")
        loss.backward()
        optimizer.step()
        context["pred_train"] = pred.detach()
        return f"generator_loss={loss.item():.6f}"

    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4)

    optimizer_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake_for_d = model.generator(rgb)
    real_pred = model.discriminator(rgb, hsi)
    fake_pred = model.discriminator(rgb, fake_for_d)
    disc_loss = disc_criterion(real_pred, fake_pred)
    if not torch.isfinite(disc_loss):
        raise RuntimeError(f"non-finite discriminator loss: {disc_loss}")
    disc_loss.backward()
    optimizer_d.step()

    optimizer_g.zero_grad(set_to_none=True)
    pred = model.generator(rgb)
    disc_real = model.discriminator(rgb, hsi).detach()
    disc_fake = model.discriminator(rgb, pred)
    gen_loss, _ = criterion(pred, hsi, disc_real=disc_real, disc_fake=disc_fake, current_iteration=1)
    if not torch.isfinite(gen_loss):
        raise RuntimeError(f"non-finite generator loss: {gen_loss}")
    gen_loss.backward()
    optimizer_g.step()
    context["pred_train"] = pred.detach()
    return f"disc_loss={disc_loss.item():.6f} gen_loss={gen_loss.item():.6f}"


def assert_amp_step(context: dict[str, object], enabled: bool) -> str:
    if not enabled:
        return "skipped by --no-amp-check"

    model = context["model"]
    criterion = context["criterion"]
    rgb = context["rgb"]
    hsi = context["hsi"]
    assert isinstance(model, NoiseRobustCSWinModel)
    assert isinstance(criterion, NoiseRobustLoss)
    assert isinstance(rgb, torch.Tensor)
    assert isinstance(hsi, torch.Tensor)

    model.train()
    optimizer = (
        torch.optim.Adam(model.generator.parameters(), lr=1e-4)
        if context.get("generator_only", False)
        else torch.optim.Adam(model.generator.parameters(), lr=1e-4)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True):
        pred = model.generator(rgb)
        if context.get("generator_only", False):
            generator_criterion = context["generator_criterion"]
            assert isinstance(generator_criterion, torch.nn.Module)
            loss = generator_criterion(pred.float(), hsi.float())
        else:
            loss, _ = criterion(pred, hsi, current_iteration=2)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite AMP loss: {loss}")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return f"amp_loss={loss.item():.6f} scaler={scaler.get_scale():.1f}"


def assert_metrics_finite(context: dict[str, object]) -> str:
    pred = context.get("pred_eval", context.get("pred_train"))
    hsi = context["hsi"]
    assert isinstance(pred, torch.Tensor)
    assert isinstance(hsi, torch.Tensor)
    metrics = compute_metrics(
        torch.clamp(pred[:1], 0.0, 1.0),
        torch.clamp(hsi[:1], 0.0, 1.0),
        compute_all=True,
    )
    bad = {key: value for key, value in metrics.items() if not torch.isfinite(torch.tensor(value))}
    if bad:
        raise RuntimeError(f"non-finite metrics: {bad}")
    return " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


def run_preflight(args: argparse.Namespace) -> list[CheckResult]:
    results: list[CheckResult] = []
    context: dict[str, object] = {}

    cuda_result = run_check("CUDA visible", lambda: assert_cuda_available(args))
    results.append(cuda_result)
    if cuda_result.passed:
        results.append(run_check("GPU memory budget", lambda: assert_memory_budget(args)))
    else:
        results.append(CheckResult("GPU memory budget", False, "blocked by failed CUDA prerequisite", 0.0))
    results.append(run_check("Training data paths", lambda: assert_data_paths(args)))

    if all(result.passed for result in results[:2]):
        def build_context() -> str:
            context.update(build_gpu_context(args))
            return f"device={context['device']}"

        train_step_name = (
            "Generator train step"
            if args.trainer == "generator"
            else "Sinkhorn train step"
        )
        gpu_checks: list[tuple[str, Callable[[], str]]] = [
            ("GPU context allocation", build_context),
            ("Model instantiation", lambda: assert_model_instantiates(context)),
            ("Generator forward finite", lambda: assert_forward_finite(context)),
            (train_step_name, lambda: assert_training_step(context)),
            ("AMP train step", lambda: assert_amp_step(context, not args.no_amp_check)),
            ("Metrics finite", lambda: assert_metrics_finite(context)),
        ]
        for name, fn in gpu_checks:
            results.append(run_check(name, fn))
    else:
        for name in (
            "GPU context allocation",
            "Model instantiation",
            "Generator forward finite",
            (
                "Generator train step"
                if args.trainer == "generator"
                else "Sinkhorn train step"
            ),
            "AMP train step",
            "Metrics finite",
        ):
            results.append(CheckResult(name, False, "blocked by failed CUDA prerequisite", 0.0))

    return results


def print_results(results: Iterable[CheckResult]) -> None:
    print("\nGPU PRE-FLIGHT RESULTS")
    print("-" * 88)
    print(f"{'STATUS':<8} {'CHECK':<28} {'TIME':>8}  DETAILS")
    print("-" * 88)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status:<8} {result.name:<28} {result.seconds:>7.2f}s  {result.details}")
    print("-" * 88)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_preflight(args)
    print_results(results)

    if not all(result.passed for result in results):
        print("GPU preflight failed. Training was not started.")
        return 1

    cmd = training_command(args)
    print("All GPU preflight checks passed.")
    print("Training is starting now:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    sys.stdout.flush()

    if args.dry_run:
        print("Dry run requested; training command was not executed.")
        return 0

    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""CAS-HSI benchmark harness (spec 9.10).

Measures, per input size, for a given variant/backend:
    parameter count, MACs, serialized model size, peak memory,
    mean / median / P90 / P99 latency, throughput, startup time.

Runtime rows of the spec's 9.10 matrix that this harness covers:
    PyTorch eager FP32          --dtype fp32   (default)
    PyTorch eager BF16/FP16     --dtype bf16 | fp16   (torch.autocast)

Rows deliberately NOT covered, because the toolchains are absent from this repo
(the harness always prints the runtime it actually measured, so a row is never
silently mislabelled):
    torch.compile, ONNX Runtime FP32/FP16, TensorRT FP16/INT8, target edge runtime.

Power consumption (spec: "where measurable") is NOT measurable from pure
PyTorch on this platform and is reported as null rather than as a fake zero.

Examples
--------
    python benchmark_cas.py --variant tiny
    python benchmark_cas.py --variant base --backend edge --json bench.json
    python benchmark_cas.py --variant tiny --sizes 64x64,128x128 --warmup 2 --iters 5
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager

import torch
import torch.nn as nn

from cas_hsi import build_cas_hsi

# Spec 9.10 benchmark conditions.
SPEC_WARMUP_ITERATIONS = 50
SPEC_MEASURED_ITERATIONS = 200
SPEC_BATCH_SIZE = 1
SPEC_INPUT_SIZES = [(128, 128), (256, 256), (482, 512), (512, 512)]

RGB_CHANNELS = 3

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class OutputInvalid(RuntimeError):
    """The model produced a wrong-shaped or non-finite output.

    This is a correctness failure, not a resource failure: it is deliberately
    NOT swallowed into a results row, because latency numbers for a model that
    does not compute the right thing are worthless.
    """


def autocast_context(device: torch.device, dtype: torch.dtype) -> ContextManager:
    if dtype is torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


# --------------------------------------------------------------------------
# MAC counting
# --------------------------------------------------------------------------
# Primary: torch.utils.flop_counter.FlopCounterMode. It ships with torch (no
# optional dependency) and counts at the aten level, so it sees BOTH the
# convolutions and every matmul -- which here means cross-channel attention
# (4.6) and stripe attention (4.5.4).
#
# It does NOT see dilated local attention (4.5.3). The research backend computes
# neighborhood attention with a shift-and-gather -- `(q * shifted_k).sum(dim=2)`
# and `weights * shifted_v` (see cas_hsi/blocks/spatial_attention.py) -- i.e.
# elementwise multiply + reduce, never a matmul. No matmul-based or conv-based
# counter can see those FLOPs; verified empirically: FlopCounterMode reports only
# aten.convolution and aten.bmm for this model. So we add that term analytically
# from the module's own head-group configuration.
#
# fvcore and thop are deliberately NOT used:
#   * thop only patches nn.Module types, so it is structurally blind to the
#     functional matmuls in both attention blocks -- it would silently undercount
#     by ~7% -- and it mutates the model by attaching total_ops/total_params
#     buffers, which would then corrupt the serialized-size measurement.
#   * fvcore duplicates what torch's own aten-level counter already does.


def gather_attention_macs(model: nn.Module, x: torch.Tensor, dtype: torch.dtype) -> int:
    """MACs of the shift-gather neighborhood attention, invisible to every counter.

    For one local head group at [B, C, H, W] with K = kernel_area neighbours:
        logits  = sum_d q * shifted_k   ->  B * heads * head_dim * H * W * K MACs
        output  = sum_K w * shifted_v   ->  B * heads * head_dim * H * W * K MACs

    Stripe head groups are excluded: they use torch.matmul and are already
    counted by any matmul-aware counter.

    Duck-typed on the attention modules' public attributes so this never has to
    import a private class from the (frozen) model package.
    """
    total = 0
    handles = []

    def hook(module, inputs, _output):
        nonlocal total
        tensor = inputs[0]
        batch, _channels, height, width = tensor.shape
        for group in module.head_groups:
            if group.kind != "local":
                continue
            total += (
                2
                * batch
                * group.heads
                * module.head_dim
                * height
                * width
                * module.kernel_area
            )

    for module in model.modules():
        if (
            hasattr(module, "head_groups")
            and hasattr(module, "head_dim")
            and hasattr(module, "kernel_area")
        ):
            handles.append(module.register_forward_hook(hook))

    if not handles:  # edge backend: no attention mixers at all
        return 0

    try:
        with torch.no_grad(), autocast_context(x.device, dtype):
            model(x)
    finally:
        for handle in handles:
            handle.remove()

    return total


def _macs_torch_flop_counter(model: nn.Module, x: torch.Tensor, dtype: torch.dtype) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad(), autocast_context(x.device, dtype):
        model(x)
    total_flops = counter.get_total_flops()
    if total_flops <= 0:
        raise RuntimeError("FlopCounterMode returned no flops")
    # torch's counter reports FLOPs with 1 MAC == 2 FLOPs.
    return int(total_flops // 2)


def _macs_conv_hooks(model: nn.Module, x: torch.Tensor, dtype: torch.dtype) -> int:
    """Emergency fallback: Conv/ConvTranspose/Linear MACs via module hooks.

    Only reached if torch.utils.flop_counter is unavailable. Undercounts, because
    the attention matmuls (cross-channel 4.6, stripe 4.5.4) are functional ops and
    fire no module hook. The caller labels and warns about this.
    """
    total = 0
    handles = []

    def conv_hook(module, _inputs, output):
        nonlocal total
        out_elems = output.numel()  # N * C_out * H_out * W_out
        in_ch_per_group = module.in_channels // module.groups
        kernel = 1
        for k in module.kernel_size:
            kernel *= k
        total += out_elems * in_ch_per_group * kernel

    def deconv_hook(module, inputs, _output):
        nonlocal total
        # ConvTranspose: MACs scale with the INPUT spatial extent.
        in_elems = inputs[0].numel()  # N * C_in * H_in * W_in
        out_ch_per_group = module.out_channels // module.groups
        kernel = 1
        for k in module.kernel_size:
            kernel *= k
        total += in_elems * out_ch_per_group * kernel

    def linear_hook(module, _inputs, output):
        nonlocal total
        total += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, nn.modules.conv._ConvTransposeNd):
            handles.append(module.register_forward_hook(deconv_hook))
        elif isinstance(module, nn.modules.conv._ConvNd):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    try:
        with torch.no_grad(), autocast_context(x.device, dtype):
            model(x)
    finally:
        for handle in handles:
            handle.remove()

    if total <= 0:
        raise RuntimeError("hook counter matched no conv/linear modules")
    return total


_MAC_BACKENDS: list[tuple[str, Callable[[nn.Module, torch.Tensor, torch.dtype], int]]] = [
    ("torch.flop_counter", _macs_torch_flop_counter),
    ("conv_hooks(conv+linear only)", _macs_conv_hooks),
]


def count_macs(
    model: nn.Module, x: torch.Tensor, dtype: torch.dtype
) -> tuple[int | None, str]:
    """Return (macs, method). ``macs`` is None only if every backend failed.

    The gather-attention term is added on top of whichever backend wins, because
    no counter can see it.
    """
    errors = []
    for name, fn in _MAC_BACKENDS:
        try:
            macs = fn(model, x, dtype)
        except ImportError:
            errors.append(f"{name}: not installed")
            continue
        except Exception as exc:  # counter present but blew up on this model
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if macs <= 0:
            errors.append(f"{name}: returned 0")
            continue

        gather = gather_attention_macs(model, x, dtype)
        method = name if gather == 0 else f"{name}+gather_attention"
        return macs + gather, method

    print(
        "  [warn] MAC counting failed on every backend:\n    " + "\n    ".join(errors),
        file=sys.stderr,
    )
    return None, "unavailable"


# --------------------------------------------------------------------------
# Peak memory
# --------------------------------------------------------------------------


def measure_peak_memory(
    model: nn.Module, x: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> tuple[int | None, str]:
    """Allocator high-water mark for one forward pass, in bytes.

    CUDA: torch.cuda.max_memory_allocated -- the allocator high-water mark.

    CPU : reconstructed from torch.profiler's allocation timeline. Every CREATE
          event adds its size, every DESTROY subtracts it, and we take the running
          maximum. This is the same *semantics* as the CUDA number (peak bytes
          live in the allocator, model weights included), so the two are directly
          comparable.

          It replaces a psutil RSS-delta, which does not measure this at all: the
          caching allocator reuses already-resident blocks, so RSS barely moves
          across a warmed-up loop. Measured on tiny/research it read 0.1 MiB at
          64x64 and 0.0 MiB at 128x128 -- non-monotone in input size, and 2-3
          orders of magnitude below the true peak (16.7 / 46.5 MiB).

    Runs its own forward pass, deliberately outside the timing loop: the profiler
    adds large overhead and would otherwise corrupt the latency numbers.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad(), autocast_context(device, dtype):
            model(x)
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device)), "cuda_max_memory_allocated"

    try:
        from torch.profiler import ProfilerActivity, profile
        from torch.profiler._memory_profiler import Action

        with profile(
            activities=[ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad(), autocast_context(device, dtype):
                model(x)

        live = 0
        peak = 0
        for _timestamp, action, _key, num_bytes in prof._memory_profile().timeline:
            if action in (Action.PREEXISTING, Action.CREATE):
                live += num_bytes
            elif action == Action.DESTROY:
                live -= num_bytes
            peak = max(peak, live)

        if peak <= 0:
            raise RuntimeError("profiler produced an empty allocation timeline")
        return int(peak), "cpu_torch_profiler_peak_allocated"
    except Exception as exc:
        # Report nothing rather than a number that does not mean what it says.
        print(
            f"  [warn] CPU peak-memory probe failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None, "unavailable"


# --------------------------------------------------------------------------
# Benchmark core
# --------------------------------------------------------------------------


@dataclass
class SizeResult:
    height: int
    width: int
    macs: int | None
    mac_method: str
    peak_memory_bytes: int | None
    peak_memory_kind: str
    latencies_ms: list[float]
    error: str | None = None

    def stats(self) -> dict[str, float | None]:
        if not self.latencies_ms:
            return {
                k: None
                for k in (
                    "mean_ms", "median_ms", "p90_ms", "p99_ms",
                    "min_ms", "max_ms", "stddev_ms", "throughput_img_per_s",
                )
            }
        lat = sorted(self.latencies_ms)
        mean = statistics.fmean(lat)
        return {
            "mean_ms": mean,
            "median_ms": statistics.median(lat),
            "p90_ms": _percentile(lat, 90.0),
            "p99_ms": _percentile(lat, 99.0),
            "min_ms": lat[0],
            "max_ms": lat[-1],
            "stddev_ms": statistics.pstdev(lat) if len(lat) > 1 else 0.0,
            # batch_size images per measured iteration.
            "throughput_img_per_s": (SPEC_BATCH_SIZE * 1000.0 / mean) if mean > 0 else None,
        }


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile on an already-sorted list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def serialized_size_bytes(model: nn.Module) -> int:
    """Size of the serialized state_dict, measured in memory (no temp files)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def validate_output(
    output: torch.Tensor, height: int, width: int, bands: int
) -> None:
    """A benchmark must not report latencies for a model that computes garbage.

    Spec: output is [B, bands, H, W] at the ORIGINAL size (8.1, 8.4, 7.7) with no
    terminal activation (7.4). Without this check the harness would happily time a
    model whose pad/crop was broken, or one that had silently started emitting NaN.
    """
    expected = (SPEC_BATCH_SIZE, bands, height, width)
    if tuple(output.shape) != expected:
        raise OutputInvalid(
            f"output shape {tuple(output.shape)} != expected {expected} "
            f"for a {height}x{width} input (spec 8.1/8.4: the model must return "
            "the ORIGINAL spatial size)"
        )
    if not torch.isfinite(output).all():
        raise OutputInvalid(
            f"output at {height}x{width} contains NaN or Inf "
            f"({int((~torch.isfinite(output)).sum())} of {output.numel()} elements)"
        )


def benchmark_size(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
    warmup: int,
    iters: int,
    bands: int,
    skip_memory: bool,
) -> SizeResult:
    x = torch.randn(SPEC_BATCH_SIZE, RGB_CHANNELS, height, width, device=device)

    macs, mac_method = count_macs(model, x, dtype)

    if skip_memory:
        peak, mem_kind = None, "skipped"
    else:
        peak, mem_kind = measure_peak_memory(model, x, device, dtype)

    latencies: list[float] = []

    try:
        with torch.no_grad(), autocast_context(device, dtype):
            for _ in range(warmup):
                output = model(x)
            _sync(device)

            # Correctness gate, on a warmed-up forward, before any timing.
            validate_output(model(x).float(), height, width, bands)
            _sync(device)

            for _ in range(iters):
                start = time.perf_counter()
                model(x)
                _sync(device)  # synchronize_device: true
                latencies.append((time.perf_counter() - start) * 1000.0)
    except OutputInvalid:
        raise  # correctness failure: never demote it to a table row
    except Exception as exc:  # OOM or anything else: record and keep going
        return SizeResult(
            height, width, macs, mac_method, None, mem_kind, [],
            error=f"{type(exc).__name__}: {exc}",
        )

    return SizeResult(height, width, macs, mac_method, peak, mem_kind, latencies)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_sizes(raw: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise argparse.ArgumentTypeError(
                f"bad size {token!r}: expected HxW, e.g. 482x512"
            )
        h_str, _, w_str = token.partition("x")
        try:
            height, width = int(h_str), int(w_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"bad size {token!r}: expected HxW, e.g. 482x512"
            ) from None
        if height <= 0 or width <= 0:
            raise argparse.ArgumentTypeError(f"bad size {token!r}: must be positive")
        sizes.append((height, width))
    if not sizes:
        raise argparse.ArgumentTypeError("no sizes given")
    return sizes


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _jsonable(value: Any) -> Any:
    """Keep dicts/lists structured in the JSON payload instead of str()-ing them."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CAS-HSI benchmark (spec 9.10), PyTorch eager.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--backend", choices=["research", "edge"], default="research")
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=SPEC_INPUT_SIZES,
        help="comma-separated HxW list, e.g. 128x128,482x512 "
             "(default: the spec's 128x128,256x256,482x512,512x512)",
    )
    parser.add_argument(
        "--warmup", type=int, default=SPEC_WARMUP_ITERATIONS,
        help="spec value is 50; lower it for CPU runs",
    )
    parser.add_argument(
        "--iters", type=int, default=SPEC_MEASURED_ITERATIONS,
        help="spec value is 200; lower it for CPU runs",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=default_device())
    parser.add_argument(
        "--dtype", choices=list(_DTYPES), default="fp32",
        help="fp32 = eager FP32; bf16/fp16 run the forward under torch.autocast",
    )
    parser.add_argument("--threads", type=int, default=None,
                        help="torch.set_num_threads override (CPU)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random input and model init (reproducibility)")
    parser.add_argument("--skip-memory", action="store_true",
                        help="skip the peak-memory probe (it runs an extra profiled forward)")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="also write results to this JSON file")
    args = parser.parse_args(argv)

    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be >= 0 and --iters must be >= 1")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but torch.cuda.is_available() is False")

    device = torch.device(args.device)
    dtype = _DTYPES[args.dtype]
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    runtime = f"PyTorch eager {args.dtype.upper()}" + (
        "" if dtype is torch.float32 else " (torch.autocast)"
    )

    # Startup time: cold model construction + move to device.
    build_start = time.perf_counter()
    model = build_cas_hsi(args.variant, backend=args.backend)
    model.eval().to(device)
    _sync(device)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    info = model.get_model_info()
    bands = int(info["output_bands"])

    # Startup time: first (cold) forward, at the first requested size.
    cold_h, cold_w = args.sizes[0]
    cold_x = torch.randn(SPEC_BATCH_SIZE, RGB_CHANNELS, cold_h, cold_w, device=device)
    cold_start = time.perf_counter()
    with torch.no_grad(), autocast_context(device, dtype):
        cold_out = model(cold_x)
    _sync(device)
    cold_forward_ms = (time.perf_counter() - cold_start) * 1000.0
    validate_output(cold_out.float(), cold_h, cold_w, bands)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ser_bytes = serialized_size_bytes(model)

    spec_conditions = (
        args.warmup == SPEC_WARMUP_ITERATIONS and args.iters == SPEC_MEASURED_ITERATIONS
    )

    print()
    print("=" * 96)
    print(f"CAS-HSI benchmark  --  spec 9.10  --  runtime: {runtime} ({torch.__version__})")
    print("=" * 96)
    print(f"  variant / backend   : {args.variant} / {args.backend}  ({info.get('name')})")
    print(f"  device              : {device}"
          + (f"  [{torch.cuda.get_device_name(device)}]" if device.type == "cuda" else
             f"  [{platform.processor() or platform.machine()}, "
             f"torch threads={torch.get_num_threads()}]"))
    print(f"  parameters          : {n_params:,} total / {n_trainable:,} trainable")
    print(f"  serialized size     : {ser_bytes / 1024 / 1024:.2f} MiB "
          f"({ser_bytes:,} bytes, state_dict, fp32)")
    print(f"  widths / depths     : {info.get('widths')} / {info.get('depths')}")
    print(f"  startup time        : {build_ms:.1f} ms build+to(device), "
          f"{cold_forward_ms:.1f} ms first forward @ {cold_h}x{cold_w}")
    print(f"  batch size          : {SPEC_BATCH_SIZE}   synchronize_device: True")
    print(f"  warmup / measured   : {args.warmup} / {args.iters} iterations"
          + ("  [spec values]" if spec_conditions
             else f"  [OVERRIDDEN -- spec is {SPEC_WARMUP_ITERATIONS}/{SPEC_MEASURED_ITERATIONS}]"))
    print(f"  output check        : shape [B,{bands},H,W] at original size + all-finite, "
          f"asserted per size")
    print("  power consumption   : not measurable from PyTorch on this platform "
          "(reported as null)")
    print()

    results: list[SizeResult] = []
    for height, width in args.sizes:
        print(f"  running {height}x{width} ...", end="", flush=True)
        result = benchmark_size(
            model, device, dtype, height, width,
            args.warmup, args.iters, bands, args.skip_memory,
        )
        results.append(result)
        print(" done" if result.error is None else f" FAILED ({result.error})")
    print()

    mem_kinds = {r.peak_memory_kind for r in results}
    mac_methods = {r.mac_method for r in results}

    header = (
        f"{'size':>9}  {'MACs (G)':>10}  {'peak mem':>12}  {'mean ms':>9}  "
        f"{'median':>9}  {'P90':>9}  {'P99':>9}  {'img/s':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        if result.error is not None:
            print(f"{result.height:>4}x{result.width:<4}  {'-- failed: ' + result.error:<60}")
            continue
        stats = result.stats()
        macs_g = None if result.macs is None else result.macs / 1e9
        mem_mib = (
            None if result.peak_memory_bytes is None
            else result.peak_memory_bytes / 1024 / 1024
        )
        print(
            f"{result.height:>4}x{result.width:<4}  {_fmt(macs_g, 3):>10}  "
            f"{(_fmt(mem_mib, 1) + ' MiB') if mem_mib is not None else 'n/a':>12}  "
            f"{_fmt(stats['mean_ms']):>9}  {_fmt(stats['median_ms']):>9}  "
            f"{_fmt(stats['p90_ms']):>9}  {_fmt(stats['p99_ms']):>9}  "
            f"{_fmt(stats['throughput_img_per_s']):>8}"
        )
    print()
    print(f"  MACs method   : {', '.join(sorted(mac_methods))}")
    if any("gather_attention" in m for m in mac_methods):
        print("                  (torch's counter sees conv + matmul only; the shift-gather")
        print("                  neighborhood attention of spec 4.5.3 uses neither, so its")
        print("                  MACs are added analytically from each block's head groups.)")
    if any("conv_hooks" in m for m in mac_methods):
        print("                  WARNING: conv/linear hooks only -- the attention MATMULS")
        print("                  (cross-channel 4.6, stripe 4.5.4) are NOT counted, so this")
        print("                  is an UNDERCOUNT.")
    print(f"  peak memory   : {', '.join(sorted(mem_kinds))}")
    if "cpu_torch_profiler_peak_allocated" in mem_kinds:
        print("                  Allocator high-water mark for one forward, reconstructed")
        print("                  from torch.profiler's alloc/free timeline (weights included).")
        print("                  Same semantics as CUDA's max_memory_allocated, so the two")
        print("                  are comparable. Measured outside the timing loop.")
    print()

    if args.json_path:
        payload: dict[str, Any] = {
            "spec": "9.10",
            "runtime": f"pytorch_eager_{args.dtype}",
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "device_name": (torch.cuda.get_device_name(device)
                            if device.type == "cuda"
                            else (platform.processor() or platform.machine())),
            "torch_threads": torch.get_num_threads(),
            "variant": args.variant,
            "backend": args.backend,
            "dtype": args.dtype,
            "seed": args.seed,
            "model_info": _jsonable(info),
            "parameters_total": n_params,
            "parameters_trainable": n_trainable,
            "serialized_size_bytes": ser_bytes,
            "startup_build_ms": build_ms,
            "startup_first_forward_ms": cold_forward_ms,
            "power_consumption_w": None,
            "conditions": {
                "warmup_iterations": args.warmup,
                "measured_iterations": args.iters,
                "batch_size": SPEC_BATCH_SIZE,
                "synchronize_device": True,
                "matches_spec_conditions": spec_conditions,
                "spec_warmup_iterations": SPEC_WARMUP_ITERATIONS,
                "spec_measured_iterations": SPEC_MEASURED_ITERATIONS,
            },
            "results": [
                {
                    "height": r.height,
                    "width": r.width,
                    "macs": r.macs,
                    "mac_method": r.mac_method,
                    "peak_memory_bytes": r.peak_memory_bytes,
                    "peak_memory_kind": r.peak_memory_kind,
                    "error": r.error,
                    **r.stats(),
                }
                for r in results
            ],
        }
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"  wrote JSON -> {args.json_path}")
        print()

    return 1 if any(r.error is not None for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

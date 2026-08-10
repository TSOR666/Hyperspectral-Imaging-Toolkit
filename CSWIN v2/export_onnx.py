#!/usr/bin/env python3
"""Export any CSWIN v2 checkpoint - including pre-refactor ones - to ONNX.

The architecture is recovered from the checkpoint's own tensor keys and shapes,
so this works for weights whose config no longer builds the same model under the
current code (changed defaults, the SSTB block rewrite, the 0-d
``iteration_count`` buffer, GAN-era ``generator.``-prefixed state dicts).

Inspect first (no files written)::

    python export_onnx.py --checkpoint artifacts/checkpoints/old_best.pth --inspect

Export at the tile size you will run inference at::

    python export_onnx.py \
        --checkpoint artifacts/checkpoints/old_best.pth \
        --output artifacts/onnx/old_best_fp32.onnx \
        --height 128 --width 128 --precision fp32

Half precision (weights stored as fp16, inputs/outputs stay fp32)::

    python export_onnx.py --checkpoint old_best.pth --output old_best_fp16.onnx \
        --height 482 --width 512 --precision fp16

Then score it on ARAD-1K with ``onnx_test_ntire.py``.

The exporter verifies that the ONNX output is finite. If a legacy FP16 graph
fails that check on CPU, re-export with ``--precision fp32``; the NTIRE tester
also has an in-memory CPU FP32 recovery for existing FP16 artifacts.

Notes
-----
* The spatial size is baked into the graph: the model's reflect-padding to a
  multiple of ``split_size`` is data-dependent control flow that tracing
  resolves to constants. Batch stays dynamic. ``--dynamic-hw`` is available but
  only correct for sizes that need the same padding decisions - verify before
  trusting it.
* A few knobs leave no trace in the weights (``norm_groups``,
  ``output_activation``, ``cascade_stages``, ...). They come from the embedded
  config when the checkpoint has one; otherwise they fall back to code defaults
  and are printed as assumptions. Correct any of them with ``--set KEY=VALUE``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hsi_model.utils.onnx_export import (  # noqa: E402
    PRECISIONS,
    export_checkpoint_to_onnx,
    load_checkpoint_payload,
    rebuild_generator,
)

LOGGER = logging.getLogger("export_onnx")


def _parse_override(text: str) -> tuple[str, Any]:
    """Parse ``KEY=VALUE`` where VALUE is JSON, falling back to a plain string."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"--set expects KEY=VALUE, got {text!r} (e.g. --set norm_groups=8)"
        )
    key, _, raw = text.partition("=")
    key = key.strip()
    raw = raw.strip()
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.lower()
        if lowered in ("true", "false"):
            value = lowered == "true"
        elif lowered in ("none", "null"):
            value = None
        else:
            value = raw
    return key, value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        "--input",
        dest="checkpoint",
        required=True,
        help="Checkpoint, weights bundle, or bare state dict to export.",
    )
    parser.add_argument(
        "--output",
        "--onnx",
        dest="output",
        default=None,
        help="Destination .onnx path (required unless --inspect).",
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        help=f"Weight precision in the graph: {', '.join(PRECISIONS)}.",
    )
    parser.add_argument(
        "--fp16-io",
        action="store_true",
        help="With --precision fp16, also make the graph inputs/outputs fp16 "
        "(default keeps them fp32 so consumers need not care).",
    )
    parser.add_argument("--height", type=int, default=128, help="Baked input height.")
    parser.add_argument("--width", type=int, default=128, help="Baked input width.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Example batch size used for tracing (the batch axis stays dynamic).",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--exporter",
        default="torchscript",
        choices=["torchscript", "dynamo"],
        help="torchscript (default) traces the eval path, which is what bakes "
        "the padding decisions in correctly. dynamo uses torch.export and also "
        "needs the onnxscript package.",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Bake the batch size instead of leaving that axis dynamic.",
    )
    parser.add_argument(
        "--dynamic-hw",
        action="store_true",
        help="Mark height/width dynamic. Only valid when every size you will "
        "feed needs the same internal padding; verify before relying on it.",
    )
    parser.add_argument(
        "--clamp-output",
        action="store_true",
        help="Bake clamp(0, 1) into the graph. Off by default so unclamped MRAE "
        "stays computable, matching cswin_test_ntire.py.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a recovered/assumed config key (repeatable). VALUE is "
        "parsed as JSON, e.g. --set norm_groups=8 --set split_sizes=[7,7,7].",
    )
    parser.add_argument(
        "--architecture-config",
        default=None,
        help="JSON/YAML architecture config used as a hint when the checkpoint "
        "embeds none. Tensor evidence still wins over it.",
    )
    parser.add_argument(
        "--no-prefer-ema",
        action="store_true",
        help="Export the raw weights instead of a complete EMA shadow.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Export even if some tensors do not match the recovered "
        "architecture. Unmatched modules keep their random initialization.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the eager-vs-ONNX parity check (needs onnxruntime).",
    )
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=None,
        help="Relative-L2 budget for the parity check (default: 1e-4 fp32, 2e-2 fp16).",
    )
    parser.add_argument(
        "--fail-on-parity",
        action="store_true",
        help="Exit non-zero when the parity check exceeds the tolerance.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the recovered architecture and exit without writing files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only log warnings and errors.",
    )
    return parser.parse_args(argv)


def _print_recovery(recovery, load_report, metadata: Dict[str, Any]) -> None:
    print("Checkpoint")
    for key in ("source_checkpoint", "epoch", "ema_applied", "embedded_config_present"):
        if metadata.get(key) is not None:
            print(f"  {key}: {metadata[key]}")
    if metadata.get("val_metrics"):
        print(f"  val_metrics: {metadata['val_metrics']}")

    print("\nRecovered architecture")
    for key in sorted(recovery.config):
        source = recovery.evidence.get(key, "?")
        print(f"  {key:34s} = {recovery.config[key]!r}   [{source}]")

    if recovery.conflicts:
        print("\nEmbedded config disagreed with the weights (weights win)")
        for key, (before, after) in recovery.conflicts.items():
            print(f"  {key}: config={before!r} -> tensors={after!r}")

    if recovery.assumptions:
        print("\nNot recoverable from the weights - override with --set if wrong")
        for line in recovery.assumptions:
            print(f"  - {line}")

    if recovery.notes:
        print("\nNotes")
        for line in recovery.notes:
            print(f"  - {line}")

    print(f"\nState-dict match: {load_report.describe()}")
    if load_report.adapted:
        for line in load_report.adapted:
            print(f"  adapted: {line}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    overrides: Dict[str, Any] = {}
    if args.architecture_config:
        from hsi_model.utils.inference import load_architecture_config

        hint = load_architecture_config(args.architecture_config)
    else:
        hint = None
    for item in args.overrides:
        key, value = _parse_override(item)
        overrides[key] = value

    if args.inspect:
        state, embedded_config, metadata = load_checkpoint_payload(
            args.checkpoint, prefer_ema=not args.no_prefer_ema
        )
        if hint:
            embedded_config = {**hint, **embedded_config}
            metadata["architecture_config_hint"] = args.architecture_config
        try:
            generator, recovery, load_report = rebuild_generator(
                state,
                embedded_config,
                overrides=overrides,
                strict=not args.allow_partial,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        metadata["parameters"] = int(sum(p.numel() for p in generator.parameters()))
        _print_recovery(recovery, load_report, metadata)
        print(f"Parameters: {metadata['parameters']:,}")
        return 0

    if not args.output:
        print(
            "ERROR: --output is required unless --inspect is given.", file=sys.stderr
        )
        return 2

    try:
        result = export_checkpoint_to_onnx(
            args.checkpoint,
            args.output,
            height=args.height,
            width=args.width,
            batch_size=args.batch_size,
            precision=args.precision,
            keep_io_fp32=not args.fp16_io,
            clamp_output=args.clamp_output,
            opset=args.opset,
            dynamic_batch=not args.static_batch,
            dynamic_hw=args.dynamic_hw,
            exporter=args.exporter,
            overrides=overrides,
            architecture_config_hint=hint,
            prefer_ema=not args.no_prefer_ema,
            strict=not args.allow_partial,
            verify=not args.no_verify,
            parity_tolerance=args.parity_tolerance,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _print_recovery(
        result.recovery, result.load_report, result.manifest["checkpoint"]
    )
    print(f"Parameters: {result.manifest['checkpoint'].get('parameters'):,}")

    export = result.manifest["export"]
    print(f"\nWrote {result.onnx_path} ({export['file_bytes'] / 1e6:.2f} MB)")
    print(f"  precision: {export['precision']}  io dtype: {export['io_dtype']}")
    print(
        f"  input '{export['input_name']}': {export['input_shape']} "
        f"(batch dynamic: {export['dynamic_batch']}, hw dynamic: {export['dynamic_hw']})"
    )
    print(f"  output '{export['output_name']}': clamped in graph: {export['clamp_output']}")
    if result.manifest_path:
        print(f"  manifest: {result.manifest_path}")

    if result.parity:
        parity = result.parity
        verdict = "PASS" if parity["within_tolerance"] else "FAIL"
        print(
            f"  parity vs eager torch: {verdict} "
            f"rel_l2={parity['rel_l2']:.3g} (tol {parity['tolerance']:.3g}), "
            f"max_abs={parity['max_abs_diff']:.3g}"
        )
        if args.fail_on_parity and not parity["within_tolerance"]:
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

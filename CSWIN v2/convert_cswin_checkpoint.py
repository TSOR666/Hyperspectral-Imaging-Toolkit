#!/usr/bin/env python3
"""Convert a CSWIN checkpoint into generator-only inference weights.

Examples
--------
Checkpoint with an embedded config::

    python convert_cswin_checkpoint.py \
        --checkpoint artifacts/checkpoints/best_model.pth \
        --output artifacts/checkpoints/cswin_generator_weights.pth

Older checkpoint or bare weights plus a known architecture config::

    python convert_cswin_checkpoint.py \
        --checkpoint old_model.pth \
        --architecture_config old_architecture.yaml \
        --output old_generator_weights.pth

The default output embeds the architecture config. ``--raw_state_dict`` writes
only the tensor mapping; that form must be loaded with ``--architecture_config``
when running inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hsi_model.utils.inference import convert_checkpoint_to_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CSWIN checkpoint or weights file to generator-only weights."
    )
    parser.add_argument(
        "--checkpoint",
        "--input",
        dest="checkpoint_path",
        required=True,
        help="Source checkpoint or weights file.",
    )
    parser.add_argument(
        "--output",
        "--weights_path",
        dest="output_path",
        required=True,
        help="Destination generator weights file.",
    )
    parser.add_argument(
        "--architecture_config",
        "--config",
        dest="architecture_config",
        default=None,
        help="JSON/YAML architecture config when the source has no embedded config.",
    )
    parser.add_argument(
        "--no_prefer_ema",
        action="store_true",
        help="Use raw checkpoint weights instead of a complete EMA shadow.",
    )
    parser.add_argument(
        "--non_strict",
        action="store_true",
        help="Allow missing/unexpected keys, while still requiring at least one match.",
    )
    parser.add_argument(
        "--raw_state_dict",
        action="store_true",
        help="Write only the bare tensor state dict; config is not embedded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = convert_checkpoint_to_weights(
        args.checkpoint_path,
        args.output_path,
        architecture_config=args.architecture_config,
        device=torch.device("cpu"),
        prefer_ema=not args.no_prefer_ema,
        strict=not args.non_strict,
        embed_config=not args.raw_state_dict,
    )
    print(f"Wrote {info['weights_format']} to {info['output_path']}")
    print(f"Generator tensors: {info['weights_keys']}")
    print(f"Architecture config: {info['config_source']}")
    print(f"EMA applied: {info['ema_applied']}")


if __name__ == "__main__":
    main()

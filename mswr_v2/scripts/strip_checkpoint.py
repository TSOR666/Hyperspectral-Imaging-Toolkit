#!/usr/bin/env python3
"""Strip an MSWR checkpoint to model weights only for warm-restart fine-tuning.

The trainer's resume path loads optimizer / scheduler / EMA / iteration / epoch
state whenever they are present in the checkpoint. For a warm restart with a
fresh LR schedule, those keys must be absent — the trainer's `.get()` defaults
will then start training from iteration 0 with a clean cosine cycle.

Usage:
    python scripts/strip_checkpoint.py \
        --input  /path/to/best_model.pth \
        --output /path/to/best_model_weights_only.pth

The output retains only `state_dict` (and `best_mrae` for logging continuity).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def strip(input_path: Path, output_path: Path, keep_best_mrae: bool = True) -> None:
    try:
        ckpt = torch.load(input_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(
            f"weights_only=True load failed ({exc}); retrying with weights_only=False. "
            "Only do this for checkpoints you trust.",
            file=sys.stderr,
        )
        ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    if "state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint at {input_path} has no 'state_dict' key; "
            f"present keys: {sorted(ckpt.keys())}"
        )

    stripped: dict = {"state_dict": ckpt["state_dict"]}
    if keep_best_mrae and "best_mrae" in ckpt:
        # The trainer reads this for logging but doesn't use it to gate training.
        stripped["best_mrae"] = ckpt["best_mrae"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, output_path)

    kept = sorted(stripped.keys())
    dropped = sorted(k for k in ckpt.keys() if k not in stripped)
    print(f"Wrote {output_path}")
    print(f"  kept keys:    {kept}")
    print(f"  dropped keys: {dropped}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Stripped output path")
    parser.add_argument(
        "--no-best-mrae",
        action="store_true",
        help="Drop best_mrae as well (default: keep it for logging only).",
    )
    args = parser.parse_args()
    strip(args.input, args.output, keep_best_mrae=not args.no_best_mrae)
    return 0


if __name__ == "__main__":
    sys.exit(main())

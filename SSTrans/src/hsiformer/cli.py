from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import Sequence

from torch.utils.data import DataLoader

from .checkpoint import build_model_from_checkpoint
from .data import ARAD1KDataset, RGBImageDataset, load_arad_manifest
from .ntire import (
    evaluate_loader,
    infer_loader,
    resolve_device,
    write_metric_reports,
)
from .training import TrainingConfig, train

PRESETS = (
    "legacy",
    "ablation_no_rpe",
    "corrected_rpe",
    "optimized_candidate",
)


def train_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train HSIFormer with an MST++-style iteration schedule."
    )
    parser.add_argument("--config", help="Training JSON configuration.")
    parser.add_argument("--data-root", help="Override config data_root.")
    parser.add_argument("--output-dir", help="Override config output_dir.")
    parser.add_argument("--resume", help="Resume from a trainer checkpoint.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)

    if args.config:
        values = json.loads(Path(args.config).read_text(encoding="utf-8"))
    else:
        text = (
            files("hsiformer")
            .joinpath("resources", "train_arad1k.json")
            .read_text(encoding="utf-8")
        )
        values = json.loads(text)
    if args.data_root:
        values["data_root"] = args.data_root
    if args.output_dir:
        values["output_dir"] = args.output_dir
    if args.no_amp:
        values["amp"] = False

    config = TrainingConfig.from_mapping(values)
    if config.data_root == "path/to/ARAD_1K":
        raise ValueError("Set --data-root or provide it in the training config.")
    latest = train(config, resume=args.resume, device=args.device)
    print(f"training complete: {latest}")


def infer_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct NTIRE-format hyperspectral cubes from RGB images."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        help="Use the packaged ARAD-1K scene order for this split.",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip", action="store_true")
    args = parser.parse_args(argv)

    if args.split and args.manifest:
        raise ValueError("Pass --split or --manifest, not both.")
    scene_ids = load_arad_manifest(args.split) if args.split else None
    device = resolve_device(args.device)
    dataset = RGBImageDataset(
        args.rgb_dir,
        scene_ids=scene_ids,
        manifest_path=args.manifest,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model, _ = build_model_from_checkpoint(
        args.checkpoint,
        preset=args.preset,
        map_location=device,
    )
    model.to(device)
    scene_ids = infer_loader(
        model,
        loader,
        device=device,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        amp=args.amp,
        clip=args.clip,
    )
    output_dir = Path(args.output_dir)
    (output_dir / "inference.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "count": len(scene_ids),
                "scene_ids": scene_ids,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {len(scene_ids)} cubes to {output_dir}")


def test_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the public ARAD-1K test split and save NTIRE 2022 .mat cubes."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip", action="store_true")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    dataset = ARAD1KDataset(
        args.data_root,
        split="test",
        manifest_path=args.manifest,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model, _ = build_model_from_checkpoint(
        args.checkpoint,
        preset=args.preset,
        map_location=device,
    )
    model.to(device)

    output_dir = Path(args.output_dir)
    summary, rows = evaluate_loader(
        model,
        loader,
        device=device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        amp=args.amp,
        output_dir=output_dir / "cubes",
        clip=args.clip,
    )
    write_metric_reports(output_dir, summary, rows)
    print(
        f"test scenes={int(summary['count'])} "
        f"MRAE={summary['mrae']:.6f} RMSE={summary['rmse']:.6f} "
        f"PSNR={summary['psnr']:.4f} SAM={summary['sam']:.6f}"
    )

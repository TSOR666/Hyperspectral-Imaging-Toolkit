from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import Sequence

from torch.utils.data import DataLoader

from .checkpoint import (
    build_model_from_checkpoint,
    checkpoint_rgb_normalization,
)
from .data import (
    ARAD1KDataset,
    RGBImageDataset,
    load_arad_manifest,
    resolve_arad_directory,
)
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
    "source_reproduction",
    "corrected_rpe",
    "optimized_candidate",
    "recommended_retrain",
    "rectangular_candidate",
)


def train_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train HSIFormer with a configured iteration schedule."
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
    parser.add_argument("--rgb-dir", help="Directory of RGB images.")
    parser.add_argument(
        "--data-root",
        help=(
            "ARAD-1K root. The RGB directory for --split is resolved from it "
            "(Test_RGB, Valid_RGB, Train_RGB, ...)."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        help=(
            "Use the packaged ARAD-1K scene order for this split, and pick the "
            "split's RGB directory when --data-root is given."
        ),
    )
    parser.add_argument(
        "--manifest",
        help="Scene order file; overrides the packaged --split manifest.",
    )
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument(
        "--rgb-normalization",
        choices=("scale_255", "per_image"),
        help=(
            "Override RGB preprocessing. By default this is recovered from "
            "trainer checkpoint metadata."
        ),
    )
    args = parser.parse_args(argv)

    if bool(args.rgb_dir) == bool(args.data_root):
        raise ValueError("Pass --rgb-dir or --data-root, not both.")
    if args.rgb_dir and args.split and args.manifest:
        raise ValueError("Pass --split or --manifest, not both.")
    if args.data_root and not args.split:
        raise ValueError(
            "--data-root needs --split to select the RGB directory."
        )

    rgb_dir = args.rgb_dir or resolve_arad_directory(
        args.data_root,
        args.split,
        kind="rgb",
    )
    # --manifest wins: with --data-root, --split still selects the directory.
    scene_ids = (
        None
        if args.manifest or not args.split
        else load_arad_manifest(args.split)
    )
    device = resolve_device(args.device)
    model, payload = build_model_from_checkpoint(
        args.checkpoint,
        preset=args.preset,
        map_location="cpu",
    )
    rgb_normalization = args.rgb_normalization or checkpoint_rgb_normalization(
        payload
    )
    dataset = RGBImageDataset(
        rgb_dir,
        scene_ids=scene_ids,
        manifest_path=args.manifest,
        rgb_normalization=rgb_normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    del payload
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
    _write_inference_report(
        output_dir,
        checkpoint=args.checkpoint,
        rgb_normalization=rgb_normalization,
        rgb_dir=rgb_dir,
        scene_ids=scene_ids,
    )
    print(f"saved {len(scene_ids)} cubes to {output_dir}")


def test_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the public ARAD-1K split and save NTIRE-format cubes."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help=(
            "ARAD manifest to evaluate. The reported 0.1468 result is on "
            "validation/ARAD-origin (the default); pass test for the public "
            "held-out split."
        ),
    )
    parser.add_argument("--manifest")
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument(
        "--metric-profile",
        choices=("source_arad_origin", "ntire_center", "legacy_full"),
        default="source_arad_origin",
        help=(
            "Named scoring protocol. source_arad_origin reproduces the "
            "full-frame denominator used by the reported SSTrans 0.1468 result."
        ),
    )
    parser.add_argument(
        "--crop-border",
        type=int,
        default=None,
        help=(
            "Override only the named profile's metric crop. Inference and "
            "export still use the complete frame."
        ),
    )
    parser.add_argument(
        "--rgb-normalization",
        choices=("scale_255", "per_image"),
        help=(
            "Override RGB preprocessing. By default this is recovered from "
            "trainer checkpoint metadata."
        ),
    )
    parser.add_argument(
        "--spectral-dir",
        action="append",
        metavar="DIR",
        help=(
            "Directory holding the ground-truth cubes, as a name under "
            "--data-root or an absolute path. Searched before the standard "
            "ARAD names; repeatable. Use this when the split's cubes live "
            "outside Test_Spec/Valid_spectral/Train_spectral."
        ),
    )
    parser.add_argument(
        "--require-targets",
        action="store_true",
        help=(
            "Fail instead of falling back to prediction-only export when the "
            "split ships no spectral cubes (ARAD-1K Test_Spec holds the MSFA "
            "'mosaic' payload, not ground truth)."
        ),
    )
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model, payload = build_model_from_checkpoint(
        args.checkpoint,
        preset=args.preset,
        map_location="cpu",
    )
    rgb_normalization = args.rgb_normalization or checkpoint_rgb_normalization(
        payload
    )
    dataset = ARAD1KDataset(
        args.data_root,
        split=args.split,
        manifest_path=args.manifest,
        augment=False,
        rgb_normalization=rgb_normalization,
        require_targets=args.require_targets,
        spectral_dirs=args.spectral_dir,
    )
    if dataset.unusable_targets:
        first = list(dataset.unusable_targets.items())[0]
        print(
            f"warning: {len(dataset.unusable_targets)}/{len(dataset.scene_ids)} "
            f"{args.split} scenes have no usable ground-truth cube "
            f"({first[0]}: {first[1]}); they are reconstructed but not scored."
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    del payload
    model.to(device)

    metric_profiles = {
        "source_arad_origin": {
            "crop_border": 0,
            "mrae_denominator": "source_additive",
            "mrae_epsilon": 1e-5,
            "psnr_clip": False,
        },
        "ntire_center": {
            "crop_border": 128,
            "mrae_denominator": "clamp_abs",
            "mrae_epsilon": 1e-6,
            "psnr_clip": True,
        },
        "legacy_full": {
            "crop_border": 0,
            "mrae_denominator": "clamp_abs",
            "mrae_epsilon": 1e-6,
            "psnr_clip": False,
        },
    }
    metric_protocol = dict(metric_profiles[args.metric_profile])
    if args.crop_border is not None:
        metric_protocol["crop_border"] = args.crop_border

    output_dir = Path(args.output_dir)
    if not dataset.scene_ids_with_targets:
        # ARAD-1K's public test split has no cubes at all: reconstruct and
        # export submission files instead of failing on the missing targets.
        exported = infer_loader(
            model,
            loader,
            device=device,
            output_dir=output_dir / "cubes",
            tile_size=args.tile_size,
            overlap=args.overlap,
            amp=args.amp,
            clip=args.clip,
        )
        _write_inference_report(
            output_dir,
            checkpoint=args.checkpoint,
            rgb_normalization=rgb_normalization,
            rgb_dir=dataset.rgb_root,
            scene_ids=exported,
        )
        print(
            f"no ground truth in split={args.split}; saved {len(exported)} "
            f"cubes to {output_dir / 'cubes'} without metrics"
        )
        return

    summary, rows = evaluate_loader(
        model,
        loader,
        device=device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        amp=args.amp,
        output_dir=output_dir / "cubes",
        clip=args.clip,
        **metric_protocol,
    )
    report_summary = {
        **summary,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "rgb_normalization": rgb_normalization,
        "tile_size": args.tile_size,
        "overlap": args.overlap if args.tile_size is not None else None,
        "amp": args.amp,
        "export_clip": args.clip,
        "metric_profile": args.metric_profile,
        **metric_protocol,
    }
    write_metric_reports(output_dir, report_summary, rows)
    print(
        f"test scenes={int(summary['count'])} "
        f"skipped={int(summary.get('skipped', 0.0))} "
        f"crop_border={int(summary['crop_border'])} "
        f"MRAE={summary['mrae']:.6f} RMSE={summary['rmse']:.6f} "
        f"PSNR={summary['psnr']:.4f} SAM={summary['sam']:.6f}"
    )


def _write_inference_report(
    output_dir: Path,
    *,
    checkpoint: str,
    rgb_normalization: str,
    rgb_dir: str | Path | None,
    scene_ids: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inference.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(checkpoint).resolve()),
                "rgb_dir": str(rgb_dir) if rgb_dir is not None else None,
                "rgb_normalization": rgb_normalization,
                "count": len(scene_ids),
                "scene_ids": list(scene_ids),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

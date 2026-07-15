#!/usr/bin/env python3
"""NTIRE-style evaluation of a trained CAS-HSI checkpoint on ARAD-1K.

Reports MRAE, PSNR, RMSE, SSIM (plus SAM and MAE) under both protocols:

    full   whole native-resolution frame        (NTIRE-style)
    crop   MST++ centre region, 128-px border   (the selection protocol)

Per-scene rows, per-band breakdown, and a bootstrap 95% CI over scenes -- 50 test
scenes is a small sample, and a mean without a spread invites over-reading a 0.003
MRAE difference between two runs.

    python test_cas_ntire.py --checkpoint experiments/.../checkpoints/best.pth \
                             --data_root /path/to/ARAD_1K --split test

The default split is ``test``. Predictions are NOT clamped (NTIRE convention);
``--clamp_eval`` matches spec 7.4's evaluation-only clamp and will report a
slightly better MRAE that is not comparable to unclamped numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cas_hsi import CASHSIConfig, CASHSI  # noqa: E402
from dataloader import EvalDataset  # noqa: E402
from evaluation import (  # noqa: E402
    METRICS_SOURCE,
    MST_CROP_BORDER,
    average_rows,
    evaluate_scene,
    forward_scene,
    format_metric_table,
)

try:
    from hsi_benchmark.metrics import bootstrap_confidence_interval  # type: ignore
except ImportError:  # pragma: no cover
    bootstrap_confidence_interval = None  # type: ignore


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[CASHSI, Dict[str, Any]]:
    """Rebuild the exact architecture the checkpoint was trained with."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config")
    if not model_config:
        raise RuntimeError(
            f"{checkpoint_path} carries no 'model_config'. Rebuilding the architecture "
            "would be guesswork; re-train with train_cas_hsi.py or add the config by hand."
        )

    config = CASHSIConfig.from_dict(model_config)
    model = CASHSI(config)

    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise RuntimeError(f"{checkpoint_path} has no model weights.")

    incompatible = model.load_state_dict(state, strict=True)
    del incompatible

    model.to(device).eval()
    return model, checkpoint


def save_mat(path: Path, cube: np.ndarray) -> None:
    """Write a prediction as a v7.3 .mat with a 'cube' variable (NTIRE submission format)."""
    try:
        import hdf5storage
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "--save_mat needs hdf5storage (pip install hdf5storage)."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    # NTIRE expects (bands, width, height) -- the transpose the loader undoes.
    hdf5storage.savemat(
        str(path), {"cube": np.transpose(cube, (0, 2, 1))}, format="7.3", store_python_metadata=True
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crop_border", type=int, default=MST_CROP_BORDER)
    parser.add_argument("--mrae_eps", type=float, default=1e-6)
    parser.add_argument("--clamp_eval", action="store_true", help="clamp predictions to [0,1] (spec 7.4)")
    parser.add_argument("--tile_size", type=int, default=0, help=">0 uses tiled inference (spec 8.7)")
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--rgb_norm", default="minmax", choices=["minmax", "div255"])
    parser.add_argument("--use_ema", action="store_true", help="evaluate the EMA weights if present")
    parser.add_argument("--save_mat", type=Path, default=None, help="directory for .mat predictions")
    parser.add_argument("--json", type=Path, default=None, help="write the full report here")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)

    if args.use_ema:
        ema_state = checkpoint.get("ema_state_dict")
        if not ema_state:
            print("ERROR: --use_ema requested but the checkpoint has no EMA weights.", file=sys.stderr)
            return 2
        model.load_state_dict(ema_state)
        model.eval()

    info = model.get_model_info()
    print("=" * 78)
    print(f"CAS-HSI {info['name']} ({info['backend']} backend)")
    print(f"  checkpoint    : {args.checkpoint}")
    print(f"  epoch         : {checkpoint.get('epoch', '?')}   "
          f"best val selection: {checkpoint.get('best_selection', float('nan')):.6f}")
    print(f"  parameters    : {info['total_parameters']:,}")
    print(f"  weights       : {'EMA' if args.use_ema else 'raw'}")
    print(f"  metrics source: {METRICS_SOURCE}")
    print(f"  clamping      : {'clamped to [0,1]' if args.clamp_eval else 'UNCLAMPED (NTIRE convention)'}")
    print("=" * 78)

    dataset = EvalDataset(args.data_root, split=args.split, rgb_norm=args.rgb_norm)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"\nEvaluating {len(dataset)} '{args.split}' scenes\n")

    rows: Dict[str, List[Dict[str, float]]] = {"full": [], "crop": []}
    per_scene: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for index, (rgb, hsi) in enumerate(tqdm(loader, desc="scenes")):
        stem = dataset.stems[index]
        rgb = rgb.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            prediction = forward_scene(model, rgb, tile_size=args.tile_size, overlap=args.overlap)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)

        scene = evaluate_scene(
            prediction, hsi,
            crop_border=args.crop_border,
            epsilon=args.mrae_eps,
            clamp=args.clamp_eval,
        )
        for protocol, values in scene.items():
            rows[protocol].append(values)

        record: Dict[str, Any] = {"scene": stem}
        for protocol, values in scene.items():
            record.update({f"{protocol}/{k}": v for k, v in values.items()})
        per_scene.append(record)

        if args.save_mat:
            cube = prediction.detach().float().cpu().squeeze(0).numpy()
            if args.clamp_eval:
                cube = np.clip(cube, 0.0, 1.0)
            save_mat(Path(args.save_mat) / f"{stem}.mat", cube)

    protocols = {name: average_rows(values) for name, values in rows.items() if values}

    print("\n" + "=" * 78)
    print(f"RESULTS -- {len(dataset)} scenes, split='{args.split}'")
    print("=" * 78)
    print(format_metric_table(protocols))

    if bootstrap_confidence_interval is not None:
        print("\n95% bootstrap CI over scenes (the mean of 50 scenes is a noisy estimate):")
        for protocol, values in rows.items():
            if not values:
                continue
            parts = []
            for key in ("mrae", "psnr", "ssim"):
                low, high = bootstrap_confidence_interval([row[key] for row in values])
                parts.append(f"{key.upper()} [{low:.4f}, {high:.4f}]")
            print(f"  {protocol:<6} " + "  ".join(parts))

    if latencies:
        print(
            f"\nInference: mean {1000 * np.mean(latencies):.1f} ms/scene, "
            f"median {1000 * np.median(latencies):.1f} ms  (device={device})"
        )

    worst = sorted(per_scene, key=lambda r: r.get("full/mrae", 0.0), reverse=True)[:5]
    print("\nWorst 5 scenes by full-frame MRAE:")
    for record in worst:
        print(f"  {record['scene']:<18} MRAE {record['full/mrae']:.4f}  PSNR {record['full/psnr']:.2f}")

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "scenes": len(dataset),
        "clamped": bool(args.clamp_eval),
        "metrics_source": METRICS_SOURCE,
        "weights": "ema" if args.use_ema else "raw",
        "model_info": info,
        "protocols": protocols,
        "per_scene": per_scene,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

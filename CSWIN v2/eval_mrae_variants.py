#!/usr/bin/env python3
"""Report MRAE four ways for a trained checkpoint, to separate metric
definition from real reconstruction error.

The 0.30-vs-0.16 gap against MST++ may be partly a metric-definition
artifact: this repo's ``compute_mrae`` divides by ``target.abs().clamp_min(1e-8)``
while the MST++ reference divides by the raw ``target``. Near-zero target
pixels (dark HSI bands/shadows) inflate the clamped variant. This script
runs ONE validation pass and reports MRAE under both denominators, on both
the full image and the MST++ center crop (226×256), so you can see exactly
how much of the gap is definition vs error.

Usage:
    python eval_mrae_variants.py \
        --checkpoint /path/to/best_model.pth \
        --data-dir /work3/paulgob/dataset \
        [--device cuda:0] [--limit 50] [--clamp-pred]

Read-only: loads weights and runs inference, writes nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hsi_model.models import NoiseRobustCSWinModel  # noqa: E402
from hsi_model.utils.data import create_training_datasets, mst_to_gan_batch  # noqa: E402


# MST++ center-crop dims for ARAD-1K (482×512 -> 226×256).
CROP_H, CROP_W = 226, 256


def center_crop(t: torch.Tensor, h: int = CROP_H, w: int = CROP_W) -> torch.Tensor:
    """Center-crop (B, C, H, W) to (B, C, h, w); pass through if already smaller."""
    _, _, H, W = t.shape
    if H < h or W < w:
        return t
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    return t[:, :, y0:y0 + h, x0:x0 + w]


def mrae_clamped_abs(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """This repo's compute_mrae: divide by |target| clamped to eps."""
    denom = target.abs().clamp_min(eps)
    return torch.mean(torch.abs(pred - target) / denom)


def mrae_raw_target(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MST++ reference: divide by raw target. May be inf if target has zeros."""
    return torch.mean(torch.abs(pred - target) / target)


def mrae_raw_target_safe(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Raw-target division but ignoring pixels where target < eps (a common
    MST++-compatible guard that drops near-zero denominators rather than
    clamping them)."""
    mask = target >= eps
    if mask.sum() == 0:
        return torch.tensor(float("nan"))
    return torch.mean(torch.abs(pred[mask] - target[mask]) / target[mask])


def load_model(checkpoint_path: str, device: torch.device) -> NoiseRobustCSWinModel:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ck.get("config")
    if config is None:
        raise KeyError(f"No 'config' in checkpoint {checkpoint_path}; cannot rebuild model.")
    state_dict = ck.get("state_dict") or ck.get("model_state_dict")
    if not state_dict:
        raise KeyError(f"No model state in {checkpoint_path}.")
    # Strip DDP prefix if present.
    if next(iter(state_dict)).startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model = NoiseRobustCSWinModel(config).to(device)
    # Generator-only checkpoints (train_generator.py) store BARE generator
    # keys; this wrapper expects 'generator.*'. Adapt instead of letting
    # strict=False silently load nothing and score random weights.
    model_keys = set(model.state_dict().keys())
    if not (set(state_dict) & model_keys):
        prefixed = {f"generator.{k}": v for k, v in state_dict.items()}
        if set(prefixed) & model_keys:
            print("[info] bare generator checkpoint detected; prefixing keys with 'generator.'")
            state_dict = prefixed
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) >= len(model_keys):
        raise RuntimeError(
            f"Checkpoint matched 0/{len(model_keys)} model keys - refusing to "
            "evaluate randomly initialized weights. Use a checkpoint produced "
            "by the GAN trainer, or load generator-only checkpoints through "
            "hsi_model.utils.inference.load_generator."
        )
    if missing:
        print(f"[warn] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    model.eval()
    print(f"Loaded checkpoint: iter={ck.get('iter', '?')}, "
          f"epoch={ck.get('epoch', '?')}, stored best_mrae={ck.get('best_mrae', '?')}")
    return model, config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--data-dir", required=True, help="ARAD-1K / MST++ dataset root")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="Max val scenes to evaluate (0 = all)")
    ap.add_argument("--clamp-pred", action="store_true",
                    help="Clamp predictions to [0,1] before scoring (off by default to match training-time val).")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device)

    # Build the validation dataset exactly as the trainers do.
    cfg = dict(config)
    cfg["data_dir"] = args.data_dir
    _, val_dataset = create_training_datasets(cfg, seed=int(cfg.get("seed", 42)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Accumulators: (denominator_variant, region) -> running sum, count.
    keys = [
        ("clamped_abs", "full"),
        ("clamped_abs", "crop"),
        ("raw_target", "full"),
        ("raw_target", "crop"),
        ("raw_safe_1e-3", "crop"),
    ]
    sums: Dict[Tuple[str, str], float] = {k: 0.0 for k in keys}
    counts: Dict[Tuple[str, str], int] = {k: 0 for k in keys}
    min_target = float("inf")
    n_scenes = 0

    with torch.no_grad():
        for idx, (bgr, hyper) in enumerate(val_loader):
            if args.limit and idx >= args.limit:
                break
            rgb, hsi = mst_to_gan_batch(bgr, hyper)
            rgb = rgb.to(device, non_blocking=True)
            hsi = hsi.to(device, non_blocking=True)

            pred = model.generator(rgb)
            if args.clamp_pred:
                pred = pred.clamp(0.0, 1.0)

            pred = pred.float()
            hsi = hsi.float()
            min_target = min(min_target, hsi.min().item())

            pred_crop = center_crop(pred)
            hsi_crop = center_crop(hsi)

            variants = {
                ("clamped_abs", "full"): mrae_clamped_abs(pred, hsi),
                ("clamped_abs", "crop"): mrae_clamped_abs(pred_crop, hsi_crop),
                ("raw_target", "full"): mrae_raw_target(pred, hsi),
                ("raw_target", "crop"): mrae_raw_target(pred_crop, hsi_crop),
                ("raw_safe_1e-3", "crop"): mrae_raw_target_safe(pred_crop, hsi_crop),
            }
            for k, v in variants.items():
                val = v.item()
                if val == val and val != float("inf"):  # skip nan/inf
                    sums[k] += val
                    counts[k] += 1
            n_scenes += 1

    print(f"\nEvaluated {n_scenes} validation scenes.")
    print(f"Minimum target value seen across all scenes: {min_target:.3e}")
    print(f"(Targets near 0 are what inflate the clamped-abs denominator.)\n")

    print(f"{'denominator':<16}{'region':<8}{'mean MRAE':>12}{'#finite':>10}")
    print("-" * 46)
    for k in keys:
        denom, region = k
        n = counts[k]
        mean = sums[k] / n if n else float("nan")
        print(f"{denom:<16}{region:<8}{mean:>12.4f}{n:>10}")

    print("\nInterpretation:")
    print("  - 'clamped_abs/crop' should reproduce your training-time val MRAE (~0.30).")
    print("  - 'raw_target/crop' is the MST++ reference definition.")
    print("  - If raw_target/crop is much lower (~0.16-0.20), the gap to MST++ is")
    print("    largely a denominator-definition artifact, not real reconstruction error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Unified inference / evaluation for HSIFusion v2.5.3 and SHARP v3.2.2 checkpoints.

Counterpart of unified_training.py; supersedes the SHARP-only sharp_inference.py.
The model family is auto-detected (checkpoint 'model' key for unified checkpoints,
state_dict key signature for legacy ones), and inputs are reflect-padded to a
multiple of 8 so the real ARAD-1K 482x512 frames run through both models.

Modes:
  evaluate     -- run the ARAD-1K validation split and report MRAE, RMSE, PSNR,
                  SAM (degrees), and SSIM per scene and averaged, on BOTH the
                  full frame and the MST++ selection crop (border 128 ->
                  226x256 on 482x512). Optional CSV/JSON dumps.
  reconstruct  -- reconstruct HSI cubes from RGB image(s) and save them as
                  MATLAB v7.3 .mat files with a (H, W, 31) float32 'cube'
                  (the MST++/NTIRE submission layout).

Legacy checkpoints (from the retired per-model scripts) are supported:
HSIFusion ones reload through HSIFusionNetV25LightningPro.from_pretrained;
SHARP ones through the same config fallbacks sharp_inference used
(tanh output, key_rbf_mode 'mean', no pass-5 modules).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from hsi_benchmark.metrics import summarize_metric_rows

from optimized_dataloader import OptimizedValDataset
from unified_training import (
    METRIC_KEYS,
    build_model,
    evaluate_scene,
    format_metric_table,
    forward_reconstruction,
    unwrap_model,
)


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def detect_model_family(checkpoint: Mapping[str, Any]) -> str:
    """'model' key for unified checkpoints, state_dict signature for legacy ones."""
    family = checkpoint.get("model")
    if isinstance(family, str) and family in ("hsifusion", "sharp"):
        return family
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    if isinstance(state, Mapping):
        keys = list(state.keys())
        if any(k.startswith(("encoder_stages.", "_orig_mod.encoder_stages.")) for k in keys):
            return "hsifusion"
        if any(k.startswith(("stages.", "_orig_mod.stages.")) for k in keys):
            return "sharp"
    raise ValueError(
        "Could not auto-detect the model family from the checkpoint; pass --model."
    )


def _load_legacy_sharp(checkpoint: Mapping[str, Any]) -> torch.nn.Module:
    """Rebuild a pre-unified SHARP checkpoint with the legacy architecture fallbacks."""
    from sharp_inference import _config_get

    config = checkpoint.get("config", {})
    kwargs = {
        "sparse_sparsity_ratio": _config_get(config, "sparse_sparsity_ratio", 0.9),
        "rbf_centers_per_head": _config_get(config, "rbf_centers_per_head", 32),
        "sparse_k_cap": _config_get(config, "sparse_k_cap", 1024),
        "sparse_block_size": _config_get(config, "sparse_block_size", 2048),
        "sparse_q_block_size": _config_get(config, "sparse_q_block_size", 1024),
        "sparse_window_size": _config_get(config, "sparse_window_size", 49),
        "sparse_max_tokens": _config_get(config, "sparse_max_tokens", 8192),
        "sparse_exact_topk_max_tokens": _config_get(config, "sparse_exact_topk_max_tokens", 1024),
        "sparse_landmark_tokens": _config_get(config, "sparse_landmark_tokens", 256),
        "max_global_tokens": _config_get(config, "max_global_tokens", None),
        "key_rbf_mode": _config_get(config, "key_rbf_mode", "mean"),
        "sparsemax_pad_value": _config_get(config, "sparsemax_pad_value", None),
        "output_activation": _config_get(config, "output_activation", "tanh"),
        "attn2_prenorm": _config_get(config, "attn2_prenorm", False),
        "spectral_head_rank": _config_get(config, "spectral_head_rank", 0),
    }
    model = build_model(
        "sharp", _config_get(config, "model_size", "base"), kwargs, compile_model=False
    )
    state = checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"]
    unwrap_model(model).load_state_dict(state)
    return model


def load_checkpoint_model(
    checkpoint_path: str,
    model_override: Optional[str] = None,
    model_size_override: Optional[str] = None,
    prefer_ema: bool = True,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """Build the right model for a unified or legacy checkpoint and load its weights."""
    checkpoint = _torch_load(checkpoint_path)
    family = model_override or detect_model_family(checkpoint)

    info: Dict[str, Any] = {"model": family, "checkpoint": str(checkpoint_path)}
    if checkpoint.get("unified_version"):
        size = model_size_override or checkpoint.get("model_size", "base")
        model = build_model(family, size, checkpoint.get("model_kwargs") or {}, False)
        state = (
            checkpoint.get("ema_model_state_dict")
            if prefer_ema and checkpoint.get("ema_model_state_dict")
            else checkpoint["model_state_dict"]
        )
        info["ema_weights"] = prefer_ema and bool(checkpoint.get("ema_model_state_dict"))
        info["model_size"] = size
        info["epoch"] = checkpoint.get("epoch")
        unwrap_model(model).load_state_dict(state)
    elif family == "hsifusion":
        # Legacy HSIFusion checkpoints know how to rebuild themselves.
        from hsifusion_v252_complete import HSIFusionNetV25LightningPro

        model = HSIFusionNetV25LightningPro.from_pretrained(checkpoint)
        info["ema_weights"] = False
        info["legacy"] = True
    else:
        model = _load_legacy_sharp(checkpoint)
        info["ema_weights"] = bool(checkpoint.get("ema_model_state_dict"))
        info["legacy"] = True

    model.eval()
    return model, info


# ============================================================================
# Evaluate mode
# ============================================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_root: str,
    device: torch.device,
    crop_border: int = 128,
    mrae_eps: float = 1e-6,
    out_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    dataset = OptimizedValDataset(data_root=data_root, memory_mode="standard")
    model = model.to(device)
    scene_rows: Dict[str, List[Dict[str, float]]] = {"full": [], "crop": []}
    per_scene_records: List[Dict[str, Any]] = []

    start = time.time()
    for idx in range(len(dataset)):
        rgb, hsi = dataset[idx]
        pred = forward_reconstruction(model, rgb.unsqueeze(0).to(device))
        rows = evaluate_scene(pred, hsi.unsqueeze(0), crop_border, mrae_eps)
        name = Path(dataset.hsi_files[idx]).stem
        for proto, row in rows.items():
            scene_rows[proto].append(row)
            per_scene_records.append({"scene": name, "protocol": proto, **row})
        crop = rows.get("crop", rows["full"])
        print(
            f"  [{idx + 1:>3}/{len(dataset)}] {name:<12} "
            + " ".join(f"{k}={crop[k]:.4f}" for k in METRIC_KEYS)
        )
    elapsed = time.time() - start

    summary: Dict[str, Dict[str, float]] = {}
    detailed: Dict[str, Dict[str, Dict[str, float]]] = {}
    for proto, rows in scene_rows.items():
        if not rows:
            continue
        stats = summarize_metric_rows(rows)
        summary[proto] = {key: stats[key]["mean"] for key in METRIC_KEYS}
        detailed[proto] = stats

    print(f"\nEvaluation summary ({len(dataset)} scenes, {elapsed:.1f}s):")
    print(format_metric_table(summary))
    for proto, stats in detailed.items():
        mrae = stats["mrae"]
        print(
            f"  {proto}: MRAE mean {mrae['mean']:.4f} +/- {mrae['std']:.4f} "
            f"(95% CI [{mrae['ci95_low']:.4f}, {mrae['ci95_high']:.4f}], n={mrae['count']})"
        )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "per_scene_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["scene", "protocol", *METRIC_KEYS])
            writer.writeheader()
            writer.writerows(per_scene_records)
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(detailed, f, indent=2)
        print(f"Wrote per_scene_metrics.csv and summary.json to {out_dir}")
    return summary


# ============================================================================
# Reconstruct mode
# ============================================================================

@torch.no_grad()
def reconstruct(
    model: torch.nn.Module,
    rgb_path: Path,
    out_dir: Path,
    device: torch.device,
) -> List[Path]:
    import cv2
    import h5py

    paths = (
        sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png") for p in rgb_path.glob(ext)
        )
        if rgb_path.is_dir()
        else [rgb_path]
    )
    if not paths:
        raise FileNotFoundError(f"No RGB images found at {rgb_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    written: List[Path] = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read RGB image {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        cube = forward_reconstruction(model, tensor)[0].float().cpu().numpy()
        # MST++/NTIRE submission layout: (H, W, 31) float32 under 'cube'.
        cube_hwc = np.ascontiguousarray(cube.transpose(1, 2, 0))
        target = out_dir / f"{path.stem}.mat"
        with h5py.File(target, "w") as f:
            f.create_dataset("cube", data=cube_hwc, dtype="float32")
        written.append(target)
        print(f"  {path.name} -> {target} cube{cube_hwc.shape}")
    return written


# ============================================================================
# CLI
# ============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Unified HSIFusion/SHARP inference and NTIRE-style evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=str, help="Path to a .pth checkpoint")
    parser.add_argument("--model", type=str, default=None, choices=["hsifusion", "sharp"],
                        help="Override auto-detected model family")
    parser.add_argument("--model_size", type=str, default=None,
                        help="Override checkpoint model size (unified checkpoints)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="ARAD-1K root; enables evaluate mode on the valid split")
    parser.add_argument("--rgb", type=str, default=None,
                        help="RGB image or directory; enables reconstruct mode")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to write metrics dumps / reconstructed cubes")
    parser.add_argument("--crop_border", type=int, default=128,
                        help="MST++ selection crop border for evaluate mode")
    parser.add_argument("--no_ema", action="store_true",
                        help="Use raw weights even when EMA weights are stored")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    if not args.data_root and not args.rgb:
        parser.error("Provide --data_root (evaluate) and/or --rgb (reconstruct).")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model, info = load_checkpoint_model(
        args.checkpoint,
        model_override=args.model,
        model_size_override=args.model_size,
        prefer_ema=not args.no_ema,
    )
    params = sum(p.numel() for p in model.parameters())
    print("Loaded checkpoint:")
    for key, value in {**info, "parameters": f"{params / 1e6:.2f}M", "device": str(device)}.items():
        print(f"  {key:<12}: {value}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if args.data_root:
        evaluate(
            model, args.data_root, device,
            crop_border=args.crop_border, out_dir=out_dir,
        )
    if args.rgb:
        reconstruct(model, Path(args.rgb), out_dir or Path("./reconstructed"), device)


if __name__ == "__main__":
    main()

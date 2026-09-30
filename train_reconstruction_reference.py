#!/usr/bin/env python3
"""One reference data, optimization, and evaluation loop for the reconstruction models."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diagnose_reconstruction import ROOT, build_model, file_hash
from hsi_benchmark.metrics import mrae_breakdown, relative_absolute_error
from hsi_benchmark.training_health import TrainingHealthMonitor


@torch.inference_mode()
def validate_reference(model, loader, *, device, border=128):
    """Full-image inference, raw exact MRAE; paired border crop only for scoring."""
    was_training = model.training
    model.eval()
    scenes = []
    try:
        for index, (rgb, target) in enumerate(loader):
            rgb, target = rgb.to(device), target.to(device)
            if getattr(model, '_requires_reconstruction_padding', False):
                from unified_training import forward_reconstruction
                pred = forward_reconstruction(model, rgb)
            else:
                pred = model(rgb)
            if border < 0 or min(target.shape[-2:]) <= 2 * border:
                raise ValueError("Scoring border leaves an empty validation region")
            if border:
                pred = pred[..., border:-border, border:-border]
                target = target[..., border:-border, border:-border]
            result = mrae_breakdown(pred, target)
            result["scene_index"] = index
            refs = getattr(getattr(loader, "dataset", None), "_scene_refs", [])
            if index < len(refs):
                result["scene_id"] = refs[index].rgb_path.stem
            scenes.append(result)
        if not scenes:
            raise RuntimeError("Reference validation has no scenes")
        return {"raw_mrae": float(np.mean([item["raw_mrae"] for item in scenes])),
                "clamped_mrae": float(np.mean([item["clamped_mrae"] for item in scenes])),
                "scenes": scenes}
    finally:
        model.train(was_training)


def atomic_checkpoint(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mswr", "cswin", "mstpp", "hsifusion", "sharp"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mst-root", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--health-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--crop-border", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exclude-scene", action="append", default=[])
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.patch_size, args.stride, args.validate_every, args.health_every) < 1:
        parser.error("steps, geometry, batch size, and intervals must be positive")
    if args.workers < 0 or args.crop_border < 0 or not (0 < args.min_lr <= args.lr < float("inf")):
        parser.error("Invalid worker count, border, or learning rates")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Use a new output directory for each experiment; existing manifests are never overwritten")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for mechanical checks")
    sys.path.insert(0, str(ROOT / "CSWIN v2" / "src"))
    from hsi_model.utils.data.mst_dataset import MST_TrainDataset, MST_ValidDataset
    from hsi_model.utils.data import make_worker_init_fn
    # Lazy keeps the supplied HSI in FP32 and normalizes each entire RGB scene
    # before cropping. Strict files + explicit exclusions prevent silent drift.
    dataset_options = dict(data_root=str(args.data_root), bgr2rgb=True, memory_mode="lazy",
                           strict_files=True, excluded_scene_stems=args.exclude_scene)
    train = MST_TrainDataset(crop_size=args.patch_size, stride=args.stride, arg=True, **dataset_options)
    valid = MST_ValidDataset(**dataset_options)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
                        worker_init_fn=make_worker_init_fn(args.seed),
                        generator=torch.Generator().manual_seed(args.seed),
                        persistent_workers=args.workers > 0)
    if len(loader) == 0:
        raise ValueError("Training split does not contain a complete optimizer batch")
    val_loader = DataLoader(valid, batch_size=1, num_workers=0)
    model, model_config = build_model(args.model, args.config, args.mst_root, diagnostic=False)
    model.to(args.device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.min_lr)
    split_files = [args.data_root / "split_txt" / "train_list.txt"]
    split_files.append(next(path for path in (args.data_root / "split_txt" / "valid_list.txt",
                                             args.data_root / "split_txt" / "val_list.txt") if path.exists()))
    policy = {"optimizer": "Adam", "epsilon": 0.0, "amp": False, "ema": False,
              "weight_decay": 0.0, "gradient_clip": None, "warmup_steps": 0,
              "inference": "full_image", "crop_border": args.crop_border,
              "checkpoint_source": "raw", "rgb_normalization": "whole_scene_minmax",
              "target_normalization": "none", "excluded_scenes": args.exclude_scene}
    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_config": model_config, "protocol": policy, "torch_version": str(torch.__version__),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "working_tree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()),
        "config_sha256": file_hash(args.config) if args.config else None,
        "split_sha256": {str(path): file_hash(path) for path in split_files},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    best = float("inf")
    monitor = TrainingHealthMonitor(model)
    iterator = iter(loader)
    try:
        with (args.output / "steps.jsonl").open("w", encoding="utf-8") as log:
            for step in range(1, args.steps + 1):
                try:
                    rgb, target = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    rgb, target = next(iterator)
                if hasattr(model, "set_iteration"):
                    model.set_iteration(step - 1)
                optimizer.zero_grad(set_to_none=True)
                sampled = step == 1 or step % args.health_every == 0
                monitor.enabled = sampled
                pred = model(rgb.to(args.device))
                loss = relative_absolute_error(pred, target.to(args.device)).mean()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                if sampled:
                    monitor.before_step()
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                record = {"step": step, "lr": lr, "train_exact_mrae": float(loss.detach().item()),
                          "pre_clip_gradient_norm": float(grad_norm.item())}
                if sampled:
                    record["health"] = monitor.after_step()
                if step % args.validate_every == 0 or step == args.steps:
                    monitor.enabled = True
                    result = validate_reference(model, val_loader, device=args.device, border=args.crop_border)
                    record["validation"] = result
                    record["validation_operators"] = {name: value for name, value in monitor.activations.items()
                                                       if "attention_operator" in value}
                    is_best = result["raw_mrae"] < best
                    best = min(best, result["raw_mrae"])
                    payload = {"state_dict": model.state_dict(), "model_config": model_config,
                               "config": {**model_config, "objective": "exact_mrae"},
                               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                               "iter": step, "best_mrae": best, "ema_applied": False, "protocol": policy}
                    if args.model in {'hsifusion', 'sharp'}:
                        recipe = json.loads(args.config.read_text(encoding='utf-8'))
                        payload.update(unified_version=2, checkpoint_kind='reference', model=args.model,
                                       model_size=recipe.get('model_size', 'base'),
                                       model_state_dict=payload['state_dict'], resolved_model_config=model_config,
                                       optimizer_step=step,
                                       run_manifest={'code_revision': manifest['commit'],
                                                     'data': {'rgb_normalization': 'scene_minmax',
                                                              'strict_files': True, 'exclude_samples': args.exclude_scene},
                                                     'protocol': {'selection_epsilon': 0., 'val_crop_border': args.crop_border}})
                    atomic_checkpoint(payload, args.output / "latest.pth")
                    if is_best:
                        atomic_checkpoint(payload, args.output / "best.pth")
                    print(f"step={step} train={record['train_exact_mrae']:.6f} val_exact_raw={result['raw_mrae']:.6f}", flush=True)
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
    finally:
        monitor.close()
        train.close()
        valid.close()


if __name__ == "__main__":
    main()

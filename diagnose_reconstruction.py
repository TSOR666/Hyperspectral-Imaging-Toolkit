#!/usr/bin/env python3
"""Overfit fixed RGB/HSI patches with one FP32 exact-MRAE implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import fields, asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from hsi_benchmark.metrics import mrae_breakdown, relative_absolute_error
from hsi_benchmark.training_health import TrainingHealthMonitor

ROOT = Path(__file__).resolve().parent


def build_model(kind: str, config_path: Path | None, mst_root: Path | None = None, *, diagnostic=True):
    if kind == "mstpp":
        if mst_root is None:
            raise ValueError("--mst-root is required for the official MST++ control")
        from hsi_benchmark.models import _build_mst_zoo
        model, _, _ = _build_mst_zoo(mst_root, "mst_plus_plus")
        return model, {"architecture": "official MST_Plus_Plus", "mst_root": str(mst_root)}
    if config_path is None:
        raise ValueError("--config is required for the local models")
    if kind in {'hsifusion', 'sharp'}:
        sys.path.insert(0, str(ROOT / 'HSIFUSION&SHARP'))
        from unified_training import build_model as build_fusion_model, unwrap_model
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if config.get('model', kind) != kind:
            raise ValueError('Recipe model family differs from --model')
        kwargs = dict(config.get('model_kwargs', {}))
        if diagnostic:
            if kind == 'hsifusion':
                kwargs.update(dropout=0.0, drop_path=0.0, use_moe=False, estimate_uncertainty=False)
            else:
                kwargs['drop_path_rate'] = 0.0
        model = build_fusion_model(kind, config.get('model_size', 'base'), kwargs, False)
        model._requires_reconstruction_padding = True
        return model, json.loads(json.dumps(asdict(unwrap_model(model).config), default=str))
    if kind == "mswr":
        sys.path.insert(0, str(ROOT / "mswr_v2"))
        from model.mswr_net_v212 import IntegratedMSWRNet, MSWRDualConfig
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        sizes = {"tiny": (32, 2, 4, 4, 32), "small": (48, 3, 6, 8, 49),
                 "base": (64, 3, 8, 8, 64), "large": (96, 4, 12, 12, 128)}
        dimensions = dict(zip(("base_channels", "num_stages", "num_heads", "window_size", "num_landmarks"),
                              sizes[config.get("model_size", "base")]))
        names = {field.name for field in fields(MSWRDualConfig)}
        architecture = {key: value for key, value in config.items() if key in names}
        architecture.update(dimensions)
        architecture.update(performance_monitoring=False, compile_model=False, mixed_precision=False)
        if diagnostic:
            architecture.update(dropout=0.0, attention_dropout=0.0, drop_path=0.0)
        resolved = MSWRDualConfig(**architecture)
        return IntegratedMSWRNet(resolved), resolved.to_dict()
    sys.path.insert(0, str(ROOT / "CSWIN v2" / "src"))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
    with initialize_config_dir(config_dir=str(config_path.resolve().parent), version_base=None):
        config = OmegaConf.to_container(compose(config_name=config_path.stem), resolve=True)
    config.update(mixed_precision=False, spectral_attention_sanitize_nonfinite=False)
    model = NoiseRobustCSWinGenerator(config)
    # Fix update-count-dependent activation behavior throughout the probe.
    if diagnostic:
        model.set_iteration(0)
    return model, config


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_probe(model, rgb, target, *, steps=1000, lr=4e-4, log_every=100):
    """Same fixed batch, raw outputs, Adam, no AMP/EMA/clipping/weight decay."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    monitor = TrainingHealthMonitor(model)
    history = []
    try:
        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            sampled = step % log_every == 0 or step == steps
            monitor.enabled = sampled
            if getattr(model, '_requires_reconstruction_padding', False):
                from unified_training import forward_reconstruction
                output = forward_reconstruction(model, rgb)
            else:
                output = model(rgb)
            loss = relative_absolute_error(output, target).mean()
            record = {"step": step, "mrae": float(loss.detach().item())}
            if sampled:
                record["metrics"] = mrae_breakdown(output.detach(), target)
            if step < steps:
                loss.backward()
                # max_norm=inf only measures; it does not clip finite gradients.
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                if sampled:
                    monitor.before_step()
                optimizer.step()
                if sampled:
                    record["pre_clip_gradient_norm"] = float(grad_norm.item())
                    record["health"] = monitor.after_step()
            if sampled:
                history.append(record)
                print(f"step={step} exact_mrae={record['mrae']:.6f}", flush=True)
    finally:
        monitor.close()
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mswr", "cswin", "mstpp", "hsifusion", "sharp"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mst-root", type=Path)
    parser.add_argument("--patches", type=Path, required=True,
                        help="NPZ with fixed normalized rgb[N,3,H,W] and target[N,C,H,W]; recommended N=4..8")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.log_every < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("steps, log-every, and lr must be positive")
    torch.manual_seed(args.seed)
    with np.load(args.patches, allow_pickle=False) as patches:
        rgb = torch.as_tensor(patches["rgb"], dtype=torch.float32, device=args.device)
        target = torch.as_tensor(patches["target"], dtype=torch.float32, device=args.device)
    if rgb.ndim != 4 or target.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("Expected rgb[N,3,H,W] and target[N,C,H,W]")
    if rgb.shape[0] != target.shape[0] or rgb.shape[-2:] != target.shape[-2:]:
        raise ValueError("RGB and HSI patches must be paired")
    if not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
        raise ValueError("RGB must be finite and already normalized over the source image to [0,1]")
    relative_absolute_error(target, target)  # validate exact-objective prerequisites
    model, config = build_model(args.model, args.config, args.mst_root)
    model.to(args.device)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {
        "model": args.model, "resolved_config": config, "commit": commit,
        "working_tree_dirty": bool(dirty), "seed": args.seed,
        "patches_sha256": file_hash(args.patches),
        "config_sha256": file_hash(args.config) if args.config else None,
        "split_sha256": {str(path): file_hash(path) for path in args.split_file},
        "protocol": {"optimizer": "Adam", "lr": args.lr, "steps": args.steps,
                     "epsilon": 0.0, "amp": False, "ema": False, "weight_decay": 0.0,
                     "gradient_clipping": False, "augmentation": False, "checkpoint_source": "raw"},
        "history": run_probe(model, rgb, target, steps=args.steps, lr=args.lr, log_every=args.log_every),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

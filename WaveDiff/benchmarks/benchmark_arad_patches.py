"""Benchmark WaveDiff on aligned RGB/HSI patches from an ARAD-1K tree."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


WAVEDIFF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WAVEDIFF_ROOT))

from inference import evaluate_metrics, load_hsi_ground_truth, load_model, run_inference


def _load_rgb(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _sample_patches(data_dir, patch_size, sample_count, seed):
    rgb_dir = data_dir / "RGB"
    hsi_dir = data_dir / "HSI"
    files = sorted(
        path
        for path in rgb_dir.glob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not files:
        raise FileNotFoundError(f"No RGB images found in {rgb_dir}")

    generator = torch.Generator().manual_seed(seed)
    cache = {}
    patches = []
    attempts = 0
    while len(patches) < sample_count and attempts < sample_count * 10:
        attempts += 1
        index = int(torch.randint(len(files), (1,), generator=generator))
        rgb_path = files[index]
        if rgb_path not in cache:
            rgb = _load_rgb(rgb_path)
            hsi = load_hsi_ground_truth(hsi_dir / rgb_path.stem)
            if hsi is None:
                continue
            cache[rgb_path] = (rgb, hsi.squeeze(0))
        rgb, hsi = cache[rgb_path]
        height = min(rgb.shape[-2], hsi.shape[-2])
        width = min(rgb.shape[-1], hsi.shape[-1])
        if height < patch_size or width < patch_size:
            continue
        top = int(
            torch.randint(
                height - patch_size + 1,
                (1,),
                generator=generator,
            )
        )
        left = int(
            torch.randint(
                width - patch_size + 1,
                (1,),
                generator=generator,
            )
        )
        rgb_patch = rgb[:, top:top + patch_size, left:left + patch_size]
        hsi_patch = hsi[:, top:top + patch_size, left:left + patch_size]
        patches.append((rgb_patch * 2.0 - 1.0, hsi_patch))
    if len(patches) < sample_count:
        raise RuntimeError(
            f"Only found {len(patches)} valid aligned patches; "
            f"requested {sample_count}"
        )
    return patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument(
        "--latent_mode",
        choices=["direct", "diffusion"],
        default="direct",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.data_dir.exists() or not args.checkpoint.exists():
        print(json.dumps({
            "status": "skipped",
            "reason": "ARAD data directory or checkpoint does not exist",
            "data_dir": str(args.data_dir),
            "checkpoint": str(args.checkpoint),
        }, indent=2))
        return
    if args.patch_size % 4:
        raise ValueError("patch_size must be divisible by 4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)

    load_start = time.perf_counter()
    patches = _sample_patches(
        args.data_dir,
        args.patch_size,
        args.samples,
        args.seed,
    )
    load_seconds = time.perf_counter() - load_start

    predictions = []
    targets = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for offset in range(0, len(patches), args.batch_size):
        batch = patches[offset:offset + args.batch_size]
        rgb = torch.stack([item[0] for item in batch])
        target = torch.stack([item[1] for item in batch])
        prediction, _ = run_inference(
            model,
            rgb,
            device,
            sampling_steps=args.sampling_steps,
            latent_mode=args.latent_mode,
        )
        predictions.append(prediction.cpu())
        targets.append(target)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - start

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    metrics = evaluate_metrics(prediction, target, config=config)
    result = {
        "status": "ok",
        "device": str(device),
        "samples": len(patches),
        "patch_size": args.patch_size,
        "latent_mode": args.latent_mode,
        "decode_patches_per_second": len(patches) / load_seconds,
        "inference_ms_per_patch": inference_seconds * 1000.0 / len(patches),
        "throughput_patches_per_second": len(patches) / inference_seconds,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        ),
        **metrics,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

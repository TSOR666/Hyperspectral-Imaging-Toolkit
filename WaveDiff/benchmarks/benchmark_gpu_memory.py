"""Measure WaveDiff CUDA latency, throughput, and peak allocated memory."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


WAVEDIFF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WAVEDIFF_ROOT))

from inference import load_model, run_inference
from models.base_model import HSILatentDiffusionModel
from train import combine_weighted_losses


def _synchronize(device):
    torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument(
        "--latent_mode",
        choices=["direct", "diffusion"],
        default="direct",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({
            "status": "skipped",
            "reason": "CUDA is not available",
        }, indent=2))
        return

    device = torch.device("cuda")
    if args.checkpoint:
        model, _ = load_model(args.checkpoint, device)
    else:
        model = HSILatentDiffusionModel(
            norm_type="group",
            cross_attention_mode="channel",
            conditional_residual_diffusion=True,
        ).to(device)

    rgb = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )
    hsi = torch.randn(
        args.batch_size,
        31,
        args.image_size,
        args.image_size,
        device=device,
    )

    model.eval()
    for _ in range(args.warmup):
        run_inference(
            model,
            rgb,
            device,
            sampling_steps=args.sampling_steps,
            latent_mode=args.latent_mode,
        )
    _synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.iterations):
        run_inference(
            model,
            rgb,
            device,
            sampling_steps=args.sampling_steps,
            latent_mode=args.latent_mode,
        )
    _synchronize(device)
    inference_seconds = time.perf_counter() - start
    inference_peak = torch.cuda.max_memory_allocated(device) / 2**20

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(rgb, hsi_target=hsi)
        losses = model.calculate_losses(outputs, rgb, hsi)
        combine_weighted_losses(losses, {}).backward()
        optimizer.step()
    _synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(rgb, hsi_target=hsi)
        losses = model.calculate_losses(outputs, rgb, hsi)
        combine_weighted_losses(losses, {}).backward()
        optimizer.step()
    _synchronize(device)
    train_seconds = time.perf_counter() - start

    print(json.dumps({
        "status": "ok",
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "inference_ms_per_batch": inference_seconds * 1000 / args.iterations,
        "inference_images_per_second": (
            args.batch_size * args.iterations / inference_seconds
        ),
        "inference_peak_memory_mib": inference_peak,
        "train_ms_per_step": train_seconds * 1000 / args.iterations,
        "train_images_per_second": (
            args.batch_size * args.iterations / train_seconds
        ),
        "train_peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
        ),
    }, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import torch

from hsiformer import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a HSIFormer forward smoke test.")
    parser.add_argument(
        "--preset",
        default="legacy",
        choices=(
            "legacy",
            "ablation_no_rpe",
            "corrected_rpe",
            "optimized_candidate",
            "recommended_retrain",
        ),
    )
    parser.add_argument("--size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(
        args.preset,
        hidden_dim=8,
        input_resolution=(args.size, args.size),
        n_blocks=(1,),
        bottle_depth=1,
        n_refine=1,
        patch_size=2,
        use_checkpoint=False,
    ).eval()
    inputs = torch.rand(1, 3, args.size, args.size)
    with torch.inference_mode():
        outputs = model(inputs)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"preset={args.preset}")
    print(f"input={tuple(inputs.shape)} output={tuple(outputs.shape)}")
    print(f"parameters={parameters:,}")


if __name__ == "__main__":
    main()

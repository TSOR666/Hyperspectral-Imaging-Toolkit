from __future__ import annotations

import argparse

from hsiformer import ARAD1KDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an ARAD-1K directory and print one sample."
    )
    parser.add_argument("root")
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--crop-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ARAD1KDataset(
        args.root,
        split=args.split,
        crop_size=args.crop_size,
        random_crop=False,
        augment=False,
        # Diagnostics should report unusable ground truth, not fail on it.
        require_targets=False,
        probe_targets=True,
    )
    sample = dataset[0]
    print(f"split={args.split} samples={len(dataset)}")
    print(f"rgb_dir={dataset.rgb_root} spectral_dir={dataset.spectral_root}")
    print(f"scene={sample['scene_id']}")
    print(f"cond={tuple(sample['cond'].shape)}")
    label = sample.get("label")
    print(f"label={tuple(label.shape) if label is not None else 'unavailable'}")

    if dataset.unusable_targets:
        print(
            f"scenes without ground truth: {len(dataset.unusable_targets)}/"
            f"{len(dataset.scene_ids)}"
        )
        for scene_id, reason in list(dataset.unusable_targets.items())[:3]:
            print(f"  {scene_id}: {reason}")


if __name__ == "__main__":
    main()

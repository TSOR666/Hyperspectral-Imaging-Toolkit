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
    )
    sample = dataset[0]
    print(f"split={args.split} samples={len(dataset)}")
    print(f"scene={sample['scene_id']}")
    print(f"cond={tuple(sample['cond'].shape)}")
    print(f"label={tuple(sample['label'].shape)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify (or, if needed, materialize) the ARAD-1K 900 / 50 / 50 splits.

This project assumes the **900 / 50 / 50** protocol:

    train_list.txt   900 scenes
    valid_list.txt    50 scenes   -> checkpoint selection
    test_list.txt     50 scenes   -> reported once, at the end

The default mode is ``verify``: it reads the split lists you already have, checks
that every scene resolves to both an RGB image and a ground-truth cube, and -- the
part that actually matters -- checks that the three lists are **disjoint**. A single
scene leaking from train into test silently converts a held-out number into a
training number, and nothing downstream would ever tell you.

Other modes exist for datasets that do not already ship a ground-truthed test split
(the public NTIRE 2022 release does not: it publishes 950 GT scenes, 900 + 50, and
withholds the challenge test cubes). They are not needed here.

    python prepare_splits.py --data_root /path/to/ARAD_1K
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_RGB_SUFFIXES = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")

# Where each split's files may live. First hit wins; the fallbacks let a dataset
# keep every scene in Train_RGB/Train_Spec and separate the splits purely by list.
_RGB_DIRS: Dict[str, Tuple[str, ...]] = {
    "train": ("Train_RGB",),
    "valid": ("Valid_RGB", "Validation_RGB", "Val_RGB", "Train_RGB"),
    "test": ("Test_RGB", "Valid_RGB", "Validation_RGB", "Train_RGB"),
}
_SPEC_DIRS: Dict[str, Tuple[str, ...]] = {
    "train": ("Train_Spec",),
    "valid": ("Valid_Spec", "Validation_Spec", "Val_Spec", "Train_Spec"),
    "test": ("Test_Spec", "Valid_Spec", "Validation_Spec", "Train_Spec"),
}
_LIST_NAMES: Dict[str, Tuple[str, ...]] = {
    "train": ("train_list.txt",),
    "valid": ("valid_list.txt", "val_list.txt"),
    "test": ("test_list.txt",),
}

EXPECTED = {"train": 900, "valid": 50, "test": 50}


def _read_list(split_dir: Path, split: str) -> Tuple[Path | None, List[str]]:
    for name in _LIST_NAMES[split]:
        path = split_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return path, [Path(line.strip()).stem for line in handle if line.strip()]
    return None, []


def _write_list(path: Path, stems: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(stems) + "\n", encoding="utf-8")
    print(f"  wrote {path.name:<16} {len(stems):>4} scenes")


def _resolve(root: Path, stem: str, split: str) -> Tuple[bool, bool]:
    """(has_rgb, has_gt) for one scene, honouring the directory fallbacks."""
    has_rgb = any(
        (root / directory / f"{stem}{suffix}").exists()
        for directory in _RGB_DIRS[split]
        for suffix in _RGB_SUFFIXES
    )
    has_gt = any(
        (root / directory / f"{stem}.mat").exists() for directory in _SPEC_DIRS[split]
    )
    return has_rgb, has_gt


def _scene_stems(rgb_dir: Path, spec_dir: Path) -> List[str]:
    if not rgb_dir.is_dir():
        return []
    rgb: set[str] = set()
    for suffix in _RGB_SUFFIXES:
        rgb.update(path.stem for path in rgb_dir.glob(f"*{suffix}"))
    spec = {path.stem for path in spec_dir.glob("*.mat")} if spec_dir.is_dir() else set()
    return sorted(rgb & spec)


def verify(root: Path, strict_counts: bool) -> int:
    split_dir = root / "split_txt"
    print(f"ARAD-1K root : {root}")
    print(f"split_txt    : {split_dir}\n")

    lists: Dict[str, List[str]] = {}
    problems: List[str] = []

    for split in ("train", "valid", "test"):
        path, stems = _read_list(split_dir, split)
        if path is None:
            problems.append(
                f"{split}: no list file found (looked for {', '.join(_LIST_NAMES[split])}). "
                f"Use --mode holdout to create one."
            )
            lists[split] = []
            continue

        lists[split] = stems
        missing_rgb = [s for s in stems if not _resolve(root, s, split)[0]]
        missing_gt = [s for s in stems if not _resolve(root, s, split)[1]]

        status = "ok"
        if missing_rgb or missing_gt:
            status = "INCOMPLETE"
        print(
            f"  {split:<6} {path.name:<16} {len(stems):>4} scenes  "
            f"[{status}]"
            + (f"  missing RGB: {len(missing_rgb)}" if missing_rgb else "")
            + (f"  missing GT: {len(missing_gt)}" if missing_gt else "")
        )
        if missing_rgb:
            problems.append(f"{split}: {len(missing_rgb)} scene(s) have no RGB, e.g. {missing_rgb[:3]}")
        if missing_gt:
            problems.append(
                f"{split}: {len(missing_gt)} scene(s) have no ground-truth cube, e.g. {missing_gt[:3]}. "
                f"Searched {list(_SPEC_DIRS[split])}."
            )

        expected = EXPECTED[split]
        if len(stems) != expected:
            message = f"{split}: expected {expected} scenes for the 900/50/50 protocol, found {len(stems)}"
            if strict_counts:
                problems.append(message)
            else:
                print(f"         NOTE: {message}")

    # The check that actually protects the reported number.
    print("\nLeakage check (splits must be disjoint):")
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = sorted(set(lists[a]) & set(lists[b]))
        if overlap:
            print(f"  {a} n {b}: {len(overlap)} OVERLAPPING scene(s)  <-- LEAK")
            problems.append(
                f"{a} and {b} share {len(overlap)} scene(s) (e.g. {overlap[:3]}). "
                f"Any metric on {b} is contaminated by {a}."
            )
        else:
            print(f"  {a} n {b}: disjoint")

    for split in ("train", "valid", "test"):
        duplicates = len(lists[split]) - len(set(lists[split]))
        if duplicates:
            problems.append(f"{split}: {duplicates} duplicate entrie(s) in the list file")

    protocol = split_dir / "PROTOCOL.txt"
    if split_dir.is_dir():
        protocol.write_text(
            "\n".join(
                [
                    "mode: verify (pre-existing 900/50/50 split)",
                    f"train: {len(lists['train'])} scenes",
                    f"valid: {len(lists['valid'])} scenes  (checkpoint selection)",
                    f"test:  {len(lists['test'])} scenes  (reported once, at the end)",
                    "",
                    "Splits verified disjoint." if not problems else "PROBLEMS FOUND -- see below:",
                    *problems,
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nFix these before trusting any reported metric.")
        return 1

    print("\nOK: 900/50/50 split verified, disjoint, and fully ground-truthed.")
    return 0


def holdout(root: Path, test_size: int, seed: int) -> int:
    """Carve a held-out test split out of the training scenes (for datasets lacking one)."""
    split_dir = root / "split_txt"
    all_gt = _scene_stems(root / "Train_RGB", root / "Train_Spec")
    if not all_gt:
        print(f"ERROR: no matched RGB+GT pairs under {root}/Train_RGB and {root}/Train_Spec.", file=sys.stderr)
        return 2

    _, train = _read_list(split_dir, "train")
    _, valid = _read_list(split_dir, "valid")
    if not train or not valid:
        shuffled = list(all_gt)
        random.Random(seed).shuffle(shuffled)
        valid = sorted(shuffled[:50])
        train = sorted(shuffled[50:])

    pool = sorted(train)
    random.Random(seed).shuffle(pool)
    test = sorted(pool[:test_size])
    train = sorted(pool[test_size:])

    print(f"Carving a {test_size}-scene held-out test split (seed={seed}):")
    _write_list(split_dir / "train_list.txt", train)
    _write_list(split_dir / "valid_list.txt", valid)
    _write_list(split_dir / "test_list.txt", test)
    print(
        f"\nNOTE: train is now {len(train)} scenes, not 900 -- the test scenes came out of it.\n"
        "      Re-run with --mode verify to confirm."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_root", required=True, help="ARAD-1K root (holds Train_RGB/ etc.)")
    parser.add_argument(
        "--mode",
        default="verify",
        choices=["verify", "holdout"],
        help="verify: check the existing 900/50/50 lists (default). "
             "holdout: carve a test split out of train (only for datasets without one).",
    )
    parser.add_argument("--test_size", type=int, default=50, help="held-out test scenes (holdout mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict_counts",
        action="store_true",
        help="fail if the split sizes are not exactly 900/50/50",
    )
    args = parser.parse_args(argv)

    root = Path(args.data_root)
    if not root.is_dir():
        print(f"ERROR: data_root does not exist: {root}", file=sys.stderr)
        return 2

    if args.mode == "holdout":
        return holdout(root, args.test_size, args.seed)
    return verify(root, args.strict_counts)


if __name__ == "__main__":
    raise SystemExit(main())
